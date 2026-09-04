"""A second geometry generator: construct microsolvated clusters, don't sample them.

`n_sweep.py` explores by thermal sampling -- gas-phase Langevin inside a
confining wall -- and quenches whatever basin the trajectory happened to
visit. That is the right tool for "what does this system actually do at
298 K", but it has a specific failure as a *generator* for downstream DFT
refinement: it cannot report a basin the trajectory never visited, and a
missing basin is a failure no amount of downstream refinement can repair --
DFT can re-rank the candidates it is handed, but it cannot invent one.

Measured on pyrazine + 2 chloroform, GFN2-xTB/ALPB(chcl3): 10 ps of Langevin,
three stratified packings, 14 deduped candidates, all one motif -- a single
H-bond at 1.90 A / 177 deg, the second chloroform 3.8-5.6 A away and unbound.
The both-nitrogens arrangement (one chloroform H-bonded to each ring nitrogen)
never appears, even though it is 1.56 kcal/mol *lower*: built by hand
(inverting the bound chloroform through the ring centroid) and relaxed, it
optimises to -13.00 kcal/mol against the sweep's pooled -11.44, with both
H...N contacts at 1.94 A / 178 deg.

This module finds that basin by constructing instead of sampling: place a
solvent molecule at a random position and orientation around the (already
relaxed) parent cluster, and optimise. BFGS only descends, so it cannot climb
out of the well it lands in -- which is exactly why it works where a *seeded*
MD run would not: `run_one_job` discards 5 ps of equilibration before
recording the first frame, so a seeded both-N arrangement would already be
gone by the time anything was written. One random pose lands in the both-N
basin about 7% of the time (measured: 4/60 placements onto the relaxed n = 1
parent), and every hit outranks every miss on energy, so the basin sorts to
the top on its own without ever needing to be suspected.

Both programs report the same *kind* of thing -- a continuum-relaxed,
wall-free local minimum, scored identically (`ensemble.relax`, no wall, even
the contact count computed on the relaxed geometry rather than the raw
frame) -- so neither report has to caveat the other. They differ only in how
the starting geometry was found: thermal exploration versus random
construction. `n_sweep.py` is not modified; this is a standalone, either/or
alternative beside it, sharing `ensemble.py`'s optimiser and `report.py`'s
formatting and dedupe/Boltzmann machinery, and only reading (never writing)
`solvate_md.py`.

Docking is a chain, not a sweep: n = 1's survivors become n = 2's starting
points, and so on. That is a **greedy** search -- the best structure at n
need not be the best structure at n - 1 plus one molecule -- so
`Docking.n_parents` carries more than one candidate forward, and the MD
sweep remains available as an independent, non-greedy check on the same
system.

Applicability is limited by the same GFN2 gradient cost that limits the MD
sweep (~N^2.5, measured), compounded by BFGS step count, which is why this
targets small n on small-to-moderate solutes -- comfortable to ~90 atoms,
painful at 180, and *combinatorially* (not just computationally) wrong once
n approaches a monolayer, where no single minimum dominates and
solvent-solvent cohesion takes over. See CLAUDE.md's Applicability section
for the numbers.

Run one from the command line:

    python docking.py examples/pyrazine.xyz examples/chloroform.xyz \
      --solvent chcl3 --n 1 2 3 --out pyrazine_dock/ --placements 64

`dock_at_n` fans placements out through `solvate_md.pool_map`, which uses
spawn, so a script calling `run_docking` itself MUST guard the call:

    if __name__ == "__main__":
        run_docking(...)

`main()` below is under such a guard already.
"""

import single_thread  # noqa: F401  -- must precede numpy; see its docstring

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.optimize import BFGS

from ensemble import (
    CONTACT_GAP_A,
    Scoring,
    reference_energies,
    relax,
    solvent_molecule_gaps,
)
from report import (
    DEDUPE_TOL_EV,
    EV_TO_KCAL,
    VERSION,
    dedupe_energies,
    dock_row_from_summary,
    ensemble_energy,
    format_dock_report,
    git_commit,
    library_versions,
    timestamp,
    write_dock_best_geometries,
)
from shell_capacity import monolayer_capacity
from solvate_md import (
    _random_rotation,
    _vdw_volume,
    align_to_principal_axes,
    bulk_molecular_volume,
    get_calculator,
    pool_map,
    shell_padding,
    solute_semi_axes,
    solvent_radius,
)


@dataclass
class Docking:
    """Everything that shapes a docking run, owned here and only here.

    `Scoring` still owns the final optimiser criterion (`fmax`, `opt_steps`)
    and the Boltzmann temperature -- docking and the MD sweep relax to the
    identical criterion, so their minima are comparable -- and this
    dataclass owns everything specific to *construction*: how many random
    poses to try, how many to carry forward, and the loose screening pass
    that makes trying dozens of them affordable.
    """

    # Random placements per parent per n. 1 - 0.93^K gives 90% confidence of
    # hitting a basin found in 7% of random poses (the measured both-N rate
    # on pyrazine/chloroform) at K = 32 and 99% at K = 64.
    n_placements: int = 64
    # Top-m deduped minima carried forward as next n's parents. The chain is
    # greedy -- the best structure at n need not descend from the best at
    # n - 1 -- and this is the mitigation.
    n_parents: int = 3
    # Placements re-relaxed at the scorer's tight fmax, after screening.
    n_refine: int = 10
    # Loose first pass so paying for `n_placements` per parent is cheap;
    # CLAUDE.md measures 1.3 s vs 4.5 s per candidate at 0.05 vs 0.002 on the
    # pyrazine + 2 chloroform system.
    screen_fmax: float = 0.05
    # FixAtoms on the solute during screening only -- an approximation, off
    # by default. A large aromatic solute's soft modes make BFGS crawl
    # without being relevant to where a solvent molecule binds; refinement is
    # always unconstrained regardless of this flag.
    freeze_solute: bool = False
    # Same meaning as Condition.shell_fill: how full the placement region is
    # at bulk density for n_solvent molecules.
    shell_fill: float = 0.5
    # Minimum interatomic distance at placement, packmol's `tolerance` in the
    # same units and for the same reason -- and the same consequence: it
    # forbids an H-bond distance at t = 0, so BFGS forms the contact itself.
    tolerance: float = 2.0
    solvent: str = "chcl3"
    calculator: str = "gfn2-xtb"
    calculator_kwargs: dict = field(default_factory=dict)


# Above this fraction of a monolayer, docking's combinatorics stop being
# targeted microsolvation: no single minimum dominates, and solvent-solvent
# cohesion (already a known limitation of E_int at n >= 2) takes over. Not a
# hard cutoff -- a warning, matching how `run_sweep` treats `cover`.
MONOLAYER_WARN_FRACTION = 1.0 / 3.0


def _random_point_in_ellipsoid(semi_axes, rng, max_tries=10000):
    """A point uniform over the volume of the ellipsoid with these semi-axes.

    Rejection sampling against the enclosing box -- about 52% acceptance in
    3D -- rather than a closed-form draw, because it needs nothing beyond
    numpy and the region here is always a modest few hundred cubic Angstrom.
    """
    semi_axes = np.asarray(semi_axes, dtype=float)
    for _ in range(max_tries):
        p = rng.uniform(-1.0, 1.0, size=3)
        if p @ p <= 1.0:
            return p * semi_axes
    raise RuntimeError("could not sample a point inside the shell ellipsoid")


def place_one(parent_atoms, solvent_unit, region, tolerance, rng, max_tries=2000):
    """One random placement of `solvent_unit` around `parent_atoms`.

    A random point inside the ellipsoidal shell `region` (semi-axes, centred
    at the origin -- callers pass an already solute-centred `parent_atoms`),
    a random orientation from `solvate_md._random_rotation`, redrawn whenever
    any interatomic distance to an existing atom would fall below
    `tolerance`. There is no separate inner-exclusion test: a point too close
    to the solute simply fails the same distance check, exactly as
    `pack_solvent` relies on packmol's own `tolerance` to keep solvent off
    the fixed solute rather than carving out a second region for it.

    Pure numpy, no packmol and no subprocess -- what makes it cheap enough to
    try dozens of poses per parent. Returns the combined `Atoms`.
    """
    parent_positions = parent_atoms.get_positions()
    unit_positions = solvent_unit.get_positions()
    centered = unit_positions - unit_positions.mean(axis=0)

    for _ in range(max_tries):
        point = _random_point_in_ellipsoid(region, rng)
        rotation = _random_rotation(rng)
        positions = centered @ rotation.T + point
        d = np.linalg.norm(
            positions[:, None, :] - parent_positions[None, :, :], axis=2)
        if d.min() >= tolerance:
            placed = solvent_unit.copy()
            placed.set_positions(positions)
            return parent_atoms + placed

    raise RuntimeError(
        f"place_one: no placement clearing tolerance={tolerance} A in "
        f"{max_tries} tries; the shell region may be too tight for this "
        "many molecules -- raise shell_fill or lower n."
    )


def _align_parent(atoms, n_solute):
    """Re-centre and re-rotate `atoms` onto its own solute block's principal axes.

    Duplicates the handful of lines in `solvate_md`'s principal-frame
    computation rather than reaching into that private helper: a parent here
    is the *whole* complex (solute plus whatever solvent has already been
    docked), and only the solute slice should define the frame, but the
    rotation has to be applied to every atom. `pack_solvent` never needs
    this because packmol always starts from a freshly aligned solute; a
    docking parent has been through one or more rounds of BFGS and may have
    drifted or rotated slightly (nothing here constrains its centre of
    mass), so the shell region computed from `n_solute` atoms is re-centred
    every generation to stay correct.
    """
    solute_positions = atoms.get_positions()[:n_solute]
    centroid = solute_positions.mean(axis=0)
    centered = solute_positions - centroid
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    order = np.argsort(eigenvalues)[::-1]
    rotation = eigenvectors[:, order].T
    if np.linalg.det(rotation) < 0:
        rotation[2] *= -1.0
    aligned = atoms.copy()
    aligned.set_positions((atoms.get_positions() - centroid) @ rotation.T)
    return aligned


def _screen_relax(atoms, calculator, solvation, calculator_kwargs, fmax, steps,
                  n_solute, freeze_solute):
    """The loose screening pass. A plain top-level function for `pool_map`.

    Local rather than `ensemble.relax` because this is the one place that
    needs an optional frozen-solute constraint; `ensemble.relax` stays
    exactly the scorer's optimiser, unconstrained, always -- refinement below
    calls it directly, unchanged.

    Returns `(atoms, energy_eV)` rather than a `Relaxed`: the screening pass
    exists only to rank placements before the expensive tight optimisation,
    so nothing here is precise enough to report -- `screen_fmax` never
    reaches `scored.json`.
    """
    a = atoms.copy()
    a.calc = get_calculator(calculator, solvation=solvation,
                            **(calculator_kwargs or {}))
    if freeze_solute:
        a.set_constraint(FixAtoms(indices=list(range(n_solute))))
    opt = BFGS(a, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    energy = float(a.get_potential_energy())
    a.set_constraint()  # cleared before returning, so nothing downstream inherits it
    results = {k: v for k, v in a.calc.results.items() if k in all_properties}
    a.calc = SinglePointCalculator(a, **results)
    return a, energy


def dock_at_n(parents, n_solute, solvent_unit, n_total, docking, scoring,
              solvation, calculator, calculator_kwargs, seed, n_workers=None):
    """Grow every parent by one solvent molecule, screen, and refine.

    Builds `len(parents) * docking.n_placements` random placements (see
    `place_one`), screens all of them at `docking.screen_fmax`, and refines
    only the best `docking.n_refine` at the scorer's tight criterion.

    Returns `(refined, sources, n_tried)`: `refined` is a list of
    `ensemble.Relaxed` objects, `sources[i]` is the index into `parents` that
    `refined[i]` descended from, and `n_tried` is the total number of random
    placements attempted (screened, not all of which were refined) -- the
    denominator the docking report shows the search effort against.
    """
    v_solvent = bulk_molecular_volume(docking.solvent, solvent_unit)
    r_solvent = solvent_radius(docking.solvent, solvent_unit)
    rng = np.random.default_rng(seed)

    placements, sources = [], []
    for parent_index, parent in enumerate(parents):
        aligned = _align_parent(parent, n_solute)
        solute_only = aligned[:n_solute]
        semi_axes = solute_semi_axes(solute_only)
        v_solute = _vdw_volume(solute_only)
        padding = shell_padding(semi_axes, v_solute, n_total, v_solvent,
                                docking.shell_fill, min_padding=r_solvent)
        region = semi_axes + padding
        for _ in range(docking.n_placements):
            placements.append(place_one(aligned, solvent_unit, region,
                                        docking.tolerance, rng))
            sources.append(parent_index)

    screen_tasks = [(atoms, calculator, solvation, calculator_kwargs,
                     docking.screen_fmax, scoring.opt_steps, n_solute,
                     docking.freeze_solute) for atoms in placements]
    screened = pool_map(_screen_relax, screen_tasks, n_workers)

    order = sorted(range(len(screened)), key=lambda i: screened[i][1])
    top = order[:docking.n_refine]

    refine_tasks = [(screened[i][0], calculator, solvation, calculator_kwargs,
                     scoring.fmax, scoring.opt_steps) for i in top]
    refined = pool_map(relax, refine_tasks, n_workers)
    refined_sources = [sources[i] for i in top]
    return refined, refined_sources, len(placements)


def _assemble_dock_n(out_root, label, n, pairs, references, solvation,
                     docking, scoring, n_tried, n_parents_used, n_solute, aps):
    """Turn one n's refined placements into `scored.json` and its files.

    `pairs` is `[(Relaxed, parent_index), ...]`, already sorted lowest energy
    first, for every refined placement at this n (not yet deduped). Mirrors
    `ensemble.assemble`'s summary shape field-for-field where a field applies
    -- `run_dir`, `e_solute_ref_eV`, `candidates`, and so on -- so
    `dft_export` reads a sweep run and a docking run identically. Where it
    does not apply (there is no MD frame, no sampling wall, no packing seed)
    the field is written as `None` rather than defaulted, per the repo's
    "missing is broken, not old" convention -- except here it genuinely is
    absent, not broken, which is exactly `pack_mode: "dock"`'s job to say.
    """
    e_solute, e_solvent, ref_solute_atoms, ref_solvent_atoms = references
    n_dir = out_root / f"{label}_n{n}_dock"
    n_dir.mkdir(parents=True, exist_ok=True)

    interactions = [r.energy_eV - e_solute - n * e_solvent for r, _ in pairs]
    keep_idx = dedupe_energies([r.energy_eV for r, _ in pairs])
    kept = [pairs[i] for i in keep_idx]
    kept_interactions = [interactions[i] for i in keep_idx]

    candidates = []
    for i, ((result, parent_index), interaction) in enumerate(
            zip(kept, kept_interactions)):
        gaps = solvent_molecule_gaps(result.atoms, n_solute, aps)
        candidates.append({
            "frame": i,
            "energy_eV": result.energy_eV,
            "interaction_eV": interaction,
            "converged": result.converged,
            "fmax": result.fmax,
            "n_contacts": int((gaps < CONTACT_GAP_A).sum()),
            "n_solvent": n,
            "min_gap_A": float(gaps.min()) if len(gaps) else float("nan"),
            "wall_energy_eV": None,
            "gnorm_Eh_bohr": result.gnorm_Eh_bohr,
            "n_opt_steps": result.n_opt_steps,
            "parent": parent_index,
        })

    absolutes = [c["energy_eV"] for c in candidates]
    e_min = min(kept_interactions)
    best_i = min(range(len(candidates)),
                key=lambda i: candidates[i]["interaction_eV"])
    write(n_dir / "best.xyz", kept[best_i][0].atoms)
    write(n_dir / "scored_candidates.xyz", [r.atoms for r, _ in kept])
    write(n_dir / "ref_solute.xyz", ref_solute_atoms)
    write(n_dir / "ref_solvent.xyz", ref_solvent_atoms)

    per_parent = {}
    for (_, parent_index), interaction in zip(pairs, interactions):
        per_parent.setdefault(parent_index, []).append(interaction)
    parent_detail = [
        {"parent": pi,
         "n_placements": len(vals),
         "e_int_min_kcal": min(vals) * EV_TO_KCAL,
         "best": abs(min(vals) - e_min) <= DEDUPE_TOL_EV}
        for pi, vals in sorted(per_parent.items())
    ]

    found_by = sum(1 for v in interactions if abs(v - e_min) <= DEDUPE_TOL_EV)

    summary = {
        "run_dir": str(n_dir),
        "label": label,
        "pack_mode": "dock",
        "seed": None,
        "n_solvent": n,
        "sampling_solvation": None,
        "scoring_solvation": list(solvation),
        "calculator": docking.calculator,
        "temperature_K": scoring.temperature_K,
        "stride": None,
        "max_frames": None,
        "opt_fmax": scoring.fmax,
        "opt_steps": scoring.opt_steps,
        "n_frames_scored": len(pairs),
        "n_unique": len(candidates),
        "e_solute_ref_eV": e_solute,
        "e_solvent_ref_eV": e_solvent,
        "min_interaction_kcal": e_min * EV_TO_KCAL,
        "ensemble_interaction_kcal": ensemble_energy(
            kept_interactions, scoring.temperature_K) * EV_TO_KCAL,
        "min_energy_eV": min(absolutes),
        "ensemble_energy_eV": ensemble_energy(absolutes, scoring.temperature_K),
        "mean_contacts": float(np.mean([c["n_contacts"] for c in candidates])),
        "dissolved_fraction": float(
            np.mean([c["n_contacts"] == 0 for c in candidates])),
        "converged_fraction": float(np.mean([r.converged for r, _ in pairs])),
        "n_unconverged_unique": sum(1 for c in candidates if not c["converged"]),
        "sampling_wall": None,
        "n_scored_wall_active": 0,
        "candidates": candidates,
        # Docking-only bookkeeping, harmless extra keys for anything reading
        # this as a plain ensemble.assemble summary.
        "n_placements_tried": n_tried,
        "n_refined": len(pairs),
        "n_parents": n_parents_used,
        "found_by": found_by,
        "parent_detail": parent_detail,
    }
    (n_dir / "scored.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_docking(solute_path, solvent_path, solvent, n_values, out_root,
                docking=None, scoring=None, n_workers=None, label=None):
    """Dock one solute in one solvent, chained upward over n.

    n = 1's parent is the bare relaxed solute (the same reference
    `E_int(0) = 0` anchors); each subsequent n grows every surviving parent
    by one random placement, screens, refines, and keeps the top
    `docking.n_parents` deduped minima as the next generation's parents. The
    chain always walks every integer from 1 to `max(n_values)` -- a later n
    depends on the actual geometry an earlier one settled into, not merely on
    its energy -- but only writes an output directory and a report row for
    the n values actually requested.

    Writes `<out_root>/dock.json` -- `{"params": ..., "runs": [...]}`,
    matching `sweep.json`'s shape -- and `<out_root>/dock_report.txt`, plus
    one `best_n<N>.xyz` per requested n. Returns the list of summaries, one
    per requested n.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    n_values = sorted(set(int(n) for n in n_values))
    if not n_values or min(n_values) < 1:
        raise ValueError(
            "docking n values must be >= 1; n = 0 is the bare relaxed "
            "solute reference, computed once rather than swept.")
    label = label or Path(solute_path).stem
    docking = docking or Docking(solvent=solvent)
    scoring = scoring or Scoring()
    solvation = ("alpb", docking.solvent)

    solute = align_to_principal_axes(read(solute_path))
    n_solute = len(solute)
    solvent_unit = read(solvent_path)
    aps = len(solvent_unit)

    _, _, capacity = monolayer_capacity(solute, docking.solvent)
    max_n = max(n_values)
    if max_n > MONOLAYER_WARN_FRACTION * capacity:
        print(
            f"WARNING: n = {max_n} is {100 * max_n / capacity:.0f}% of a "
            f"monolayer (~{capacity:.0f} {docking.solvent} molecules). "
            "Docking targets targeted microsolvation, not a full shell: "
            "no single minimum dominates near a monolayer, and "
            "solvent-solvent cohesion takes over. The MD sweep is the "
            "right tool past roughly a third of capacity -- see "
            "CLAUDE.md's Applicability section.")

    start = time.time()
    e_solute, e_solvent, ref_solute_atoms, ref_solvent_atoms = reference_energies(
        solute_path, solvent_path, docking.calculator, solvation,
        docking.calculator_kwargs, scoring.fmax, scoring.opt_steps)
    references = (e_solute, e_solvent, ref_solute_atoms, ref_solvent_atoms)

    parents = [ref_solute_atoms]
    all_n_min_kcal = {0: 0.0}  # E_int(0) = 0 by construction
    summaries = []
    n_tried_total = 0

    for n in range(1, max_n + 1):
        n_parents_used = len(parents)
        refined, sources, n_tried = dock_at_n(
            parents, n_solute, solvent_unit, n, docking, scoring, solvation,
            docking.calculator, docking.calculator_kwargs, seed=n,
            n_workers=n_workers)
        n_tried_total += n_tried

        pairs = sorted(zip(refined, sources), key=lambda pair: pair[0].energy_eV)
        summary = _assemble_dock_n(
            out_root, label, n, pairs, references, solvation, docking,
            scoring, n_tried, n_parents_used, n_solute, aps)
        all_n_min_kcal[n] = summary["min_interaction_kcal"]

        keep_idx = dedupe_energies([r.energy_eV for r, _ in pairs])
        parents = [pairs[i][0].atoms for i in keep_idx[:docking.n_parents]]

        if n in n_values:
            summaries.append(summary)

    elapsed = time.time() - start
    print(f"docking: {n_tried_total} placements screened, "
          f"{sum(s['n_refined'] for s in summaries)} refined at requested n "
          f"in {elapsed:.1f} s")

    params = dock_params(docking, scoring, n_values, label, capacity,
                         solute_path, solvent_path, all_n_min_kcal)
    (out_root / "dock.json").write_text(
        json.dumps({"params": params, "runs": summaries}, indent=2))
    dock_rows = [dock_row_from_summary(s) for s in summaries]
    (out_root / "dock_report.txt").write_text(
        format_dock_report(params, dock_rows))
    write_dock_best_geometries(out_root, dock_rows)
    return summaries


def dock_params(docking, scoring, n_values, label, capacity, solute_path,
                solvent_path, all_n_min_kcal):
    """Everything needed to reproduce the run, recorded with its results.

    Built from `asdict` of the real `Docking` and `Scoring`, the same
    convention `n_sweep.sweep_params` uses, so a docking run and a sweep are
    reproducible from their own params block by the same rule -- including
    the `library_versions` that say which build of the Hamiltonian ran.
    """
    params = {k: v for k, v in asdict(docking).items()}
    params.update(asdict(scoring))
    params.update({
        "solute_label": label,
        "solute_path": str(solute_path),
        "solvent_path": str(solvent_path),
        "n_values": list(n_values),
        "monolayer_capacity": capacity,
        # Every n walked internally, not just the requested rows -- what lets
        # dE_int be computed for a requested n even when n - 1 was not itself
        # requested.
        "all_n_min_kcal": all_n_min_kcal,
        "version": VERSION,
        "git_commit": git_commit(),
        "timestamp": timestamp(),
    })
    params.update(library_versions(docking.calculator))
    return params


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Dock solvent onto one solute in one solvent, chained "
                    "upward over n -- a constructive alternative to the MD "
                    "sweep's thermal sampling.",
        epilog="Docking wins at every n by construction (BFGS only "
               "descends), so its numbers are never pooled with an MD "
               "sweep's; read the two reports side by side instead.")
    parser.add_argument("solute", help="solute geometry file")
    parser.add_argument("solvent_geometry", help="one solvent molecule")
    parser.add_argument("--solvent", default="chcl3",
                        help="solvent name; the ALPB key and the bulk-density "
                             "key both (default: %(default)s)")
    parser.add_argument("--n", dest="n_values", type=int, nargs="+",
                        required=True, metavar="N",
                        help="explicit solvent counts to dock, e.g. 1 2 3 "
                             "(n = 0 is the bare relaxed solute and is not "
                             "swept)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--placements", type=int, default=Docking.n_placements,
                        help="random placements tried per parent per n "
                             "(default: %(default)s)")
    parser.add_argument("--parents", type=int, default=Docking.n_parents,
                        help="deduped minima carried forward as the next n's "
                             "parents (default: %(default)s)")
    parser.add_argument("--refine", type=int, default=Docking.n_refine,
                        help="best-screened placements re-relaxed at the "
                             "scorer's tight fmax (default: %(default)s)")
    parser.add_argument("--screen-fmax", type=float,
                        default=Docking.screen_fmax,
                        help="loose optimiser convergence for the screening "
                             "pass, eV/A (default: %(default)s)")
    parser.add_argument("--freeze-solute", action="store_true",
                        help="FixAtoms the solute during screening only "
                             "(approximation; refinement is always "
                             "unconstrained)")
    parser.add_argument("--fmax", type=float,
                        default=Scoring.__dataclass_fields__["fmax"].default,
                        help="refinement optimiser convergence, eV/A "
                             "per-atom max force (default: %(default)s)")
    parser.add_argument("--opt-steps", type=int,
                        default=Scoring.__dataclass_fields__["opt_steps"].default,
                        help="max optimiser steps per candidate "
                             "(default: %(default)s)")
    parser.add_argument("--temperature", type=float,
                        default=Scoring.__dataclass_fields__["temperature_K"].default,
                        help="K, for the Boltzmann weights "
                             "(default: %(default)s)")
    parser.add_argument("--calculator", default=Docking.calculator,
                        help="default: %(default)s")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel workers (default: all cores)")
    parser.add_argument("--label", default=None,
                        help="solute name in output directories and the "
                             "table (default: the solute filename stem)")
    parser.add_argument("--export-dft", action="store_true",
                        help="also export deduped, near-minimum candidates "
                             "for DFT refinement to <out>/dft_export/")
    args = parser.parse_args(argv)

    docking = Docking(
        n_placements=args.placements,
        n_parents=args.parents,
        n_refine=args.refine,
        screen_fmax=args.screen_fmax,
        freeze_solute=args.freeze_solute,
        solvent=args.solvent,
        calculator=args.calculator,
    )
    scoring = Scoring(fmax=args.fmax, opt_steps=args.opt_steps,
                      temperature_K=args.temperature)

    run_docking(
        args.solute, args.solvent_geometry, args.solvent, args.n_values,
        args.out, docking=docking, scoring=scoring, n_workers=args.workers,
        label=args.label,
    )
    print(Path(args.out) / "dock_report.txt")

    if args.export_dft:
        from dft_export import export_dft
        out_dir = Path(args.out) / "dft_export"
        export_dft(args.out, out_dir)
        print(out_dir / "manifest.json")


if __name__ == "__main__":
    main()
