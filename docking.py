"""A second geometry generator: construct microsolvated clusters, don't sample them.

`n_sweep.py` explores by thermal sampling and quenches whatever basin the
trajectory happened to visit. That is the right tool for "what does this
system actually do at 298 K", but as a *generator* for downstream DFT
refinement it has a specific failure: it cannot report a basin the trajectory
never visited, and a missing basin is the one thing no amount of downstream
refinement repairs -- DFT can re-rank the candidates it is handed, but it
cannot invent one.

This module finds basins by constructing instead: place a solvent molecule at
a random position and orientation around the (already relaxed) parent
cluster, and optimise. BFGS only descends, so it cannot climb out of the well
it lands in -- which is exactly why it works where a *seeded* MD run would
not: `run_one_job` discards 5 ps of equilibration before recording the first
frame, so a seeded arrangement would already be gone by the time anything was
written. Every hit outranks every miss on energy, so a basin sorts to the top
on its own without ever needing to be suspected.

**Docking owns minimum-finding**, and the MD sweep keeps the two jobs docking
cannot do -- an independently drawn, non-greedy check, and basin occupancy,
which a constructed minimum has no sense of at all (`_assemble_dock_n` writes
`n_frames` / `frames` as `None` for exactly that reason). CLAUDE.md's
`docking.py` section carries the measurements behind all of that: the
both-nitrogens result on pyrazine + 2 chloroform, the staged
screen-then-refine cost, the `1 - 0.93^K` placement-count argument, and where
this stops working. Not repeated here.

`n_sweep.py` and `solvate_md.py` are unmodified by this; it only reads from
them, adds no Hamiltonian of its own, and relaxes its candidates to
`Scoring.fmax` -- the criterion the scorer uses -- so a docked and a swept
minimum stay comparable.

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
from ase.io import read, write

from ensemble import (
    CONTACT_GAP_A,
    Candidate,
    Scoring,
    reference_energies,
    relax,
    solvent_molecule_gaps,
    summarise,
)
from report import (
    DEDUPE_TOL_EV,
    EV_TO_KCAL,
    VERSION,
    dedupe_energies,
    format_report,
    library_versions,
    pool_by_n,
    timestamp,
    write_best_geometries,
)
from shell_capacity import monolayer_capacity
from solvate_md import (
    _random_rotation,
    _vdw_volume,
    align_to_principal_axes,
    bulk_molecular_volume,
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
    # Placements re-relaxed at the scorer's tight fmax, after screening --
    # **per parent**, and chosen among that parent's *distinct* screened
    # basins (see `screen_dedupe_tol_eV`) rather than by raw screened energy.
    # It used to be a total over all parents, taken off the raw ranking: ten
    # of 192 at n >= 2, which could be ten copies of one basin. That starved
    # `n_parents` -- the next generation's parents were the deduped top-3 of
    # at most ten refinements -- and capped `dft_export`'s window at ten
    # candidates per n whatever the window said.
    n_refine: int = 10
    # Two screened energies this close are one basin for the purpose of
    # choosing what to refine. Deliberately looser than `report.DEDUPE_TOL_EV`
    # (1 meV): at `screen_fmax = 0.05` two placements in the same basin can
    # still differ by up to ~0.6 kcal/mol, so this errs toward refining a
    # duplicate rather than dropping a basin. ~0.1 kcal/mol. The screened
    # energies never reach a `scored.json`, so neither does this criterion;
    # the refined candidates are deduped at the 1 meV one like everything else.
    screen_dedupe_tol_eV: float = 4e-3
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


def dock_at_n(parents, n_solute, solvent_unit, n_total, docking, scoring,
              solvation, calculator, calculator_kwargs, seed, n_workers=None):
    """Grow every parent by one solvent molecule, screen, and refine.

    Builds `len(parents) * docking.n_placements` random placements (see
    `place_one`), screens all of them at `docking.screen_fmax`, and refines
    up to `docking.n_refine` *per parent* at the scorer's tight criterion:
    the parent's screened energies are deduped at
    `docking.screen_dedupe_tol_eV` and the lowest representative of each
    screened basin is refined, best-first, so the refined set carries every
    distinct basin the screen found rather than the best basin several times
    over. Fewer than `n_refine` are refined when a parent's placements
    collapsed into fewer screened basins than that.

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
        # Only the solute block defines the frame the shell region is
        # measured in, but the whole complex moves with it.
        aligned = align_to_principal_axes(parent, n_solute)
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

    # The same optimiser as the refinement below and as the scorer, at a
    # looser `fmax` and optionally with the solute held fixed. Nothing it
    # produces reaches a `scored.json`: it exists only to rank placements, so
    # only its geometry and its energy are read back.
    freeze_n = n_solute if docking.freeze_solute else 0
    screen_tasks = [(atoms, calculator, solvation, calculator_kwargs,
                     docking.screen_fmax, scoring.opt_steps, freeze_n)
                    for atoms in placements]
    screened = pool_map(relax, screen_tasks, n_workers)

    # Per parent, so a parent whose placements all screened a little higher
    # than another's still gets its basins refined -- that is what keeps the
    # chain's `n_parents` lineages alive into the next generation. Within a
    # parent, one representative per screened basin, lowest first.
    by_parent = {}
    for i, parent_index in enumerate(sources):
        by_parent.setdefault(parent_index, []).append(i)
    top = []
    for parent_index in sorted(by_parent):
        members = by_parent[parent_index]
        representatives = dedupe_energies(
            [screened[i].energy_eV for i in members],
            docking.screen_dedupe_tol_eV)
        top += [members[r] for r in representatives[:docking.n_refine]]

    refine_tasks = [(screened[i].atoms, calculator, solvation,
                     calculator_kwargs, scoring.fmax, scoring.opt_steps, 0)
                    for i in top]
    refined = pool_map(relax, refine_tasks, n_workers)
    refined_sources = [sources[i] for i in top]
    return refined, refined_sources, len(placements)


def _assemble_dock_n(out_root, label, n, pairs, references, solvation,
                     docking, scoring, n_tried, n_parents_used, n_solute, aps):
    """Turn one n's refined placements into `scored.json` and its files.

    `pairs` is `[(Relaxed, parent_index), ...]`, already sorted lowest energy
    first, for every refined placement at this n (not yet deduped). The
    summary itself is built by `ensemble.summarise`, the same function
    `ensemble.assemble` calls, so a docking run and a sweep run cannot drift
    apart in shape or in what a field means -- which is what `dft_export` and
    `report` both rely on when they read either without branching.

    Everything specific to construction is what this adds: the placement
    bookkeeping in `extra`, and the per-parent table that makes the greedy
    chain visible. What it deliberately does *not* add is an occupancy: a
    docking dedupe groups constructed placements, not thermal samples, so
    counting them would look like a frame-weighted population and would not
    be one. `Candidate.n_frames` is `None` for these, which is what makes
    `summarise` report no occupancy at all rather than a plausible number.
    """
    e_solute, e_solvent, ref_solute_atoms, ref_solvent_atoms = references
    n_dir = out_root / f"{label}_n{n}_dock"
    n_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for i, (result, parent_index) in enumerate(pairs):
        gaps = solvent_molecule_gaps(result.atoms, n_solute, aps)
        candidates.append(Candidate(
            # No trajectory to index into: this is which refined placement
            # the geometry came from, in energy order.
            frame=i,
            energy_eV=result.energy_eV,
            interaction_eV=result.energy_eV - e_solute - n * e_solvent,
            converged=result.converged,
            fmax=result.fmax,
            n_contacts=int((gaps < CONTACT_GAP_A).sum()),
            n_solvent=n,
            min_gap_A=float(gaps.min()) if len(gaps) else float("nan"),
            # A docked structure has no sampling frame, so no wall energy and
            # no basin occupancy -- real absences, which is exactly what
            # `pack_mode: "dock"` exists to say.
            wall_energy_eV=None,
            gnorm_Eh_bohr=result.gnorm_Eh_bohr,
            n_opt_steps=result.n_opt_steps,
            parent=parent_index,
            n_frames=None,
            frames=None,
        ))

    keep_idx = dedupe_energies([c.energy_eV for c in candidates])
    unique = [candidates[i] for i in keep_idx]
    e_min = min(c.interaction_eV for c in unique)

    write(n_dir / "best.xyz", pairs[keep_idx[0]][0].atoms)
    write(n_dir / "scored_candidates.xyz", [pairs[i][0].atoms for i in keep_idx])
    write(n_dir / "ref_solute.xyz", ref_solute_atoms)
    write(n_dir / "ref_solvent.xyz", ref_solvent_atoms)

    per_parent = {}
    for c in candidates:
        per_parent.setdefault(c.parent, []).append(c.interaction_eV)
    parent_detail = [
        {"parent": pi,
         "n_placements": len(vals),
         "e_int_min_kcal": min(vals) * EV_TO_KCAL,
         "best": abs(min(vals) - e_min) <= DEDUPE_TOL_EV}
        for pi, vals in sorted(per_parent.items())
    ]

    summary = summarise(
        run_dir=n_dir,
        label=label,
        pack_mode="dock",
        seed=None,
        n_solvent=n,
        candidates=candidates,
        unique=unique,
        references=(e_solute, e_solvent),
        sampling_solvation=None,
        scoring_solvation=solvation,
        calculator=docking.calculator,
        scoring=scoring,
        sampling_wall=None,
        scored_frame_spacing_fs=None,
        extra={
            "n_placements_tried": n_tried,
            "n_refined": len(pairs),
            "n_parents": n_parents_used,
            # The constructive analogue of the sweep's `found by`: refined
            # placements that landed on this n's minimum. Not independent
            # corroboration the way agreeing packings are -- see
            # `report.format_dock_parent_detail`.
            "found_by": sum(1 for c in candidates
                            if abs(c.interaction_eV - e_min) <= DEDUPE_TOL_EV),
            "parent_detail": parent_detail,
        },
    )
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
    (out_root / "dock_report.txt").write_text(
        format_report(params, summaries))
    write_best_geometries(out_root, pool_by_n(summaries), "dock")
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
    # `max_frames` describes how a trajectory is subsampled, and docking has
    # no trajectory: `summarise` already writes it as `None` per run, and the
    # params block should not advertise a setting that did nothing either.
    params.update({k: v for k, v in asdict(scoring).items()
                   if k != "max_frames"})
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
                        help="screened placements re-relaxed at the scorer's "
                             "tight fmax, per parent: one per distinct "
                             "screened basin, best-first, up to this many "
                             "(default: %(default)s)")
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
