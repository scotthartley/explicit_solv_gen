"""Score geometries under a Hamiltonian other than the one that generated them.

The pipeline in `solvate_md.py` samples and evaluates with a single calculator.
That is the wrong shape for a cluster-continuum study, because the two jobs
want different Hamiltonians:

  * sampling wants whatever actually finds the contact geometries. Gas-phase
    MD does this well -- measured on methanol + 4 water, 10 ps of gas-phase
    Langevin forms ~4 hydrogen bonds and never touches the confining wall.
  * scoring wants the continuum, because the bulk dielectric is a large,
    strongly solvent-dependent term that a finite shell cannot supply.

So: generate without the continuum, score with it. The seam is a *file*
boundary rather than a function call, for two reasons. Importing torch (MACE)
and tblite into one process raises an OpenMP duplicate-runtime error whose
only workaround is documented as unsafe, so a MACE generator and a GFN2 scorer
have to be separate processes anyway. And a trajectory on disk can be rescored
under a different continuum without regenerating it.

The scorer deliberately applies **no wall**. The wall exists to stop a shell
dissociating during dynamics; here, dissociation is the signal. A solvent
molecule that optimises away from the solute is reporting that it had no
specific interaction, and the interaction energy below is constructed so that
such a molecule contributes exactly zero.
"""

import single_thread  # noqa: F401  -- must precede numpy; see its docstring

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import BFGS
from ase.units import Bohr, Hartree

from report import (
    EV_TO_KCAL,
    dedupe_groups,
    ensemble_energy,
    format_scored_log,
    wall_stats,
)
from solvate_md import (
    _vdw_radii_array,
    align_to_principal_axes,
    get_calculator,
    pool_map,
)

# EV_TO_KCAL, the Boltzmann helpers and the dedupe criterion live in `report`
# so that formatting a number, or regenerating a log from JSON, never drags in
# ASE -- and so the live and the regenerated logs weight and dedupe identically
# by construction rather than by comment. `report` pools candidates across
# seeds with the same criterion this module applies within one run.

# A solvent molecule counts as touching the solute when some atom pair is
# within this much of van der Waals contact. Generous on purpose: it is a
# bookkeeping diagnostic, not a bond criterion.
CONTACT_GAP_A = 0.5

# 1 Eh/bohr in eV/A. ASE reports forces in eV/A and converges on the largest
# per-atom force; xtb reports a gradient norm over all 3N components in
# Eh/bohr. The two criteria are not comparable by eye -- fmax 0.05 eV/A is
# about 2.5x looser than xtb's `--opt normal` -- so both are recorded.
EV_A_PER_EH_BOHR = Hartree / Bohr


@dataclass
class Scoring:
    """Everything that sets how a trajectory is rescored.

    One owner per default, for the same reason the MD lengths live only on
    `Condition`: an internal function that defaulted `max_frames` for itself
    would be a second, silently different opinion, and this pipeline has been
    bitten by exactly that twice.
    """

    # Cap on frames scored per run, spread evenly over the whole trajectory.
    # This is what sets scoring cost, and the cost is independent of run
    # length: a longer trajectory is scored at wider spacing for the same
    # price. There used to be a `stride` beside it, which selected every Nth
    # dump before this cap was applied -- but the cap selects by `linspace`
    # over the whole trajectory, so it discarded whatever thinning `stride`
    # had done, and at any usable setting the cap always bites.
    max_frames: int = 50
    # Optimiser convergence, eV/A per-atom max force. Every term of E_int has
    # to be relaxed to convergence, not merely to a stationary-ish geometry:
    # at the old 0.05 a frame is left hanging on whichever soft mode it was
    # descending, which on one pyrazine + 2 chloroform frame put the reported
    # minimum 0.58 kcal/mol above the true one. It does not cancel between
    # solvents or between conformers, because it depends on the mode rather
    # than on the chemistry. xtb's `--opt normal` stops at a gradient norm of
    # 1e-3 Eh/a, which this is comfortably inside -- the two criteria are not
    # comparable by eye, so `scored.log` prints both.
    fmax: float = 0.002
    opt_steps: int = 1000
    # For the Boltzmann weights. Normally the same as the MD temperature.
    temperature_K: float = 298.0


@dataclass
class Candidate:
    """One optimised geometry, scored in the continuum.

    Both generators build these, so `scored.json`'s candidate shape has one
    definition rather than two kept in step by hand. Where a field cannot
    apply to one of them it is `None` -- a real absence, which is what
    `pack_mode` exists to explain -- never a plausible-looking zero.
    """

    # For an MD candidate, the trajectory dump this was quenched from. For a
    # docked one there is no trajectory: it is the index of the refined
    # random placement instead, which is the only provenance a constructed
    # geometry has.
    frame: int
    energy_eV: float
    # E(cluster) - E(solute) - n * E(solvent), every term relaxed in the same
    # environment. This is the quantity that makes an n-sweep meaningful:
    # absolute energies at different n differ by whole solvent molecules, and
    # a solvent molecule that dissolves into the continuum contributes ~0 here
    # rather than a large constant. So "the shell dissolved" and "there was no
    # explicit solvent" converge to the same number, as they should.
    interaction_eV: float
    converged: bool
    fmax: float
    n_contacts: int
    n_solvent: int
    min_gap_A: float
    # The confinement energy of the *sampling* frame this came from -- the
    # scorer applies no wall. A nonzero value says this geometry was being
    # squeezed by the wall when it was recorded, which is worth seeing beside
    # the weight it carries.
    wall_energy_eV: float
    # The same convergence, in xtb's units: ||grad|| over all components, in
    # Eh/bohr, for comparison against `xtb --opt normal`'s 1e-3 threshold.
    gnorm_Eh_bohr: float
    # BFGS steps taken. Says whether the hour a sweep spends in the optimiser
    # goes to the tight `fmax` or to BFGS being the wrong optimiser for a
    # floppy cluster -- worth knowing before anyone tries swapping it on a
    # larger system.
    n_opt_steps: int
    # Which of the previous n's surviving parents a docked candidate grew
    # from -- the greedy chain made visible. `None` for an MD candidate,
    # which descends from a packing rather than from another structure.
    parent: int = None
    # How many scored frames quenched into this minimum, and their sampling
    # dump indices -- the inherent-structure occupancy `n_sweep`'s docstring
    # discusses. 1 / [frame] until `assemble` groups this candidate with its
    # duplicates and calls `dataclasses.replace` on the survivor;
    # a docked candidate (no sampling frame at all) carries `None` for both,
    # a real absence rather than a population of one.
    n_frames: int = 1
    frames: list = field(default_factory=list)


def solvent_molecule_gaps(atoms, n_solute, atoms_per_solvent):
    """Closest van der Waals gap (d - r_i - r_j) from each solvent molecule.

    Negative means interpenetrating vdW spheres, i.e. real contact; large
    positive means the molecule has drifted off into the continuum.
    """
    n_solvent_atoms = len(atoms) - n_solute
    if atoms_per_solvent <= 0 or n_solvent_atoms <= 0:
        return np.zeros(0)

    positions = atoms.get_positions()
    radii = _vdw_radii_array(atoms)
    solute_p, solute_r = positions[:n_solute], radii[:n_solute]

    n_mol = n_solvent_atoms // atoms_per_solvent
    gaps = np.empty(n_mol)
    for m in range(n_mol):
        lo = n_solute + m * atoms_per_solvent
        sel = slice(lo, lo + atoms_per_solvent)
        d = np.linalg.norm(positions[sel][:, None, :] - solute_p[None, :, :], axis=2)
        gaps[m] = float((d - radii[sel][:, None] - solute_r[None, :]).min())
    return gaps


@dataclass
class Relaxed:
    """One finished optimisation, detached from the calculator that ran it.

    `relax` runs in a pool worker, so its result has to cross a pickle. A live
    tblite calculator cannot -- it holds a cffi handle -- so the geometry comes
    back carrying a `SinglePointCalculator` with the same results dict instead.
    That is not merely a workaround: it is what lets `best.xyz` and
    `scored_candidates.xyz` keep their energy, forces and charges, written by
    the parent from a geometry optimised in a worker.
    """

    atoms: object
    energy_eV: float
    converged: bool
    fmax: float
    gnorm_Eh_bohr: float
    # How many BFGS steps this candidate took. The number that says whether
    # the hour a sweep spends here goes to the tight `fmax` or to a poor
    # optimiser choice, which is worth knowing before anyone tries swapping
    # BFGS for LBFGS on a larger cluster.
    n_opt_steps: int


def relax(atoms, calculator, solvation, calculator_kwargs, fmax, steps,
          freeze_n):
    """Optimise `atoms` in the scoring environment. No wall -- see module doc.

    This is the unit of work both grids fan out: one candidate, one of the two
    references, or one docking placement per call. It is a plain top-level
    function taking picklable arguments for exactly that reason.

    `freeze_n` holds the first that many atoms fixed. Zero -- every candidate
    the scorer reports and every refinement `docking.py` does -- is the
    unconstrained optimisation `Scoring.fmax` names, and the only one whose
    result is comparable between the two generators. Nonzero is docking's
    optional frozen-solute screening pass, whose energies exist only to rank
    placements before the expensive tight optimisation and never reach a
    `scored.json`. Required rather than defaulted, like every other argument
    here: the caller has to say which of the two it means.

    The constraint is cleared before returning, so nothing downstream --
    a refinement, a `best.xyz` -- inherits it.
    """
    a = atoms.copy()
    a.calc = get_calculator(calculator, solvation=solvation,
                            **(calculator_kwargs or {}))
    if freeze_n:
        a.set_constraint(FixAtoms(indices=list(range(freeze_n))))
    opt = BFGS(a, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    # Read all of these before anything else touches the geometry:
    # `converged()` re-evaluates against current positions rather than
    # reporting on the run.
    converged = bool(opt.converged())
    forces = a.get_forces()
    energy = float(a.get_potential_energy())
    results = {k: v for k, v in a.calc.results.items() if k in all_properties}
    a.set_constraint()
    a.calc = SinglePointCalculator(a, **results)
    return Relaxed(
        atoms=a,
        energy_eV=energy,
        converged=converged,
        fmax=float(np.linalg.norm(forces, axis=1).max()),
        gnorm_Eh_bohr=float(np.linalg.norm(forces) / EV_A_PER_EH_BOHR),
        n_opt_steps=int(opt.nsteps),
    )


def reference_energies(solute_path, solvent_path, calculator, solvation,
                       calculator_kwargs, fmax, steps):
    """Relaxed isolated solute and solvent, in the scoring environment.

    Returns `(e_solute_eV, e_solvent_eV, solute_atoms, solvent_atoms)`. The
    geometries ride along, not just the energies: `assemble` persists them as
    `ref_solute.xyz` / `ref_solvent.xyz` in every run directory that shares
    this reference, which is what lets a downstream DFT export reconstruct
    `E(solute) + n E(solvent)` against the same relaxed geometry the sweep
    used, rather than re-relaxing (possibly to a different minimum) itself.

    The solute reference is geometry-specific by construction -- it comes
    from `solute_path`, so two conformers of one molecule each get their
    own, which is what makes an interaction energy comparable between them.
    """
    solute = align_to_principal_axes(read(solute_path))
    solute_relaxed = relax(solute, calculator, solvation,
                           calculator_kwargs, fmax, steps, 0)
    solvent_relaxed = relax(read(solvent_path), calculator, solvation,
                            calculator_kwargs, fmax, steps, 0)
    return (solute_relaxed.energy_eV, solvent_relaxed.energy_eV,
            solute_relaxed.atoms, solvent_relaxed.atoms)


def select_frames(run_dir, max_frames):
    """The frames of one run directory that will be scored, and their context.

    Returns `(meta, records, indices, frames)`. Split out of the scoring pass
    so the parent can do it for every job in a sweep before any optimisation
    starts: reading a trajectory is milliseconds, and doing it up front is
    what turns a sweep into one flat list of independent candidate
    optimisations rather than a handful of serial per-directory loops.
    """
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "metadata.json").read_text())
    # Trajectory dump i and records[i] are written by the same `record()`
    # closure in `solvate_md`, so a candidate's dump index reads its wall
    # energy straight out of this list.
    records = json.loads((run_dir / "energies.json").read_text())

    frames = read(str(run_dir / "traj.xyz"), index=":")
    # Carried alongside so a candidate can name the dump it actually came
    # from -- `max_frames` below subsamples, and an index into the subsampled
    # list would label a 300-dump trajectory 0..49.
    indices = list(range(len(frames)))
    if max_frames is not None and len(frames) > max_frames:
        # Spread over the whole trajectory rather than taking a prefix.
        sel = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in sel]
        indices = [indices[i] for i in sel]
    return meta, records, indices, frames


def summarise(run_dir, label, pack_mode, seed, n_solvent, candidates, unique,
              references, sampling_solvation, scoring_solvation, calculator,
              scoring, sampling_wall, scored_frame_spacing_fs, extra):
    """The `scored.json` shape, built once for both generators.

    `candidates` is every optimised geometry at this point of the sweep or
    chain; `unique` is the same list after the basin dedupe, lowest first.
    Everything here is derived from those two, the references and `scoring`,
    so a docking run and an MD run cannot disagree about what a field means
    -- which they could, and did, while `docking.py` rebuilt this dict field
    by field beside `assemble`'s.

    Where a field describes something one generator does not do, it comes out
    `None` from the inputs themselves rather than from a branch here: a
    docked candidate has no sampling frame, so its `n_frames` is `None` and
    the occupancy fields below are too; it has no sampling wall, so
    `sampling_wall` is `None` and no candidate has a wall energy to count.
    `pack_mode` is what says which generator ran, and therefore which
    absences to expect. `extra` carries the keys only one generator has --
    docking's placement bookkeeping -- appended rather than interleaved, so
    the shared shape stays a contiguous block.
    """
    e_solute, e_solvent = references
    interactions = [c.interaction_eV for c in unique]
    # Absolute energies over the same candidates. `boltzmann_weights`
    # subtracts the minimum before exponentiating and E_int differs from
    # E(cluster) only by the constant e_solute + n * e_solvent, so these two
    # averages carry identical weights and cannot disagree.
    absolutes = [c.energy_eV for c in unique]
    n_frames_scored = len(candidates)

    # Frame-weighted, over the deduped basins: how much of the *trajectory*
    # actually sat in a contact state, as opposed to how many distinct
    # contact states were found (`mean_contacts` / `dissolved_fraction`
    # below). Quarantined from every energy in this summary -- occupancy
    # never enters `interaction_eV` or the Boltzmann averages, only these two
    # diagnostic fields. `None` throughout when the candidates carry no frame
    # counts, i.e. for a docked run, where a count of *placements* would look
    # like a population and would not be one.
    counted = all(c.n_frames is not None for c in unique)
    occupancy_mean_contacts = (
        float(sum(c.n_contacts * c.n_frames for c in unique) / n_frames_scored)
        if counted else None)
    occupancy_dissolved_fraction = (
        (float(sum(c.n_frames for c in unique if c.n_contacts == 0)
               / n_frames_scored) if n_solvent else 1.0)
        if counted else None)

    return {
        "run_dir": str(run_dir),
        "label": label,
        "pack_mode": pack_mode,
        "seed": seed,
        "n_solvent": n_solvent,
        "sampling_solvation": sampling_solvation,
        "scoring_solvation": (list(scoring_solvation) if scoring_solvation
                              else None),
        "calculator": calculator,
        "temperature_K": scoring.temperature_K,
        # Describes how a trajectory was subsampled, so it is a real absence
        # for a generator that has no trajectory.
        "max_frames": scoring.max_frames if pack_mode == "md" else None,
        "opt_fmax": scoring.fmax,
        "opt_steps": scoring.opt_steps,
        "n_frames_scored": n_frames_scored,
        "n_unique": len(unique),
        "e_solute_ref_eV": e_solute,
        "e_solvent_ref_eV": e_solvent,
        "min_interaction_kcal": min(interactions) * EV_TO_KCAL,
        "ensemble_interaction_kcal":
            ensemble_energy(interactions, scoring.temperature_K) * EV_TO_KCAL,
        "min_energy_eV": min(absolutes),
        "ensemble_energy_eV": ensemble_energy(absolutes, scoring.temperature_K),
        # Over the *distinct minima* -- how many kinds of basin the search
        # turned up, not how the trajectory's time was actually spent. See
        # `occupancy_mean_contacts` / `occupancy_dissolved_fraction` below
        # for the frame-weighted analogues.
        "mean_contacts": float(np.mean([c.n_contacts for c in unique])),
        # At n = 0 there is no solvent to dissolve, which is 1.0 here rather
        # than a division by zero.
        "dissolved_fraction": (
            float(np.mean([c.n_contacts == 0 for c in unique]))
            if n_solvent else 1.0),
        "scored_frame_spacing_fs": scored_frame_spacing_fs,
        "occupancy_mean_contacts": occupancy_mean_contacts,
        "occupancy_dissolved_fraction": occupancy_dissolved_fraction,
        "converged_fraction": float(np.mean([c.converged for c in candidates])),
        # Counted over the unique candidates because those are the ones that
        # carry Boltzmann weight. A candidate left with residual force is
        # exactly the contamination the tight fmax exists to remove, so it is
        # surfaced rather than dropped -- dropping would bias the ensemble
        # toward whichever basins happen to relax easily.
        "n_unconverged_unique": sum(1 for c in unique if not c.converged),
        # Named `sampling_*` because the scorer applies no wall: this
        # qualifies the geometries, not the energies computed from them here.
        "sampling_wall": sampling_wall,
        "n_scored_wall_active": sum(1 for c in candidates if c.wall_energy_eV),
        "candidates": [asdict(c) for c in unique],
        **extra,
    }


def assemble(run_dir, meta, records, indices, relaxed, references, solvation,
             calculator, scoring, out_name, reference_atoms=None):
    """Turn one run's optimised candidates into its summary and its files.

    Everything after the optimisations, and nothing that needs a calculator:
    contacts, dedupe, Boltzmann numbers, `best.xyz`,
    `<out_name>_candidates.xyz`, `<out_name>` and its `.log`. Contacts are
    microseconds of numpy, so this stays in the parent whether the
    optimisations ran here or in a pool.

    `reference_atoms`, when given, is `(solute_atoms, solvent_atoms)` -- the
    same relaxed geometries `references`' energies came from -- and gets
    written to `ref_solute.xyz` / `ref_solvent.xyz` in `run_dir`. Every run in
    a sweep shares one reference, so this writes the same two small files
    into every run directory rather than trying to pick one place to own
    them; a downstream DFT export reads either copy.
    """
    run_dir = Path(run_dir)
    if reference_atoms is not None:
        ref_solute_atoms, ref_solvent_atoms = reference_atoms
        write(run_dir / "ref_solute.xyz", ref_solute_atoms)
        write(run_dir / "ref_solvent.xyz", ref_solvent_atoms)
    n_solute = meta["n_solute"]
    aps = meta["atoms_per_solvent"]
    n_solvent = meta["n_solvent"]
    e_solute, e_solvent = references

    pairs = []
    for index, result in zip(indices, relaxed):
        gaps = solvent_molecule_gaps(result.atoms, n_solute, aps)
        pairs.append((Candidate(
            frame=index,
            energy_eV=result.energy_eV,
            interaction_eV=result.energy_eV - e_solute - n_solvent * e_solvent,
            converged=result.converged,
            fmax=result.fmax,
            n_contacts=int((gaps < CONTACT_GAP_A).sum()),
            n_solvent=n_solvent,
            min_gap_A=float(gaps.min()) if len(gaps) else float("nan"),
            wall_energy_eV=float(records[index]["wall_energy_eV"]),
            gnorm_Eh_bohr=result.gnorm_Eh_bohr,
            n_opt_steps=result.n_opt_steps,
            frames=[index],
        ), result.atoms))

    candidates = [c for c, _ in pairs]
    # Consecutive MD frames mostly relax into the same basin, so grouping
    # them is what turns a candidate list into an inherent-structure
    # population: each survivor's `n_frames` / `frames` (1 / [its own frame]
    # on a bare `Candidate`, which knows only about itself) is replaced with
    # its whole group's membership, which is what makes the occupancy fields
    # below a population over *scored frames* rather than over distinct
    # minima. `Σ n_frames == n_frames_scored` by construction: the groups
    # partition `pairs`.
    #
    # The counts used to be discarded here, on the grounds that a frame count
    # at the old 5 fs dump interval was pure autocorrelation -- ~100x
    # oversampled against a ~0.55 ps shell decorrelation time. At today's
    # defaults (50 fs dumps, `max_frames = 50` selecting down to ~200 fs of
    # scored-frame spacing on a 10 ps trajectory) that ratio is ~2.75x,
    # roughly the ~18 independent shell configurations the MD lengths were
    # chosen to buy, and every energy behind the count was computed anyway.
    #
    # `report.dedupe_groups` rather than a criterion of our own, because the
    # same question -- is this the same minimum? -- is asked again by
    # `report.pool_by_n` when the candidates of every packing at one n are
    # pooled, and the two answers have to agree by construction.
    unique = [
        (replace(pairs[g[0]][0], n_frames=len(g),
                 frames=sorted(pairs[i][0].frame for i in g)),
         pairs[g[0]][1])
        for g in dedupe_groups([c.energy_eV for c, _ in pairs])
    ]
    # How far apart, in fs, the frames actually sent to the optimiser are --
    # what the occupancy section has to be read against a decorrelation time
    # measured on the system, never a fabricated one. `indices` is already
    # ascending (built by `select_frames`); `None` for a single scored frame,
    # which has no spacing to report.
    scored_frame_spacing_fs = (
        float(np.mean(np.diff(indices))) * meta["dump_interval"]
        * meta["timestep_fs"] if len(indices) > 1 else None)

    best = min(range(len(pairs)), key=lambda i: pairs[i][0].interaction_eV)
    write(run_dir / "best.xyz", pairs[best][1])
    # Same order as summary["candidates"] below, so the two can be zipped by
    # index: frame i there is candidates[i].
    write(run_dir / f"{Path(out_name).stem}_candidates.xyz",
          [geometry for _, geometry in unique])

    summary = summarise(
        run_dir=run_dir,
        label=meta["label"],
        pack_mode="md",
        seed=meta["seed"],
        n_solvent=n_solvent,
        candidates=candidates,
        unique=[c for c, _ in unique],
        references=references,
        sampling_solvation=meta["solvation"],
        scoring_solvation=solvation,
        calculator=calculator,
        scoring=scoring,
        sampling_wall=wall_stats(records),
        scored_frame_spacing_fs=scored_frame_spacing_fs,
        extra={},
    )
    (run_dir / out_name).write_text(json.dumps(summary, indent=2))
    (run_dir / (Path(out_name).stem + ".log")).write_text(
        format_scored_log(summary, meta))
    return summary


def score_run_grid(jobs, scoring, n_workers=None):
    """Score many run directories at once, parallel over *candidates*.

    Scoring is the expensive half of a sweep: every candidate is a full
    geometry optimisation to `fmax = 0.002`, and at the defaults a four-point
    sweep spends a few minutes on MD and the better part of an hour here.

    The granularity matters as much as the parallelism. Fanning out over run
    *directories* gives a default sweep twelve tasks (4 n x 3 seeds) on an
    eighteen-core machine, and they are badly unequal -- n = 0 is a rigid
    10-atom molecule, n = 3 a floppy 25-atom cluster -- so cores idle from the
    start and the tail is one whole run directory long. Flattening to one task
    per candidate gives ~600 near-equal tasks instead, which is what the
    machine can actually balance, and the tail shrinks to a single
    optimisation. It also scales the right way with the solute, because
    `max_frames` fixes the task count regardless of system size.

    Measured on six run directories -- pyrazine, n = 0..2, two seeds, ten
    frames each, so 62 optimisations on 18 cores: 69.6 s serial, 34.5 s over
    run directories, 10.2 s over candidates.

    The two **references go into the same pool**, at the front of the task
    list, rather than being computed serially in the parent between the two
    phases: they are ordinary optimisations of the same shape and cost as any
    candidate. They are computed once per distinct (solute, solvent,
    calculator, continuum) rather than once per job -- recomputing per job
    would be n_jobs times the work and a way for two rows of one table to end
    up measured against slightly different zeros -- and a job that already
    carries `references` skips them entirely.

    `jobs` is a list of keyword dicts (`run_dir`, `solvation`, and optionally
    `calculator`, `calculator_kwargs`, `references`, `out_name`) rather than
    tuples of positional arguments, so a caller adding one more per-job knob
    does not have to widen a positional signature to reach it.

    Each job writes `<run_dir>/scored.json`, a human-readable `.log` beside
    it, the lowest-interaction-energy geometry as `<run_dir>/best.xyz`, and
    every deduped candidate as a multi-frame `scored_candidates.xyz` -- frame
    i there is `summary["candidates"][i]`. All are named after `out_name`, so
    rescoring the same trajectory in a second continuum gives
    `scored_acetone.json` / `.log` / `scored_acetone_candidates.xyz` rather
    than appending to one growing file. One job in the list is the
    rescore-by-hand path.

    Like every grid here the pool uses spawn, so a driver script must guard
    its call with `if __name__ == "__main__":`. See `solvate_md.pool_map`.
    """
    jobs = [dict(job) for job in jobs]
    if not jobs:
        return []

    selections = [select_frames(job["run_dir"], scoring.max_frames)
                  for job in jobs]
    for job, (meta, *_) in zip(jobs, selections):
        job["calculator"] = job.get("calculator") or meta["calculator"]

    def task(atoms, job):
        # freeze_n = 0: the scorer never constrains anything. Only docking's
        # screening pass does, and it is not this grid.
        return (atoms, job["calculator"], job["solvation"],
                job.get("calculator_kwargs"), scoring.fmax, scoring.opt_steps,
                0)

    # References first, so they are in flight before the candidates rather
    # than behind them: `assemble` cannot run for any job until its pair
    # lands.
    tasks, reference_at = [], {}
    for job, (meta, *_) in zip(jobs, selections):
        if job.get("references") is not None:
            continue
        # Everything the two reference optimisations depend on. `scoring`
        # cannot vary between jobs -- there is one of it -- so it is not in
        # the key.
        key = (meta["solute_path"], meta["solvent_path"], job["calculator"],
               tuple(job["solvation"] or ()),
               json.dumps(job.get("calculator_kwargs"), sort_keys=True))
        if key not in reference_at:
            reference_at[key] = len(tasks)
            tasks.append(task(align_to_principal_axes(
                read(meta["solute_path"])), job))
            tasks.append(task(read(meta["solvent_path"]), job))
        job["references_at"] = reference_at[key]

    starts = []
    for job, (_meta, _records, _indices, frames) in zip(jobs, selections):
        starts.append(len(tasks))
        tasks += [task(frame, job) for frame in frames]

    results = pool_map(relax, tasks, n_workers)

    summaries = []
    for job, selection, start in zip(jobs, selections, starts):
        meta, records, indices, frames = selection
        references = job.get("references")
        reference_atoms = None
        if references is None:
            at = job["references_at"]
            references = (results[at].energy_eV, results[at + 1].energy_eV)
            reference_atoms = (results[at].atoms, results[at + 1].atoms)
        summaries.append(assemble(
            job["run_dir"], meta, records, indices,
            results[start:start + len(frames)], references, job["solvation"],
            job["calculator"], scoring, job.get("out_name", "scored.json"),
            reference_atoms=reference_atoms))
    return summaries
