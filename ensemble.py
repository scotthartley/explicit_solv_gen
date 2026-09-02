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
import multiprocessing
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS

from report import (
    EV_TO_KCAL,
    boltzmann_weights,
    ensemble_energy,
    format_scored_log,
    wall_stats,
)
from solvate_md import _vdw_radii_array, align_to_principal_axes, get_calculator

# EV_TO_KCAL and the two Boltzmann helpers live in `report` so that formatting
# a number, or regenerating a log from JSON, never drags in ASE. They are
# imported here so callers can keep taking them from `ensemble`, and so the
# live and the regenerated logs weight identically by construction.

# A solvent molecule counts as touching the solute when some atom pair is
# within this much of van der Waals contact. Generous on purpose: it is a
# bookkeeping diagnostic, not a bond criterion.
CONTACT_GAP_A = 0.5

# 1 Eh/bohr in eV/A. ASE reports forces in eV/A and converges on the largest
# per-atom force; xtb reports a gradient norm over all 3N components in
# Eh/bohr. The two criteria are not comparable by eye -- fmax 0.05 eV/A is
# about 2.5x looser than xtb's `--opt normal` -- so both are recorded.
EV_A_PER_EH_BOHR = 51.42208619


@dataclass
class Candidate:
    """One optimised geometry, scored in the continuum."""

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
    # the weight it carries. Defaulted and last so that a `scored.json` written
    # before this existed still loads into `Candidate(**d)`.
    wall_energy_eV: float = None
    # The same convergence, in xtb's units: ||grad|| over all components, in
    # Eh/bohr, for comparison against `xtb --opt normal`'s 1e-3 threshold.
    # Defaulted and last for the same backward-compatibility reason.
    gnorm_Eh_bohr: float = None


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


def relax(atoms, calculator, solvation, calculator_kwargs=None, fmax=0.002,
          steps=1000):
    """Optimise `atoms` in the scoring environment. No wall -- see module doc."""
    a = atoms.copy()
    a.calc = get_calculator(calculator, solvation=solvation,
                            **(calculator_kwargs or {}))
    opt = BFGS(a, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    # Read both before anything else touches the geometry: `converged()`
    # re-evaluates against current positions rather than reporting on the run.
    converged = bool(opt.converged())
    forces = a.get_forces()
    final_fmax = float(np.linalg.norm(forces, axis=1).max())
    gnorm = float(np.linalg.norm(forces) / EV_A_PER_EH_BOHR)
    return a, float(a.get_potential_energy()), converged, final_fmax, gnorm


def reference_energies(solute_path, solvent_path, calculator, solvation,
                       calculator_kwargs=None, fmax=0.002, steps=1000):
    """Relaxed isolated solute and solvent, in the scoring environment.

    The solute reference is conformer-specific by construction -- it comes
    from `solute_path`, so AAA and BBB each get their own -- which is what
    makes an interaction energy comparable between conformers.
    """
    solute = align_to_principal_axes(read(solute_path))
    _, e_solute, _, _, _ = relax(solute, calculator, solvation,
                                 calculator_kwargs, fmax, steps)
    _, e_solvent, _, _, _ = relax(read(solvent_path), calculator, solvation,
                                  calculator_kwargs, fmax, steps)
    return e_solute, e_solvent


def dedupe(candidates, energy_tol_eV=1e-3):
    """Drop candidates that optimised into the same minimum.

    Consecutive MD frames mostly relax to the same structure, so without this
    a Boltzmann average is really an average over how long the trajectory
    happened to loiter somewhere. Energy-based rather than RMSD-based: it
    cannot distinguish two genuinely different structures that happen to be
    isoenergetic, which for ranking purposes costs nothing.
    """
    kept = []
    for c in sorted(candidates, key=lambda c: c.energy_eV):
        if all(abs(c.energy_eV - k.energy_eV) > energy_tol_eV for k in kept):
            kept.append(c)
    return kept


def score_run(run_dir, solvation, calculator=None, calculator_kwargs=None,
              stride=10, max_frames=40, fmax=0.002, steps=1000,
              temperature_K=298.0, references=None, out_name="scored.json"):
    """Rescore one `run_one_job` output directory in the continuum.

    Returns the parsed summary; also writes `<run_dir>/scored.json`, a
    human-readable `<run_dir>/scored.log` beside it, the
    lowest-interaction-energy geometry as `<run_dir>/best.xyz`, and every
    other deduped candidate as a multi-frame `<run_dir>/scored_candidates.xyz`
    -- frame i there is `summary["candidates"][i]`, so a structure of
    interest in the JSON (an odd contact count, a low weight that's still
    non-negligible) can be pulled back out rather than re-run from scratch.

    Both are named after `out_name`, so rescoring the same trajectory in a
    second continuum gives `scored_acetone.json` / `scored_acetone.log` /
    `scored_acetone_candidates.xyz` rather than appending to one growing
    file.
    """
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "metadata.json").read_text())
    calculator = calculator or meta["calculator"]
    n_solute = meta["n_solute"]
    aps = meta["atoms_per_solvent"]
    n_solvent = meta["n_solvent"]

    # Trajectory dump i and energies[i] are written by the same `record()`
    # closure in `solvate_md`, so a candidate's dump index reads its wall
    # energy straight out of this list. Absent for an old or hand-made run
    # directory, in which case every wall field below stays None.
    energies_path = run_dir / "energies.json"
    records = json.loads(energies_path.read_text()) if energies_path.exists() else []

    if references is None:
        references = reference_energies(
            meta["solute_path"], meta["solvent_path"], calculator, solvation,
            calculator_kwargs, fmax, steps)
    e_solute, e_solvent = references

    frames = read(str(run_dir / "traj.xyz"), index=f"::{stride}")
    # Carried alongside so a candidate can name the dump it actually came
    # from. `i * stride` would index the subsampled list instead, and label a
    # 300-dump trajectory 0..70 as soon as `max_frames` bites.
    indices = list(range(0, len(frames) * stride, stride))
    if max_frames is not None and len(frames) > max_frames:
        # Spread over the whole trajectory rather than taking a prefix.
        sel = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in sel]
        indices = [indices[i] for i in sel]

    candidates, geometries = [], []
    for index, frame in zip(indices, frames):
        opt_atoms, energy, converged, final_fmax, gnorm = relax(
            frame, calculator, solvation, calculator_kwargs, fmax, steps)
        gaps = solvent_molecule_gaps(opt_atoms, n_solute, aps)
        candidates.append(Candidate(
            frame=index,
            energy_eV=energy,
            interaction_eV=energy - e_solute - n_solvent * e_solvent,
            converged=converged,
            fmax=final_fmax,
            n_contacts=int((gaps < CONTACT_GAP_A).sum()),
            n_solvent=n_solvent,
            min_gap_A=float(gaps.min()) if len(gaps) else float("nan"),
            wall_energy_eV=(float(records[index]["wall_energy_eV"])
                            if index < len(records) else None),
            gnorm_Eh_bohr=gnorm,
        ))
        geometries.append(opt_atoms)

    # Keyed by frame rather than zipped positionally: `dedupe` sorts by
    # energy, so `unique`'s order no longer matches `candidates`'/`geometries`'.
    # `frame` is the trajectory dump index and unique per candidate.
    geometry_by_frame = {c.frame: g for c, g in zip(candidates, geometries)}
    unique = dedupe(candidates)
    interactions = [c.interaction_eV for c in unique]
    # Absolute energies over the same candidates. `boltzmann_weights`
    # subtracts the minimum before exponentiating and E_int differs from
    # E(cluster) only by the constant e_solute + n * e_solvent, so these two
    # averages carry identical weights and cannot disagree.
    absolutes = [c.energy_eV for c in unique]
    best = min(range(len(candidates)), key=lambda i: candidates[i].interaction_eV)
    write(run_dir / "best.xyz", geometries[best])
    # Same order as summary["candidates"] below, so the two can be zipped by
    # index. write() to a multi-frame .xyz appends frames in list order, not
    # dict order, hence building the list from `unique` rather than the dict.
    write(run_dir / f"{Path(out_name).stem}_candidates.xyz",
          [geometry_by_frame[c.frame] for c in unique])

    summary = {
        "run_dir": str(run_dir),
        "label": meta["label"],
        "seed": meta["seed"],
        "n_solvent": n_solvent,
        "sampling_solvation": meta["solvation"],
        "scoring_solvation": list(solvation) if solvation else None,
        "calculator": calculator,
        "temperature_K": temperature_K,
        "stride": stride,
        "max_frames": max_frames,
        "opt_fmax": fmax,
        "opt_steps": steps,
        "n_frames_scored": len(candidates),
        "n_unique": len(unique),
        "e_solute_ref_eV": e_solute,
        "e_solvent_ref_eV": e_solvent,
        "min_interaction_kcal": min(interactions) * EV_TO_KCAL,
        "ensemble_interaction_kcal":
            ensemble_energy(interactions, temperature_K) * EV_TO_KCAL,
        "min_energy_eV": min(absolutes),
        "ensemble_energy_eV": ensemble_energy(absolutes, temperature_K),
        "mean_contacts": float(np.mean([c.n_contacts for c in unique])),
        "dissolved_fraction":
            float(np.mean([c.n_contacts == 0 for c in unique])) if n_solvent else 1.0,
        "converged_fraction": float(np.mean([c.converged for c in candidates])),
        # Counted over the unique candidates because those are the ones that
        # carry Boltzmann weight. A candidate left with residual force is
        # exactly the contamination the tight fmax exists to remove, so it is
        # surfaced rather than dropped -- dropping would bias the ensemble
        # toward whichever basins happen to relax easily.
        "n_unconverged_unique": sum(1 for c in unique if not c.converged),
        # Named `sampling_*` because the scorer applies no wall: this qualifies
        # the geometries, not the energies computed from them here.
        "sampling_wall": wall_stats(records) if records else None,
        "n_scored_wall_active": (
            sum(1 for c in candidates if c.wall_energy_eV) if records else None),
        "candidates": [asdict(c) for c in unique],
    }
    (run_dir / out_name).write_text(json.dumps(summary, indent=2))
    (run_dir / (Path(out_name).stem + ".log")).write_text(
        format_scored_log(summary, meta))
    return summary


def _score_run_worker(kwargs):
    return score_run(**kwargs)


def score_run_grid(jobs, n_workers=None):
    """Score many run directories at once, one single-threaded worker each.

    Scoring is the expensive half of a sweep. Each candidate is a full
    geometry optimisation to `fmax = 0.002`, and there are `max_frames` of
    them per run directory against a single MD trajectory, so leaving this
    loop serial made a sweep single-threaded in practice however many cores
    the MD grid had just used. Measured on a 9-run pyrazine/chloroform sweep,
    scoring serially took 108.8 s against 35.8 s here, a 3.0x wall-clock win
    on a job mix dominated by its three largest runs.

    The work is embarrassingly parallel -- each `score_run` reads and writes
    only its own run directory, and every candidate is an independent
    optimisation -- so there is nothing to coordinate. Results are unchanged
    to within tblite's own run-to-run jitter, which is ~1e-11 eV and shows up
    equally in the references computed serially in the parent.

    `jobs` is a list of keyword dicts for `score_run`, rather than a tuple of
    positional arguments, so this does not have to restate that signature and
    cannot fall behind it.

    Hand the workers the `references` rather than letting each recompute them:
    they are the same two optimisations for every job in a sweep, so
    recomputing is both n_jobs times the work and a way for two rows of one
    table to end up measured against slightly different zeros.

    Like `run_job_grid`, the pool uses "spawn", so each worker re-imports the
    calling module and **a driver script must guard its call** with
    `if __name__ == "__main__":`. `n_sweep.main()` already is.
    """
    jobs = list(jobs)
    if not jobs:
        return []
    n_workers = min(n_workers or os.cpu_count(), len(jobs))
    # One job, or one worker, in this process: spawning an interpreter to run
    # a single job costs more than it saves, and an exception arrives with a
    # useful traceback instead of a pickled one.
    if n_workers == 1:
        return [score_run(**job) for job in jobs]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        # `map` preserves input order, so the summaries come back in the same
        # order the serial loop produced them and the report tables are
        # unchanged.
        return pool.map(_score_run_worker, jobs)
