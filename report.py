"""Human-readable logs for the solvation pipeline.

Everything else in this package writes JSON, which is the right format for the
next program and the wrong one for a person. A 74 KB array of per-dump energy
dictionaries technically contains `wall_energy_eV` -- the one diagnostic that
says whether a run's energies are trustworthy at all -- but nobody reads it,
and a long run gives no sign of life until it finishes (piping it through
`tail` does not help: the pipe buffers until it closes).

So this module renders the same numbers as text, in the Gaussian-ish idiom of
a header, a streaming table, and a footer. `RunLogger` flushes every write, so
`run.log` is genuinely tailable while MD is running.

Deliberately import-light -- no ASE, no tblite, no torch at module scope --
because `solvate_md` imports it into every spawned MD worker. That is also why
the Boltzmann weighting lives here rather than in `ensemble`: it needs nothing
from ASE but a constant, and putting it beside the formatting means the live
log and a log regenerated from JSON weight identically by construction rather
than by comment. `ensemble` imports both functions back.

Regenerate logs for anything already on disk, without running any MD:

    python -m report path/to/sweep_or_run_dir/
"""

import single_thread  # noqa: F401  -- must precede numpy; see its docstring

import json
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

# Bump on any change to the pipeline's numerics or output shapes -- it lands
# in every sweep's params block via `n_sweep.sweep_params`, so a report can be
# matched back to the code that produced it.
VERSION = "0.4.0"

# Live here rather than in `ensemble` so that a text-only consumer never has to
# import ASE to format or weight a number. `ensemble` re-exports both.
EV_TO_KCAL = 23.060548
KB_EV_PER_K = 8.617333262e-5   # Boltzmann constant, == ase.units.kB

# Above this fraction of frames with a nonzero wall energy, the confinement is
# load-bearing rather than a safety net. Measured regimes: 4-5% for gas-phase
# sampling (shell self-bound), 50-60% for ALPB(water) (shell dissociating).
WALL_WARN_FRACTION = 0.20

_RULE = "=" * 78


def git_commit():
    """Short HEAD of the repo this module lives in, or None."""
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent),
             "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def kv_block(title, mapping, indent="  "):
    """An aligned key/value section -- the common building block of every log.

    A value containing newlines is wrapped onto continuation lines indented to
    the value column, so a key whose explanation runs to several lines is one
    entry in the mapping. It used to take one entry per line, keyed by "",
    " ", "  " and so on -- distinct strings so the dict kept them all, padded
    to the key width so they printed as blanks.
    """
    rows = [(str(k), "-" if v is None else str(v)) for k, v in mapping.items()]
    width = max((len(k) for k, _ in rows), default=0)
    lines = [title, "-" * len(title)]
    for key, value in rows:
        first, *rest = value.split("\n")
        lines.append(f"{indent}{key:<{width}}   {first}".rstrip())
        lines += [f"{indent}{'':<{width}}   {line}".rstrip() for line in rest]
    return "\n".join(lines)


def banner(text):
    return f"{_RULE}\n {text}\n{_RULE}"


# Two optimised energies this close are one minimum. The criterion lives here,
# beside the Boltzmann weighting and for the same reason: `ensemble` dedupes
# the candidates of one run with it, and `pool_by_n` dedupes the pooled
# candidates of every seed at one n with it, and the two must be the same test
# by construction rather than by comment.
DEDUPE_TOL_EV = 1e-3


def dedupe_energies(energies_eV, tol_eV=DEDUPE_TOL_EV):
    """Indices of the distinct minima, lowest first.

    Energy-based rather than RMSD-based: it cannot tell two genuinely
    different structures apart when they happen to be isoenergetic, which for
    ranking purposes costs nothing, and it needs nothing but the numbers
    already in `scored.json` -- no geometries, no ASE, no calculator.
    """
    order = sorted(range(len(energies_eV)), key=lambda i: energies_eV[i])
    kept = []
    for i in order:
        if all(abs(energies_eV[i] - energies_eV[k]) > tol_eV for k in kept):
            kept.append(i)
    return kept


def boltzmann_weights(energies_eV, temperature_K=298.0):
    """Weights over a set of energies, minimum subtracted before exponentiating.

    Subtracting the minimum is not only for numerical range: it makes the
    weights invariant under a constant shift of the energies. E_int differs
    from E(cluster) by exactly such a constant -- E(solute) + n E(solvent) --
    so an average over one and an average over the other carry identical
    weights and cannot disagree.
    """
    e = np.asarray(energies_eV, dtype=float)
    w = np.exp(-(e - e.min()) / (KB_EV_PER_K * temperature_K))
    return w / w.sum()


def ensemble_energy(energies_eV, temperature_K=298.0):
    """Boltzmann-weighted mean.

    Not a free energy: there is no vibrational or configurational entropy in
    here, and the weights come from whatever geometries the generator found.
    It is an ensemble-averaged energy, and should be reported as one.
    """
    w = boltzmann_weights(energies_eV, temperature_K)
    return float(np.dot(w, np.asarray(energies_eV, dtype=float)))


def _num(value, spec):
    return "-" if value is None else format(value, spec)


def _solvation_str(solvation):
    if not solvation:
        return "gas phase (no continuum)"
    model, name = solvation
    return f"{model.upper()}({name})"


# --------------------------------------------------------------------------
# MD runs
# --------------------------------------------------------------------------

_MD_COLUMNS = (
    f"{'step':>8} {'time/ps':>9} {'E_pot/eV':>15} {'E_kin/eV':>10} "
    f"{'E_tot/eV':>15} {'T/K':>8} {'E_wall/eV':>11}"
)


def format_run_header(meta, extra=None):
    """Header for one MD run, built from a `metadata.json`-shaped mapping.

    Taking the metadata dict rather than the `Condition` and `Packing` objects
    keeps the live log and the post-hoc regeneration on one code path, and
    guarantees the log cannot describe a run differently from the JSON beside
    it. `extra` carries the few `Condition` fields metadata.json does not
    record (`shell_fill`, `wall_slack`), and is omitted when regenerating.
    """
    extra = dict(extra or {})
    n_solvent = meta["n_solvent"]
    aps = meta["atoms_per_solvent"]
    axes = meta["shell_semi_axes"]
    padding = meta["shell_padding"]
    dump = meta["dump_interval"]
    n_steps = meta["n_steps"]
    dt = meta["timestep_fs"]

    parts = [banner(f"Explicit-solvent MD: {meta['label']} "
                    f"(seed {meta['seed']})")]

    parts.append(kv_block("System", {
        "solute": f"{meta['solute_path']}  ({meta['n_solute']} atoms)",
        "solvent": f"{meta['solvent_path']}  ({aps} atoms x {n_solvent})",
        "solvent name": meta["solvent"],
        "total atoms": meta["n_atoms"],
    }))

    packing = {
        "solute semi-axes/A": "  ".join(f"{a:.3f}" for a in axes),
        "shell padding/A": f"{padding:.3f}",
        "region semi-axes/A": "  ".join(f"{a + padding:.3f}" for a in axes),
        "packmol tolerance/A": f"{meta['tolerance']:.2f}",
    }
    if "shell_fill" in extra:
        packing["shell fill"] = f"{extra['shell_fill']:.2f}"
    if meta["pack_directions"]:
        packing["arrangement"] = (
            f"stratified, clustering {meta['pack_clustering']:.2f} "
            "(0 = spread, 1 = one face)")
        packing["hemisphere axes"] = "\n".join(
            "  ".join(f"{x:+.3f}" for x in d) for d in meta["pack_directions"])
    else:
        packing["arrangement"] = "unconstrained (one packmol block)"
    packing["wall distance/A"] = f"{meta['wall_distance']:.3f}"
    if "wall_slack" in extra:
        packing["wall slack"] = f"{extra['wall_slack']:.2f} solvent diameters"
    packing["wall k/eV A^-2"] = f"{meta['wall_k']:.3f}"
    parts.append(kv_block("Packing", packing))

    parts.append(kv_block("Sampling Hamiltonian", {
        "calculator": meta["calculator"],
        # None is a real value here -- gas-phase sampling -- not a missing one.
        "continuum": _solvation_str(meta["solvation"]),
        "wall": "SolventShellWall on solvent atoms only",
    }))

    parts.append(kv_block("Pre-MD relaxation (clash relief, fmax 0.5)", {
        "converged": "yes" if meta["relax_converged"] else "no",
        "final fmax/eV A^-1": f"{meta['relax_final_fmax']:.4f}",
    }))

    parts.append(kv_block("Molecular dynamics", {
        "ensemble": "Langevin NVT, FixCom",
        "temperature/K": f"{meta['temperature_K']:.1f}",
        "timestep/fs": f"{dt:.2f}",
        "friction": f"{meta['friction']:.4f}",
        "equilibration": (f"{meta['n_equilibrate_steps']} steps "
                          f"({meta['n_equilibrate_steps'] * dt / 1000:.2f} ps)"),
        "production": f"{n_steps} steps ({n_steps * dt / 1000:.2f} ps)",
        "dump interval": f"{dump} steps ({n_steps // dump} frames expected)",
    }))

    parts.append("Trajectory\n----------")
    parts.append(_MD_COLUMNS)
    return "\n\n".join(parts)


def format_md_row(record):
    return (f"{record['step']:>8d} "
            f"{record['time_fs'] / 1000.0:>9.3f} "
            f"{record['potential_energy_eV']:>15.6f} "
            f"{record['kinetic_energy_eV']:>10.6f} "
            f"{record['total_energy_eV']:>15.6f} "
            f"{record['temperature_K']:>8.1f} "
            f"{record['wall_energy_eV']:>11.6f}")


def wall_stats(records):
    """Aggregate `wall_energy_eV` over a list of `energies.json` records.

    The one number that says whether a run's energies are trustworthy, so it
    is computed once here and rendered in three places -- the MD footer, the
    scored log and the sweep table -- rather than recomputed in each.
    """
    n = len(records)
    walls = [r["wall_energy_eV"] for r in records]
    n_active = sum(1 for w in walls if w > 0.0)
    return {
        "n_frames": n,
        "n_wall_active": n_active,
        "wall_active_fraction": (n_active / n) if n else None,
        "max_wall_energy_eV": max(walls) if walls else None,
    }


def wall_warning(fraction, indent="  "):
    """The load-bearing-wall warning, or None when the run is under threshold."""
    if fraction is None or fraction <= WALL_WARN_FRACTION:
        return None
    return "\n".join(indent + line for line in (
        f"** WARNING: the confining wall is active in {100 * fraction:.0f}% of "
        "frames.",
        "** Measured regimes are 4-5% (gas-phase sampling, shell self-bound,",
        "** wall a safety net) versus 50-60% (ALPB(water), shell dissociating).",
        f"** Above {100 * WALL_WARN_FRACTION:.0f}% the wall is load-bearing: it "
        "is holding together a shell",
        "** the Hamiltonian wants to disperse, and these energies are",
        "** contaminated by the confinement. Sample in gas phase, or accept",
        "** that there is no bound shell to find.",
    ))


def unconverged_warning(n_unconverged, n_unique, indent="  "):
    """Warning for candidates that hit the optimiser step cap, or None.

    Unconverged candidates are kept rather than dropped -- dropping them would
    bias the ensemble toward whichever basins relax easily -- so the Boltzmann
    average has to say out loud that some of its weight sits on geometries
    that are not minima.
    """
    if not n_unconverged:
        return None
    return "\n".join(indent + line for line in (
        f"** WARNING: {n_unconverged} of {n_unique} unique candidates hit the "
        "optimiser step cap",
        "** without converging (the NO rows in the conv column above). They "
        "still carry",
        "** Boltzmann weight, and residual force is worth up to a few tenths "
        "of a",
        "** kcal/mol per candidate -- the same contamination the tight fmax "
        "exists to",
        "** remove. Raise --opt-steps, or discount these rows by hand.",
    ))


def format_run_footer(records, target_K=None, elapsed_s=None):
    if not records:
        return kv_block("MD summary", {"frames written": 0})

    n = len(records)
    temps = [r["temperature_K"] for r in records]
    epots = [r["potential_energy_eV"] for r in records]
    mean_T = sum(temps) / n
    mean_E = sum(epots) / n
    sd_E = (sum((e - mean_E) ** 2 for e in epots) / n) ** 0.5
    wall = wall_stats(records)
    frac = wall["wall_active_fraction"]

    summary = {
        "frames written": n,
        "mean temperature/K": (f"{mean_T:.1f}" + (f"  (target {target_K:.1f})"
                                                  if target_K else "")),
        "E_pot/eV": f"{mean_E:.6f} +/- {sd_E:.6f}",
        "wall active": (f"{wall['n_wall_active']} / {n} frames "
                        f"({100 * frac:.1f}%)"),
        "max wall energy/eV": f"{wall['max_wall_energy_eV']:.6f}",
    }
    if elapsed_s is not None:
        summary["elapsed"] = f"{elapsed_s:.1f} s"
    block = kv_block("MD summary", summary)

    warning = wall_warning(frac)
    if warning:
        block += "\n\n" + warning
    return block


class RunLogger:
    """Streams `run.log` for one MD job.

    Every write is flushed, so the file can be followed with `tail -f` while
    the run is in progress. Per-dump I/O is negligible against an SCF.
    """

    def __init__(self, path, live=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w")
        self._live = live
        self._t0 = time.time() if live else None
        self._target_K = None

    def _write(self, text):
        self._fh.write(text + "\n")
        self._fh.flush()

    def header(self, meta, extra=None):
        self._target_K = meta.get("temperature_K")
        if not self._live:
            self._write(f"[regenerated from JSON on {timestamp()}]\n")
        self._write(format_run_header(meta, extra))

    def md_row(self, record):
        """Log one trajectory dump. Takes the same dict `energies.json` gets."""
        self._write(format_md_row(record))

    def footer(self, records):
        """Summarise the run. `records` is the list `md_row` was fed.

        Passed in rather than accumulated here: the caller already holds that
        list -- it is what becomes `energies.json` -- and one copy cannot fall
        out of step with the other.
        """
        elapsed = None if self._t0 is None else time.time() - self._t0
        self._write("")
        self._write(format_run_footer(records, self._target_K, elapsed))

    def close(self):
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _candidate_table(candidates, temperature_K):
    weights = boltzmann_weights([c["energy_eV"] for c in candidates],
                                temperature_K)
    lines = [f"{'frame':>7} {'E(cluster)/eV':>16} {'E_int/kcal':>12} "
             f"{'weight':>8} {'contacts':>9} {'min gap/A':>10} {'conv':>5} "
             f"{'fmax':>7} {'gnorm':>9} {'steps':>6} {'E_wall/eV':>11}"]
    for c, w in zip(candidates, weights):
        # NaN when there is no solvent to measure a gap to, i.e. at n = 0.
        gap = c["min_gap_A"]
        gap_s = "-" if gap != gap else f"{gap:.3f}"
        # fmax and gnorm are the same convergence in ASE's and xtb's units:
        # per-atom max force in eV/A, and the norm over all components in
        # Eh/bohr.
        gnorm_s = f"{c['gnorm_Eh_bohr']:.2e}"
        # Optimiser steps, so a slow sweep can be attributed to the tight fmax
        # or to BFGS before anyone swaps optimisers on a guess.
        steps_s = str(c["n_opt_steps"])
        # Of the *sampling* frame this candidate came from -- the scorer
        # applies no wall. "-" for a docked candidate, which has no sampling
        # frame and so no wall energy to report -- a real absence, not a
        # zero.
        wall_s = _num(c["wall_energy_eV"], ".6f")
        lines.append(
            f"{c['frame']:>7d} {c['energy_eV']:>16.6f} "
            f"{c['interaction_eV'] * EV_TO_KCAL:>12.2f} "
            f"{w:>8.3f} {c['n_contacts']:>9d} {gap_s:>10} "
            f"{'yes' if c['converged'] else 'NO':>5} {c['fmax']:>7.4f} "
            f"{gnorm_s:>9} {steps_s:>6} {wall_s:>11}".rstrip())
    return "\n".join(lines)


def _sampling_wall_str(summary):
    """One-line summary of the wall during the MD that produced these frames."""
    # None for a docked candidate -- there is no sampling frame at all, and
    # for an MD run that dumped no frames, which `wall_stats` reports rather
    # than guessing at.
    wall = summary["sampling_wall"]
    if wall is None or wall["wall_active_fraction"] is None:
        return None
    return (f"active in {100 * wall['wall_active_fraction']:.1f}% of "
            f"{wall['n_frames']} frames "
            f"(max {wall['max_wall_energy_eV']:.6f} eV)"
            f"; {summary['n_scored_wall_active']} of "
            f"{summary['n_frames_scored']} scored frames affected")


def format_scored_log(summary, meta):
    """The scoring section: provenance, references, candidates, result."""
    n = summary["n_solvent"]
    T = summary["temperature_K"]
    e_solute = summary["e_solute_ref_eV"]
    e_solvent = summary["e_solvent_ref_eV"]
    candidates = summary["candidates"]

    parts = [banner(f"Continuum rescoring: {summary['label']} "
                    f"(seed {summary['seed']})")]

    max_frames = summary["max_frames"]
    parts.append(kv_block("Provenance", {
        "run directory": summary["run_dir"],
        # Missing on anything scored before 0.4.0 -- defaults to "md" rather
        # than failing, since it is informational and never enters a
        # computation, unlike the mandatory wall fields.
        "pack mode": summary.get("pack_mode", "md"),
        "trajectory": (f"traj.xyz, every {summary['stride']}th frame"
                       + (f", at most {max_frames}" if max_frames else "")
                       + f"  ->  {summary['n_frames_scored']} scored"),
        "sampling Hamiltonian": (
            f"{summary['calculator']}, "
            f"{_solvation_str(summary['sampling_solvation'])}"),
        # The frames scored below were drawn from a walled trajectory, so how
        # hard that wall was working qualifies every energy in this file.
        "sampling wall": _sampling_wall_str(summary),
        "sampling MD": (f"{meta['n_steps']} steps x {meta['timestep_fs']} fs "
                        f"at {meta['temperature_K']} K, dumped every "
                        f"{meta['dump_interval']}"),
    }))

    # `best.xyz` is the geometry relaxed here, so state the surface it was
    # relaxed on: reproducing it in the xtb binary needs the same continuum,
    # and the two codes converge on criteria that are not comparable by eye.
    solvation = summary["scoring_solvation"] or []
    alpb = f" --alpb {solvation[1]}" if len(solvation) > 1 and solvation[1] else ""
    parts.append(kv_block("Scoring", {
        "calculator": summary["calculator"],
        "continuum": _solvation_str(summary["scoring_solvation"]),
        "optimiser": (f"BFGS, fmax {summary['opt_fmax']} eV/A "
                      "(ASE: largest per-atom force). xtb's\n"
                      "`--opt normal` instead stops at a gradient norm of "
                      "1e-3 Eh/a -- see\n"
                      f"the gnorm column. Max {summary['opt_steps']} steps"),
        "reproduce": (f"xtb best.xyz --gfn 2{alpb} --sp\n"
                      "best.xyz was relaxed WITH the continuum -- omitting "
                      + (f"--alpb {solvation[1]}" if alpb else "it")
                      + "\nputs it on a different surface"),
        "wall": ("none -- dissolution is the signal, not a failure;\n"
                 "a solvent molecule that optimises away into the "
                 "continuum contributes ~0 to E_int"),
    }))

    offset = e_solute + n * e_solvent
    parts.append(kv_block("References (each relaxed in the scoring environment)", {
        "E(solute)/eV": f"{e_solute:.6f}",
        "E(solvent)/eV": f"{e_solvent:.6f}",
        "offset/eV": f"E(solute) + {n} x E(solvent) = {offset:.6f}",
    }))

    if candidates:
        parts.append("Candidates (unique minima, lowest first)\n"
                     "----------------------------------------\n"
                     + _candidate_table(candidates, T))

    result = {
        "E_int      (Boltzmann)": (
            f"{summary['ensemble_interaction_kcal']:.2f} kcal/mol"
            "     <- the reported number"),
        "E_int      (minimum)": f"{summary['min_interaction_kcal']:.2f} kcal/mol",
        "E(cluster) (Boltzmann)": f"{summary['ensemble_energy_eV']:.6f} eV",
        "E(cluster) (minimum)": f"{summary['min_energy_eV']:.6f} eV",
        "references": (f"E(solute) {e_solute:.6f} eV, "
                       f"{n} x E(solvent) {e_solvent:.6f} eV"),
        "frames scored": (f"{summary['n_frames_scored']} "
                          f"({summary['n_unique']} unique, "
                          f"{100 * summary['converged_fraction']:.0f}"
                          "% converged)"),
        "mean contacts": f"{summary['mean_contacts']:.2f}",
        "dissolved": (f"{100 * summary['dissolved_fraction']:.0f}"
                      "% of unique candidates"),
    }
    parts.append(kv_block("Result", result) + f"""

  The Boltzmann average is over the unique candidates at {T:.1f} K. It is an
  ensemble-averaged *energy*, not a free energy: no vibrational or
  configurational entropy is in it, and the weights come from whatever
  geometries the generator happened to find.

  E_int is the reported number because it alone is comparable across n.
  E(cluster) differs from it only by the constant offset above, so the two
  averages carry identical weights -- it is shown as a raw number to
  sanity-check the small differences of large energies against.""")

    # After the Result block rather than up in Provenance: these are the
    # energies the confinement contaminated. None for a docked candidate,
    # which has no sampling wall at all.
    warning = (wall_warning(summary["sampling_wall"]["wall_active_fraction"])
              if summary["sampling_wall"] is not None else None)
    if warning:
        parts.append(warning)
    # Same reason, one line up the stack: these are the energies a residual
    # force contaminated.
    warning = unconverged_warning(summary["n_unconverged_unique"],
                                  summary["n_unique"])
    if warning:
        parts.append(warning)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------

def _wall_fraction(summary):
    """Wall-active fraction of the run behind this summary, or None."""
    return summary["sampling_wall"]["wall_active_fraction"]


def _row_order(summary):
    """Sort key for every per-run table: by n, then by seed."""
    return summary["n_solvent"], summary["seed"]


def _increments(pooled):
    """`dE_int(n) = E_int(min, n) - E_int(min, n-1)`, keyed by n.

    The increment is the readable quantity in an n-sweep, because the level is
    not: see the note under the table for why `E_int(n)` tends to a line
    rather than to a plateau.

    Keyed by n alone, over the pooled minima, because a seed is a search and
    not a replica -- seed 0 at n = 2 is not seed 0 at n = 1 with a molecule
    added, it is an independent packing, so pairing the two by index paired
    nothing physical and threw away every seed that had no counterpart. Only
    the minimum gets an increment: a second one on the ensemble average would
    only invite reporting whichever of the two looks better.
    """
    by_n = {p["n_solvent"]: p["e_int_min_kcal"] for p in pooled}
    return {n: value - previous for n, value in by_n.items()
            if (previous := by_n.get(n - 1)) is not None}


def pool_by_n(summaries):
    """Pool every seed's candidates at each n, and score the pool as one set.

    Seeds are independent *searches*, not replicas of a physical system: they
    differ in packing, and under stratified packing they differ in packing by
    design. Averaging over them therefore penalises searching more widely --
    it dilutes the one packing that found the both-nitrogens geometry with the
    six that missed it -- and the mean of several minima is a number no
    geometry has. So the reported number at each n is the minimum over the
    pooled candidates of every seed, and the ensemble average is taken over
    the same pool.

    Pooling requires a cross-seed dedupe, not just a concatenation: seven
    seeds that all find one basin would otherwise multiply its Boltzmann
    weight by seven, which is an artefact of how the search was spent rather
    than a degeneracy. `dedupe_energies` is the same criterion `ensemble`
    applies within a run.

    That is only legitimate because the references are sweep-wide:
    `score_run_grid` computes them once per distinct (solute, solvent,
    calculator, continuum), so every `interaction_eV` in a sweep shares a
    zero. A sweep whose rows were measured against different zeros is broken
    rather than old, so this raises rather than pooling them anyway.

    Returns one record per n, sorted by n.
    """
    references = {(s["e_solute_ref_eV"], s["e_solvent_ref_eV"])
                  for s in summaries}
    if len(references) > 1:
        raise ValueError(
            "the runs in this sweep were scored against different references "
            f"({sorted(references)}); their interaction energies do not share "
            "a zero and cannot be pooled. Rescore the sweep in one pass.")

    groups = {}
    for s in summaries:
        groups.setdefault(s["n_solvent"], []).append(s)

    pooled = []
    for n, group in sorted(groups.items()):
        # Deterministic ties: `dedupe_energies` sorts by energy before
        # keeping anything, so this ordering never changes which candidates
        # survive -- only which packing gets the credit when several tie.
        group = sorted(group, key=_row_order)
        tagged = [(s, c) for s in group for c in s["candidates"]]
        keep = [tagged[i] for i in
                dedupe_energies([c["energy_eV"] for _, c in tagged])]
        interactions = [c["interaction_eV"] for _, c in keep]
        absolutes = [c["energy_eV"] for _, c in keep]
        e_min = min(interactions)
        temperature = group[0]["temperature_K"]
        # A seed "found" the pooled minimum when its own best candidate is the
        # same minimum by the same test used everywhere else.
        seed_minima = [min(c["interaction_eV"] for c in s["candidates"])
                       for s in group]
        walls = [f for s in group if (f := _wall_fraction(s)) is not None]
        # `dedupe_energies` sorts ascending, so `keep[0]` is not just *a*
        # minimum -- it is the pooled minimum, and therefore its own
        # packing's minimum too: that packing's `best.xyz` is the geometry
        # behind `e_int_min_kcal` below.
        weights = boltzmann_weights([c["energy_eV"] for _, c in keep],
                                    temperature)
        s_best, c_best = keep[0]
        pooled.append({
            "n_solvent": n,
            "n_seeds": len(group),
            "pool": len(keep),
            "e_int_min_kcal": e_min * EV_TO_KCAL,
            "e_int_ens_kcal":
                ensemble_energy(interactions, temperature) * EV_TO_KCAL,
            # Identical weights to the line above, by the minimum-subtraction
            # argument in `boltzmann_weights`.
            "e_cluster_ens_eV": ensemble_energy(absolutes, temperature),
            "found_by": sum(1 for m in seed_minima
                            if abs(m - e_min) <= DEDUPE_TOL_EV),
            "seed_minima_kcal": [m * EV_TO_KCAL for m in seed_minima],
            "mean_contacts": float(np.mean([c["n_contacts"] for _, c in keep])),
            # At n = 0 there is no solvent to dissolve, and `assemble` calls
            # that 1.0 rather than dividing by zero.
            "dissolved_fraction": (
                float(np.mean([c["n_contacts"] == 0 for _, c in keep]))
                if n else 1.0),
            # The worst of the seeds, matching `format_wall_diagnostic`: the
            # question is whether any of the pooled energies is contaminated.
            "wall": max(walls) if walls else None,
            "best": {"run": Path(s_best["run_dir"]).name,
                     "seed": s_best["seed"],
                     "weight": float(weights[0]),
                     "candidate": c_best},
        })
    return pooled


def format_table(params, pooled):
    """E_int vs n over the pooled candidates, with the search effort behind it.

    One sweep is one solute in one solvent, so what is being measured is
    stated once above the table rather than repeated down a column of
    identical values. It comes from `params`, which is also why no summary
    carries a `solute_label` of its own.
    """
    deltas = _increments(pooled)
    capacity = params["monolayer_capacity"]
    header = (f"{'n':>3} {'cover':>6} {'E_int(min)':>12} {'E_int(ens)':>12} "
              f"{'dE_int':>10} {'found by':>9} {'pool':>6} "
              f"{'E(cluster)/eV':>15} {'contacts':>9} {'dissolved':>10} "
              f"{'wall':>6}")
    lines = [f"leg: {params['solute_label']}/{params['solvent']}",
             f"full first shell: ~{capacity:.0f} solvent molecules", "",
             header, "-" * len(header)]
    for p in pooled:
        n = p["n_solvent"]
        delta = deltas.get(n)
        found = f"{p['found_by']}/{p['n_seeds']}"
        lines.append(
            f"{n:>3} "
            f"{100 * n / capacity:>5.0f}% "
            f"{p['e_int_min_kcal']:>12.2f} "
            f"{p['e_int_ens_kcal']:>12.2f} "
            + (f"{'-':>10} " if delta is None else f"{delta:>10.2f} ")
            + f"{found:>9} "
            f"{p['pool']:>6} "
            f"{p['e_cluster_ens_eV']:>15.6f} "
            f"{p['mean_contacts']:>9.2f} "
            f"{100 * p['dissolved_fraction']:>9.0f}% "
            + (f"{'-':>6}" if p["wall"] is None
               else f"{100 * p['wall']:>5.0f}%"))
    lines.append(
        "\nOne row per n, over the candidates of every packing at that n "
        "pooled together\nand deduped by energy. Seeds are independent "
        "*searches*, not repeat measurements\nof one system, so they are "
        "pooled rather than averaged: the mean of several\nsearches penalises "
        "searching more widely, and it is exactly the packing that\nfinds the "
        "geometry nobody else found that the stratification exists to buy.\n"
        "'found by' = how many of that n's packings reached the pooled "
        "minimum; 'pool' =\nhow many distinct minima the pooled search "
        "turned up. The per-packing numbers\nare in the table below this one.\n"
        "\nE_int in kcal/mol, E(cluster) in eV (ASE's native unit). 'cover' = n "
        "as a\npercentage of a full first-shell monolayer; well under 100% is "
        "targeted\nmicrosolvation, where every explicit molecule sits at the "
        "continuum boundary.\n'contacts' = solvent molecules touching the "
        "solute after optimisation;\n'dissolved' = fraction of pooled "
        "candidates with no contact at all. 'wall' =\nthe worst seed's "
        "fraction of sampling frames with a nonzero wall energy.\n"
        "\ndE_int = E_int(min) at this n minus E_int(min) at n - 1: what the "
        "nth molecule\nwas worth. Read the increment, not the level: E_int(n) "
        "does not plateau. Every\nadded molecule also collects the continuum's "
        "per-molecule bias (measured at ~1\nkcal/mol for chloroform, and of "
        "either sign -- see the binding table in the\ndocs) and, from n = 2 "
        "on, solvent-solvent cohesion, and neither term switches\noff once the "
        "specific sites are filled. So the curve tends to a line with a\n"
        "nonzero slope, not to a flat. Convergence is dE_int settling to a "
        "*constant*:\nthe specific interaction is exhausted and each further "
        "molecule is only being\ncondensed out of the continuum into a "
        "bulk-like site.\n"
        "\nA minimum is a running minimum over *search effort*, and the effort "
        "is not\nconstant across n -- the pool column above says how far it "
        "went at each. Some\nof dE_int's slope is therefore search depth "
        "rather than chemistry, which is the\nhonest form of the objection "
        "that would otherwise argue for averaging the seeds.\nThe answer is to "
        "show the effort, not to average it away: a step is real only "
        "if\nit survives that reading and the search convergence below.\n"
        "\nWhat does plateau is a difference taken at fixed n between two legs "
        "-- two\nconformers, two tautomers, bound and free, one solute in two "
        "solvents. Both\nlegs carry n molecules in comparable environments, so "
        "the bias and the cohesion\nlargely cancel and what is left is the "
        "specific interaction. Judge 'how much\nexplicit solvent is enough' on "
        "that difference, not on one leg's E_int(n).\n"
        "\nE(cluster) is Boltzmann-averaged over the same weights as E_int, and is "
        "here\nto sanity-check the magnitudes it came from. It is NOT comparable "
        "across rows\nof different n -- successive rows differ by a whole solvent "
        "molecule. Making\nthat comparison meaningful is exactly what E_int is "
        "for; subtract those\ninstead.")
    return "\n".join(lines)


def format_best_geometry(pooled):
    """Where the answer lives: the file behind each n's pooled minimum.

    `pool_by_n` already worked out that its `keep[0]` -- the pooled minimum --
    is necessarily that candidate's own packing's minimum too, so this is
    pure formatting of the `best` block it attached: no new computation, no
    filesystem access. It exists because the report otherwise names a number
    at each n without ever saying which geometry produced it, which used to
    mean grepping every `scored.json` in the sweep by hand.
    """
    header = (f"{'n':>3} {'E_int(min)':>12} {'weight':>7} {'contacts':>9} "
             f"{'min gap/A':>10} {'frame':>6} {'E_wall/eV':>11} {'packing'}")
    lines = ["Best geometry at each n", "-----------------------",
             header, "-" * len(header)]
    for p in pooled:
        best = p["best"]
        c = best["candidate"]
        # NaN when there is no solvent to measure a gap to, i.e. at n = 0.
        gap = c["min_gap_A"]
        gap_s = "-" if gap != gap else f"{gap:.3f}"
        lines.append(
            f"{p['n_solvent']:>3} "
            f"{p['e_int_min_kcal']:>12.2f} "
            f"{best['weight']:>7.3f} "
            f"{c['n_contacts']:>9d} "
            f"{gap_s:>10} "
            f"{c['frame']:>6d} "
            f"{c['wall_energy_eV']:>11.6f} "
            f"{best['run']}")
    lines.append(
        "\n  The geometry is <packing>/best.xyz above, and equivalently frame "
        "0 of that\n  packing's scored_candidates.xyz: the pooled minimum at "
        "an n is necessarily that\n  packing's own minimum, since "
        "dedupe_energies sorts lowest first. 'weight' is\n  the candidate's "
        "Boltzmann weight within the pooled set at this n -- near 1 means\n  "
        "E_int(min) is effectively the ensemble, small means it is one of "
        "several\n  comparable minima and E_int(ens) is the number to read "
        "instead. 'frame' is the\n  MD dump index into that packing's "
        "traj.xyz / energies.json, in the same sense\n  the candidate table "
        "and sampling convergence use it -- not an index into any xyz\n  of "
        "candidates. 'E_wall/eV' is that sampling frame's wall energy: "
        "nonzero means\n  the wall was holding this particular geometry "
        "together, the one row where the\n  wall diagnostic bites on the "
        "reported number itself rather than on an average.\n  best.xyz is "
        "not named after out_name, so a second continuum scored into the "
        "same\n  run directories overwrites it -- it reflects the most "
        "recent scoring pass.\n\n  The same geometry is also copied to "
        "best_n<N>.xyz in the sweep directory, with\n  sweep_n=, "
        "sweep_E_int_kcal=, and sweep_packing= appended to its comment "
        "line.\n  That is one file per n, not one file for the whole sweep, "
        "because the frames\n  differ in atom count as n grows and a "
        "viewer that reads a multi-frame xyz as a\n  trajectory -- Avogadro, "
        "VMD, most others -- takes the first frame's atom count\n  and "
        "applies it to the rest, silently dropping every n after the "
        "first.")
    return "\n".join(lines)


def format_seed_detail(summaries, pooled):
    """Per packing, under the pooled table: what each search on its own found.

    These rows used to be the report's primary table, back when the reported
    number was a per-run one. They are kept because a pooled number hides
    which packing produced it -- a row whose E_int(min) is alone at the bottom
    of its n is the arrangement lottery being won once -- but they are not the
    answer, so they sit below it. No dE_int here: the increment is a pooled
    quantity now, and a per-packing one would be pairing independent searches
    by index.
    """
    tol_kcal = DEDUPE_TOL_EV * EV_TO_KCAL
    best_by_n = {p["n_solvent"]: p["e_int_min_kcal"] for p in pooled}
    header = (f"{'n':>3} {'seed':>4} {'best':>4} {'E_int(ens)':>12} "
              f"{'E_int(min)':>12} {'E(cluster)/eV':>15} {'contacts':>9} "
              f"{'dissolved':>10} {'uniq':>5} {'wall':>6}")
    lines = ["Per-packing detail", "------------------", header,
             "-" * len(header)]
    for s in sorted(summaries, key=_row_order):
        frac = _wall_fraction(s)
        e_min = best_by_n.get(s["n_solvent"])
        star = ("*" if e_min is not None
                and abs(s["min_interaction_kcal"] - e_min) <= tol_kcal
                else "")
        lines.append(
            f"{s['n_solvent']:>3} "
            f"{s['seed']:>4} "
            f"{star:>4} "
            f"{s['ensemble_interaction_kcal']:>12.2f} "
            f"{s['min_interaction_kcal']:>12.2f} "
            f"{s['ensemble_energy_eV']:>15.6f} "
            f"{s['mean_contacts']:>9.2f} "
            f"{100 * s['dissolved_fraction']:>9.0f}% "
            f"{s['n_unique']:>5} "
            + (f"{'-':>6}" if frac is None else f"{100 * frac:>5.0f}%"))
    lines.append(
        "\n  One row per packing: its own candidates, its own Boltzmann "
        "average over them,\n  and 'uniq' its own count of distinct minima. "
        "These do not average to the table\n  above and are not meant to -- "
        "the pooled minimum is the lowest of the\n  E_int(min) column, not "
        f"the mean of it. 'best' marks a packing within {1000 * DEDUPE_TOL_EV:.0f} "
        "meV of\n  the pooled minimum at its n -- the same test 'found by' "
        "counts, so the number\n  of '*' at an n equals its 'found by' "
        "numerator. The Best geometry section\n  above names the one of "
        "these whose file to open when several tie.")
    return "\n".join(lines)


def format_wall_diagnostic(summaries):
    """The worst wall-active fraction in the sweep, and what it means.

    A sweep is many runs, and the per-run footer in each `run.log` is exactly
    where nobody looks after the fact. This puts the one number that says
    whether any of these energies are contaminated next to the energies.
    """
    known = [(f, s) for s in summaries
             if (f := _wall_fraction(s)) is not None]
    if not known:
        return None

    worst_frac, worst = max(known, key=lambda pair: pair[0])
    lines = ["Wall diagnostic", "---------------",
             f"  max wall-active fraction   {100 * worst_frac:.1f}%  "
             f"({worst.get('label', '?')} seed {worst.get('seed', '?')})"]

    warning = wall_warning(worst_frac)
    if warning:
        lines += ["", warning]
    else:
        lines.append(
            f"  Under the {100 * WALL_WARN_FRACTION:.0f}% threshold, so the wall "
            "is a safety net rather than load-bearing.\n"
            "  Reference regimes: 4-5% for gas-phase sampling of a self-bound "
            "shell, 50-60%\n  for a shell the Hamiltonian is dissociating.")
    return "\n".join(lines)


# A minimum first found in the last quarter of the trajectory, or a
# last-quarter gain bigger than this, means the run was still turning up new
# basins when it stopped. 0.25 kcal/mol is far under the ~3 kcal/mol signal
# this pipeline exists to resolve, and well over the noise a converged
# optimisation leaves behind.
LATE_TRAJECTORY_FRACTION = 0.75
LATE_GAIN_WARN_KCAL = 0.25


def sampling_convergence(summary):
    """Where in the trajectory `E_int(min)` was found, and what arrived late.

    `E_int(min)` is a running minimum over candidates, so it can only ever
    fall as sampling continues. That makes *when* it last fell a usable
    convergence test: a minimum found in the middle of a trajectory that then
    ran on without improving is evidence the sampling saturated, and a
    minimum found on the last frame is evidence it did not -- the reported
    number is an upper bound and a longer run would beat it.

    Computed from the candidate list already in `scored.json`, so it costs no
    MD and no calculator. `dedupe` drops only energy-duplicates, and a
    duplicate cannot move a running minimum by more than the dedup tolerance,
    so working from the deduped list is safe.

    Returns None when there is nothing to say: too few candidates to see a
    trend, or a summary too old to carry frame indices.
    """
    candidates = summary.get("candidates") or []
    if len(candidates) < 4:
        return None
    frames = [c.get("frame") for c in candidates]
    if any(f is None for f in frames):
        return None
    last = max(frames)
    if last <= 0:
        return None

    cut = LATE_TRAJECTORY_FRACTION * last
    best = min(candidates, key=lambda c: c["interaction_eV"])
    early = [c["interaction_eV"] for c in candidates if c["frame"] <= cut]
    late = [c["interaction_eV"] for c in candidates if c["frame"] > cut]
    # Undefined rather than zero when the trajectory has no candidates on one
    # side of the cut: there was nothing there to improve on. Clamped at zero
    # because the quantity being reported is a *running* minimum, which cannot
    # rise -- a late quarter that found nothing better improved it by nothing,
    # and printing how much worse its own best was would only invite reading
    # the sign backwards.
    gain = (min(min(late) - min(early), 0.0) * EV_TO_KCAL
            if early and late else None)
    return {
        "best_at": best["frame"] / last,
        "late_gain_kcal": gain,
        "still_improving": (best["frame"] > cut
                            or (gain is not None
                                and gain < -LATE_GAIN_WARN_KCAL)),
    }


def format_sampling_convergence(summaries):
    """Per-run: was `E_int(min)` still falling when the trajectory stopped?"""
    rows = [(s, conv) for s in summaries if (conv := sampling_convergence(s))]
    if not rows:
        return None

    lines = ["Sampling convergence", "--------------------",
             f"{'n':>3} {'seed':>4} {'best found at':>14} "
             f"{'late gain':>10} {'verdict':>16}",
             "-" * 52]

    for s, conv in sorted(rows, key=lambda r: _row_order(r[0])):
        gain = conv["late_gain_kcal"]
        lines.append(
            f"{s['n_solvent']:>3} "
            f"{s['seed']:>4} "
            f"{100 * conv['best_at']:>13.0f}% "
            + (f"{'-':>10} " if gain is None else f"{gain:>10.2f} ")
            + f"{'STILL FALLING' if conv['still_improving'] else 'settled':>16}")

    lines.append(
        "\n  'best found at' = how far into the sampling trajectory the "
        "lowest-E_int\n  candidate was found. 'late gain' = what the final "
        f"{100 - 100 * LATE_TRAJECTORY_FRACTION:.0f}% of the trajectory\n  "
        "took off E_int(min), in kcal/mol; negative means it was still "
        "finding\n  better geometries at the end.")

    unsettled = [s for s, conv in rows if conv["still_improving"]]
    if unsettled:
        lines.append(
            "\n" + "\n".join("  " + line for line in (
                f"** WARNING: {len(unsettled)} of {len(rows)} runs were still "
                "improving E_int(min)",
                f"** in the last {100 - 100 * LATE_TRAJECTORY_FRACTION:.0f}% "
                "of their trajectory. Those rows are upper bounds, not",
                "** converged values, and the amount by which they are wrong "
                "is unknown --",
                "** it is bounded below by the late gain, not by it. Raise "
                "--steps until",
                "** the minima stop moving, and add seeds: an independent "
                "trajectory finds",
                "** a different basin, where a longer one may only sit in the "
                "same well.",
            )))
    return "\n".join(lines)


def format_search_convergence(params, pooled):
    """Do the independent packings at each n agree on the minimum?

    A packing is a search, not a replica, so the useful question about a set
    of them is not how far apart their answers scattered but how many of them
    arrived at the same minimum. Agreement is the convergence evidence: a
    minimum several independent packings reached is one the search is finding
    reliably, and a minimum one packing reached and six missed rests entirely
    on that packing -- the reported number is then an upper bound held up by a
    single draw, and the honest response is more packings.

    That is also why this is not an error bar. The spread of the per-packing
    minima is printed because it is worth seeing, but under stratified packing
    the packings are *designed* to differ, so their scatter measures the
    arrangement lottery as much as the sampling noise. A sweep with one
    packing has none of this evidence at all and must say so, rather than
    leaving the absence to read as precision.
    """
    if not pooled:
        return None

    lines = ["Search convergence", "------------------"]
    if max(p["n_seeds"] for p in pooled) < 2:
        return "\n".join(lines + [
            "  Single packing per n: no agreement to measure. Every E_int above is",
            "  one search's answer, and the difference between two such answers at",
            "  the same n has been measured at whole kcal/mol -- at n = 2 the",
            "  arrangement that puts one solvent molecule on each of pyrazine's",
            "  nitrogens turns up in well under half of packings. Re-run with",
            "  --seeds 3 before subtracting two sweeps from each other: a double",
            "  difference is four of these numbers and inherits the weakness of all",
            "  four."])

    header = (f"{'n':>3} {'packings':>9} {'E_int(min)':>12} {'found by':>9} "
              f"{'spread':>8} {'pool':>6}")
    lines += [header, "-" * len(header)]

    worst = 0.0
    alone = []
    for p in pooled:
        minima = p["seed_minima_kcal"]
        spread = max(minima) - min(minima) if len(minima) > 1 else None
        if spread is not None:
            worst = max(worst, spread)
        if p["n_seeds"] > 1 and p["found_by"] == 1:
            alone.append(p["n_solvent"])
        found = f"{p['found_by']}/{p['n_seeds']}"
        lines.append(
            f"{p['n_solvent']:>3} {p['n_seeds']:>9} "
            f"{p['e_int_min_kcal']:>12.2f} "
            f"{found:>9} "
            + (f"{'-':>8} " if spread is None else f"{spread:>8.2f} ")
            + f"{p['pool']:>6}")

    lines.append(
        "\n  'found by' = how many of that n's packings reached the pooled "
        "minimum, by the\n  same energy criterion that dedupes candidates "
        f"within a run ({1000 * DEDUPE_TOL_EV:.0f} meV).\n  'spread' = the "
        "full range of the per-packing minima, kcal/mol -- printed to be\n"
        "  seen, not to be used as an error bar, because independent packings "
        "are\n  independent searches rather than repeat measurements of one "
        "system.\n"
        f"\n  Widest spread in this sweep: {worst:.2f} kcal/mol. A difference "
        "you go on to\n  take between sweeps has to be credible against that, "
        "and a double difference\n  of four such numbers accumulates it four "
        "times.")

    if alone:
        lines.append(
            "\n" + "\n".join("  " + line for line in (
                f"** WARNING: at n = {', '.join(str(n) for n in alone)} the "
                "pooled minimum was reached",
                "** by exactly one packing. That number rests on a single "
                "draw of the",
                "** arrangement lottery, and the packings that missed it are "
                "not evidence",
                "** against it -- they are evidence that the search is not "
                "finding it",
                "** reliably. Add packings (--seeds) rather than steps; a "
                "longer trajectory",
                "** mostly sits in the basin it was packed into.",
            )))

    if params["pack_stratified"]:
        lines.append(
            "\n  Packing was stratified across the packings at each n (see "
            "the params block),\n  so they are not repeat draws from one "
            "distribution: they are deliberately\n  different solvent "
            "arrangements, spread and clustered. That is what makes "
            "their\n  agreement meaningful -- searches that started from "
            "different arrangements and\n  converged on one minimum -- and it "
            "is why disagreement at small n reads as an\n  arrangement "
            "lottery not yet won often enough. The fix for that is more "
            "packings,\n  not more steps.")
    return "\n".join(lines)


def format_sweep_report(params, summaries):
    """Params block plus the E_int(n) table.

    `pooled` is computed once here and threaded through every section that
    needs it, rather than each calling `pool_by_n(summaries)` on its own --
    they would recompute the same dedupe over the same candidates three times
    over.
    """
    parts = [banner("n-sweep report")]

    shown = {k: ("-" if v is None else v) for k, v in params.items()}
    parts.append(kv_block("Parameters", shown))

    pooled = pool_by_n(summaries)

    parts.append("E_int(n) = E(solute + n solvent) - E(solute) - n E(solvent)\n"
                 + "-" * 58 + "\n" + format_table(params, pooled))

    parts.append(format_best_geometry(pooled))

    parts.append(format_seed_detail(summaries, pooled))

    for section in (format_sampling_convergence(summaries),
                    format_search_convergence(params, pooled),
                    format_wall_diagnostic(summaries)):
        if section:
            parts.append(section)

    zeros = [s for s in summaries if s["n_solvent"] == 0]
    if zeros:
        values = ", ".join(f"{s['ensemble_interaction_kcal']:.2f}"
                           for s in sorted(zeros, key=_row_order))
        parts.append(
            "Self-consistency check\n"
            "----------------------\n"
            f"  E_int(0) = {values} kcal/mol.\n"
            "  Zero by construction: the n = 0 cluster *is* the solute reference.\n"
            "  A departure of more than a few hundredths means the reference\n"
            "  optimisations and the n = 0 run landed in different minima, and\n"
            "  every other row is offset by that much.")
    return "\n\n".join(parts) + "\n"


def write_best_geometries(sweep_dir, pooled):
    """Copy each n's winning `best.xyz` to a sweep-level `best_n<N>.xyz`.

    The deliverable of a sweep, not just its report: `pooled` already knows,
    per n, which packing's `best.xyz` is the pooled minimum (see `pool_by_n`),
    so this only has to read those files and copy them out, one per n.

    One file per n, not one multi-frame xyz, because a sweep's frames differ
    in atom count -- n = 0..k goes 10, 15, 20, ... atoms for a fixed solute
    and solvent -- and a multi-frame xyz has no way to say "these frames are
    unrelated". A viewer reads it as a trajectory, takes the first frame's
    atom count, and applies it to the rest, so Avogadro (and VMD, and most
    others) render only n = 0 and silently drop every row after it. A set of
    chemically distinct clusters is not a trajectory, and no single-file XYZ
    shape expresses that it isn't one.

    Pure text handling -- no ASE, keeping this module's import list at module
    scope unchanged -- because every field needed is already a line of text:
    the natoms line, the comment line, and the coordinate block. Each frame's
    comment line (extended-xyz's `key=value` line) gains `sweep_n=`,
    `sweep_E_int_kcal=`, and `sweep_packing=` (the run directory name, plain
    `[A-Za-z0-9_]` so it needs no quoting), appended after whatever ASE
    already wrote there (`Properties=...`, `energy=...`), which xtb reads
    back and this does not disturb. `sweep_packing=` is what still says which
    packing produced a file once it has been lifted out of the sweep
    directory.

    Skips an n whose run directory or `best.xyz` is missing -- a `sweep.json`
    copied without its run directories, say -- and returns an empty list if
    none are present.
    """
    sweep_dir = Path(sweep_dir)
    written = []
    for p in pooled:
        run = p["best"]["run"]
        best_xyz = sweep_dir / run / "best.xyz"
        if not best_xyz.is_file():
            continue
        lines = best_xyz.read_text().splitlines()
        if len(lines) < 2:
            continue
        lines[1] = (f"{lines[1].rstrip()} sweep_n={p['n_solvent']} "
                    f"sweep_E_int_kcal={p['e_int_min_kcal']:.6f} "
                    f"sweep_packing={run}")
        out = sweep_dir / f"best_n{p['n_solvent']}.xyz"
        out.write_text("\n".join(lines) + "\n")
        written.append(out)
    return written


# --------------------------------------------------------------------------
# Docking (docking.py's report)
# --------------------------------------------------------------------------

def dock_row_from_summary(s):
    """One docking `scored.json` summary -> the row shape the report reads.

    `docking.py` writes every field this needs straight into the summary
    (`found_by`, `n_refined`, `parent_detail`, ...) precisely so this
    reconstruction needs nothing but the JSON already on disk -- the same
    reason `pool_by_n` works from `scored.json` alone. Used both by
    `docking.run_docking` right after building a fresh summary, and by
    `render_dock_dir` when regenerating a report with no MD, no calculator,
    and no optimiser.
    """
    best_i = min(range(len(s["candidates"])),
                key=lambda i: s["candidates"][i]["interaction_eV"])
    weights = boltzmann_weights([c["energy_eV"] for c in s["candidates"]],
                                s["temperature_K"])
    return {
        "n_solvent": s["n_solvent"],
        "run_dir": s["run_dir"],
        "pool": s["n_unique"],
        "n_refined": s.get("n_refined", s["n_frames_scored"]),
        "found_by": s["found_by"],
        "e_int_min_kcal": s["min_interaction_kcal"],
        "e_int_ens_kcal": s["ensemble_interaction_kcal"],
        "e_cluster_ens_eV": s["ensemble_energy_eV"],
        "mean_contacts": s["mean_contacts"],
        "dissolved_fraction": s["dissolved_fraction"],
        "best": {
            "candidate": s["candidates"][best_i],
            "weight": float(weights[best_i]),
            "parent": s["candidates"][best_i]["parent"],
        },
        "parent_detail": s["parent_detail"],
    }


def _dock_increments(params, dock_rows):
    """`dE_int(n)` for docking, seeded with `E_int(0) = 0` by construction.

    Same idea as `_increments`, but keyed against `params["all_n_min_kcal"]`
    rather than the reported rows alone: a docking chain walks every integer
    n internally even when only some are requested (`--n 1 3` still docks
    n = 2 to get there), so the predecessor a requested row needs may not be
    one of the rows being printed. `all_n_min_kcal` carries the full walk,
    which is what lets dE_int be exact rather than "-" at every gap.
    """
    all_min = {0: 0.0}
    all_min.update({int(k): v for k, v in params.get("all_n_min_kcal", {}).items()})
    return {p["n_solvent"]: all_min[p["n_solvent"]] - all_min[p["n_solvent"] - 1]
            for p in dock_rows if (p["n_solvent"] - 1) in all_min}


def format_dock_table(params, dock_rows):
    """E_int vs n over one docking chain -- the docking analogue of `format_table`."""
    deltas = _dock_increments(params, dock_rows)
    capacity = params["monolayer_capacity"]
    header = (f"{'n':>3} {'cover':>6} {'E_int(min)':>12} {'dE_int':>10} "
              f"{'E_int(ens)':>12} {'found by':>10} {'pool':>6} "
              f"{'E(cluster)/eV':>15} {'contacts':>9} {'dissolved':>10}")
    lines = [f"leg: {params['solute_label']}/{params['solvent']} (docked)",
             f"full first shell: ~{capacity:.0f} solvent molecules", "",
             header, "-" * len(header)]
    for p in dock_rows:
        n = p["n_solvent"]
        delta = deltas.get(n)
        found = f"{p['found_by']}/{p['n_refined']}"
        lines.append(
            f"{n:>3} "
            f"{100 * n / capacity:>5.0f}% "
            f"{p['e_int_min_kcal']:>12.2f} "
            + (f"{'-':>10} " if delta is None else f"{delta:>10.2f} ")
            + f"{p['e_int_ens_kcal']:>12.2f} "
            f"{found:>10} "
            f"{p['pool']:>6} "
            f"{p['e_cluster_ens_eV']:>15.6f} "
            f"{p['mean_contacts']:>9.2f} "
            f"{100 * p['dissolved_fraction']:>9.0f}%")
    lines.append(
        "\nOne row per requested n. Unlike the MD sweep's pooled search over "
        "several\nindependent packings, a docking row is one constructed "
        "chain: 'found by' =\nrefined placements within "
        f"{1000 * DEDUPE_TOL_EV:.0f} meV of this n's minimum, out\nof the "
        "placements actually refined at the tight criterion ('pool' is the "
        "number\nof distinct minima among them -- most placements are only "
        "screened, at a\nlooser fmax, and are not part of either count). "
        "dE_int = E_int(min) at this n\nminus E_int(min) at n - 1, computed "
        "over every n walked internally even when\nan intermediate n was "
        "not requested. E_int(0) = 0 by construction: the chain\nalways "
        "starts from the bare relaxed solute.\n"
        "\nDocking constructs rather than samples -- BFGS only descends, so "
        "it wins at\nevery n it is pointed at by construction -- so its "
        "numbers are never pooled\nwith an MD sweep's. Read the two reports "
        "side by side: a basin docking finds\nthat the sweep never visits "
        "is the failure this program exists to catch; a\nbasin only "
        "docking finds is not automatically more real than one only the\n"
        "sweep finds, since docking never samples thermal motion at all. "
        "'cover' = n\nas a percentage of a full first-shell monolayer -- "
        "docking targets targeted\nmicrosolvation and warns past roughly a "
        "third of capacity.")
    return "\n".join(lines)


def format_dock_best_geometry(dock_rows):
    """Where the answer lives: the file behind each n's constructed minimum."""
    header = (f"{'n':>3} {'E_int(min)':>12} {'weight':>7} {'contacts':>9} "
             f"{'min gap/A':>10} {'parent':>6} {'directory'}")
    lines = ["Best geometry at each n", "-----------------------",
             header, "-" * len(header)]
    for p in dock_rows:
        best = p["best"]
        c = best["candidate"]
        gap = c["min_gap_A"]
        gap_s = "-" if gap != gap else f"{gap:.3f}"
        lines.append(
            f"{p['n_solvent']:>3} "
            f"{p['e_int_min_kcal']:>12.2f} "
            f"{best['weight']:>7.3f} "
            f"{c['n_contacts']:>9d} "
            f"{gap_s:>10} "
            f"{best['parent']:>6d} "
            f"{Path(p['run_dir']).name}")
    lines.append(
        "\n  The geometry is <directory>/best.xyz above. 'parent' is the "
        "0-based index\n  into the previous n's surviving parents that this "
        "minimum grew from -- see\n  the per-parent detail table below for "
        "what each of those parents found on\n  its own. Also copied to "
        "best_n<N>.xyz alongside dock_report.txt, with dock_n=,\n  "
        "dock_E_int_kcal=, and dock_parent= appended to its comment line -- "
        "one file\n  per n, not one multi-frame xyz, for the same reason "
        "the MD sweep's\n  best_n<N>.xyz is: atom count changes with n, and "
        "a viewer that reads a\n  multi-frame xyz as a trajectory keeps "
        "only the first frame's atom count.")
    return "\n".join(lines)


def format_dock_parent_detail(dock_rows):
    """Per parent, under the main table: the greedy chain made visible."""
    header = (f"{'n':>3} {'parent':>6} {'best':>4} {'placements':>10} "
             f"{'E_int(min)':>12}")
    lines = ["Per-parent detail", "-----------------", header,
             "-" * len(header)]
    for p in dock_rows:
        for parent in p["parent_detail"]:
            star = "*" if parent["best"] else ""
            lines.append(
                f"{p['n_solvent']:>3} "
                f"{parent['parent']:>6} "
                f"{star:>4} "
                f"{parent['n_placements']:>10} "
                f"{parent['e_int_min_kcal']:>12.2f}")
    lines.append(
        "\n  One row per parent used to grow to that n: its own best "
        "E_int(min) among\n  its own random placements. 'best' marks the "
        "parent whose descendant became\n  the n's reported minimum -- the "
        "greedy chain made visible, and the reason\n  docking carries more "
        "than one parent forward: the best structure at n need\n  not "
        "descend from the best structure at n - 1. Unlike the MD sweep's "
        "'found\n  by', several parents landing near the same minimum here "
        "is not independent\n  corroboration -- every parent explores the "
        "same shell region with\n  independent random poses, not a "
        "differently-arranged packing -- so read it\n  as which branch of "
        "the chain the reported minimum came from, not as agreement.")
    return "\n".join(lines)


def format_dock_report(params, dock_rows):
    """Params block plus the docking E_int(n) table, for one docking chain."""
    parts = [banner("docking report")]

    shown = {k: ("-" if v is None else v) for k, v in params.items()
            if k != "all_n_min_kcal"}
    parts.append(kv_block("Parameters", shown))

    parts.append("E_int(n) = E(solute + n solvent) - E(solute) - n E(solvent)\n"
                 + "-" * 58 + "\n" + format_dock_table(params, dock_rows))
    parts.append(format_dock_best_geometry(dock_rows))
    parts.append(format_dock_parent_detail(dock_rows))
    return "\n\n".join(parts) + "\n"


def write_dock_best_geometries(out_root, dock_rows):
    """Copy each n's winning `best.xyz` to `<out_root>/best_n<N>.xyz`.

    Mirrors `write_best_geometries`: one file per n rather than one
    multi-frame xyz, because atom count grows with n and a trajectory viewer
    would keep only the first frame.
    """
    out_root = Path(out_root)
    written = []
    for p in dock_rows:
        run_dir = Path(p["run_dir"])
        best_xyz = (run_dir if run_dir.is_absolute()
                    else out_root / run_dir.name) / "best.xyz"
        if not best_xyz.is_file():
            continue
        lines = best_xyz.read_text().splitlines()
        if len(lines) < 2:
            continue
        lines[1] = (f"{lines[1].rstrip()} dock_n={p['n_solvent']} "
                    f"dock_E_int_kcal={p['e_int_min_kcal']:.6f} "
                    f"dock_parent={p['best']['parent']}")
        out = out_root / f"best_n{p['n_solvent']}.xyz"
        out.write_text("\n".join(lines) + "\n")
        written.append(out)
    return written


def render_dock_dir(path):
    """Rewrite `dock_report.txt` and `best_n<N>.xyz`, from `dock.json` alone."""
    path = Path(path)
    data = _load(path / "dock.json")
    params, runs = data["params"], data["runs"]
    dock_rows = [dock_row_from_summary(s) for s in runs]

    report = path / "dock_report.txt"
    report.write_text(format_dock_report(params, dock_rows))
    written = [report]
    written += write_dock_best_geometries(path, dock_rows)
    return written


# --------------------------------------------------------------------------
# Post-hoc regeneration
# --------------------------------------------------------------------------

def _load(path):
    return json.loads(Path(path).read_text())


def render_run_dir(path):
    """Rewrite `run.log`, and a `.log` for every scored summary, from JSON.

    No MD and no calculator: everything comes off disk. Both JSON files are
    required -- `run_one_job` always writes both, and a directory missing one
    is a broken run rather than an old one.
    """
    path = Path(path)
    meta = _load(path / "metadata.json")
    records = _load(path / "energies.json")

    with RunLogger(path / "run.log", live=False) as log:
        log.header(meta)
        for record in records:
            log.md_row(record)
        log.footer(records)
    written = [path / "run.log"]

    for scored in sorted(path.glob("scored*.json")):
        out = scored.with_suffix(".log")
        out.write_text(format_scored_log(_load(scored), meta))
        written.append(out)
    return written


def render_sweep_dir(path):
    """Rewrite `report.txt` plus every run's logs, from `sweep.json`."""
    path = Path(path)
    data = _load(path / "sweep.json")
    params, runs = data["params"], data["runs"]

    written = []
    for summary in runs:
        # By name under the sweep directory rather than by the absolute path
        # recorded at write time, so a sweep that has been moved or copied
        # still re-renders -- but the name itself still comes from the
        # summary, so it is never rebuilt from `<label>_seed<n>` here.
        run_dir = path / Path(summary["run_dir"]).name
        if run_dir.is_dir():
            written += render_run_dir(run_dir)

    report = path / "report.txt"
    report.write_text(format_sweep_report(params, runs))
    written.append(report)

    written += write_best_geometries(path, pool_by_n(runs))
    return written


def render(path):
    path = Path(path)
    if (path / "dock.json").exists():
        return render_dock_dir(path)
    if (path / "sweep.json").exists():
        return render_sweep_dir(path)
    if (path / "metadata.json").exists():
        return render_run_dir(path)
    written = []
    for child in sorted(p for p in path.iterdir() if p.is_dir()):
        if (child / "metadata.json").exists():
            written += render_run_dir(child)
    if not written:
        raise SystemExit(f"{path}: no dock.json, sweep.json, metadata.json, "
                         "or run subdirectories found")
    return written


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m report <run-or-sweep-dir> ...")
    for target in sys.argv[1:]:
        for written in render(target):
            print(written)
