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
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

# Bump on any change to the pipeline's numerics or output shapes -- it lands
# in every sweep's params block via `n_sweep.sweep_params`, so a report can be
# matched back to the code that produced it.
VERSION = "0.8.0"

# Live here rather than in `ensemble` so that a text-only consumer never has to
# import ASE to format or weight a number. `ensemble` re-exports both.
EV_TO_KCAL = 23.060548
KB_EV_PER_K = 8.617333262e-5   # Boltzmann constant, == ase.units.kB

# Above this fraction of frames with a nonzero wall energy, the confinement is
# load-bearing rather than a safety net. Measured regimes: 4-5% for gas-phase
# sampling (shell self-bound), 50-60% for ALPB(water) (shell dissociating).
WALL_WARN_FRACTION = 0.20

_RULE = "=" * 78


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


# The libraries that can move a number, keyed by the calculator that uses
# them. ASE and numpy are common to every calculator: the optimiser, the
# integrator and the unit conversions all live there.
_CALCULATOR_LIBRARIES = {
    "gfn1-xtb": ("tblite",),
    "gfn2-xtb": ("tblite",),
    "mace-off23": ("mace-torch", "torch"),
}


def library_versions(calculator):
    """`{"<dist>_version": ...}` for the libraries behind `calculator`.

    Recorded beside `VERSION` because a version does not pin them. Two
    sweeps from identical code under an identical `Condition` still rest on
    different Hamiltonians if one ran against a different GFN2 build, and
    nothing else on disk would say so. Measured: a cluster env that had
    drifted to tblite 0.4.0 against 0.7.0 locally -- the params blocks were
    indistinguishable.

    Read from installed package metadata rather than by importing anything,
    so this module keeps its promise of no ASE, tblite or torch at any scope
    and a report still regenerates in an environment that has none of them.
    `None` where the metadata is unreadable, which is a visible gap rather
    than a silently absent key.
    """
    names = ("ase", "numpy") + _CALCULATOR_LIBRARIES.get(calculator.lower(), ())
    versions = {}
    for name in names:
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            found = None
        versions[f"{name.replace('-', '_')}_version"] = found
    return versions


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


def dedupe_groups(energies_eV, tol_eV=DEDUPE_TOL_EV):
    """Partition indices into distinct minima, lowest first within and across.

    Each inner list is one basin's member indices, representative (its own
    lowest-energy member) first; the groups themselves are ordered by that
    representative's energy ascending. Walking the indices in ascending
    energy order, an index joins the first existing group whose
    representative is within `tol_eV`, else starts a new group -- exactly the
    rule `dedupe_energies` used to decide keep-or-drop, just with the merged
    members kept rather than thrown away.

    Energy-based rather than RMSD-based: it cannot tell two genuinely
    different structures apart when they happen to be isoenergetic, which for
    ranking purposes costs nothing, and it needs nothing but the numbers
    already in `scored.json` -- no geometries, no ASE, no calculator.
    """
    order = sorted(range(len(energies_eV)), key=lambda i: energies_eV[i])
    groups = []
    for i in order:
        for group in groups:
            if abs(energies_eV[i] - energies_eV[group[0]]) <= tol_eV:
                group.append(i)
                break
        else:
            groups.append([i])
    return groups


def dedupe_energies(energies_eV, tol_eV=DEDUPE_TOL_EV):
    """Indices of the distinct minima, lowest first.

    A thin wrapper over `dedupe_groups` that keeps only the representatives --
    for the many callers that only ever wanted keep-or-drop and never needed
    which candidates a representative absorbed.
    """
    return [group[0] for group in dedupe_groups(energies_eV, tol_eV)]


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
    mean_T = float(np.mean([r["temperature_K"] for r in records]))
    epots = np.array([r["potential_energy_eV"] for r in records])
    mean_E, sd_E = float(epots.mean()), float(epots.std())
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
        self._target_K = meta["temperature_K"]
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
        "pack mode": summary["pack_mode"],
        "trajectory": ("traj.xyz"
                       + (f", at most {max_frames} frames, spread evenly"
                          if max_frames else "")
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

    occ_contacts = summary["occupancy_mean_contacts"]
    occ_dissolved = summary["occupancy_dissolved_fraction"]
    result = {
        "E_int (minimum)": (f"{summary['min_interaction_kcal']:.2f} kcal/mol"
                            "     <- the reported number"),
        "references": (f"E(solute) {e_solute:.6f} eV, "
                       f"{n} x E(solvent) {e_solvent:.6f} eV"),
        "frames scored": (f"{summary['n_frames_scored']} "
                          f"({summary['n_unique']} unique, "
                          f"{100 * summary['converged_fraction']:.0f}"
                          "% converged)"),
        # Frame-weighted, so it says how much of this run's sampling sat in a
        # contact state -- not how many kinds of contact it turned up, which
        # is a fact about the search and is `n_unique` above.
        "contacts": _num(occ_contacts, ".2f") + " (frame-weighted)",
        "dissolved": ("-" if occ_dissolved is None
                      else f"{100 * occ_dissolved:.0f}% of scored frames"),
    }
    # The Boltzmann numbers and E(cluster) are in `scored.json` and are
    # deliberately not printed here: E_int(min) is the reported number, and
    # showing an average beside it only invites reporting whichever looks
    # better. See README: Reading report.txt.
    parts.append(kv_block("Result", result))

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
    """Wall-active fraction of the run behind this summary, or None.

    `None` both for an MD run that dumped no frames, which `wall_stats`
    reports rather than guessing at, and for a docked structure, which was
    placed rather than sampled and so has no sampling wall at all.
    """
    wall = summary["sampling_wall"]
    return None if wall is None else wall["wall_active_fraction"]


def _row_order(summary):
    """Sort key for every per-run table: by n, then by seed.

    A docked run has no packing seed -- it is one constructed chain, one run
    per n -- so it sorts as -1 rather than tripping over a `None`.
    """
    seed = summary["seed"]
    return summary["n_solvent"], -1 if seed is None else seed


def _increments(params, pooled):
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

    A docking chain walks every integer n internally even when only some are
    requested (`--n 1 3` still docks n = 2 to get there), so the predecessor
    a requested row needs may not be one of the rows being printed. Its
    params therefore carry `all_n_min_kcal`, the full walk, and it is used in
    place of the rows when present -- which is what lets dE_int be exact
    rather than "-" at every gap. A sweep runs exactly the n it prints and
    has no such key.
    """
    by_n = {p["n_solvent"]: p["e_int_min_kcal"] for p in pooled}
    if "all_n_min_kcal" in params:
        by_n = {0: 0.0}  # E_int(0) = 0 by construction: the chain starts bare
        by_n.update({int(k): v for k, v in params["all_n_min_kcal"].items()})
    return {p["n_solvent"]: by_n[p["n_solvent"]] - by_n[p["n_solvent"] - 1]
            for p in pooled if (p["n_solvent"] - 1) in by_n}


def pool_by_n(summaries):
    """Pool every run's candidates at each n, and score the pool as one set.

    Both generators go through here, because both produce the same object: a
    set of continuum-relaxed, wall-free local minima with potential energies,
    scored to the identical criterion. A docking run is simply one run per n,
    so the pool at each n is that one chain's deduped candidates and the
    dedupe below is a no-op across runs. CLAUDE.md's rule that docked and
    swept numbers are never pooled is about mixing the two generators' runs
    in one table, which one `sweep.json` or one `dock.json` never does.

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
    than a degeneracy. `dedupe_groups` is the same criterion `ensemble`
    applies within a run, kept as groups rather than representatives-only so
    that a basin found by several seeds can have their frame counts summed --
    each summary's candidates already carry `n_frames`, the number of scored
    frames that quenched into it within that one run, and a basin found by
    two packings should report the sum, not either half.

    That is only legitimate because the references are sweep-wide:
    `score_run_grid` computes them once per distinct (solute, solvent,
    calculator, continuum), so every `interaction_eV` in a sweep shares a
    zero. A sweep whose rows were measured against different zeros is broken
    rather than old, so this raises rather than pooling them anyway.

    Each record also carries `basins` -- one entry per pooled distinct
    minimum, richest first would defeat the point, so `format_basin_occupancy`
    sorts by frame share -- and `occupancy_mean_contacts` /
    `occupancy_dissolved_fraction`, the frame-weighted analogues of
    `mean_contacts` / `dissolved_fraction` below. The two pairs answer
    different questions and must not be confused: `mean_contacts` is a
    property of the *search* (how many kinds of basin were found), the
    `occupancy_*` pair a property of the *trajectory* (how much of it sat in
    each). Quarantined from every energy in this record -- occupancy never
    enters `e_int_min_kcal` or `e_int_ens_kcal`.

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
        # Deterministic ties: `dedupe_groups` sorts by energy before
        # forming any group, so this ordering never changes which candidates
        # survive -- only which packing gets the credit when several tie.
        group = sorted(group, key=_row_order)
        tagged = [(s, c) for s in group for c in s["candidates"]]
        basin_groups = dedupe_groups([c["energy_eV"] for _, c in tagged])
        keep = [tagged[g[0]] for g in basin_groups]
        interactions = [c["interaction_eV"] for _, c in keep]
        absolutes = [c["energy_eV"] for _, c in keep]
        e_min = min(interactions)
        temperature = group[0]["temperature_K"]
        pack_mode = group[0]["pack_mode"]
        # A seed "found" the pooled minimum when its own best candidate is the
        # same minimum by the same test used everywhere else.
        seed_minima = [min(c["interaction_eV"] for c in s["candidates"])
                       for s in group]
        walls = [f for s in group if (f := _wall_fraction(s)) is not None]
        # `basin_groups` sorts ascending, so `keep[0]` is not just *a*
        # minimum -- it is the pooled minimum, and therefore its own
        # packing's minimum too: that packing's `best.xyz` is the geometry
        # behind `e_int_min_kcal` below.
        weights = boltzmann_weights([c["energy_eV"] for _, c in keep],
                                    temperature)

        # Occupancy: sum `n_frames` -- the scored frames that quenched into
        # this basin within their own run -- over every candidate a basin
        # absorbed, possibly from several packings. A run's own candidates
        # are already deduped by `ensemble.assemble`, so no packing can
        # contribute two candidates to one basin here: counting distinct
        # seeds per group is counting packings that visited it.
        #
        # `None` throughout for a docked chain, whose candidates carry no
        # frame counts because they were placed rather than visited -- see
        # `docking._assemble_dock_n`.
        counted = all(c["n_frames"] is not None for _, c in tagged)
        n_frames_pooled = (sum(c["n_frames"] for _, c in tagged)
                           if counted else None)
        basins = []
        for basin, (_, rep_c), weight in zip(basin_groups, keep, weights):
            members = [tagged[i] for i in basin]
            frames = (sum(c["n_frames"] for _, c in members)
                      if counted else None)
            basins.append({
                "e_int_kcal": rep_c["interaction_eV"] * EV_TO_KCAL,
                "n_frames": frames,
                "frame_share": (frames / n_frames_pooled
                                if n_frames_pooled else None),
                "n_seeds_hit": len({s["seed"] for s, _ in members}),
                "weight": float(weight),
                "n_contacts": rep_c["n_contacts"],
            })

        # What corroborates the reported minimum, and out of how many tries.
        # For a sweep those are independent packings agreeing on it. A
        # docking row is one constructed chain instead, so the analogue is
        # how many of its refined random placements landed on the same
        # minimum -- the number `docking` already computed as `found_by`. The
        # two are the same column and emphatically not the same evidence;
        # `format_dock_parent_detail` is where that is spelt out.
        if pack_mode == "dock":
            found_by = sum(s["found_by"] for s in group)
            n_searches = sum(s["n_refined"] for s in group)
        else:
            found_by = sum(1 for m in seed_minima
                           if abs(m - e_min) <= DEDUPE_TOL_EV)
            n_searches = len(group)

        s_best, c_best = keep[0]
        pooled.append({
            "n_solvent": n,
            "pack_mode": pack_mode,
            "n_seeds": len(group),
            "n_searches": n_searches,
            "pool": len(keep),
            "e_int_min_kcal": e_min * EV_TO_KCAL,
            "e_int_ens_kcal":
                ensemble_energy(interactions, temperature) * EV_TO_KCAL,
            # Identical weights to the line above, by the minimum-subtraction
            # argument in `boltzmann_weights`.
            "e_cluster_ens_eV": ensemble_energy(absolutes, temperature),
            "found_by": found_by,
            "seed_minima_kcal": [m * EV_TO_KCAL for m in seed_minima],
            # Over the *distinct minima* -- a property of how many kinds of
            # basin the search turned up, not of how the trajectory's time
            # was spent. See `occupancy_*` below for the latter.
            "mean_contacts": float(np.mean([c["n_contacts"] for _, c in keep])),
            # At n = 0 there is no solvent to dissolve, and `assemble` calls
            # that 1.0 rather than dividing by zero.
            "dissolved_fraction": (
                float(np.mean([c["n_contacts"] == 0 for _, c in keep]))
                if n else 1.0),
            "n_frames_pooled": n_frames_pooled,
            "basins": basins,
            # Frame-weighted, over the same pooled+deduped basins: how much of
            # the *sampling* actually sat in a contact state, as opposed to
            # how many distinct contact states were found. At n = 0 every
            # frame's basin has n_contacts = 0, so this comes out 1.0 the same
            # way `dissolved_fraction` is defined to, with no special case.
            "occupancy_mean_contacts": (
                sum(b["n_contacts"] * b["n_frames"] for b in basins)
                / n_frames_pooled if n_frames_pooled else None),
            "occupancy_dissolved_fraction": (
                sum(b["n_frames"] for b in basins if b["n_contacts"] == 0)
                / n_frames_pooled if n_frames_pooled else None),
            # The worst of the seeds, matching `format_wall_diagnostic`: the
            # question is whether any of the pooled energies is contaminated.
            # `None` for a docked chain, which has no sampling wall at all.
            "wall": max(walls) if walls else None,
            "best": {"run": Path(s_best["run_dir"]).name,
                     "seed": s_best["seed"],
                     "weight": float(weights[0]),
                     "candidate": c_best},
        })
    return pooled


def format_table(params, pooled):
    """E_int vs n over the pooled candidates, with the search effort behind it.

    One run is one solute in one solvent, so what is being measured is stated
    once above the table rather than repeated down a column of identical
    values. It comes from `params`, which is also why no summary carries a
    `solute_label` of its own.

    Both generators render through here: the columns are properties of a set
    of continuum-relaxed minima, which is what each produces.

    The columns are the load-bearing ones only. `E_int(ens)` and `E(cluster)`
    used to sit here too and are still in `sweep.json` / `dock.json`, but
    showing two averages beside each other invites reporting whichever looks
    better -- the objection this repo already raises against giving the
    ensemble average its own `dE_int` -- and `E(cluster)` is not comparable
    across rows anyway, which is the whole reason `E_int` exists. The
    contacts pair is the frame-weighted one, over scored frames rather than
    over distinct minima, because that is the one that answers a question
    about the system rather than about the search; `pool` already reports the
    search.
    """
    docked = pooled[0]["pack_mode"] == "dock"
    deltas = _increments(params, pooled)
    capacity = params["monolayer_capacity"]
    header = (f"{'n':>3} {'cover':>6} {'E_int(min)':>12} {'dE_int':>10} "
              f"{'found by':>9} {'pool':>6} {'contacts':>9} "
              f"{'dissolved':>10} {'wall':>6}")
    lines = [f"leg: {params['solute_label']}/{params['solvent']}"
             + (" (docked)" if docked else ""),
             f"full first shell: ~{capacity:.0f} solvent molecules", "",
             header, "-" * len(header)]
    for p in pooled:
        n = p["n_solvent"]
        delta = deltas.get(n)
        found = f"{p['found_by']}/{p['n_searches']}"
        diss = p["occupancy_dissolved_fraction"]
        lines.append(
            f"{n:>3} "
            f"{100 * n / capacity:>5.0f}% "
            f"{p['e_int_min_kcal']:>12.2f} "
            + (f"{'-':>10} " if delta is None else f"{delta:>10.2f} ")
            + f"{found:>9} "
            f"{p['pool']:>6} "
            f"{_num(p['occupancy_mean_contacts'], '.2f'):>9} "
            + (f"{'-':>10} " if diss is None else f"{100 * diss:>9.0f}% ")
            + (f"{'-':>6}" if p["wall"] is None
               else f"{100 * p['wall']:>5.0f}%"))
    lines.append(
        "\nE_int in kcal/mol. cover = n as a fraction of a full first-shell "
        "monolayer.\ndE_int = E_int(min) here minus E_int(min) at n - 1: read "
        "the increment, not\nthe level. found by = independent searches that "
        "reached the pooled minimum, of\nhow many"
        + (" refined placements;\npool = distinct minima among them."
           if docked else
           " packings; pool = distinct minima in the pool.")
        + " contacts / dissolved\nare frame-weighted over the scored frames"
        + (" -- blank here, and so is wall: a docked\nstructure was placed, "
           "not sampled, so it has neither an occupancy nor a wall."
           if docked else
           ", not averaged over distinct minima.\nwall = the worst packing's "
           "fraction of sampling frames with a nonzero wall\nenergy.")
        + "\nSee README: Reading report.txt.")
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
    docked = pooled[0]["pack_mode"] == "dock"
    header = (f"{'n':>3} {'E_int(min)':>12} {'weight':>7} {'contacts':>9} "
              f"{'min gap/A':>10} {'frame':>6} {'parent':>7} "
              f"{'E_wall/eV':>11} {'run'}")
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
            f"{_num(c['parent'], 'd'):>7} "
            f"{_num(c['wall_energy_eV'], '.6f'):>11} "
            f"{best['run']}")
    lines.append(
        "\n  The geometry is <run>/best.xyz, and frame 0 of that run's "
        "scored_candidates.xyz.\n  weight = its Boltzmann weight within the "
        "pooled set at this n; contacts and min\n  gap/A are its own, not an "
        "average. frame = "
        + ("which refined placement it was,\n  in energy order; parent = "
           "which of the previous n's survivors it grew from (see\n  the "
           "table below). A docked structure has no sampling frame, hence no "
           "E_wall/eV."
           if docked else
           "the MD dump it was quenched\n  from; E_wall/eV is that frame's "
           "wall energy -- the one place the wall diagnostic\n  bites on the "
           "reported number itself rather than on an average. parent is\n  "
           "blank: an MD candidate descends from a packing, not from another "
           "structure.")
        + "\n  Also copied to best_n<N>.xyz beside this report, one file per "
        "n. See README:\n  Reading report.txt.")
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
    header = (f"{'n':>3} {'seed':>4} {'best':>4} {'E_int(min)':>12} "
              f"{'uniq':>5} {'contacts':>9} {'dissolved':>10} {'wall':>6}")
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
            f"{s['min_interaction_kcal']:>12.2f} "
            f"{s['n_unique']:>5} "
            f"{_num(s['occupancy_mean_contacts'], '.2f'):>9} "
            f"{100 * s['occupancy_dissolved_fraction']:>9.0f}% "
            + (f"{'-':>6}" if frac is None else f"{100 * frac:>5.0f}%"))
    lines.append(
        "\n  One row per packing, its own search alone. These do not average "
        "to the\n  table above and are not meant to: the pooled minimum is "
        "the lowest of this\n  E_int(min) column, not the mean of it. uniq = "
        "its own count of distinct\n  minima; contacts / dissolved are "
        "frame-weighted over its own scored frames.\n  'best' marks a packing "
        f"within {1000 * DEDUPE_TOL_EV:.0f} meV of the pooled minimum at its "
        "n -- the\n  same test 'found by' counts, so the number of '*' at "
        "an n equals its 'found by'\n  numerator.")
    return "\n".join(lines)


_OCCUPANCY_TOP_N = 5


def format_basin_occupancy(params, summaries, pooled):
    """How the trajectory's time was actually spent, per n -- not just which
    distinct minima the search turned up.

    `pool_by_n` already summed `n_frames` -- scored frames that quenched into
    each basin -- across every packing at this n, into `pooled[n]["basins"]`.
    This only formats it: the top 5 basins by frame share, an "other" row for
    the rest, and a line contrasting `E_int(min)` against the frame-share-
    weighted mean and the two contact statistics (over distinct minima versus
    over frames) against each other. That contrast is the point of the
    section -- a basin can be the reported minimum and still be a rare
    outlier the shell almost never visits.

    A table of frame counts invites being read as a Boltzmann population and
    is not one; the caveats that go with it are in the README, under Reading
    report.txt, once rather than once here, once in the rendered output and
    once in CLAUDE.md as they used to be.

    Guarded by the caller on `pack_mode == "md"`: a docked run has no
    sampling frames and so no occupancy to report, and `_assemble_dock_n`
    writes `None` for exactly that reason -- a real absence, not a zero.
    """
    lines = ["Basin occupancy", "----------------"]
    header = (f"  {'E_int/kcal':>11} {'frames':>7} {'share':>7} "
              f"{'seeds':>6} {'weight':>7} {'contacts':>9}")
    rule = "  " + "-" * (len(header) - 2)

    for p in pooled:
        basins = sorted(p["basins"], key=lambda b: -(b["frame_share"] or 0.0))
        top, rest = basins[:_OCCUPANCY_TOP_N], basins[_OCCUPANCY_TOP_N:]
        lines.append(f"n = {p['n_solvent']}  ({p['n_frames_pooled']} scored "
                     "frames pooled)")
        lines += [header, rule]
        for b in top:
            lines.append(
                f"  {b['e_int_kcal']:>11.2f} {b['n_frames']:>7} "
                f"{100 * b['frame_share']:>6.1f}% {b['n_seeds_hit']:>6} "
                f"{b['weight']:>7.3f} {b['n_contacts']:>9}")
        if rest:
            other_frames = sum(b["n_frames"] for b in rest)
            other_share = sum(b["frame_share"] for b in rest)
            other_weight = sum(b["weight"] for b in rest)
            lines.append(
                f"  {'other (' + str(len(rest)) + ')':>11} {other_frames:>7} "
                f"{100 * other_share:>6.1f}% {'-':>6} {other_weight:>7.3f} "
                f"{'-':>9}")
        occ_mean = p["occupancy_mean_contacts"]
        occ_diss = p["occupancy_dissolved_fraction"]
        weighted_mean = sum(b["e_int_kcal"] * (b["frame_share"] or 0.0)
                            for b in basins)
        lines.append(
            f"  E_int(min) {p['e_int_min_kcal']:.2f} kcal/mol vs "
            f"frame-share-weighted mean {weighted_mean:.2f} kcal/mol")
        lines.append(
            f"  contacts -- distinct minima {p['mean_contacts']:.2f}, "
            f"occupancy (frame-weighted) {occ_mean:.2f}")
        lines.append(
            f"  dissolved -- distinct minima {100 * p['dissolved_fraction']:.0f}%, "
            f"occupancy (frame-weighted) {100 * occ_diss:.0f}%")
        lines.append("")

    spacings = sorted({s["scored_frame_spacing_fs"] for s in summaries
                       if s["scored_frame_spacing_fs"] is not None})
    spacing_str = ("-" if not spacings else
                   f"{spacings[0]:.0f} fs" if len(spacings) == 1 else
                   ", ".join(f"{s:.0f} fs" for s in spacings))
    lines.append(f"scored frame spacing: {spacing_str}")

    lines.append(
        "\n  How many of the scored frames quenched into each basin, pooled "
        "across the\n  packings at that n. 'seeds' = how many of them visited "
        "it, the same idiom as\n  'found by'. These are inherent-structure "
        "populations of a gas-phase search,\n  not Boltzmann populations and "
        "not free energies, and they are a diagnostic\n  only -- no E_int "
        "here is built from them. The caveats that come with\n  reading them "
        "are in README: Reading report.txt. Compare the frame spacing\n  "
        "above against a decorrelation time measured on *this* system "
        "(0.55 ps for\n  pyrazine + 3 CHCl3).")
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
             f"({worst['label']} seed {worst['seed']})"]

    warning = wall_warning(worst_frac)
    if warning:
        lines += ["", warning]
    else:
        lines.append(
            f"  Under the {100 * WALL_WARN_FRACTION:.0f}% threshold, so the "
            "wall is a safety net rather than load-bearing.")
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
    MD and no calculator. Each candidate's basin was first *reached* at
    `min(c["frames"])` -- the representative's own `frame` is arbitrary among
    a group of members agreeing to within 1 meV, but the earliest of them is
    when the trajectory first visited this minimum, which is what "best found
    at" means. A duplicate cannot move a running minimum by more than the
    dedup tolerance, so working from the deduped list is safe.

    Returns None when there is nothing to say: too few candidates to see a
    trend, or a trajectory too short for the late-quarter cut to mean
    anything.
    """
    candidates = summary.get("candidates") or []
    if len(candidates) < 4:
        return None
    firsts = [min(c["frames"]) for c in candidates]
    last = max(firsts)
    if last <= 0:
        return None

    cut = LATE_TRAJECTORY_FRACTION * last
    best_i = min(range(len(candidates)),
                key=lambda i: candidates[i]["interaction_eV"])
    early = [candidates[i]["interaction_eV"] for i in range(len(candidates))
             if firsts[i] <= cut]
    late = [candidates[i]["interaction_eV"] for i in range(len(candidates))
            if firsts[i] > cut]
    # Undefined rather than zero when the trajectory has no candidates on one
    # side of the cut: there was nothing there to improve on. Clamped at zero
    # because the quantity being reported is a *running* minimum, which cannot
    # rise -- a late quarter that found nothing better improved it by nothing,
    # and printing how much worse its own best was would only invite reading
    # the sign backwards.
    gain = (min(min(late) - min(early), 0.0) * EV_TO_KCAL
            if early and late else None)
    return {
        "best_at": firsts[best_i] / last,
        "late_gain_kcal": gain,
        "still_improving": (firsts[best_i] > cut
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
        "\n  'best found at' = how far into the trajectory the lowest-E_int "
        "candidate was\n  found; 'late gain' = what the final "
        f"{100 - 100 * LATE_TRAJECTORY_FRACTION:.0f}% took off E_int(min) in "
        "kcal/mol,\n  negative meaning it was still finding better "
        "geometries at the end.")

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
        f"minimum, by the\n  same {1000 * DEDUPE_TOL_EV:.0f} meV criterion "
        "that dedupes candidates within a run. Their\n  agreement is the "
        "evidence, not their scatter: 'spread' is the full range of\n  the "
        "per-packing minima, printed to be seen rather than used as an error "
        f"bar.\n  Widest here: {worst:.2f} kcal/mol -- a difference you go on "
        "to take between sweeps\n  has to be credible against that, and a "
        "double difference accumulates it four\n  times. See README: Reading "
        "report.txt.")

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


def format_report(params, summaries):
    """The whole report for either generator, from `sweep.json` / `dock.json`.

    One pipeline, not two. Both generators produce the same object -- a set of
    continuum-relaxed, wall-free local minima, scored to the identical
    criterion -- so both pool, dedupe, increment and tabulate the same way.
    What legitimately differs is what corroborates a number (independent
    packings agreeing, against independent random placements agreeing) and
    what provenance a geometry has, and that is what the two branches below
    are for. The rule that docked and swept *numbers* are never pooled is
    about mixing the two generators' runs in one table; one `sweep.json` or
    one `dock.json` never does.

    `pooled` is computed once here and threaded through every section that
    needs it, rather than each calling `pool_by_n(summaries)` on its own --
    they would recompute the same dedupe over the same candidates three times
    over.
    """
    docked = summaries[0]["pack_mode"] == "dock"
    parts = [banner("docking report" if docked else "n-sweep report")]

    # `all_n_min_kcal` is the docking chain's full internal walk, which
    # `_increments` reads and nobody needs to see spelt out in a params block.
    shown = {k: ("-" if v is None else v) for k, v in params.items()
             if k != "all_n_min_kcal"}
    parts.append(kv_block("Parameters", shown))

    pooled = pool_by_n(summaries)

    parts.append("E_int(n) = E(solute + n solvent) - E(solute) - n E(solvent)\n"
                 + "-" * 58 + "\n" + format_table(params, pooled))

    parts.append(format_best_geometry(pooled))

    if docked:
        # One run per n, so there are no per-packing rows to show; what a
        # docking chain has instead is per-parent rows.
        parts.append(format_parent_detail(summaries))
        return "\n\n".join(parts) + "\n"

    parts.append(format_seed_detail(summaries, pooled))
    parts.append(format_basin_occupancy(params, summaries, pooled))

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
            f"  E_int(0) = {values} kcal/mol, zero by construction: the "
            "n = 0 cluster *is* the\n  solute reference. More than a few "
            "hundredths off means the reference\n  optimisations and the "
            "n = 0 run landed in different minima, and every other\n  row is "
            "offset by that much.")
    return "\n\n".join(parts) + "\n"


def write_best_geometries(out_dir, pooled, prefix):
    """Copy each n's winning `best.xyz` to `<out_dir>/best_n<N>.xyz`.

    The deliverable of a run, not just its report: `pooled` already knows, per
    n, which run directory's `best.xyz` is the minimum (see `pool_by_n`), so
    this only has to read those files and copy them out, one per n.

    One file per n, not one multi-frame xyz, because the frames differ in atom
    count -- n = 0..k goes 10, 15, 20, ... atoms for a fixed solute and
    solvent -- and a multi-frame xyz has no way to say "these frames are
    unrelated". A viewer reads it as a trajectory, takes the first frame's
    atom count, and applies it to the rest, so Avogadro (and VMD, and most
    others) render only n = 0 and silently drop every row after it. A set of
    chemically distinct clusters is not a trajectory, and no single-file XYZ
    shape expresses that it isn't one.

    Pure text handling -- no ASE, keeping this module's import list at module
    scope unchanged -- because every field needed is already a line of text:
    the natoms line, the comment line, and the coordinate block. Each frame's
    comment line (extended-xyz's `key=value` line) gains `<prefix>_n=`,
    `<prefix>_E_int_kcal=` and one field naming where it came from, appended
    after whatever ASE already wrote there (`Properties=...`, `energy=...`),
    which xtb reads back and this does not disturb. That last field is what
    still says which search produced a file once it has been lifted out of
    the output directory, so it differs by generator: `sweep_packing=` names
    the run directory a packing wrote, `dock_parent=` the branch of the chain
    a construction descended from.

    Skips an n whose run directory or `best.xyz` is missing -- a `sweep.json`
    copied without its run directories, say -- and returns an empty list if
    none are present. The run directory is looked up by name under `out_dir`
    rather than by the absolute path recorded at write time, so a run that
    has been moved or copied still resolves.
    """
    out_dir = Path(out_dir)
    written = []
    for p in pooled:
        run = p["best"]["run"]
        best_xyz = out_dir / run / "best.xyz"
        if not best_xyz.is_file():
            continue
        lines = best_xyz.read_text().splitlines()
        if len(lines) < 2:
            continue
        origin = (f"dock_parent={p['best']['candidate']['parent']}"
                  if prefix == "dock" else f"sweep_packing={run}")
        lines[1] = (f"{lines[1].rstrip()} {prefix}_n={p['n_solvent']} "
                    f"{prefix}_E_int_kcal={p['e_int_min_kcal']:.6f} {origin}")
        out = out_dir / f"best_n{p['n_solvent']}.xyz"
        out.write_text("\n".join(lines) + "\n")
        written.append(out)
    return written


def format_parent_detail(summaries):
    """Per parent, under the main table: the greedy chain made visible.

    Docking's counterpart to `format_seed_detail`, and the one section a
    sweep has no analogue of -- it reads `parent_detail`, which only
    `docking._assemble_dock_n` writes.
    """
    header = (f"{'n':>3} {'parent':>6} {'best':>4} {'placements':>10} "
             f"{'E_int(min)':>12}")
    lines = ["Per-parent detail", "-----------------", header,
             "-" * len(header)]
    for p in sorted(summaries, key=_row_order):
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
        "E_int(min) among its\n  own random placements. 'best' marks the "
        "parent whose descendant became the n's\n  reported minimum -- the "
        "greedy chain made visible, and the reason docking\n  carries more "
        "than one parent forward. Unlike the sweep's 'found by', several\n  "
        "parents landing near one minimum is not independent corroboration: "
        "every\n  parent explores the same shell region with independent "
        "random poses, not a\n  differently-arranged packing.")
    return "\n".join(lines)


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

    log = RunLogger(path / "run.log", live=False)
    log.header(meta)
    for record in records:
        log.md_row(record)
    log.footer(records)
    log.close()
    written = [path / "run.log"]

    for scored in sorted(path.glob("scored*.json")):
        out = scored.with_suffix(".log")
        out.write_text(format_scored_log(_load(scored), meta))
        written.append(out)
    return written


def render_output_dir(path, name, report_name, prefix):
    """Rewrite one generator's report and `best_n<N>.xyz`, from its JSON.

    `name` is `sweep.json` or `dock.json` -- the two carry the same
    `{"params": ..., "runs": [...]}` shape, which is why one function reads
    either. A sweep's run directories also carry the MD trajectory logs,
    which get regenerated on the way through; a docking run directory has no
    trajectory and none to regenerate.
    """
    path = Path(path)
    data = _load(path / name)
    params, runs = data["params"], data["runs"]

    written = []
    for summary in runs:
        # By name under the output directory rather than by the absolute path
        # recorded at write time, so a run that has been moved or copied
        # still re-renders -- but the name itself still comes from the
        # summary, so it is never rebuilt from `<label>_seed<n>` here.
        run_dir = path / Path(summary["run_dir"]).name
        if (run_dir / "metadata.json").is_file():
            written += render_run_dir(run_dir)

    report = path / report_name
    report.write_text(format_report(params, runs))
    written.append(report)

    written += write_best_geometries(path, pool_by_n(runs), prefix)
    return written


def render(path):
    path = Path(path)
    if (path / "dock.json").exists():
        return render_output_dir(path, "dock.json", "dock_report.txt", "dock")
    if (path / "sweep.json").exists():
        return render_output_dir(path, "sweep.json", "report.txt", "sweep")
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
