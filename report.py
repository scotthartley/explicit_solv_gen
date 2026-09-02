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

    python -m report pyrazine_sweep_output/
"""

import single_thread  # noqa: F401  -- must precede numpy; see its docstring

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

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
    return out.stdout.strip() or None if out.returncode == 0 else None


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def kv_block(title, mapping, indent="  "):
    """An aligned key/value section -- the common building block of every log."""
    rows = [(str(k), "-" if v is None else str(v)) for k, v in mapping.items()]
    width = max((len(k) for k, _ in rows), default=0)
    lines = [title, "-" * len(title)]
    lines += [f"{indent}{k:<{width}}   {v}".rstrip() for k, v in rows]
    return "\n".join(lines)


def banner(text):
    return f"{_RULE}\n {text}\n{_RULE}"


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


def _mean(values):
    return sum(values) / len(values)


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
    n_solvent = meta.get("n_solvent", 0)
    aps = meta.get("atoms_per_solvent")
    axes = meta.get("shell_semi_axes") or []
    padding = meta.get("shell_padding")
    dump = meta.get("dump_interval") or 1
    n_steps = meta.get("n_steps") or 0
    dt = meta.get("timestep_fs")

    parts = [banner(f"Explicit-solvent MD: {meta.get('label', '?')} "
                    f"(seed {meta.get('seed', '?')})")]

    parts.append(kv_block("System", {
        "solute": f"{meta.get('solute_path')}  ({meta.get('n_solute')} atoms)",
        "solvent": f"{meta.get('solvent_path')}  ({aps} atoms x {n_solvent})",
        "solvent name": meta.get("solvent"),
        "total atoms": meta.get("n_atoms"),
    }))

    packing = {
        "solute semi-axes/A": "  ".join(f"{a:.3f}" for a in axes) or None,
        "shell padding/A": _num(padding, ".3f"),
        "region semi-axes/A": ("  ".join(f"{a + padding:.3f}" for a in axes)
                               if axes and padding is not None else None),
        "packmol tolerance/A": _num(meta.get("tolerance"), ".2f"),
    }
    if "shell_fill" in extra:
        packing["shell fill"] = _num(extra["shell_fill"], ".2f")
    packing["wall distance/A"] = _num(meta.get("wall_distance"), ".3f")
    if "wall_slack" in extra:
        packing["wall slack"] = f"{extra['wall_slack']:.2f} solvent diameters"
    packing["wall k/eV A^-2"] = _num(meta.get("wall_k"), ".3f")
    parts.append(kv_block("Packing", packing))

    parts.append(kv_block("Sampling Hamiltonian", {
        "calculator": meta.get("calculator"),
        "continuum": _solvation_str(meta.get("solvation")),
        "wall": "SolventShellWall on solvent atoms only",
    }))

    parts.append(kv_block("Pre-MD relaxation (clash relief, fmax 0.5)", {
        "converged": {True: "yes", False: "no"}.get(meta.get("relax_converged")),
        "final fmax/eV A^-1": _num(meta.get("relax_final_fmax"), ".4f"),
    }))

    parts.append(kv_block("Molecular dynamics", {
        "ensemble": "Langevin NVT, FixCom",
        "temperature/K": _num(meta.get("temperature_K"), ".1f"),
        "timestep/fs": _num(dt, ".2f"),
        "friction": _num(meta.get("friction"), ".4f"),
        "equilibration": (f"{meta.get('n_equilibrate_steps')} steps"
                          + (f" ({meta['n_equilibrate_steps'] * dt / 1000:.2f} ps)"
                             if dt and meta.get("n_equilibrate_steps") else "")),
        "production": (f"{n_steps} steps"
                       + (f" ({n_steps * dt / 1000:.2f} ps)" if dt else "")),
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
        self._records = []
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
        self._records.append(record)
        self._write(format_md_row(record))

    def footer(self):
        elapsed = None if self._t0 is None else time.time() - self._t0
        self._write("")
        self._write(format_run_footer(self._records, self._target_K, elapsed))

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
             f"{'fmax':>7} {'gnorm':>9} {'E_wall/eV':>11}"]
    for c, w in zip(candidates, weights):
        gap = c.get("min_gap_A")
        gap_s = "-" if gap is None or gap != gap else f"{gap:.3f}"
        # Of the *sampling* frame this candidate came from -- the scorer
        # applies no wall. Blank rather than 0 when the run predates the
        # records being carried through, so "not known" cannot read as "zero".
        wall = c.get("wall_energy_eV")
        wall_s = "" if wall is None else f"{wall:.6f}"
        # fmax and gnorm are the same convergence in ASE's and xtb's units:
        # per-atom max force in eV/A, and the norm over all components in
        # Eh/bohr. Blank rather than 0 for a `scored.json` written before the
        # gradient norm was recorded.
        gnorm = c.get("gnorm_Eh_bohr")
        gnorm_s = "" if gnorm is None else f"{gnorm:.2e}"
        lines.append(
            f"{c['frame']:>7d} {c['energy_eV']:>16.6f} "
            f"{c['interaction_eV'] * EV_TO_KCAL:>12.2f} "
            f"{w:>8.3f} {c['n_contacts']:>9d} {gap_s:>10} "
            f"{'yes' if c['converged'] else 'NO':>5} {c['fmax']:>7.4f} "
            f"{gnorm_s:>9} {wall_s:>11}".rstrip())
    return "\n".join(lines)


def _sampling_wall_str(summary):
    """One-line summary of the wall during the MD that produced these frames."""
    wall = summary.get("sampling_wall")
    if not wall or wall.get("wall_active_fraction") is None:
        return None
    line = (f"active in {100 * wall['wall_active_fraction']:.1f}% of "
            f"{wall['n_frames']} frames "
            f"(max {wall['max_wall_energy_eV']:.6f} eV)")
    n_scored_active = summary.get("n_scored_wall_active")
    if n_scored_active is not None:
        line += (f"; {n_scored_active} of {summary.get('n_frames_scored')} "
                 "scored frames affected")
    return line


def backfill_wall_stats(summary, records):
    """Fill a scored summary's wall diagnostic from the run's `energies.json`.

    For `scored.json` files written before the scorer recorded any of this.
    Only ever fills what is missing, so a summary written by the current code
    re-renders to exactly the log it was first written with -- and
    `n_scored_wall_active`, which needs to know which frames were scored,
    stays absent rather than being guessed at from the unique candidates.

    Mutates and returns `summary`. The caller does not write it back:
    regenerating a log renders text, it does not rewrite results.
    """
    if not records:
        return summary
    if summary.get("sampling_wall") is None:
        summary["sampling_wall"] = wall_stats(records)
    # Trajectory dump i and energies[i] come from the same `record()` closure
    # in `solvate_md`, and `Candidate.frame` is the true dump index, so the
    # records index directly.
    for candidate in summary.get("candidates") or []:
        frame = candidate.get("frame")
        if (candidate.get("wall_energy_eV") is None
                and frame is not None and 0 <= frame < len(records)):
            candidate["wall_energy_eV"] = float(records[frame]["wall_energy_eV"])
    return summary


def format_scored_log(summary, meta=None):
    """The scoring section: provenance, references, candidates, result."""
    n = summary.get("n_solvent", 0)
    T = summary.get("temperature_K", 298.0)
    e_solute = summary.get("e_solute_ref_eV")
    e_solvent = summary.get("e_solvent_ref_eV")
    candidates = summary.get("candidates") or []

    parts = [banner(f"Continuum rescoring: {summary.get('label', '?')} "
                    f"(seed {summary.get('seed', '?')})")]

    stride = summary.get("stride")
    max_frames = summary.get("max_frames")
    provenance = {
        "run directory": summary.get("run_dir"),
        "trajectory": (f"traj.xyz, every {stride}th frame"
                       if stride else "traj.xyz")
                      + (f", at most {max_frames}" if max_frames else "")
                      + f"  ->  {summary.get('n_frames_scored')} scored",
        "sampling Hamiltonian": (
            f"{summary.get('calculator')}, "
            f"{_solvation_str(summary.get('sampling_solvation'))}"),
        # The frames scored below were drawn from a walled trajectory, so how
        # hard that wall was working qualifies every energy in this file.
        "sampling wall": _sampling_wall_str(summary),
    }
    if meta:
        dt = meta.get("timestep_fs")
        provenance["sampling MD"] = (
            f"{meta.get('n_steps')} steps x {dt} fs at "
            f"{meta.get('temperature_K')} K, dumped every "
            f"{meta.get('dump_interval')}")
    parts.append(kv_block("Provenance", provenance))

    # `best.xyz` is the geometry relaxed here, so state the surface it was
    # relaxed on: reproducing it in the xtb binary needs the same continuum,
    # and the two codes converge on criteria that are not comparable by eye.
    solvation = summary.get("scoring_solvation") or []
    alpb = f" --alpb {solvation[1]}" if len(solvation) > 1 and solvation[1] else ""
    parts.append(kv_block("Scoring", {
        "calculator": summary.get("calculator"),
        "continuum": _solvation_str(summary.get("scoring_solvation")),
        "optimiser": (f"BFGS, fmax {summary.get('opt_fmax', '?')} eV/A "
                      "(ASE: largest per-atom force). xtb's"),
        "": ("`--opt normal` instead stops at a gradient norm of 1e-3 Eh/a "
             "-- see"),
        " ": f"the gnorm column. Max {summary.get('opt_steps', '?')} steps",
        "reproduce": f"xtb best.xyz --gfn 2{alpb} --sp",
        "  ": ("best.xyz was relaxed WITH the continuum -- omitting "
               + (f"--alpb {solvation[1]}" if alpb else "it")),
        "   ": "puts it on a different surface",
        "wall": "none -- dissolution is the signal, not a failure;",
        "    ": ("a solvent molecule that optimises away into the "
                 "continuum contributes ~0 to E_int"),
    }))

    offset = (None if e_solute is None or e_solvent is None
              else e_solute + n * e_solvent)
    parts.append(kv_block("References (each relaxed in the scoring environment)", {
        "E(solute)/eV": _num(e_solute, ".6f"),
        "E(solvent)/eV": _num(e_solvent, ".6f"),
        "offset/eV": (f"E(solute) + {n} x E(solvent) = {offset:.6f}"
                      if offset is not None else None),
    }))

    if candidates:
        parts.append("Candidates (unique minima, lowest first)\n"
                     "----------------------------------------\n"
                     + _candidate_table(candidates, T))

    dissolved = summary.get("dissolved_fraction")
    converged = summary.get("converged_fraction")
    result = {
        "E_int      (Boltzmann)": (
            f"{_num(summary.get('ensemble_interaction_kcal'), '.2f')} kcal/mol"
            "     <- the reported number"),
        "E_int      (minimum)": (
            f"{_num(summary.get('min_interaction_kcal'), '.2f')} kcal/mol"),
        "E(cluster) (Boltzmann)": f"{_num(summary.get('ensemble_energy_eV'), '.6f')} eV",
        "E(cluster) (minimum)": f"{_num(summary.get('min_energy_eV'), '.6f')} eV",
        "references": (f"E(solute) {_num(e_solute, '.6f')} eV, "
                       f"{n} x E(solvent) {_num(e_solvent, '.6f')} eV"),
        "frames scored": (f"{summary.get('n_frames_scored')} "
                          f"({summary.get('n_unique')} unique, "
                          f"{_num(None if converged is None else 100 * converged, '.0f')}"
                          "% converged)"),
        "mean contacts": _num(summary.get("mean_contacts"), ".2f"),
        "dissolved": (f"{_num(None if dissolved is None else 100 * dissolved, '.0f')}"
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
    # energies the confinement contaminated.
    warning = wall_warning((summary.get("sampling_wall")
                            or {}).get("wall_active_fraction"))
    if warning:
        parts.append(warning)
    # Same reason, one line up the stack: these are the energies a residual
    # force contaminated.
    warning = unconverged_warning(summary.get("n_unconverged_unique"),
                                  summary.get("n_unique"))
    if warning:
        parts.append(warning)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------

def _wall_fraction(summary):
    """Wall-active fraction of the run behind this summary, or None."""
    return (summary.get("sampling_wall") or {}).get("wall_active_fraction")


def _leg_name(summary):
    """What this row is: which solute, scored in which continuum."""
    solvation = summary.get("scoring_solvation") or [None, None]
    solvent = (solvation[1] if len(solvation) > 1 else None) or "gas"
    return f"{summary.get('solute_label') or summary['label']}/{solvent}"


def format_table(summaries):
    """E_int vs n, with the dissolution diagnostics and a raw cluster energy.

    One sweep is one solute in one solvent, so the leg is normally a constant
    and is stated once above the table rather than repeated down a column. The
    column comes back only when it is actually carrying information -- when
    summaries from several sweeps have been pooled by hand, which is how a
    double difference gets assembled now.
    """
    legs = sorted({_leg_name(s) for s in summaries})
    one_leg = len(legs) == 1

    lines = [f"leg: {legs[0]}", ""] if one_leg else []
    leg_head = "" if one_leg else f"{'leg':20s} "
    lines += [f"{leg_head}{'n':>3} {'seed':>4} {'E_int(ens)':>12} {'E_int(min)':>12} "
              f"{'E(cluster)/eV':>15} {'contacts':>9} {'dissolved':>10} "
              f"{'uniq':>5} {'wall':>6}",
              "-" * (83 if one_leg else 104)]
    for s in sorted(summaries, key=lambda s: (_leg_name(s), s["n_solvent"],
                                              s.get("seed", 0))):
        frac = _wall_fraction(s)
        lines.append(
            ("" if one_leg else f"{_leg_name(s):20s} ")
            + f"{s['n_solvent']:>3} "
            f"{s.get('seed', 0):>4} "
            f"{s['ensemble_interaction_kcal']:>12.2f} "
            f"{s['min_interaction_kcal']:>12.2f} "
            f"{_num(s.get('ensemble_energy_eV'), '.6f'):>15} "
            f"{s['mean_contacts']:>9.2f} "
            f"{100 * s['dissolved_fraction']:>9.0f}% "
            f"{s['n_unique']:>5} "
            + (f"{'-':>6}" if frac is None else f"{100 * frac:>5.0f}%"))
    lines.append(
        "\nE_int in kcal/mol, E(cluster) in eV (ASE's native unit). 'contacts' = "
        "solvent\nmolecules touching the solute after optimisation; 'dissolved' "
        "= fraction of\nunique candidates with no contact at all. 'wall' = "
        "fraction of the sampling\ntrajectory's frames with a nonzero wall "
        "energy.\n"
        "\nE(cluster) is Boltzmann-averaged over the same weights as E_int, and is "
        "here\nto sanity-check the magnitudes it came from. It is NOT comparable "
        "across rows\nof different n -- successive rows differ by a whole solvent "
        "molecule. Making\nthat comparison meaningful is exactly what E_int is "
        "for; subtract those\ninstead.")
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

    one_leg = len({_leg_name(s) for s in summaries}) == 1
    leg_head = "" if one_leg else f"{'leg':20s} "
    lines = ["Sampling convergence", "--------------------",
             f"{leg_head}{'n':>3} {'seed':>4} {'best found at':>14} "
             f"{'late gain':>10} {'verdict':>16}",
             "-" * (52 if one_leg else 73)]

    for s, conv in sorted(rows, key=lambda r: (_leg_name(r[0]),
                                               r[0]["n_solvent"],
                                               r[0].get("seed", 0))):
        gain = conv["late_gain_kcal"]
        lines.append(
            ("" if one_leg else f"{_leg_name(s):20s} ")
            + f"{s['n_solvent']:>3} "
            f"{s.get('seed', 0):>4} "
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


def format_seed_spread(summaries):
    """Run-to-run scatter of E_int at fixed n -- the only error bar here.

    Independent seeds differ only in their packing and initial velocities, so
    anything they disagree about is sampling error rather than chemistry.
    That makes the spread the honest uncertainty on every number in the table
    above, and it is worth stating loudly that a single-seed sweep has none:
    an absence of error bars reads as precision unless something says
    otherwise.
    """
    groups = {}
    for s in summaries:
        groups.setdefault((_leg_name(s), s["n_solvent"]), []).append(s)

    if not groups:
        return None

    lines = ["Seed spread", "-----------"]
    if max(len(g) for g in groups.values()) < 2:
        return "\n".join(lines + [
            "  Single seed: no error bar. Every E_int above is one trajectory's",
            "  answer, and the difference between two such answers at the same n",
            "  has been measured at whole kcal/mol. Re-run with --seeds 3 before",
            "  subtracting two sweeps from each other -- a double difference is",
            "  four of these numbers, and it inherits the error of all four."])

    one_leg = len({leg for leg, _ in groups}) == 1
    leg_head = "" if one_leg else f"{'leg':20s} "
    lines += [f"{leg_head}{'n':>3} {'seeds':>6} {'E_int(ens)':>22} "
              f"{'E_int(min)':>22}",
              "-" * (56 if one_leg else 77)]

    worst = 0.0
    for (leg, n), group in sorted(groups.items()):
        ens = [g["ensemble_interaction_kcal"] for g in group]
        mins = [g["min_interaction_kcal"] for g in group]
        worst = max(worst, max(ens) - min(ens), max(mins) - min(mins))
        lines.append(
            ("" if one_leg else f"{leg:20s} ")
            + f"{n:>3} {len(group):>6} "
            f"{_mean(ens):>10.2f} +/- {(max(ens) - min(ens)) / 2:<7.2f} "
            f"{_mean(mins):>10.2f} +/- {(max(mins) - min(mins)) / 2:<7.2f}")

    lines.append(
        "\n  Mean over seeds +/- half the full range, kcal/mol. Half-range "
        "rather than\n  a standard deviation because three seeds do not "
        "estimate one, and the\n  quantity that matters is how far apart two "
        "runs of the same thing can land."
        f"\n\n  Widest spread in this sweep: {worst:.2f} kcal/mol. Any "
        "difference you go on\n  to take between sweeps has to clear that, "
        "and a double difference of four\n  such numbers accumulates it four "
        "times.")
    return "\n".join(lines)


def format_sweep_report(params, summaries):
    """Params block plus the E_int(n) table."""
    parts = [banner("n-sweep report")]

    shown = {k: ("-" if v is None else v) for k, v in params.items()}
    parts.append(kv_block("Parameters", shown))

    parts.append("E_int(n) = E(solute + n solvent) - E(solute) - n E(solvent)\n"
                 + "-" * 58 + "\n" + format_table(summaries))

    for section in (format_sampling_convergence(summaries),
                    format_seed_spread(summaries),
                    format_wall_diagnostic(summaries)):
        if section:
            parts.append(section)

    zeros = [s for s in summaries if s.get("n_solvent") == 0]
    if zeros:
        one_leg = len({_leg_name(s) for s in summaries}) == 1
        values = ", ".join(
            (f"{s['ensemble_interaction_kcal']:.2f}" if one_leg else
             f"{_leg_name(s)} {s['ensemble_interaction_kcal']:.2f}")
            for s in sorted(zeros, key=_leg_name))
        parts.append(
            "Self-consistency check\n"
            "----------------------\n"
            f"  E_int(0) = {values} kcal/mol.\n"
            "  Zero by construction: the n = 0 cluster *is* the solute reference.\n"
            "  A departure of more than a few hundredths means the reference\n"
            "  optimisations and the n = 0 run landed in different minima, and\n"
            "  every other row is offset by that much.")
    return "\n\n".join(parts) + "\n"


# --------------------------------------------------------------------------
# Post-hoc regeneration
# --------------------------------------------------------------------------

def _load(path):
    return json.loads(Path(path).read_text())


def render_run_dir(path):
    """Rewrite `run.log`, and a `.log` for every scored summary, from JSON.

    No MD and no calculator: everything comes off disk.
    """
    path = Path(path)
    written = []

    energies_path = path / "energies.json"
    records = _load(energies_path) if energies_path.exists() else []

    meta_path = path / "metadata.json"
    if meta_path.exists():
        meta = _load(meta_path)
        with RunLogger(path / "run.log", live=False) as log:
            log.header(meta)
            for record in records:
                log.md_row(record)
            log.footer()
        written.append(path / "run.log")
    else:
        meta = None

    for scored in sorted(path.glob("scored*.json")):
        out = scored.with_suffix(".log")
        summary = backfill_wall_stats(_load(scored), records)
        out.write_text(format_scored_log(summary, meta))
        written.append(out)
    return written


def render_sweep_dir(path):
    """Rewrite `report.txt` plus every run's logs, from `sweep.json`."""
    path = Path(path)
    data = _load(path / "sweep.json")
    params, runs = data["params"], data["runs"]

    written = []
    for summary in runs:
        run_dir = path / f"{summary['label']}_seed{summary.get('seed', 0)}"
        if not run_dir.is_dir():
            run_dir = Path(summary["run_dir"])
        if run_dir.is_dir():
            written += render_run_dir(run_dir)
            # So a sweep run before the scorer recorded any of this still gets
            # a wall column. In memory only -- `sweep.json` is not rewritten.
            energies_path = run_dir / "energies.json"
            if energies_path.exists():
                backfill_wall_stats(summary, _load(energies_path))

    report = path / "report.txt"
    report.write_text(format_sweep_report(params, runs))
    written.append(report)
    return written


def render(path):
    path = Path(path)
    if (path / "sweep.json").exists():
        return render_sweep_dir(path)
    if (path / "metadata.json").exists():
        return render_run_dir(path)
    written = []
    for child in sorted(p for p in path.iterdir() if p.is_dir()):
        if (child / "metadata.json").exists():
            written += render_run_dir(child)
    if not written:
        raise SystemExit(f"{path}: no sweep.json, metadata.json, or run "
                         "subdirectories found")
    return written


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m report <run-or-sweep-dir> ...")
    for target in sys.argv[1:]:
        for written in render(target):
            print(written)
