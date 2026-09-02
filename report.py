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
because `solvate_md` imports it into every spawned MD worker. The one heavy
import (`ensemble`, for Boltzmann weights when regenerating a log from an old
summary that predates them) is done lazily inside the function that needs it,
so that both the live and the regenerated path use the *same* weighting code.
That matters: the ensemble average of E(cluster) and of E_int must rest on
identical weights or they silently disagree.

Regenerate logs for anything already on disk, without running any MD:

    python -m report pyrazine_sweep_output/
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Lives here rather than in `ensemble` so that a text-only consumer never has
# to import ASE to format a number. `ensemble` re-exports it.
EV_TO_KCAL = 23.060548

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


def format_run_footer(records, target_K=None, elapsed_s=None):
    if not records:
        return kv_block("MD summary", {"frames written": 0})

    n = len(records)
    temps = [r["temperature_K"] for r in records]
    epots = [r["potential_energy_eV"] for r in records]
    walls = [r["wall_energy_eV"] for r in records]
    mean_T = sum(temps) / n
    mean_E = sum(epots) / n
    sd_E = (sum((e - mean_E) ** 2 for e in epots) / n) ** 0.5
    n_active = sum(1 for w in walls if w > 0.0)
    frac = n_active / n

    summary = {
        "frames written": n,
        "mean temperature/K": (f"{mean_T:.1f}" + (f"  (target {target_K:.1f})"
                                                  if target_K else "")),
        "E_pot/eV": f"{mean_E:.6f} +/- {sd_E:.6f}",
        "wall active": f"{n_active} / {n} frames ({100 * frac:.1f}%)",
        "max wall energy/eV": f"{max(walls):.6f}",
    }
    if elapsed_s is not None:
        summary["elapsed"] = f"{elapsed_s:.1f} s"
    block = kv_block("MD summary", summary)

    if frac > WALL_WARN_FRACTION:
        block += (
            "\n\n"
            f"  ** WARNING: the confining wall is active in {100 * frac:.0f}% of "
            "frames.\n"
            "  ** Measured regimes are 4-5% (gas-phase sampling, shell self-bound,\n"
            "  ** wall a safety net) versus 50-60% (ALPB(water), shell dissociating).\n"
            f"  ** Above {100 * WALL_WARN_FRACTION:.0f}% the wall is load-bearing: it "
            "is holding together a shell\n"
            "  ** the Hamiltonian wants to disperse, and these energies are\n"
            "  ** contaminated by the confinement. Sample in gas phase, or accept\n"
            "  ** that there is no bound shell to find."
        )
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

def fill_absolute_energies(summary, temperature_K=298.0):
    """Add `ensemble_energy_eV` / `min_energy_eV` if a summary lacks them.

    `boltzmann_weights` subtracts the minimum before exponentiating, so the
    weights over E(cluster) and over E_int are identical -- E_int differs from
    E(cluster) only by the constant E(solute) + n E(solvent). The two averages
    therefore cannot disagree, and this fills in old summaries written before
    the absolute energies were reported.
    """
    candidates = summary.get("candidates") or []
    if not candidates or "ensemble_energy_eV" in summary:
        return summary
    from ensemble import ensemble_energy   # heavy (ASE); only on this path

    energies = [c["energy_eV"] for c in candidates]
    T = summary.get("temperature_K", temperature_K)
    summary["ensemble_energy_eV"] = ensemble_energy(energies, T)
    summary["min_energy_eV"] = min(energies)
    return summary


def _candidate_table(candidates, temperature_K):
    from ensemble import boltzmann_weights   # heavy (ASE); only on this path

    weights = boltzmann_weights([c["energy_eV"] for c in candidates],
                                temperature_K)
    lines = [f"{'frame':>7} {'E(cluster)/eV':>16} {'E_int/kcal':>12} "
             f"{'weight':>8} {'contacts':>9} {'min gap/A':>10} {'conv':>5} "
             f"{'fmax':>7}"]
    for c, w in zip(candidates, weights):
        gap = c.get("min_gap_A")
        gap_s = "-" if gap is None or gap != gap else f"{gap:.3f}"
        lines.append(
            f"{c['frame']:>7d} {c['energy_eV']:>16.6f} "
            f"{c['interaction_eV'] * EV_TO_KCAL:>12.2f} "
            f"{w:>8.3f} {c['n_contacts']:>9d} {gap_s:>10} "
            f"{'yes' if c['converged'] else 'NO':>5} {c['fmax']:>7.4f}")
    return "\n".join(lines)


def format_scored_log(summary, meta=None):
    """The scoring section: provenance, references, candidates, result."""
    summary = fill_absolute_energies(dict(summary))
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
    }
    if meta:
        dt = meta.get("timestep_fs")
        provenance["sampling MD"] = (
            f"{meta.get('n_steps')} steps x {dt} fs at "
            f"{meta.get('temperature_K')} K, dumped every "
            f"{meta.get('dump_interval')}")
    parts.append(kv_block("Provenance", provenance))

    parts.append(kv_block("Scoring", {
        "calculator": summary.get("calculator"),
        "continuum": _solvation_str(summary.get("scoring_solvation")),
        "optimiser": (f"BFGS, fmax {summary.get('opt_fmax', '?')} eV/A, "
                      f"max {summary.get('opt_steps', '?')} steps"),
        "wall": "none -- dissolution is the signal, not a failure;",
        "": ("a solvent molecule that optimises away into the "
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
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------

def _leg_name(summary):
    solvation = summary.get("scoring_solvation") or [None, None]
    solvent = solvation[1] if len(solvation) > 1 else None
    solvent = solvent or "gas"
    name = summary.get("solute_label") or summary.get("conformer")
    if name is None:
        label = summary.get("label", "?")
        suffix = f"_{solvent}_n{summary.get('n_solvent')}"
        name = label[:-len(suffix)] if label.endswith(suffix) else label
    return f"{name}/{solvent}"


def format_table(summaries, n_values=None):
    """E_int vs n, with the dissolution diagnostics and a raw cluster energy."""
    summaries = [fill_absolute_energies(dict(s)) for s in summaries]
    lines = [f"{'leg':20s} {'n':>3} {'E_int(ens)':>12} {'E_int(min)':>12} "
             f"{'E(cluster)/eV':>15} {'contacts':>9} {'dissolved':>10} "
             f"{'uniq':>5}",
             "-" * 92]
    for s in sorted(summaries, key=lambda s: (_leg_name(s), s["n_solvent"],
                                              s.get("seed", 0))):
        lines.append(
            f"{_leg_name(s):20s} {s['n_solvent']:>3} "
            f"{s['ensemble_interaction_kcal']:>12.2f} "
            f"{s['min_interaction_kcal']:>12.2f} "
            f"{_num(s.get('ensemble_energy_eV'), '.6f'):>15} "
            f"{s['mean_contacts']:>9.2f} "
            f"{100 * s['dissolved_fraction']:>9.0f}% "
            f"{s['n_unique']:>5}")
    lines.append(
        "\nE_int in kcal/mol, E(cluster) in eV (ASE's native unit). 'contacts' = "
        "solvent\nmolecules touching the solute after optimisation; 'dissolved' "
        "= fraction of\nunique candidates with no contact at all.\n"
        "\nE(cluster) is Boltzmann-averaged over the same weights as E_int, and is "
        "here\nto sanity-check the magnitudes it came from. It is NOT comparable "
        "across rows\nof different n -- successive rows differ by a whole solvent "
        "molecule. Making\nthat comparison meaningful is exactly what E_int is "
        "for; subtract those\ninstead.")
    return "\n".join(lines)


def format_sweep_report(params, summaries):
    """Params block plus the E_int(n) table."""
    parts = [banner("n-sweep report")]

    if params:
        shown = {k: ("-" if v is None else v) for k, v in params.items()}
        parts.append(kv_block("Parameters", shown))
    else:
        parts.append("Parameters\n----------\n  not recorded -- this sweep "
                     "predates the params block.")

    parts.append("E_int(n) = E(solute + n solvent) - E(solute) - n E(solvent)\n"
                 + "-" * 58 + "\n" + format_table(summaries))

    zeros = [s for s in summaries if s.get("n_solvent") == 0]
    if zeros:
        values = ", ".join(f"{_leg_name(s)} {s['ensemble_interaction_kcal']:.2f}"
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

    meta_path = path / "metadata.json"
    if meta_path.exists():
        meta = _load(meta_path)
        energies_path = path / "energies.json"
        records = _load(energies_path) if energies_path.exists() else []
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
        out.write_text(format_scored_log(_load(scored), meta))
        written.append(out)
    return written


def render_sweep_dir(path):
    """Rewrite `report.txt` plus every run's logs, from `sweep.json`."""
    path = Path(path)
    data = _load(path / "sweep.json")
    if isinstance(data, list):        # pre-params-block sweeps
        params, runs = {}, data
    else:
        params, runs = data.get("params", {}), data["runs"]

    written = []
    for summary in runs:
        run_dir = path / f"{summary['label']}_seed{summary.get('seed', 0)}"
        if not run_dir.is_dir():
            run_dir = Path(summary["run_dir"])
        if run_dir.is_dir():
            written += render_run_dir(run_dir)

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
