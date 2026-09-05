"""Hand a sweep's or a docking run's candidates to a DFT single point or reopt.

Reads **either** a `run_sweep` output directory or a `run_docking` output
directory -- both write one `scored.json` per run, in the same shape (see
`ensemble.assemble` and `docking._assemble_dock_n`), so this needs no branch
on which generator produced them.

Per n: pool every run's candidates, dedupe at `report.DEDUPE_TOL_EV` (the same
criterion `pool_by_n` uses), and keep everything within `window_kcal` of that
n's minimum. A window rather than a fixed count, so the exported set adapts to
the system -- 3 kcal/mol is ~5 kT and comfortably spans both motifs measured
on pyrazine + 2 chloroform (the both-nitrogens minimum and the pooled MD
minimum are 1.56 kcal/mol apart), which is the concrete test that it is not
too tight. `max_per_n` is a safety cap only, applied after the window.

    python -m dft_export pyrazine_chcl3/ --out pyrazine_chcl3/dft_export
    python -m dft_export pyrazine_dock/ --out pyrazine_dock/dft_export

Two things worth knowing before sending any of this to DFT, not rediscovering:

  * GFN2-xTB over-binds the C-H...N contact this pipeline was built to study
    -- 1.91 A against a literature 2.2-2.5 A -- so a DFT single point on a
    GFN2 geometry sits partway up a repulsive wall. Re-optimising at the DFT
    level is preferable wherever affordable; a single point is a lower bound
    on how much the answer will move.
  * Never export a raw MD frame for a single point. 298 K thermal strain is
    worth several kcal/mol at random and would swamp the signal this
    pipeline exists to resolve -- export only optimised candidates, which is
    all `scored_candidates.xyz` ever contains.
"""

import single_thread  # noqa: F401  -- must precede numpy; see its docstring

import argparse
import json
from pathlib import Path

from ase.io import read, write

from report import DEDUPE_TOL_EV, EV_TO_KCAL, boltzmann_weights, dedupe_energies


def _iter_run_dirs(root):
    """Every directory under `root` carrying a `scored.json`.

    Handles being pointed at a sweep/docking output directory (many run
    subdirectories) or at a single run directory directly (one `scored.json`
    right there).
    """
    root = Path(root)
    if (root / "scored.json").exists():
        yield root
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "scored.json").exists():
            yield child


def export_dft(run_or_sweep_dir, out_dir, window_kcal=3.0, max_per_n=None):
    """Export deduped, near-minimum candidates plus a reconstructable manifest.

    Writes `<out_dir>/manifest.json`, `<out_dir>/references/{solute,solvent}.xyz`
    (the relaxed references every exported `E_int` was measured against), and
    `<out_dir>/n<N>/cand<i>.xyz` for the kept structures at each n.

    Raises if the runs found were not all scored against the same references
    -- the same invariant `report.pool_by_n` enforces, for the same reason:
    an `E_int` pooled or exported across two different zeros is not one
    number.
    """
    root = Path(run_or_sweep_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = list(_iter_run_dirs(root))
    if not run_dirs:
        raise FileNotFoundError(
            f"{root}: no scored.json found directly or in a subdirectory")

    summaries = [(d, json.loads((d / "scored.json").read_text()))
                for d in run_dirs]

    references = {(s["e_solute_ref_eV"], s["e_solvent_ref_eV"])
                  for _, s in summaries}
    if len(references) > 1:
        raise ValueError(
            f"{root}: runs were scored against different references "
            f"({sorted(references)}); their interaction energies do not "
            "share a zero and cannot be exported together. Rescore in one "
            "pass, or export each reference's runs separately.")
    e_solute_ref, e_solvent_ref = references.pop()

    ref_solute_src = next(
        (d / "ref_solute.xyz" for d, _ in summaries
         if (d / "ref_solute.xyz").is_file()), None)
    ref_solvent_src = next(
        (d / "ref_solvent.xyz" for d, _ in summaries
         if (d / "ref_solvent.xyz").is_file()), None)
    if ref_solute_src is None or ref_solvent_src is None:
        raise FileNotFoundError(
            f"{root}: no run carries ref_solute.xyz / ref_solvent.xyz -- "
            "rescore with the current ensemble.py or docking.py, which "
            "persist the relaxed reference geometries alongside the energies.")
    (out_dir / "references").mkdir(exist_ok=True)
    (out_dir / "references" / "solute.xyz").write_text(ref_solute_src.read_text())
    (out_dir / "references" / "solvent.xyz").write_text(ref_solvent_src.read_text())

    manifest = {
        "source": str(root),
        "e_solute_ref_eV": e_solute_ref,
        "e_solvent_ref_eV": e_solvent_ref,
        "window_kcal": window_kcal,
        "max_per_n": max_per_n,
        "structures": [],
    }

    by_n = {}
    for run_dir, summary in summaries:
        by_n.setdefault(summary["n_solvent"], []).append((run_dir, summary))

    frame_cache = {}

    def frames_of(run_dir):
        if run_dir not in frame_cache:
            candidates_xyz = run_dir / "scored_candidates.xyz"
            frame_cache[run_dir] = (
                read(str(candidates_xyz), index=":") if candidates_xyz.is_file()
                else [])
        return frame_cache[run_dir]

    for n, group in sorted(by_n.items()):
        tagged = [(run_dir, summary, idx, c)
                  for run_dir, summary in group
                  for idx, c in enumerate(summary["candidates"])]
        if not tagged:
            continue
        keep_idx = dedupe_energies([c["energy_eV"] for _, _, _, c in tagged],
                                   DEDUPE_TOL_EV)
        deduped = [tagged[i] for i in keep_idx]
        weights = boltzmann_weights([c["energy_eV"] for _, _, _, c in deduped],
                                    deduped[0][1]["temperature_K"])

        e_min = min(c["interaction_eV"] for _, _, _, c in deduped)
        window_eV = window_kcal / EV_TO_KCAL
        selected = [(t, w) for t, w in zip(deduped, weights)
                    if t[3]["interaction_eV"] - e_min <= window_eV]
        if max_per_n is not None:
            selected = selected[:max_per_n]

        n_dir = out_dir / f"n{n}"
        n_dir.mkdir(parents=True, exist_ok=True)
        for i, ((run_dir, summary, idx, c), weight) in enumerate(selected):
            frames = frames_of(run_dir)
            if idx >= len(frames):
                raise ValueError(
                    f"{run_dir}: scored_candidates.xyz has {len(frames)} "
                    f"frames but candidate {idx} was requested -- the file "
                    "and scored.json have drifted out of step.")
            out_path = n_dir / f"cand{i:02d}.xyz"
            write(out_path, frames[idx])
            pack_mode = summary["pack_mode"]
            manifest["structures"].append({
                "n": n,
                "path": str(out_path.relative_to(out_dir)),
                "energy_eV": c["energy_eV"],
                "interaction_kcal": c["interaction_eV"] * EV_TO_KCAL,
                "boltzmann_weight": float(weight),
                "source_run": str(run_dir),
                "pack_mode": pack_mode,
                "frame": c.get("frame") if pack_mode == "md" else None,
            })

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_or_sweep_dir",
                        help="a run_sweep or run_docking output directory, "
                             "or a single run directory")
    parser.add_argument("--out", required=True, help="export directory")
    parser.add_argument("--window-kcal", type=float, default=3.0,
                        help="keep candidates within this many kcal/mol of "
                             "each n's minimum (default: %(default)s)")
    parser.add_argument("--max-per-n", type=int, default=None,
                        help="safety cap on structures exported per n, "
                             "applied after the window (default: no cap)")
    args = parser.parse_args(argv)

    out_dir = export_dft(args.run_or_sweep_dir, args.out,
                         window_kcal=args.window_kcal,
                         max_per_n=args.max_per_n)
    print(out_dir / "manifest.json")


if __name__ == "__main__":
    main()
