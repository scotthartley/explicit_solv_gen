"""How much explicit solvent is actually needed? Sweep n and watch it converge.

The point of a cluster-continuum model is to add explicit solvent only where
the continuum misses something. That is a measurable claim, not a modelling
preference: compute the interaction energy

    E_int(n) = E(solute + n solvent) - E(solute) - n E(solvent)

with every term relaxed in the same continuum, and look at how it moves with
n. E_int(0) is zero by construction, and a departure from it is exactly "what
the continuum was missing". If it never departs, the continuum was already
sufficient and the explicit shell is not what your model is missing.

**E_int(n) does not plateau.** The reference E(solvent) is one solvent
molecule relaxed in the same continuum, so the intent was that moving a
molecule from bulk into a bulk-like site costs nothing. That cancellation is
not clean: the continuum's per-molecule bias and, from n = 2 on,
solvent-solvent cohesion both survive it, both roughly linear in n and
neither switching off once the specific sites are filled. Read the
*increment* instead, dE_int(n) = E_int(n) - E_int(n-1), which `report.txt`
prints as a column over the pooled candidates of every packing at each n.
Convergence is the increment settling to a constant, not to zero. Better,
judge "how much is enough" on a difference at fixed n between two legs (two
conformers, bound and free, one solute in two solvents), where the bias and
the cohesion largely cancel and what is left does plateau.

**One sweep is one solute in one solvent.** A double difference such as
`(A - B)_solvent1 - (A - B)_solvent2` is assembled by hand from four sweeps;
each writes a params block into `sweep.json` so you can check that all four
ran under the same settings before subtracting them.

**This is a basin search whose proposal move happens to be MD, not a thermal
ensemble.** `ensemble.relax` quenches every scored frame before anything
downstream sees it, and `docking.py` finds better minima by construction. So
`E_int(ens)` is not a thermal average, and the sweep is not here to win at
minimum-finding: it is an independently drawn, non-greedy check on what
docking's single lineage might not reach, and the only source of **basin
occupancy** -- how many scored frames quenched into each minimum, which a
placed structure has no analogue of. Occupancy is a diagnostic, quarantined
from every E_int here: the wall volume moves it and leaves E_int(min)
untouched. DESIGN.md carries the measurements behind all of that, and README
carries the caveats that come with reading a report.

Sampling is gas-phase by default and scoring is in the continuum: see the
module docstring of `ensemble.py` for why those are separated.

Run one from the command line:

    python n_sweep.py examples/pyrazine.xyz examples/chloroform.xyz \
      --solvent chcl3 --n 0 1 2 3 --out pyrazine_chcl3/ --seeds 5

A script that calls `run_sweep` itself MUST guard the call, because the job
grid underneath uses "spawn" and each worker re-imports the calling module:

    if __name__ == "__main__":
        run_sweep(...)

The `main()` below is under such a guard already, so the command line above
needs no special handling.
"""

import single_thread  # noqa: F401  -- must precede numpy; see its docstring

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from ase.io import read

from ensemble import Scoring, score_run_grid
from report import (VERSION, format_report, library_versions, pool_by_n,
                    timestamp, write_best_geometries)
from shell_capacity import monolayer_capacity
from solvate_md import Condition, run_job_grid

# Independent packings at each n >= 1, the one sampling default that lives
# here rather than on `Condition`, because it is a property of the sweep and
# not of a run. One owner: `run_sweep` and `--seeds` both read it, the same
# way the MD lengths are read off `Condition` rather than restated.
#
# 5, up from 3, because every job the sweep still does scales with packings
# rather than steps. `found by` out of 3 can only say 1, 2 or 3; the chance
# that every packing at n = 2 on pyrazine/chloroform draws the same-face
# arrangement (83% per draw) is 0.57 at 3 packings and 0.39 at 5; and
# occupancy's `n_seeds_hit` counts packings. The weighted seed budget 0.8.0
# removed used to give the shipped 4-point sweep 1 / 4 / 4 / 3 packings from
# `--seeds 3`; a flat 3 lost one at n = 1 and n = 2 exactly where the
# arrangement lottery bites. Paid for by `Scoring.max_frames` going 50 -> 30:
# 5 x 30 = 3 x 50 relaxations per n.
DEFAULT_SEEDS = 5


def run_sweep(solute_path, solvent_path, solvent, n_values, out_root,
              n_seeds=DEFAULT_SEEDS, n_workers=None, scoring=None, label=None,
              condition_kwargs=None):
    """Generate and score one solute in one solvent at every n and seed.

    Everything that shapes a run lives on one of two dataclasses and is
    passed as one: `condition_kwargs` overrides `Condition` fields (the
    calculator, the temperature, the MD lengths), and `scoring` is a
    `Scoring`. This function names none of those fields itself, which is the
    point -- it used to carry seventeen parameters, three of which shadowed
    `Condition`'s documented MD lengths for every run started from the CLI.
    A parameter it cannot name is a default it cannot silently override.

    Sampling takes `Condition`'s default -- gas phase, because that is what
    reliably finds contact geometries. Override it with
    `condition_kwargs={"sample_in_continuum": True}`; scoring happens in
    ALPB(`solvent`) either way.

    `n_seeds` is the number of independent packings at each n >= 1; n = 0
    gets one, since every packing of a bare solute is the same packing.

    Writes `<out_root>/sweep.json` -- `{"params": ..., "runs": [...]}` -- and
    a human-readable `<out_root>/report.txt`, and returns the summaries.
    """
    out_root = Path(out_root)
    n_values = list(n_values)
    if not n_values:
        raise ValueError("n_values is empty; nothing to sweep.")
    label = label or Path(solute_path).stem
    scoring = scoring or Scoring()

    conditions = [
        Condition(
            solute_path=str(solute_path),
            solvent_path=str(solvent_path),
            n_solvent=n,
            solvent=solvent,
            label=f"{label}_{solvent}_n{n}",
            **(condition_kwargs or {}),
        )
        for n in n_values
    ]

    # Once per sweep, not once per job: the capacity depends only on the
    # solute and the solvent, both of which are fixed for a sweep by
    # construction. It is what the report's `cover` column is a fraction of.
    _, _, capacity = monolayer_capacity(read(solute_path), solvent)
    seeds_per_n = allocate_seeds(n_values, n_seeds)

    start = time.time()
    run_dirs = run_job_grid(conditions,
                            seeds_per_condition=[seeds_per_n[n]
                                                 for n in n_values],
                            out_root=out_root, n_workers=n_workers)
    elapsed_md = time.time() - start

    # Across all cores, like the MD grid above and for a stronger reason:
    # scoring is the expensive half of a sweep. At the default settings a
    # 4-point pyrazine/chloroform sweep spends a few minutes in MD spread over
    # every core and the better part of an hour optimising candidates.
    #
    # The run directories come from `run_job_grid` rather than being rebuilt
    # from `<label>_seed<n>` here, so that name is constructed in exactly one
    # place. The references are not computed here either: they are two more
    # optimisations of the same shape as any candidate, so `score_run_grid`
    # puts them in the same pool.
    solvation = ("alpb", solvent)
    jobs = [{"run_dir": run_dir, "solvation": solvation}
            for run_dir in run_dirs]
    start = time.time()
    summaries = score_run_grid(jobs, scoring, n_workers=n_workers)
    elapsed_scoring = time.time() - start

    # Both halves, every time, so the next time anyone wonders where a sweep's
    # hours went the measurement is already on stdout.
    n_candidates = sum(s["n_frames_scored"] for s in summaries)
    print(f"MD:      {len(run_dirs)} trajectories in {elapsed_md:.1f} s")
    print(f"scoring: {n_candidates} candidates + 2 references in "
          f"{elapsed_scoring:.1f} s")

    params = sweep_params(conditions[0], scoring, n_values, n_seeds, label,
                          capacity, [seeds_per_n[n] for n in n_values])
    (out_root / "sweep.json").write_text(
        json.dumps({"params": params, "runs": summaries}, indent=2))
    (out_root / "report.txt").write_text(format_report(params, summaries))
    write_best_geometries(out_root, pool_by_n(summaries), "sweep")
    return summaries


def allocate_seeds(n_values, n_seeds):
    """`{n: packings}` -- `n_seeds` everywhere, except one packing at n = 0.

    At n = 0 there is no solvent to arrange, so every packing is the same
    packing of a bare solute and the extra ones buy nothing but different
    velocities on a molecule whose minimum is already the reference.

    A fixed total used to be spread unevenly instead, weighted by
    `1 - n/capacity` so that small nonzero n -- where the arrangement lottery
    bites hardest -- got more than its share. That existed to serve the
    stratified packing it was spent on, and both went together: with the
    packings drawn independently again, a packing at one n is worth what a
    packing at any other n is worth, and there is nothing to prefer.
    """
    return {n: (1 if n == 0 else n_seeds) for n in n_values}


def sweep_params(condition, scoring, n_values, n_seeds, label, capacity,
                 seeds_per_n):
    """Everything needed to reproduce the sweep, recorded with its results.

    Built from `asdict` of the real `Condition` and `Scoring` rather than
    from a hand-written list, so every default and every override is captured
    and no field can be added to either dataclass without appearing here.
    Only the two `Condition` fields that vary within a sweep are dropped.

    The dataclasses cover the settings; `library_versions` covers the code
    underneath them, which they cannot.
    """
    params = {k: v for k, v in asdict(condition).items()
              if k not in ("n_solvent", "label")}
    params.update(asdict(scoring))
    params.update({
        "solute_label": label,
        "n_values": list(n_values),
        "n_seeds": n_seeds,
        # Molecules in a complete first shell, from `shell_capacity`. The
        # report turns it into the `cover` column, which is what says whether
        # a row is targeted microsolvation or a real shell.
        "monolayer_capacity": capacity,
        # Aligned with `n_values`, because n = 0 gets one packing rather
        # than `n_seeds` of them.
        "seeds_per_n": list(seeds_per_n),
        "version": VERSION,
        "timestamp": timestamp(),
    })
    # `version` pins this repo, not the Hamiltonian under it: a sweep run
    # against a different tblite is a different measurement, and without this
    # the two params blocks would agree.
    params.update(library_versions(condition.calculator))
    return params


def dataclass_default(cls, name):
    """A dataclass field's default, for the CLI to advertise as its own.

    Read off `Condition` and `Scoring` rather than restated, so `--steps` and
    friends cannot drift away from the values those dataclasses document --
    which is exactly what had happened, twice.
    """
    return cls.__dataclass_fields__[name].default


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sweep n for one solute in one solvent.",
        epilog="A double difference is four of these, subtracted by hand; "
               "check the params blocks match before subtracting.")
    parser.add_argument("solute", help="solute geometry file")
    parser.add_argument("solvent_geometry", help="one solvent molecule")
    parser.add_argument("--solvent", default="chcl3",
                        help="solvent name; the ALPB key and the bulk-density "
                             "key both (default: %(default)s)")
    parser.add_argument("--n", dest="n_values", type=int, nargs="+",
                        required=True, metavar="N",
                        help="explicit solvent counts to sweep, e.g. 0 1 2 3")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="independent packings per n (n = 0 gets one: "
                             "every packing of a bare solute is the same "
                             "packing). Packings are independent searches, "
                             "and how many of them agree on the minimum is "
                             "the only convergence evidence this pipeline "
                             "produces, so one packing reports a number "
                             "nothing corroborates; the report says so when "
                             "it sees one. Scoring cost per n is seeds x "
                             "max-frames, and a packing buys an independent "
                             "draw where a frame buys a correlated one "
                             "(default: %(default)s)")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel workers, for both halves "
                             "(default: all cores)")
    parser.add_argument("--calculator",
                        default=dataclass_default(Condition, "calculator"),
                        help="default: %(default)s")
    parser.add_argument("--steps", type=int,
                        default=dataclass_default(Condition, "n_steps"),
                        help="production MD steps. At the default 0.5 fs "
                             "timestep this is 10 ps; the shell's "
                             "configuration decorrelates in ~0.55 ps on "
                             "pyrazine + 3 CHCl3, so it buys ~18 independent "
                             "shell configurations (default: %(default)s)")
    parser.add_argument("--equilibrate", type=int,
                        default=dataclass_default(Condition,
                                                  "n_equilibrate_steps"),
                        help="equilibration steps before recording. The "
                             "default is 5 ps, ~5 Langevin relaxation times "
                             "at the default friction (default: %(default)s)")
    parser.add_argument("--dump-interval", type=int,
                        default=dataclass_default(Condition, "dump_interval"),
                        help="MD steps between trajectory frames "
                             "(default: %(default)s)")
    parser.add_argument("--temperature", type=float,
                        default=dataclass_default(Condition, "temperature_K"),
                        help="K, for both MD and the Boltzmann weights "
                             "(default: %(default)s)")
    parser.add_argument("--max-frames", type=int,
                        default=dataclass_default(Scoring, "max_frames"),
                        help="cap on frames scored per run, spread evenly over "
                             "the whole trajectory. This is what sets scoring "
                             "cost, and it is independent of run length. The "
                             "default spaces scored frames ~333 fs apart on "
                             "the 10 ps default, under the ~0.55 ps shell "
                             "decorrelation time measured on pyrazine + 3 "
                             "CHCl3; denser than that mostly recounts the "
                             "same configuration, which is why the budget "
                             "goes to --seeds instead (default: %(default)s)")
    parser.add_argument("--fmax", type=float,
                        default=dataclass_default(Scoring, "fmax"),
                        help="scoring optimiser convergence, eV/A per-atom max "
                             "force (default: %(default)s)")
    parser.add_argument("--opt-steps", type=int,
                        default=dataclass_default(Scoring, "opt_steps"),
                        help="max optimiser steps per candidate "
                             "(default: %(default)s)")
    parser.add_argument("--label", default=None,
                        help="solute name in run directories and the table "
                             "(default: the solute filename stem)")
    parser.add_argument("--export-dft", action="store_true",
                        help="also export deduped, near-minimum candidates "
                             "for DFT refinement to <out>/dft_export/")
    args = parser.parse_args(argv)

    run_sweep(
        args.solute, args.solvent_geometry, args.solvent, args.n_values,
        args.out,
        n_seeds=args.seeds,
        n_workers=args.workers,
        scoring=Scoring(max_frames=args.max_frames, fmax=args.fmax,
                        opt_steps=args.opt_steps,
                        temperature_K=args.temperature),
        label=args.label,
        condition_kwargs={
            "calculator": args.calculator,
            "temperature_K": args.temperature,
            "n_equilibrate_steps": args.equilibrate,
            "n_steps": args.steps,
            "dump_interval": args.dump_interval,
        },
    )
    print(Path(args.out) / "report.txt")

    if args.export_dft:
        from dft_export import export_dft
        out_dir = Path(args.out) / "dft_export"
        export_dft(args.out, out_dir)
        print(out_dir / "manifest.json")


if __name__ == "__main__":
    main()
