"""How much explicit solvent is actually needed? Sweep n and watch it converge.

The point of a cluster-continuum model is to add explicit solvent only where
the continuum misses something. That is a measurable claim, not a modelling
preference: compute the interaction energy

    E_int(n) = E(solute + n solvent) - E(solute) - n E(solvent)

with every term relaxed in the same continuum, and look at how it moves with
n. E_int(0) is zero by construction, and a departure from it is exactly "what
the continuum was missing". If it never departs, the continuum was already
sufficient and the explicit shell is not what your model is missing.

Two properties of E_int make this work. It is comparable across n, because
whole solvent molecules are subtracted off. And a solvent molecule that
optimises away into the continuum contributes ~0 to it, so a run where the
shell dissolves lands back on the n = 0 answer rather than on some arbitrary
offset -- "the shell dissolved" and "there was no explicit shell" agree, which
is what makes dissolution a usable null result rather than a failure mode.

**E_int(n) does not plateau**, and this docstring claimed for a long time that
it did. The reference E(solvent) is one solvent molecule relaxed in the same
continuum -- approximately a molecule of bulk liquid -- so the intent was that
moving a molecule from bulk into a bulk-like site in the shell costs nothing,
leaving only the specific sites able to move E_int. That cancellation is not
clean. Two terms survive it, both roughly linear in n, neither of which
switches off once the specific sites are filled:

  - the continuum's per-molecule bias, which is what the gas -> ALPB binding
    table in the docs measures: -0.9 kcal/mol for pyrazine...HCCl3, and +6.2
    for a water in ALPB(water), so not even fixed in sign; and
  - solvent-solvent cohesion from n = 2 on, which E_int scores as solvation
    because what it subtracts is isolated solvent molecules.

Add that E_int is a potential energy, with no entropic penalty for condensing
molecules out of the continuum, and that E_int(min) is a running minimum over
a configuration space that grows with n, and the curve tends to a line of
nonzero slope rather than to a flat.

Read the *increment* instead, dE_int(n) = E_int(n) - E_int(n-1), which
`report.txt` prints as a column of a per-n table: the candidates of every
packing at one n are pooled and deduped across packings, and the reported
E_int is the minimum over that pool rather than a mean over the seeds. Seeds
are independent *searches*, not replicas, so averaging them would penalise
searching more widely -- `report.txt` prints `found by` and `pool` instead, so
the search effort behind a minimum is on the page. Convergence is the
increment settling to a constant -- the specific interaction exhausted, every
further molecule merely being condensed into a bulk-like site -- rather than
to zero. Better, judge "how much is enough"
on a difference at fixed n between two legs (two conformers, bound and free,
one solute in two solvents): both legs carry n molecules in comparable
environments, so the bias and the cohesion largely cancel and what is left
does plateau. `mean_contacts` and `dissolved_fraction` are the other honest
convergence indicators, because the monolayer capacity bounds them and so
they saturate where E_int cannot.

Sampling is gas-phase by default and scoring is in the continuum: see the
module docstring of `ensemble.py` for why those are separated. Sampling is
`Condition.sample_in_continuum`, and there is no second name for it here.

**One sweep is one solute in one solvent.** A comparison across two solutes
and two solvents -- a double difference such as

    dd = (A - B)_solvent1 - (A - B)_solvent2

-- is assembled by hand from four sweeps, because there is no reason for this
module to know that a difference is what you eventually want. What that gives
up is the guarantee that both halves ran under identical settings, so each
sweep writes a **params block** into `sweep.json` recording the whole
`Condition` it ran under plus the package version and git commit -- cheap
insurance, given that `wall_slack` has already changed meaning once (at
c429f8a) without changing its name.

At small n the packing *is* the answer -- a chloroform bound to a pyrazine
nitrogen does not detach, migrate around the ring and rebind at the far one
within 10 ps of Langevin -- and packmol's unconstrained draw is badly biased
toward putting the molecules together (measured: 83% same-face at n = 2 on
pyrazine/chloroform, 4% opposed). So the seeds at each n are not repeats:
`solvate_md.run_job_grid` stratifies them over the degree of clustering, and
`allocate_seeds` below spreads a fixed total of packings over n by monolayer
coverage rather than giving each n the same number. `--seeds` is the
*average* per n. Because the packings differ by design, their scatter is not
an error bar: what `report.txt` reports is how many of them **agree** on the
pooled minimum.

Run one from the command line:

    python n_sweep.py examples/pyrazine.xyz examples/chloroform.xyz \
      --solvent chcl3 --n 0 1 2 3 --out pyrazine_chcl3/ --seeds 3

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
from report import (VERSION, format_sweep_report, git_commit, pool_by_n,
                    timestamp, write_best_geometries)
from shell_capacity import monolayer_capacity
from solvate_md import Condition, run_job_grid

# Every n >= 1 keeps at least this many packings, because a lone packing has
# nothing to agree with it and a row without that agreement reads as
# precision. When the budget cannot pay for it everywhere, the allocation
# falls back to uniform rather than quietly stripping the evidence off some
# rows.
MIN_SEEDS_PER_N = 2


def run_sweep(solute_path, solvent_path, solvent, n_values, out_root,
              n_seeds=3, n_workers=None, scoring=None, label=None,
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

    `n_seeds` is the *average* number of packings per n, not the number at
    each: the total `n_seeds * len(n_values)` is spread by
    `allocate_seeds`, which spends less of it on n = 0 (nothing to arrange)
    and on the n closest to a full monolayer (least room to arrange it
    differently).

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
    # construction. It sets the budget split below and the `cover` column in
    # the report.
    _, _, capacity = monolayer_capacity(read(solute_path), solvent)
    seeds_per_n = allocate_seeds(n_values, n_seeds, capacity)

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
    (out_root / "report.txt").write_text(
        format_sweep_report(params, summaries))
    write_best_geometries(out_root, pool_by_n(summaries))
    return summaries


def allocate_seeds(n_values, n_seeds, capacity):
    """Spread a fixed packing budget over n, unevenly, by how much room to arrange.

    The budget is `n_seeds * len(n_values)` -- the same compute a uniform
    `n_seeds` everywhere would have cost -- and the point is that the seeds
    are not all worth the same. At n = 0 there is no solvent to arrange, so
    every packing is the same packing and one is enough. At n close to a
    monolayer there is little freedom in where the molecules go, so the seeds
    differ mostly in velocities. The room to arrange, and therefore the value
    of another packing, is largest at small nonzero n -- which is exactly
    where the arrangement problem bites: on pyrazine + 2 CHCl3, one
    chloroform on each nitrogen is 1.3% of uniform draws.

    So n = 0 takes one, and the rest are weighted by `1 - n/capacity` and
    handed out by largest remainder over a floor of `MIN_SEEDS_PER_N`. If the
    budget cannot meet that floor -- notably `--seeds 1` -- the split is
    abandoned rather than fudged, and every n gets `n_seeds`, so a
    single-packing sweep still reports no agreement to speak of and says so.

    Returns `{n: packings}`.
    """
    n_values = list(n_values)
    arranged = [n for n in n_values if n >= 1]
    # n = 0 is one packing of a bare solute; the rest of its share is better
    # spent where there is something to arrange.
    budget = n_seeds * len(n_values) - (len(n_values) - len(arranged))
    if not arranged or budget < MIN_SEEDS_PER_N * len(arranged):
        return {n: n_seeds for n in n_values}

    weights = [1.0 - min(n / capacity, 1.0) for n in arranged]
    if sum(weights) <= 0.0:  # every n at or past a monolayer: nothing to prefer
        weights = [1.0] * len(arranged)

    extra = budget - MIN_SEEDS_PER_N * len(arranged)
    shares = [w / sum(weights) * extra for w in weights]
    counts = [MIN_SEEDS_PER_N + int(s) for s in shares]
    # Largest remainder, so the total is exactly the budget rather than
    # whatever rounding leaves.
    order = sorted(range(len(shares)), key=lambda i: int(shares[i]) - shares[i])
    for i in order[:extra - sum(int(s) for s in shares)]:
        counts[i] += 1

    allocation = dict(zip(arranged, counts))
    return {n: allocation.get(n, 1) for n in n_values}


def sweep_params(condition, scoring, n_values, n_seeds, label, capacity,
                 seeds_per_n):
    """Everything needed to reproduce the sweep, recorded with its results.

    Built from `asdict` of the real `Condition` and `Scoring` rather than
    from a hand-written list, so every default and every override is captured
    and no field can be added to either dataclass without appearing here.
    Only the two `Condition` fields that vary within a sweep are dropped.
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
        # Aligned with `n_values`: the budget was spread, so `n_seeds` alone
        # no longer says how many packings any given row had.
        "seeds_per_n": list(seeds_per_n),
        "version": VERSION,
        "git_commit": git_commit(),
        "timestamp": timestamp(),
    })
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
    parser.add_argument("--seeds", type=int, default=3,
                        help="*average* independent packings per n: the total, "
                             "N x len(--n), is spread by monolayer coverage, so "
                             "n = 0 gets one and small nonzero n get more than "
                             "N. Packings are independent searches, and "
                             "how many of them agree on the minimum is the "
                             "only convergence evidence this pipeline "
                             "produces, so one packing reports a number "
                             "nothing corroborates; the report says so when "
                             "it sees one (default: %(default)s)")
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
    parser.add_argument("--stride", type=int,
                        default=dataclass_default(Scoring, "stride"),
                        help="score every Nth dump; redundant with "
                             "--max-frames whenever that cap bites, which at "
                             "the defaults it always does "
                             "(default: %(default)s)")
    parser.add_argument("--max-frames", type=int,
                        default=dataclass_default(Scoring, "max_frames"),
                        help="cap on frames scored per run, spread evenly over "
                             "the whole trajectory. This, not --stride, is "
                             "what sets scoring cost (default: %(default)s)")
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
        scoring=Scoring(stride=args.stride, max_frames=args.max_frames,
                        fmax=args.fmax, opt_steps=args.opt_steps,
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
