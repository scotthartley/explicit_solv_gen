"""How much explicit solvent is actually needed? Sweep n and watch it converge.

The point of a cluster-continuum model is to add explicit solvent only where
the continuum misses something. That is a measurable claim, not a modelling
preference: compute the interaction energy

    E_int(n) = E(solute + n solvent) - E(solute) - n E(solvent)

with every term relaxed in the same continuum, and look at how it moves with
n. E_int(0) is zero by construction. If explicit solvent is capturing
something ALPB cannot, E_int departs from zero and then plateaus; the plateau
is the answer and the departure is exactly "what the continuum was missing".
If it never departs, the continuum was already sufficient and the explicit
shell is not what your model is missing.

Two properties of E_int make this work. It is comparable across n, because
whole solvent molecules are subtracted off. And a solvent molecule that
optimises away into the continuum contributes ~0 to it, so a run where the
shell dissolves lands back on the n = 0 answer rather than on some arbitrary
offset -- "the shell dissolved" and "there was no explicit shell" agree, which
is what makes dissolution a usable null result rather than a failure mode.

Sampling is gas-phase by default and scoring is in the continuum: see the
module docstring of `ensemble.py` for why those are separated. Sampling is
`Condition.sample_in_continuum`, and there is no second name for it here.

**One sweep is one solute in one solvent.** A double difference such as

    dd = (AAA - BBB)_chcl3 - (AAA - BBB)_acetone

is assembled by hand from four sweeps, because there is no reason for this
module to know that a difference is what you eventually want. What that gives
up is the guarantee that both halves ran under identical settings, so each
sweep writes a **params block** into `sweep.json` recording the whole
`Condition` it ran under plus the git commit -- cheap insurance, given that
`wall_slack` has already changed meaning once (at c429f8a) without changing
its name.

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

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ensemble import reference_energies, score_run_grid
from report import format_sweep_report, git_commit, timestamp
from report import format_table as format_table  # re-export; rendering lives in report
from solvate_md import Condition, run_job_grid


def run_sweep(solute_path, solvent_path, solvent, n_values, out_root,
              n_seeds=3, n_workers=None, calculator="gfn2-xtb",
              temperature_K=298.0, n_equilibrate_steps=None, n_steps=None,
              dump_interval=None, stride=1, max_frames=50, fmax=0.002,
              opt_steps=1000, label=None, condition_kwargs=None):
    """Generate and score one solute in one solvent at every n and seed.

    Sampling takes `Condition`'s default -- gas phase, because that is what
    reliably finds contact geometries. Override it with
    `condition_kwargs={"sample_in_continuum": True}`; scoring happens in
    ALPB(`solvent`) either way.

    The three MD-length arguments default to **None, meaning "whatever
    `Condition` says"**, rather than to numbers of their own. They used to
    carry literals -- 1 ps of equilibration and 3 ps of production -- which
    silently shadowed `Condition`'s documented 5 ps and 10 ps for every run
    started from here, i.e. for every run. Since the params block is built
    from the `Condition` that actually ran, the override was invisible in
    `report.txt` too. One default, in one place, is the fix.

    Writes `<out_root>/sweep.json` -- `{"params": ..., "runs": [...]}` -- and
    a human-readable `<out_root>/report.txt`, and returns the summaries.
    """
    out_root = Path(out_root)
    n_values = list(n_values)
    if not n_values:
        raise ValueError("n_values is empty; nothing to sweep.")
    label = label or Path(solute_path).stem
    condition_kwargs = dict(condition_kwargs or {})
    # Only the lengths actually asked for; the rest fall through to
    # `Condition`. Merged into `condition_kwargs` rather than passed
    # alongside it so that setting one of these both ways is an override
    # rather than a duplicate-keyword TypeError.
    condition_kwargs.update({
        key: value for key, value in (
            ("n_equilibrate_steps", n_equilibrate_steps),
            ("n_steps", n_steps),
            ("dump_interval", dump_interval),
        ) if value is not None
    })

    conditions = [
        Condition(
            solute_path=str(solute_path),
            solvent_path=str(solvent_path),
            n_solvent=n,
            solvent=solvent,
            calculator=calculator,
            temperature_K=temperature_K,
            label=f"{label}_{solvent}_n{n}",
            **condition_kwargs,
        )
        for n in n_values
    ]

    run_job_grid(conditions, n_seeds=n_seeds, out_root=out_root,
                 n_workers=n_workers)

    # The references depend on the solute, the solvent and the continuum, all
    # of which are fixed across the sweep -- and each is a full geometry
    # optimisation, so do them once rather than once per n.
    solvation = ("alpb", solvent)
    references = reference_energies(
        str(solute_path), str(solvent_path), calculator, solvation,
        fmax=fmax, steps=opt_steps)

    # Across all cores, like the MD grid above and for a stronger reason:
    # scoring is now the expensive half of a sweep. At the default settings a
    # 4-point pyrazine/chloroform sweep spends ~7 min in MD spread over every
    # core and ~56 min optimising candidates, so leaving this loop serial made
    # the whole thing single-threaded in practice.
    jobs = [
        {
            "run_dir": out_root / f"{condition.label}_seed{seed}",
            "solvation": solvation,
            "calculator": calculator,
            "stride": stride,
            "max_frames": max_frames,
            "fmax": fmax,
            "steps": opt_steps,
            "temperature_K": temperature_K,
            "references": references,
        }
        for condition in conditions
        for seed in range(n_seeds)
    ]
    summaries = score_run_grid(jobs, n_workers=n_workers)
    for summary in summaries:
        # Names the solute independently of the run label, so a table can
        # be built without re-parsing "<label>_<solvent>_n<n>".
        summary["solute_label"] = label

    params = sweep_params(conditions[0], n_values, n_seeds, stride, max_frames,
                          fmax, opt_steps, label)
    (out_root / "sweep.json").write_text(
        json.dumps({"params": params, "runs": summaries}, indent=2))
    (out_root / "report.txt").write_text(
        format_sweep_report(params, summaries))
    return summaries


def sweep_params(condition, n_values, n_seeds, stride, max_frames, fmax,
                 opt_steps, label):
    """Everything needed to reproduce the sweep, recorded with its results.

    Built from `asdict` of a real `Condition` rather than from a hand-written
    list, so every default and every `condition_kwargs` override is captured
    and no field can be added to `Condition` without appearing here. Only the
    two fields that vary within a sweep are dropped.
    """
    params = {k: v for k, v in asdict(condition).items()
              if k not in ("n_solvent", "label")}
    params.update({
        "solute_label": label,
        "n_values": list(n_values),
        "n_seeds": n_seeds,
        "stride": stride,
        "max_frames": max_frames,
        "fmax": fmax,
        "opt_steps": opt_steps,
        "git_commit": git_commit(),
        "timestamp": timestamp(),
    })
    return params


def condition_default(name):
    """A `Condition` field's default, for the CLI to advertise as its own.

    Read off the dataclass rather than restated, so `--steps` and friends
    cannot drift away from the values `Condition` documents -- which is
    exactly what had happened.
    """
    return Condition.__dataclass_fields__[name].default


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
                        help="independent MD seeds per n. Seed-to-seed scatter "
                             "is the only error bar this pipeline produces, so "
                             "one seed reports a number with no uncertainty "
                             "attached; the report says so when it sees one "
                             "(default: %(default)s)")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel MD workers (default: all cores)")
    parser.add_argument("--calculator", default="gfn2-xtb",
                        help="default: %(default)s")
    parser.add_argument("--steps", type=int,
                        default=condition_default("n_steps"),
                        help="production MD steps. At the default 0.5 fs "
                             "timestep this is 10 ps; the shell's "
                             "configuration decorrelates in ~0.55 ps on "
                             "pyrazine + 3 CHCl3, so it buys ~18 independent "
                             "shell configurations (default: %(default)s)")
    parser.add_argument("--equilibrate", type=int,
                        default=condition_default("n_equilibrate_steps"),
                        help="equilibration steps before recording. The "
                             "default is 5 ps, ~5 Langevin relaxation times "
                             "at the default friction (default: %(default)s)")
    parser.add_argument("--dump-interval", type=int,
                        default=condition_default("dump_interval"),
                        help="MD steps between trajectory frames "
                             "(default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=298.0,
                        help="K, for both MD and the Boltzmann weights "
                             "(default: %(default)s)")
    parser.add_argument("--stride", type=int, default=1,
                        help="score every Nth dump. Redundant with "
                             "--max-frames whenever that cap bites, which at "
                             "the defaults it always does, so it is 1 unless "
                             "you want to score a prefix-thinned trajectory "
                             "specifically (default: %(default)s)")
    parser.add_argument("--max-frames", type=int, default=50,
                        help="cap on frames scored per run, spread evenly over "
                             "the whole trajectory. This, not --stride, is "
                             "what sets scoring cost, and it is independent of "
                             "run length -- a longer trajectory is scored at "
                             "wider spacing for the same price "
                             "(default: %(default)s)")
    parser.add_argument("--fmax", type=float, default=0.002,
                        help="scoring optimiser convergence, eV/A per-atom max "
                             "force. xtb's `--opt normal` stops at a gradient "
                             "norm of 1e-3 Eh/a, which the default here is "
                             "comfortably inside (default: %(default)s)")
    parser.add_argument("--opt-steps", type=int, default=1000,
                        help="max optimiser steps per candidate "
                             "(default: %(default)s)")
    parser.add_argument("--label", default=None,
                        help="solute name in run directories and the table "
                             "(default: the solute filename stem)")
    args = parser.parse_args(argv)

    run_sweep(
        args.solute, args.solvent_geometry, args.solvent, args.n_values,
        args.out,
        n_seeds=args.seeds,
        n_workers=args.workers,
        calculator=args.calculator,
        temperature_K=args.temperature,
        n_equilibrate_steps=args.equilibrate,
        n_steps=args.steps,
        dump_interval=args.dump_interval,
        stride=args.stride,
        max_frames=args.max_frames,
        fmax=args.fmax,
        opt_steps=args.opt_steps,
        label=args.label,
    )
    print(Path(args.out) / "report.txt")


if __name__ == "__main__":
    main()
