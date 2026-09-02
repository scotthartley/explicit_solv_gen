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
module docstring of `ensemble.py` for why those are separated.

**One sweep is one solute in one solvent.** A double difference such as

    dd = (AAA - BBB)_chcl3 - (AAA - BBB)_acetone

is assembled by hand from four sweeps, because there is no reason for this
module to know that a difference is what you eventually want. What that gives
up is the guarantee that both halves ran under identical settings, so each
sweep writes a **params block** into `sweep.json` recording the whole
`Condition` it ran under plus the git commit -- cheap insurance, given that
`wall_slack` has already changed meaning once (at c429f8a) without changing
its name.

Any script calling `run_sweep` MUST guard the call, because the job grid
underneath uses "spawn" and each worker re-imports the calling module:

    if __name__ == "__main__":
        run_sweep(...)
"""

import json
from dataclasses import asdict
from pathlib import Path

from ensemble import reference_energies, score_run
from report import format_sweep_report, git_commit, timestamp
from report import format_table as format_table  # re-export; rendering lives in report
from solvate_md import Condition, run_job_grid


def run_sweep(solute_path, solvent_path, solvent, n_values, out_root,
              n_seeds=1, n_workers=None, calculator="gfn2-xtb",
              sample_in_continuum=False, temperature_K=298.0,
              n_equilibrate_steps=2000, n_steps=6000, dump_interval=20,
              stride=5, max_frames=20, fmax=0.05, opt_steps=300,
              label=None, condition_kwargs=None):
    """Generate and score one solute in one solvent at every n and seed.

    `sample_in_continuum=False` is the default because a gas-phase generator
    is what reliably finds contact geometries. Scoring always happens in
    ALPB(`solvent`) regardless.

    Writes `<out_root>/sweep.json` -- `{"params": ..., "runs": [...]}` -- and
    a human-readable `<out_root>/report.txt`, and returns the summaries.
    """
    out_root = Path(out_root)
    n_values = list(n_values)
    if not n_values:
        raise ValueError("n_values is empty; nothing to sweep.")
    label = label or Path(solute_path).stem
    condition_kwargs = dict(condition_kwargs or {})

    conditions = [
        Condition(
            solute_path=str(solute_path),
            solvent_path=str(solvent_path),
            n_solvent=n,
            solvent=solvent,
            implicit_solvent=sample_in_continuum,
            calculator=calculator,
            temperature_K=temperature_K,
            n_equilibrate_steps=n_equilibrate_steps,
            n_steps=n_steps,
            dump_interval=dump_interval,
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

    summaries = []
    for condition in conditions:
        for seed in range(n_seeds):
            summary = score_run(
                out_root / f"{condition.label}_seed{seed}",
                solvation=solvation,
                calculator=calculator,
                stride=stride,
                max_frames=max_frames,
                fmax=fmax,
                steps=opt_steps,
                temperature_K=temperature_K,
                references=references,
            )
            # Names the solute independently of the run label, so a table can
            # be built without re-parsing "<label>_<solvent>_n<n>".
            summary["solute_label"] = label
            summaries.append(summary)

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
