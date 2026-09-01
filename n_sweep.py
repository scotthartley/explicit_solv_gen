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

Any script calling `run_sweep` MUST guard the call, because the job grid
underneath uses "spawn" and each worker re-imports the calling module:

    if __name__ == "__main__":
        run_sweep(...)
"""

import json
from dataclasses import dataclass
from pathlib import Path

from ensemble import EV_TO_KCAL, reference_energies, score_run
from solvate_md import Condition, run_job_grid


@dataclass
class Leg:
    """One (solute, solvent) combination of a double difference."""

    conformer: str      # e.g. "AAA" -- names the solute, not the file
    solvent: str        # ALPB name; also the bulk-density key
    solute_path: str
    solvent_path: str

    def label(self, n):
        return f"{self.conformer}_{self.solvent}_n{n}"


def run_sweep(legs, n_values, out_root, n_seeds=1, n_workers=None,
              calculator="gfn2-xtb", sample_in_continuum=False,
              temperature_K=298.0, n_equilibrate_steps=2000, n_steps=6000,
              dump_interval=20, stride=5, max_frames=20, fmax=0.05,
              opt_steps=300, condition_kwargs=None):
    """Generate and score every (leg, n, seed); return the summaries.

    `sample_in_continuum=False` is the default because a gas-phase generator
    is what reliably finds contact geometries. Scoring always happens in the
    leg's own continuum regardless.
    """
    out_root = Path(out_root)
    condition_kwargs = dict(condition_kwargs or {})

    conditions, index = [], []
    for leg in legs:
        for n in n_values:
            conditions.append(Condition(
                solute_path=leg.solute_path,
                solvent_path=leg.solvent_path,
                n_solvent=n,
                solvent=leg.solvent,
                implicit_solvent=sample_in_continuum,
                calculator=calculator,
                temperature_K=temperature_K,
                n_equilibrate_steps=n_equilibrate_steps,
                n_steps=n_steps,
                dump_interval=dump_interval,
                label=leg.label(n),
                **condition_kwargs,
            ))
            index.append((leg, n))

    run_job_grid(conditions, n_seeds=n_seeds, out_root=out_root,
                 n_workers=n_workers)

    # References depend on the leg but not on n, and each is a full geometry
    # optimisation, so cache them across the sweep.
    ref_cache = {}
    summaries = []
    for (leg, n) in index:
        solvation = ("alpb", leg.solvent)
        key = (leg.solute_path, leg.solvent_path, leg.solvent)
        if key not in ref_cache:
            ref_cache[key] = reference_energies(
                leg.solute_path, leg.solvent_path, calculator, solvation,
                fmax=fmax, steps=opt_steps)
        for seed in range(n_seeds):
            summary = score_run(
                out_root / f"{leg.label(n)}_seed{seed}",
                solvation=solvation,
                calculator=calculator,
                stride=stride,
                max_frames=max_frames,
                fmax=fmax,
                steps=opt_steps,
                temperature_K=temperature_K,
                references=ref_cache[key],
            )
            summary["conformer"] = leg.conformer
            summaries.append(summary)

    (out_root / "sweep.json").write_text(json.dumps(summaries, indent=2))
    return summaries


def _pick(summaries, conformer, solvent, n, key):
    vals = [s[key] for s in summaries
            if s["conformer"] == conformer and s["scoring_solvation"][1] == solvent
            and s["n_solvent"] == n]
    if not vals:
        raise KeyError(f"no summary for {conformer}/{solvent}/n={n}")
    return sum(vals) / len(vals)   # average over seeds


def delta_delta(summaries, conformers, solvents, n_values,
                key="ensemble_interaction_kcal"):
    """Double difference per n, in kcal/mol.

    (E_int[c0] - E_int[c1]) in solvent0, minus the same in solvent1. Both
    inner differences are between conformers at fixed n and fixed solvent, so
    the solvent reference terms cancel exactly.
    """
    c0, c1 = conformers
    s0, s1 = solvents
    out = {}
    for n in n_values:
        d0 = _pick(summaries, c0, s0, n, key) - _pick(summaries, c1, s0, n, key)
        d1 = _pick(summaries, c0, s1, n, key) - _pick(summaries, c1, s1, n, key)
        out[n] = d0 - d1
    return out


def format_table(summaries, n_values):
    """Interaction energy vs n, per leg, with the dissolution diagnostics."""
    lines = [f"{'leg':22s} {'n':>3} {'E_int(ens)':>11} {'E_int(min)':>11} "
             f"{'contacts':>9} {'dissolved':>10} {'uniq':>5}",
             "-" * 76]
    for s in sorted(summaries, key=lambda s: (s["conformer"],
                                              s["scoring_solvation"][1],
                                              s["n_solvent"])):
        leg = f"{s['conformer']}/{s['scoring_solvation'][1]}"
        lines.append(
            f"{leg:22s} {s['n_solvent']:>3} "
            f"{s['ensemble_interaction_kcal']:>11.2f} "
            f"{s['min_interaction_kcal']:>11.2f} "
            f"{s['mean_contacts']:>9.2f} "
            f"{100*s['dissolved_fraction']:>9.0f}% "
            f"{s['n_unique']:>5}")
    lines.append("\nE_int in kcal/mol; 'contacts' = solvent molecules touching "
                 "the solute after\noptimisation; 'dissolved' = fraction of "
                 "candidates with no contact at all.")
    return "\n".join(lines)
