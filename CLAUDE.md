# explicit_solv_gen

Explicit-solvent MD for cluster-continuum solvation studies.
Packmol -> ASE -> tblite (GFN2-xTB) or MACE-OFF23, with a confining wall.

## The problem this exists for

Alternating o-phenylene / 2,3-pyrazinylene foldamers fold into a compact helix
(AAA) in chloroform but an extended one (BBB) in acetone. PCM does not
reproduce this, which points at specific solvation -- chloroform's acidic C-H
capping the pyrazine nitrogens -- rather than at bulk dielectric. The target
is a double difference, `dd = (AAA - BBB)_chcl3 - (AAA - BBB)_acetone`, with a
small explicit shell carrying the specific interaction and ALPB carrying the
bulk.

This replaced a CREST/QCG workflow that was abandoned after its ensemble
finalization stage was found to be broken (see `resources/crest_qcg_notes.md`).

## Environment

Conda env `solvate_md` (`~/opt/miniforge3/envs/solvate_md`). **packmol lives in
that env's bin and is not on the default PATH**, so run with:

    PATH=~/opt/miniforge3/envs/solvate_md/bin:$PATH \
      ~/opt/miniforge3/envs/solvate_md/bin/python n_sweep.py \
        examples/pyrazine.xyz examples/chloroform.xyz \
        --solvent chcl3 --n 0 1 2 3 --out pyrazine_chcl3/ --seeds 3

`n_sweep.py --help` lists the rest; every flag maps 1:1 onto a `run_sweep`
argument. That is the entry point -- there is no driver script.

## Layout

| file | role |
| --- | --- |
| `solvate_md.py` | packing + MD. The **generator**. |
| `ensemble.py` | rescoring optimised frames in a continuum. The **scorer**. |
| `n_sweep.py` | `E_int(n)` sweep for **one solute in one solvent**. |
| `report.py` | all text rendering, plus the ASE-free numeric helpers (`EV_TO_KCAL`, `boltzmann_weights`, `ensemble_energy`) that `ensemble` re-exports. No ASE/tblite at module scope. |
| `shell_capacity.py` | monolayer capacity, for choosing `n_solvent`. |

`run_sweep` deliberately does *not* assemble a double difference. One sweep is
one solute in one solvent; `dd` is four sweeps subtracted by hand. The `Leg` /
`delta_delta` machinery that used to do it had no callers and no driver script,
and the only real sweep on disk (`pyrazine_sweep_output/`) is one solute across
two solvents -- a shape `delta_delta` would have rejected.

What that gives up is the guarantee that both halves of a difference ran under
identical settings, so every sweep now records a **params block**: the whole
`Condition` (via `asdict`, so new fields are picked up automatically) plus the
git commit and a timestamp. Given `wall_slack` silently changed meaning at
`c429f8a`, this is cheap insurance -- check it before subtracting two sweeps.

## Artefacts

Per run directory: `packed.xyz`, `opt.log`, `traj.xyz`, `energies.json`,
`metadata.json`, `best.xyz`, `scored.json` -- and now

| file | written by | contents |
| --- | --- | --- |
| `run.log` | `solvate_md.run_one_job` | header (system, packing, Hamiltonian, pre-MD relaxation, MD settings), a streaming per-dump table, a footer |
| `scored.log` | `ensemble.score_run` | provenance, references, per-candidate table, result block. Named after `out_name`, so a second continuum gives `scored_acetone.json` / `.log` |
| `report.txt` | `n_sweep.run_sweep` | params block + the `E_int(n)` table |

`run.log` is flushed on every line, so a long run can be followed with
`tail -f` instead of going silent until it exits. **The footer surfaces the
wall diagnostic** -- wall-active fraction and max wall energy -- and warns
above 20%, rather than leaving it buried in a 74 KB `energies.json` array.
The same diagnostic now travels with the numbers it qualifies: a `wall` column
and a **Wall diagnostic** section in `report.txt`, and in `scored.log` a
provenance line plus a per-candidate `E_wall/eV` -- the wall energy of the
sampling frame each candidate came from. `scored.json` carries it as
`sampling_wall` / `n_scored_wall_active`, and older ones are backfilled from
`energies.json` when re-rendered.

Regenerate any of these from the JSON already on disk, no MD and no calculator:

    python -m report /path/to/sweep_or_run_dir/

(`pyrazine_sweep_output/` is the exception: its `sweep.json` is the old bare
list, so it no longer re-renders. Its `report.txt` is already on disk, and the
sweep is under-sampled and due to be redone anyway.)

`E_int` stays the reported number, because it alone is comparable across n.
`E(cluster)` is now shown beside it everywhere: within a run the two differ
only by the constant `E(solute) + n E(solvent)`, and `boltzmann_weights`
subtracts the minimum before exponentiating, so the weights -- and therefore
the two ensemble averages -- are consistent by construction. It costs nothing
and gives the large numbers the small differences came from. Absolute energies
are in eV (ASE's native unit), `E_int` in kcal/mol; the mix is deliberate.

## The central design decision: sampling and scoring use different Hamiltonians

Generate geometries **without** the continuum, score them **with** it. The two
jobs want different things: sampling wants whatever actually finds the contact
geometries, scoring wants the continuum because the bulk dielectric is a large
solvent-dependent term a finite shell cannot supply.

The seam is a **file boundary**, not a function call, because torch (MACE) and
tblite cannot share a process -- OpenMP duplicate-runtime error whose only
workaround is documented as unsafe. It also lets a trajectory be rescored in a
different continuum without regenerating it.

The scorer applies **no wall**, deliberately. Dissolution is the signal.

## Measurements to not re-derive

Binding of one solvent molecule, GFN2-xTB, gas -> ALPB (kcal/mol):

| complex | gas | ALPB | bias |
| --- | --- | --- | --- |
| methanol...water in ALPB(water) | -3.6 | **+2.6** | +6.2 |
| pyrazine...HCCl3 in ALPB(chcl3) | -5.7 | **-6.6** | -0.9 |
| pyrazine...acetone in ALPB(acetone) | -2.1 | **-3.5** | -1.3 |

**Water-in-water is pathological; it is not the general case.** ALPB(water)
reproduces a water's full hydration free energy, so a departing water loses
nothing and dissociation is downhill. The weaker continua *stabilise* the
complex instead, and the bias differs by only 0.5 kcal/mol between chloroform
and acetone against a 3.1 kcal/mol signal. Do not generalise a toy-system
failure to the real chemistry without measuring it.

Sampling Hamiltonian, methanol + 4 water, 10 ps, three seeds each -- no
overlap between the populations:

| | H-bonds | wall active |
| --- | --- | --- |
| gas | 3.84 - 4.39 | 4-5% |
| ALPB(water) | 0.00 - 0.30 | 50-60% |

**`wall_energy_eV` is the diagnostic.** Rarely nonzero = shell is self-bound,
wall is a safety net. Persistently nonzero = the wall is holding together
something the Hamiltonian wants to disperse, and the energies are contaminated.

## `E_int(n)` is what makes "how much explicit solvent?" a measurement

    E_int(n) = E(solute + n solvent) - E(solute) - n E(solvent)

every term relaxed in the same continuum. `E_int(0)` is zero by construction.
It is comparable across n because whole solvent molecules are subtracted off,
and a solvent molecule that optimises away into the continuum contributes ~0 --
so a dissolved shell lands back on the n = 0 answer. That is what makes
dissolution a usable *null result* rather than a failure mode: if solvent
drifts away, there was no specific interaction to capture.

Departure from `E_int(0)` is exactly "what the continuum was missing"; the
plateau in n is where enough explicit solvent has been added.

Every term has to be relaxed *to convergence*, not merely to a stationary-ish
geometry. The scoring optimiser therefore runs to `fmax = 0.002` eV/A
(`--fmax`, `--opt-steps 1000`), not the 0.05 it used to. At 0.05 a frame is
left hanging on whichever soft mode it happened to be descending: measured on
one pyrazine + 2 chloroform frame, that put the reported minimum **0.58
kcal/mol above the true one**, and the residual does not cancel between
solvents or between conformers, because it depends on the mode rather than on
the chemistry. It perturbs the Boltzmann weights too, so it contaminates the
ensemble average as well as the minimum. The references matter as much as the
candidates -- `E_int` subtracts `E(solute) + n E(solvent)`, so a loosely
relaxed reference puts a constant offset on every point in the sweep.

Cost on pyrazine + 2 chloroform: 4.5 s vs 1.3 s per candidate, ~90 s vs ~26 s
per run directory. Negligible at this size; revisit when the hexamer arrives,
which is why it is a flag and not a constant.

## Gotchas

- `run_job_grid` uses spawn, so **every script calling it needs an
  `if __name__ == "__main__":` guard** or workers re-run the grid on import
  and fork until the machine dies. `n_sweep.main()` is already under one, so
  the command line above is safe; a hand-written driver is not automatically.
- Sampling is **gas phase by default**: `Condition.sample_in_continuum` is
  `False`, and scoring applies the continuum regardless. It was called
  `implicit_solvent` and defaulted to `True` -- the opposite of the documented
  design -- until this commit, and the `metadata.json` key changed with it, so
  older run directories carry `implicit_solvent` instead.
- Packmol's `tolerance` (default 2.0) forbids hydrogen-bond contacts at t = 0
  (H...O/N sit at 1.8-2.0 A). Harmless here -- gas MD forms them within a few
  ps -- but it means a packed structure never starts bonded.
- `wall_slack` changed meaning at `c429f8a`. It is now measured in the wall's
  own metric and defaults to 0.25. Saved `Condition`s from before that behave
  differently. It is not just a margin: the wall volume sets the translational
  entropy of a dissociated solvent molecule, so a looser wall makes
  dissociation more favourable.
- Don't pipe a long background run through `tail` -- it buffers until the pipe
  closes, so no interim output appears no matter what `flush=True` says. Tail
  the run's own `run.log` instead; it is flushed per line.
- `sweep.json` is `{"params": {...}, "runs": [...]}`, not the bare list it was
  before. Nothing reads the old shape any more.
- `best.xyz` is relaxed **in the scoring continuum, with no wall**, so a plain
  `xtb best.xyz --opt` optimises it on a different surface and keeps moving it.
  Reproduce it with `xtb best.xyz --gfn 2 --alpb <solvent> --sp`; `scored.log`
  prints the exact command. Two further traps, both checked rather than
  assumed: ASE's `fmax` (largest per-atom force, eV/A) and xtb's gradient norm
  (all components, Eh/a) are **different criteria** -- the old 0.05 eV/A is
  ~2.5x looser than `--opt normal`'s 1e-3 Eh/a -- so `scored.log` now prints
  both per candidate. And tblite and the env's `xtb` 6.4.1 binary were
  verified to agree to <1e-6 Eh on the same geometry, gas and ALPB(chcl3)
  alike, with xtb reading the extended-xyz file momenta columns and all.
  **There is no Hamiltonian or I/O discrepancy to chase.**
- `resources/` is gitignored. `resources/handoff_md_pipeline_claude_code.md`
  contains a struck-through paragraph saying not to use ALPB with an explicit
  shell. **That advice is wrong** and was corrected in place on 2026-09-01.

## State

Validated: pyrazine, n = 0..3, gas sampling scored in each continuum.
**Those run directories predate the `fmax = 0.002` scoring default and were
not rescored**, so their energies sit up to ~0.6 kcal/mol above their true
minima; the next production sweep picks the new default up. The numbers below
are from the loose runs.
`E_int(0)` = 0.02 / 0.04 kcal/mol, confirming self-consistency. At n = 1,
chcl3 -5.59 vs acetone -3.55, matching the single-complex numbers within
sampling error; the best geometry has H...N 1.98 A at 158 deg, found from a
2.0 A packing.

Not yet established:

- **No real AAA/BBB structures exist anywhere on this machine.** Every hexamer
  number quoted so far is from a spherocylinder/lattice model. These are the
  missing input for both `shell_capacity.py` and a real sweep.
- The pyrazine sweep is **under-sampled** (one seed, 15 frames, 3 ps). The
  signature is non-monotonic contact counts -- n = 2 chcl3 had *fewer*
  contacts than n = 1. Production runs want 3-5 seeds and longer sampling.
- `E_int` **conflates solute-solvent with solvent-solvent** at n >= 2: two
  chloroforms binding each other counts as solvation. Largely cancels in the
  double difference (same solvent, same n, different conformer) but it does
  contaminate reading convergence in n directly.
- `n_solvent` is still an open choice. 12 is targeted microsolvation, ~40 a
  full first shell on the hexamer. Note the tension: microsolvation is the
  regime where *every* explicit molecule sits at the continuum boundary.
- MACE-OFF23 as a generator is **untested**. No model is cached; it needs a
  download. MPS is available.
- GFN2 likely **over-binds** the C-H...N contact -- 1.91 A is short against a
  literature 2.2-2.5 A, so absolute magnitudes may be 1.5-2x too strong. Sign
  and ordering should be robust; ratios may not be.
