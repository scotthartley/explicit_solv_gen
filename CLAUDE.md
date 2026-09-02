# explicit_solv_gen

Explicit-solvent MD for cluster-continuum solvation studies.
Packmol -> ASE -> tblite (GFN2-xTB) or MACE-OFF23, with a confining wall.

## The problem this exists for

A continuum solvation model averages the solvent away. When a solvent effect
comes from a *specific* interaction -- a hydrogen bond to one site, a halogen
bond, an acidic C-H capping a lone pair -- the continuum cannot see it, and a
conformational or binding preference that depends on such a contact comes out
wrong. The cluster-continuum answer is to put a small number of solvent
molecules in explicitly, carry the specific interaction with them, and leave
the bulk dielectric to the continuum.

That leaves one question, which this code turns into a measurement: **how much
explicit solvent is enough?** Sweep the number of explicit molecules `n`, watch
the interaction energy converge, and report the two diagnostics that say
whether the sampling behind it was sufficient.

## Environment

Conda env from `environment.yml` (packmol, xtb, ase, tblite, numpy; MACE via
pip). **packmol is installed into the env's bin and is not on the default
PATH**, so either activate the env or run with it prepended:

    conda env create -f environment.yml
    conda activate solvate_md
    python n_sweep.py examples/pyrazine.xyz examples/chloroform.xyz \
      --solvent chcl3 --n 0 1 2 3 --out pyrazine_chcl3/ --seeds 3

`n_sweep.py --help` lists the rest; every flag maps onto a field of
`Condition` or of `Scoring`, and takes its default from there. That is the
entry point -- there is no driver script, and `solvate_md.py` has no
`__main__` smoke test of its own. A fast end-to-end check is the same command
with `--n 2 --seeds 1 --steps 6000 --equilibrate 2000 --dump-interval 20
--stride 5 --max-frames 20`.

## Layout

| file | role |
| --- | --- |
| `single_thread.py` | pins the numeric stack to one thread per process. Must be imported **before numpy**; see its docstring. |
| `solvate_md.py` | packing + MD. The **generator**. |
| `ensemble.py` | rescoring optimized frames in a continuum. The **scorer**. |
| `n_sweep.py` | `E_int(n)` sweep for **one solute in one solvent**. |
| `report.py` | all text rendering, plus the ASE-free numeric helpers (`EV_TO_KCAL`, `boltzmann_weights`, `ensemble_energy`) that `ensemble` re-exports. No ASE/tblite at module scope. |
| `shell_capacity.py` | monolayer capacity, for choosing `n_solvent`. |

`run_sweep` deliberately does *not* assemble a difference of sweeps. One sweep
is one solute in one solvent; a double difference such as
`(A - B)_solvent1 - (A - B)_solvent2` is four sweeps subtracted by hand. The
`Leg` / `delta_delta` machinery that used to do it had no callers and no driver
script, and imposed a shape real sweeps did not have.

What that gives up is the guarantee that both halves of a difference ran under
identical settings, so every sweep now records a **params block**: the whole
`Condition` (via `asdict`, so new fields are picked up automatically) plus the
package version (`report.VERSION`), the git commit, and a timestamp. Given
`wall_slack` silently changed meaning at `c429f8a`, this is cheap insurance --
check it before subtracting two sweeps.

## Artefacts

Per run directory: `packed.xyz`, `opt.log`, `traj.xyz`, `energies.json`,
`metadata.json`, `best.xyz`, `scored.json` -- and now

| file | written by | contents |
| --- | --- | --- |
| `run.log` | `solvate_md.run_one_job` | header (system, packing, Hamiltonian, pre-MD relaxation, MD settings), a streaming per-dump table, a footer |
| `scored.log` | `ensemble.assemble` | provenance, references, per-candidate table (including BFGS `steps`), result block. Named after `out_name`, so a second continuum gives `scored_acetone.json` / `.log` |
| `scored_candidates.xyz` | `ensemble.assemble` | every deduped candidate, not just the best, as a multi-frame xyz in the same order as `scored.json`'s `candidates` list -- frame *i* is `candidates[i]`. Named after `out_name` like `scored.log` |
| `report.txt` | `n_sweep.run_sweep` | params block, the `E_int(n)` table, and the sampling diagnostics below |

`run.log` is flushed on every line, so a long run can be followed with
`tail -f` instead of going silent until it exits. **The footer surfaces the
wall diagnostic** -- wall-active fraction and max wall energy -- and warns
above 20%, rather than leaving it buried in a 74 KB `energies.json` array.
The same diagnostic now travels with the numbers it qualifies: a `wall` column
and a **Wall diagnostic** section in `report.txt`, and in `scored.log` a
provenance line plus a per-candidate `E_wall/eV` -- the wall energy of the
sampling frame each candidate came from. `scored.json` carries it as
`sampling_wall` / `n_scored_wall_active`.

Nothing reads a `scored.json` or a run directory older than the current
scorer. The backfill that used to reconstruct a missing wall diagnostic from
`energies.json`, and the defaults every reader used to apply to a missing
field, are gone: `run_one_job` always writes both JSON files and the scorer
always writes every field, so a missing one is a broken run rather than an old
one and now fails loudly.

Regenerate any of these from the JSON already on disk, no MD and no calculator:

    python -m report /path/to/sweep_or_run_dir/

(Anything scored before `n_opt_steps` and the mandatory wall fields no longer
re-renders, since the readers no longer default a missing field. Rescore it
rather than re-reporting it.)

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
nothing and dissociation is downhill. The weaker continua *stabilize* the
complex instead, and the bias differs by only 0.5 kcal/mol between chloroform
and acetone against a 3.1 kcal/mol signal. Do not generalize a toy-system
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
and a solvent molecule that optimizes away into the continuum contributes ~0 --
so a dissolved shell lands back on the n = 0 answer. That is what makes
dissolution a usable *null result* rather than a failure mode: if solvent
drifts away, there was no specific interaction to capture.

Departure from `E_int(0)` is exactly "what the continuum was missing"; the
plateau in n is where enough explicit solvent has been added.

Every term has to be relaxed *to convergence*, not merely to a stationary-ish
geometry. The scoring optimizer therefore runs to `fmax = 0.002` eV/A
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
per run directory. Negligible at this size; revisit on a larger solute, which
is why it is a flag and not a constant.

## How much sampling? The defaults are production defaults now

They were not. `run_sweep` carried its own MD-length literals -- 1 ps of
equilibration, 3 ps of production -- that **shadowed `Condition`'s documented
5 ps and 10 ps for every run started from the CLI**, i.e. for every run. The
params block is built from the `Condition` that actually ran, so the override
did not show up in `report.txt` either. The three length arguments now default
to `None`, meaning "whatever `Condition` says", and the CLI reads its
`--steps` / `--equilibrate` / `--dump-interval` defaults off the dataclass via
`condition_default`, so the two cannot drift apart again.

What sets them, measured rather than assumed:

| quantity | value | why |
| --- | --- | --- |
| Langevin relaxation time | 1.02 ps | `friction = 0.01` ASE^-1 is 0.98 ps^-1 |
| shell decorrelation | **0.55 ps** | solvent-COM autocorrelation (1/e) in the solute frame, pyrazine + 3 CHCl3 |
| equilibration | 5 ps | ~5 relaxation times. The old 1 ps was **one** |
| production | 10 ps | ~18 independent shell configurations. The old 3 ps was ~5 |
| dump interval | 100 steps = 50 fs | 11 dumps per decorrelation time. The old 5 fs recorded each configuration ~100 times |
| seeds | 3 | see below |

Expect all of these to grow with the solute. Re-measure the decorrelation time
on your own system rather than carrying 0.55 ps over -- it is the one number
here that is a property of the system rather than of the integrator.

### Both halves of a sweep are parallel, and scoring is parallel per candidate

`run_job_grid` fans the MD out across cores and `score_run_grid` does the same
for the scoring, because scoring is the expensive half: at the defaults a
4-point sweep spends a few minutes on MD and the better part of an hour on
candidate optimizations. Both go through `solvate_md.pool_map`, which uses
spawn, so the `if __name__ == "__main__":` guard below covers both.

**The granularity is per candidate, not per run directory.** Fanning the
scoring out over run directories gives a default sweep 12 tasks (4 n x 3
seeds) on an 18-core machine, and they are badly unequal -- n = 0 is a rigid
10-atom molecule, n = 3 a floppy 25-atom cluster -- so cores idle from the
start and the tail is one whole directory long. `score_run` therefore splits
into `select_frames` -> `relax` per frame -> `assemble`, and `score_run_grid`
selects every job in the parent, flattens to one `relax` task per candidate,
and assembles afterwards. That is ~600 near-equal tasks with a tail of one
optimization, and the count is set by `max_frames` regardless of system size.
Measured on 6 run directories, 62 optimizations, 18 cores: 69.6 s serial,
34.5 s per directory, **10.2 s per candidate**.

Selecting and assembling stay in the parent because they are milliseconds of
file reading and microseconds of numpy; only the optimizations are worth
shipping to a worker. `relax` hands its geometry back carrying a
`SinglePointCalculator` rather than the live tblite one, which is what lets
the result cross a pickle at all.

The two **references go into the same pool**, at the front of the task list,
computed once per distinct (solute, solvent, calculator, continuum) rather
than once per job -- recomputing per job would be n_jobs times the work and a
way for two rows of one table to end up measured against slightly different
zeros. `run_sweep` no longer computes them itself.

Overlapping the two halves -- starting to score a run as soon as its
trajectory lands -- was considered and rejected: MD is the cheap half, and a
shared pool would let a worker that had run MACE then run tblite and hit the
OpenMP clash the file boundary exists to avoid.

`--stride` and `--max-frames` are not independent: `max_frames` selects by
`linspace` over the whole trajectory, so **whenever it bites, `stride` does
nothing at all**. `max_frames` alone sets scoring cost, and that cost is
independent of run length -- a longer trajectory is scored at wider spacing
for the same price. So `stride` is 1 and `max_frames` is 50.

### The two diagnostics that say whether the sampling was enough

`E_int(min)` is a running minimum over candidates, so it can only fall. That
makes *when it last fell* a convergence test, and `report.txt` now runs it:

- **Sampling convergence** -- per run, how far into the trajectory the best
  candidate was found and what the last 25% took off `E_int(min)`. A minimum
  found at 100% is an upper bound, not a converged value; a minimum found
  early and never beaten is evidence the sampling saturated.
- **Seed spread** -- mean over seeds +/- half the full range, at fixed n.
  Independent seeds differ only in packing and initial velocities, so anything
  they disagree about is sampling error rather than chemistry. **This is the
  only error bar the pipeline produces**, and a single-seed sweep says so in
  place of the table rather than leaving the absence to read as precision.
  A double difference is four of these numbers and inherits the error of all
  four.

Both are computed from the candidate list already in `scored.json`, so
`python -m report <dir>` renders them for anything on disk -- no MD, no
calculator. They separate the two ways an under-sampled sweep fails, which
otherwise look alike in the table: a run still falling on the final frame
(too short) and a run whose minimum came from frame 2 and never moved (stuck
in the basin it was packed into).

## Gotchas

- Scoring settings live on `Scoring` and **only** on `Scoring`, the same way
  sampling settings live on `Condition`. `run_sweep` takes a
  `condition_kwargs` dict and a `Scoring` and names no field of either, and
  the CLI reads every default off whichever dataclass owns it via
  `dataclass_default`. Internal functions -- `pack_solvent`, `relax`,
  `score_run` and the rest -- carry no defaults at all, so a caller has to say
  what it means. `pack_solvent` used to default `wall_slack = 1.0` against
  `Condition`'s 0.25, and `score_run` `stride = 10, max_frames = 40` against
  the CLI's 1 and 50.
- Sampling lengths live on `Condition` and **only** on `Condition`. Do not
  give `run_sweep` or the CLI a literal default for one; that is exactly the
  shadowing that made every run so far 3 ps when the docs said 10.
- **`import single_thread` first, above every other import, in any module that
  imports numpy.** The OpenMP runtime numpy loads reads `OMP_NUM_THREADS`
  once, when it loads; setting it afterwards is silently ignored. `solvate_md`
  used to set it at its own module top, which was too late for every caller
  that reached numpy first -- `ensemble` imports numpy, ASE and `report`
  before it gets to `solvate_md`, and `n_sweep` imports `ensemble`. Cost was
  2.5x on anything scoring in the parent process (65.8 s -> 25.8 s on one
  n = 2 job, with system time going 230 s -> 0.4 s). Spawned workers were
  never affected: they inherit an environment where the variable is already
  set, so it lands before *their* numpy import. That asymmetry is the tell if
  it ever comes back.
- **Killing a sweep's parent process does not stop its workers.** Measured:
  `kill <parent-pid>` on a running sweep exits the parent, and the pool's
  workers are reparented to PID 1 and carry on at 100% CPU each until they
  finish the job they were given. The parent's shell wrapper even reports
  exit code 0 while they run. Kill the process **group** instead -- verified
  to leave nothing behind, no orphans and no busy python:

      kill -TERM -- -$(ps -o pgid= -p <parent-pid> | tr -d ' ')

  `pkill -f n_sweep.py` is the trap: a spawn worker's command line is
  `python -c from multiprocessing.spawn import spawn_main; ...`, so that
  pattern matches only the parent and produces exactly the orphans it was
  meant to prevent. (Ctrl-C on a foreground run should be fine, since SIGINT
  goes to the whole foreground group -- but that was not tested.)
- `run_job_grid` and `score_run_grid` both go through `solvate_md.pool_map`,
  which uses spawn, so **every script calling either needs an
  `if __name__ == "__main__":` guard** or workers re-run the grid on import
  and fork until the machine dies. `n_sweep.main()` is already under one, so
  the command line above is safe; a hand-written driver is not automatically.
  `run_sweep` spawns twice, once per half.
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
  `xtb best.xyz --opt` optimizes it on a different surface and keeps moving it.
  Reproduce it with `xtb best.xyz --gfn 2 --alpb <solvent> --sp`; `scored.log`
  prints the exact command. Two further traps, both checked rather than
  assumed: ASE's `fmax` (largest per-atom force, eV/A) and xtb's gradient norm
  (all components, Eh/a) are **different criteria** -- the old 0.05 eV/A is
  ~2.5x looser than `--opt normal`'s 1e-3 Eh/a -- so `scored.log` now prints
  both per candidate. And tblite and the env's `xtb` 6.4.1 binary were
  verified to agree to <1e-6 Eh on the same geometry, gas and ALPB(chcl3)
  alike, with xtb reading the extended-xyz file momenta columns and all.
  **There is no Hamiltonian or I/O discrepancy to chase.**
- An explicit shell **and** ALPB together is the intended configuration, not a
  double count. The continuum describes the bulk the explicit molecules do not
  represent; advice to the contrary is wrong, and the binding table above is
  the measurement that settles it.

## State

Validated on the shipped example, pyrazine in chloroform and in acetone,
n = 0..3, gas-phase sampling scored in each continuum. `E_int(0)` came out at
0.02 / 0.04 kcal/mol, which is the self-consistency check: it is zero by
construction, so a nonzero value would mean the two references and the n = 0
cluster were not being relaxed to the same place. At n = 1, chcl3 -5.59 vs
acetone -3.55, matching the single-complex numbers in the table above within
sampling error; the best geometry has H...N at 1.98 A and 158 deg, found from
a 2.0 A packing that forbade the contact at t = 0.

Those runs predate the `fmax = 0.002` scoring default and the current
`scored.json` shape, so their energies sit up to ~0.6 kcal/mol above their
true minima and they no longer re-render. They were also under-sampled -- one
seed, 15 frames, 3 ps, all well below the current defaults -- which is what
the two diagnostics were written to catch, and did.

Known limitations:

- `E_int` **conflates solute-solvent with solvent-solvent** at n >= 2: two
  chloroforms binding each other counts as solvation. It largely cancels in a
  difference taken at the same solvent and the same n, but it does contaminate
  reading convergence in n directly.
- `n_solvent` is a real choice, not a default to accept. Run
  `shell_capacity.py` first: a count well under a monolayer is targeted
  microsolvation, and that is the regime where *every* explicit molecule sits
  at the continuum boundary.
- MACE-OFF23 as a generator is **untested** here. It needs a model download,
  and it cannot share a process with tblite (see the file boundary above).
- GFN2 likely **over-binds** the C-H...N contact -- 1.91 A is short against a
  literature 2.2-2.5 A, so absolute magnitudes may be 1.5-2x too strong. Signs
  and orderings should be robust; ratios may not be.
