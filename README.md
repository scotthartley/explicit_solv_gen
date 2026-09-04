# explicit_solv_gen

Explicit-solvent MD for cluster–continuum solvation studies, and a way to
answer the question that method always raises: **how many explicit solvent
molecules do you actually need?**

Packmol packs a small solvent shell around a solute, ASE runs Langevin MD
under a confining wall with tblite (GFN2-xTB) or MACE-OFF23, and the resulting
frames are re-optimized in an ALPB continuum. The output is an interaction
energy as a function of shell size, `E_int(n)`, plus the diagnostics that say
whether the sampling behind it was good enough to believe.

## Why

A continuum solvation model averages the solvent away. When a solvent effect
comes from a *specific* interaction — a hydrogen bond to one site, a halogen
bond, an acidic C–H capping a lone pair — the continuum cannot see it, and a
conformational or binding preference that depends on that contact comes out
wrong. Cluster–continuum fixes this by putting a few solvent molecules in
explicitly and leaving the bulk dielectric to the continuum.

The usual objection is that `n` is arbitrary. It does not have to be:

    E_int(n) = E(solute + n solvent) − E(solute) − n·E(solvent)

with every term relaxed in the same continuum. `E_int(0)` is zero by
construction, and a departure from it is exactly *what the continuum was
missing*. If it never departs, the continuum was already sufficient.

Two properties make this work. `E_int` is comparable across `n`, because whole
solvent molecules are subtracted off. And a solvent molecule that optimizes
away into the continuum contributes ≈ 0, so a run whose shell dissolves lands
back on the `n = 0` answer rather than on an arbitrary offset. "The shell
dissolved" and "there was no explicit shell" agree, which makes dissolution a
usable null result instead of a failure mode.

### `E_int(n)` does not plateau — read the increment

The obvious objection to the paragraph above is that `E_int` should just keep
falling as you add molecules, and it does. The reference `E(solvent)` is one
solvent molecule relaxed in the same continuum — approximately a molecule of
bulk liquid — so the intent was that moving a molecule from bulk into a
bulk-like site in the shell costs nothing, leaving only the specific sites
able to move `E_int`. That cancellation is not clean. Two terms survive it,
both roughly linear in `n`, and neither switches off once the specific sites
are filled:

- **the continuum's per-molecule bias** — the difference between binding one
  solvent molecule in the gas phase and binding it in the continuum. Measured
  on this repo's examples at GFN2: −0.9 kcal/mol for pyrazine···HCCl₃ in
  ALPB(chcl3), and +6.2 for a water in ALPB(water). Not even fixed in sign,
  and worth measuring on your own system;
- **solvent–solvent cohesion** from `n = 2` on, which `E_int` scores as
  solvation, because what it subtracts is *isolated* solvent molecules.

Add that `E_int` is a potential energy — no entropic penalty for condensing
molecules out of the continuum — and that `E_int(min)` is a running minimum
over a configuration space that grows with `n`, and the curve tends to a line
of nonzero slope rather than to a flat.

So read the **increment**, `dE_int(n) = E_int(n) − E_int(n−1)`, which
`report.txt` prints as a column. Convergence is the increment settling to a
*constant* — the specific interaction exhausted, every further molecule merely
condensed into a bulk-like site — not to zero.

Both are taken over the **pooled** candidates of every packing at that `n`,
because the packings are independent *searches* and not replicas: averaging
them penalizes searching more widely, and it dilutes the one packing that
found the geometry the stratification below exists to buy. So the reported
number is the pooled minimum, and `found by` says how many packings reached
it. What that gives up is that a minimum is a running minimum over *search
effort*, which is not constant across `n` — the `pool` column says how far the
search went at each, so some of `dE_int`'s slope is search depth rather than
chemistry. The answer is to show the effort, not to average it away.

Better still, judge "how much is enough" on a **difference at fixed `n`**
between two legs: two conformers, two tautomers, bound and free, one solute in
two solvents. Both legs carry `n` molecules in comparable environments, so the
bias and the cohesion largely cancel and what is left does plateau. That is
also the quantity a cluster–continuum study reports in the first place, which
is why one sweep is one leg. `contacts` and `dissolved` are the other honest
convergence indicators — the monolayer capacity bounds them, so they saturate
where `E_int` cannot.

## Install

```bash
conda env create -f environment.yml
conda activate solvate_md
```

Packmol is installed into the environment's `bin` and is invoked as a
subprocess, so the environment must be *active* (or its `bin` on `PATH`) —
importing the Python packages is not enough.

## Quick start

```bash
python n_sweep.py examples/pyrazine.xyz examples/chloroform.xyz \
  --solvent chcl3 --n 0 1 2 3 --out pyrazine_chcl3/ --seeds 3
```

One sweep is **one solute in one solvent**. Comparisons — the same solute in
two solvents, two conformers in one solvent, a double difference
`(A − B)_solvent1 − (A − B)_solvent2` — are assembled by hand from several
sweeps. Every sweep records a params block (the whole `Condition`, the git
commit, a timestamp) so you can check two sweeps ran under the same settings
before subtracting them.

A fast end-to-end smoke test, a few minutes rather than an hour:

```bash
python n_sweep.py examples/solute_toy.xyz examples/water.xyz \
  --solvent water --n 2 --seeds 1 --out smoke/ \
  --steps 6000 --equilibrate 2000 --dump-interval 20 --stride 5 --max-frames 20
```

`n_sweep.py --help` lists everything. Each flag maps onto a field of
`Condition` (sampling) or `Scoring` (rescoring) and reads its default from
that dataclass, so the CLI and the library cannot drift apart.

## The design decision: sampling and scoring use different Hamiltonians

**Generate geometries without the continuum; score them with it.** The two
jobs want different things. Sampling wants whatever reliably finds the contact
geometries — gas-phase MD does this well, and a continuum can make a small
shell dissociate outright during dynamics. Scoring wants the continuum,
because the bulk dielectric is a large, strongly solvent-dependent term that a
finite shell cannot supply.

The seam is a **file boundary**, not a function call. Torch (MACE) and tblite
cannot safely share a process, so a MACE generator and a GFN2 scorer have to
be separate processes anyway — and a trajectory on disk can be rescored in a
different continuum without regenerating it.

The scorer deliberately applies **no wall**. During dynamics the wall stops a
finite cluster in vacuum from drifting apart; during scoring, dissolution is
the signal.

## Output

Per run directory (`<label>_<solvent>_n<N>_seed<S>/`):

| file | contents |
| --- | --- |
| `packed.xyz`, `traj.xyz` | the packed shell and the MD trajectory |
| `energies.json`, `metadata.json` | per-dump energies (including the wall term) and the full run configuration |
| `run.log` | header, streaming per-dump table, footer — flushed per line, so `tail -f` works |
| `scored.json` / `scored.log` | every deduped candidate with its energy, `E_int`, Boltzmann weight, contacts and convergence |
| `scored_candidates.xyz` | all candidates as a multi-frame xyz, in the same order as `scored.json` |
| `best.xyz` | the lowest candidate |

Per sweep: `sweep.json` and `report.txt` — the params block, the `E_int(n)`
table, and the diagnostics. Rescoring in a second continuum writes
`scored_<name>.json` / `.log` / `_candidates.xyz` rather than overwriting.

Everything human-readable regenerates from the JSON already on disk, with no
MD and no calculator:

```bash
python -m report path/to/sweep_or_run_dir/
```

```
E_int(n) = E(solute + n solvent) - E(solute) - n E(solvent)
----------------------------------------------------------
leg: pyrazine/chcl3
full first shell: ~13 solvent molecules

  n  cover   E_int(min)   E_int(ens)     dE_int  found by   pool   E(cluster)/eV  contacts  dissolved   wall
------------------------------------------------------------------------------------------------------------
  0     0%        -0.03        -0.01          -       1/1      2     -446.915704      0.00       100%     0%
  1     8%        -6.66        -6.54      -6.63       2/3     18     -890.445167      0.33        67%     4%
  2    15%       -13.02       -12.63      -6.36       1/2     22    -1333.955483      1.23        14%     0%
```

One row per `n`, over every packing's candidates pooled together and deduped
by energy. `cover` is `n` as a percentage of a full first-shell monolayer —
well under 100% is targeted microsolvation, the regime where every explicit
molecule sits at the continuum boundary. `E_int` is in kcal/mol and absolute
energies in eV (ASE's native unit); the mix is deliberate.
`E_int(min)` is the pooled minimum — the reported number — `E_int(ens)` is
Boltzmann-averaged over the same pool, and `dE_int` is the per-molecule
increment in the minimum against `n − 1`, the column to read for convergence
per the section above. `found by` is how many of that `n`'s packings reached
the pooled minimum and `pool` how many distinct minima the pooled search
found: a `1/2` says the number rests on a single draw of the arrangement
lottery. (This particular sweep is the fast smoke test — 3 ps, 20 frames, two
packings per `n` on average, all well under the current defaults — so its
increments are dominated by sampling noise: they do not settle because the
sampling never did.) `E(cluster)` is shown beside them to give the large
numbers the small differences came from — it is *not* comparable across rows
of different `n`, which is precisely what `E_int` is for. A **per-packing
detail** table under it carries what each search found on its own.

## Packing: the seeds are stratified, not repeated

At small `n` the packing *is* the answer. A chloroform hydrogen-bonded to a
pyrazine nitrogen at ~5.7 kcal/mol will not detach, migrate around the ring
and rebind at the far nitrogen within 10 ps of gas-phase Langevin — so if the
packing did not put one molecule at each nitrogen, the trajectory will not
find that geometry. And packmol's unconstrained draw is badly biased toward
putting them together: measured on pyrazine + 2 CHCl3 over 24 packings, 83%
came out same-face and only 4% opposed.

So the seeds are not repeats. Each one constrains every solvent molecule to
its own hemisphere of the shell, and the hemispheres are spread across the
seeds — maximally separated for the first, coincident for the last, rotated
by a random rotation so no orientation relative to the solute is assumed. A
fixed budget of packings then covers opposite-faces / perpendicular /
same-face **by design** rather than by chance: opposed arrangements go from 4%
to 29%, which over the four packings `n = 2` gets is a ~70% chance of at least
one, against ~15% before.

This is *not* an assumption that the solvent spreads out. The clustered end of
the range is exactly how "both molecules on one face, sharing a Cl···Cl
contact" gets sampled at all — a real arrangement, and the same solvent–solvent
cohesion that gives `E_int(n)` its nonzero slope. The mechanism also fades on
its own as `n` grows: two hemispheres are disjoint, four tetrahedral ones
overlap so heavily a molecule is barely confined, and by `n ≈ 8` the
constraint says nothing — which is the range over which random packing stops
being the bottleneck anyway.

`--seeds` is therefore the **average** number of packings per `n`, not the
count at each. The total, `--seeds × len(--n)`, is spread by monolayer
coverage: `n = 0` has no solvent to arrange and takes one, and the rest are
weighted toward small nonzero `n` where the room to arrange is largest, with a
floor of two so every row keeps something to agree with it. With
`--seeds 3 --n 0 1 2 3` and a 13-molecule shell that is 1 / 4 / 4 / 3 rather
than 3 / 3 / 3 / 3 — the same
compute, better spent. `--seeds 1` falls back to one everywhere, so a
single-packing sweep still reports no agreement and says so.

## The three diagnostics

Numbers from this kind of workflow are easy to over-read. Three things are
reported alongside them, and all three can turn a plausible-looking table into
an obviously unfinished one.

**Wall activity.** The confining wall is meant to be a safety net. If it is
rarely active, the shell is self-bound and the energies are clean; if it is
persistently active, the wall is holding together something the Hamiltonian
wants to disperse and the energies are contaminated. Measured regimes: 4–5% of
frames wall-active for gas-phase sampling, 50–60% for a case where the
continuum was making the shell dissociate. `run.log`'s footer warns above 20%,
and the fraction travels into `report.txt` and `scored.log` beside the numbers
it qualifies.

**Sampling convergence.** `E_int(min)` is a running minimum over candidates,
so it can only fall — which makes *when it last fell* a convergence test. The
report gives, per run, how far into the trajectory the best candidate was
found and what the last 25% took off the minimum. A minimum found on the final
frame is an upper bound, not a converged value.

**Search convergence.** Independent packings are independent *searches*, so
the useful question about a set of them is not how far apart their answers
scattered but how many of them arrived at the same minimum. The report gives,
per `n`, the pooled minimum, `found by`, the spread of the per-packing minima
and the pool size. A minimum several packings reached is one the search finds
reliably; a `1/7` rests entirely on one draw of the arrangement lottery, and
the report warns about it — the fix there is more packings, not more steps.
The spread is printed to be seen rather than propagated: since the packings
are stratified, they are *designed* to differ, so their scatter measures the
lottery as much as the sampling noise. A single-packing sweep has none of this
evidence and says so in place of the table, rather than leaving the absence to
read as precision.

## Choosing `n`

```bash
python shell_capacity.py solute.xyz --solvent chcl3 --n-solvent 12
```

reports the solvent-accessible surface, how many molecules a complete first
shell would take, and what fraction of a monolayer your `n` represents. Well
under a monolayer is *targeted microsolvation* — the regime where every
explicit molecule sits at the continuum boundary, and where the sweep above is
the only honest way to pick `n`. Comparing two conformers, note that a compact
and an extended geometry have different surface areas, so the same `n` is a
different fraction of a monolayer for each.

## Performance

Both halves run in parallel. MD fans out across cores per job; scoring — the
expensive half — fans out **per candidate optimization** rather than per run
directory, because run directories are badly unequal in cost and would leave
cores idle with a long tail. The two references go into the same pool, at the
front, computed once per distinct (solute, solvent, calculator, continuum) so
that every row of a table is measured against the same zero.

Measured on six run directories — 62 optimizations on 18 cores — as mean wall
time per run directory: 69.6 s serial, 34.5 s parallel over run directories,
**10.2 s** parallel over candidates.

Both grids use `spawn`, so **any script calling `run_sweep`, `run_job_grid` or
`score_run_grid` must guard the call**:

```python
if __name__ == "__main__":
    run_sweep(...)
```

Without the guard, workers re-import the calling module and re-run the grid.
`n_sweep.main()` is already guarded, so the command lines above are safe.

Note also that killing a sweep's parent process does **not** stop its workers —
they are reparented and keep running. Kill the process group:

```bash
kill -TERM -- -$(ps -o pgid= -p <parent-pid> | tr -d ' ')
```

## Layout

| file | role |
| --- | --- |
| `solvate_md.py` | packing + MD — the **generator** |
| `ensemble.py` | re-optimizing and scoring frames in a continuum — the **scorer** |
| `n_sweep.py` | the `E_int(n)` sweep, and the CLI |
| `report.py` | all text rendering, plus the ASE-free numeric helpers; no ASE/tblite at module scope |
| `shell_capacity.py` | monolayer capacity, for choosing `n_solvent` |
| `single_thread.py` | pins the numeric stack to one thread per process — **must be imported before numpy** |

`CLAUDE.md` carries the design rationale in full, including the measurements
behind every default and a list of gotchas worth reading before changing one.

## Gotchas worth knowing up front

- **`import single_thread` first, above every other import**, in any module
  that imports numpy. The OpenMP runtime reads `OMP_NUM_THREADS` once, when it
  loads; setting it afterwards is silently ignored, and getting this wrong
  cost 2.5× on in-process scoring.
- Stratified packing changes the packing for **every `n ≥ 2` run**, so sweeps
  from either side of it are not comparable. The `version` (`0.2.0`) and
  `pack_stratified` fields in the params block are what distinguish them, and
  `metadata.json` records the `pack_clustering` and `pack_directions` each run
  was drawn at.
- Packmol's `tolerance` (default 2.0 Å) forbids hydrogen-bond contacts at
  *t* = 0, since H···O/N sit at 1.8–2.0 Å. Harmless — gas-phase MD forms them
  within a few ps — but a packed structure never starts bonded.
- Sampling is **gas phase by default** (`Condition.sample_in_continuum =
  False`); scoring applies the continuum regardless.
- `best.xyz` is relaxed in the scoring continuum with **no wall**, so plain
  `xtb best.xyz --opt` optimizes it on a different surface. Reproduce it with
  `xtb best.xyz --gfn 2 --alpb <solvent> --sp`; `scored.log` prints the exact
  command.
- ASE's `fmax` (largest per-atom force, eV/Å) and xtb's gradient norm (all
  components, Eh/a₀) are different criteria and are not comparable by eye, so
  both are recorded per candidate.
- Every term of `E_int` must be relaxed *to convergence*, not merely to a
  stationary-ish geometry. The scoring optimizer runs to `fmax = 0.002` eV/Å
  for this reason; at 0.05 a frame is left hanging on whichever soft mode it
  was descending, which on one test case put the reported minimum 0.58 kcal/mol
  above the true one — and that residual does not cancel between solvents or
  between conformers.

## Limitations

- `E_int` **conflates solute–solvent with solvent–solvent** at `n ≥ 2`: two
  solvent molecules binding each other counts as solvation. Together with the
  continuum's per-molecule bias this is why `E_int(n)` has a nonzero
  asymptotic slope and cannot be read for a plateau — see
  [`E_int(n)` does not plateau](#e_intn-does-not-plateau--read-the-increment).
  Both terms largely cancel in a difference taken at the same solvent and the
  same `n`, which is the form to report.
- Semi-empirical Hamiltonians can over-bind close contacts, so absolute
  magnitudes may be too strong even where signs and orderings are robust.
  Check your interaction against a higher level of theory before quoting a
  ratio.
- MACE-OFF23 as a generator is supported but untested here; it needs a model
  download and cannot share a process with tblite.

## Authorship

Written by Scott Hartley in collaboration with [Claude
Code](https://claude.com/claude-code), Anthropic's agentic coding tool. The
chemistry, the design decisions and the validation are the author's; much of
the implementation and documentation was drafted by the model and reviewed,
corrected and measured against real runs before landing. The numbers quoted
throughout — binding energies, decorrelation times, timings, the cost of a
loose convergence threshold — were measured, not estimated.

## License

MIT — see [LICENSE](LICENSE).
