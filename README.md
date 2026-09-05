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
found a geometry nobody else did. So the reported number is the pooled
minimum, and `found by` says how many packings reached it. What that gives up is that a minimum is a running minimum over *search
effort*, which is not constant across `n` — the `pool` column says how far the
search went at each, so some of `dE_int`'s slope is search depth rather than
chemistry. The answer is to show the effort, not to average it away.

Better still, judge "how much is enough" on a **difference at fixed `n`**
between two legs: two conformers, two tautomers, bound and free, one solute in
two solvents. Both legs carry `n` molecules in comparable environments, so the
bias and the cohesion largely cancel and what is left does plateau. That is
also the quantity a cluster–continuum study reports in the first place, which
is why one sweep is one leg. The `contacts` / `dissolved` columns of the
report are the frame-weighted ones — see
[Basin occupancy](#basin-occupancy) — because the distinct-minima versions,
still in `sweep.json`, are convergence indicators for the *search* rather
than the *sampling*: they say how many kinds of basin were found, not how
much of the trajectory sat in one, which `pool` already reports.

### Basin occupancy

`n_sweep.py`'s candidates are quenched local minima, not a thermal ensemble —
`ensemble.relax` optimizes every scored frame to `fmax = 0.002` before
anything downstream sees it, and the cross-seed dedupe above discards
duplicate basins. What survives is structurally the same object `docking.py`
produces, and docking finds better minima by construction (dozens of
independent random poses descending by BFGS reliably outfind one thermal
trajectory: −13.00 vs −11.44 kcal/mol at `n = 2` on pyrazine/chloroform — see
[Docking](#docking-a-constructive-alternative)). So `E_int(ens)` is **not** a
thermal average; it is a Boltzmann-weighted soft minimum over a deduped,
search-effort-dependent set of quenched energies.

What the MD sweep gives that docking structurally cannot is **basin
occupancy** — how many of the scored frames quenched into each minimum,
pooled across packings and reported frame-weighted in `report.txt`'s Basin
occupancy section. At today's defaults (50 fs dumps, `max_frames = 30`
selecting down to ~333 fs of scored-frame spacing against a ~0.55 ps shell
decorrelation time) that count is only ~1.65× oversampled, close enough to
independent to be a usable inherent-structure population estimate — unlike at
the old 5 fs dump interval (~100× oversampled), which is why it used to be
discarded at dedupe. `max_frames` was 50 until `0.9.0`; the frames it lost
were buying extra chances at the lowest quench, which is docking's job, and
the relaxations they cost now go to two more packings per `n` (`--seeds 5`),
each an independent draw where a denser scoring of one trajectory is not.

Occupancy is a diagnostic, quarantined from every `E_int` in this repo: the
wall volume (`wall_slack`) sets the population of a dissociated molecule and
moves an occupancy number, while a dissolved molecule still contributes ~0 to
`E_int` regardless of box size — no thermal-average `E_int` is built from any
of this. Read it with its caveats in mind: sampling is gas-phase (see the
Hamiltonians section below); the 1 meV dedupe merges isoenergetic distinct
minima into one count; frames are correlated, so the reported spacing has to
be compared against a decorrelation time measured on the system in question;
and these are inherent-structure populations with no vibrational entropy or
ZPE. They are listed in full under
[Reading `report.txt`](#reading-reporttxt).
A docked candidate has no sampling frame and so no occupancy — its fields are
`null`, a real absence rather than a population of one.

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
  --solvent chcl3 --n 0 1 2 3 --out pyrazine_chcl3/ --seeds 5
```

One sweep is **one solute in one solvent**. Comparisons — the same solute in
two solvents, two conformers in one solvent, a double difference
`(A − B)_solvent1 − (A − B)_solvent2` — are assembled by hand from several
sweeps. Every sweep records a params block (the whole `Condition`, the
package version, the library versions, a timestamp) so you can check two
sweeps ran under the same settings before subtracting them.

A fast end-to-end smoke test, a few minutes rather than an hour:

```bash
python n_sweep.py examples/solute_toy.xyz examples/water.xyz \
  --solvent water --n 2 --seeds 1 --out smoke/ \
  --steps 6000 --equilibrate 2000 --dump-interval 20 --max-frames 20
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
table, a Best geometry at each n section naming the file behind each row, a
Basin occupancy section (how many scored frames quenched into each basin,
frame-weighted and pooled across packings — see
[Basin occupancy](#basin-occupancy) above), and the diagnostics — plus one
`best_n<N>.xyz` per n: the pooled-minimum
packing's `best.xyz` at that n, copied out with `sweep_n=` /
`sweep_E_int_kcal=` / `sweep_packing=` appended to its comment line. One
file per n, not one multi-frame file, because the atom count changes with
n and a viewer that reads a multi-frame xyz as a trajectory — Avogadro, VMD,
most others — shows only the first frame.
Rescoring in a second continuum writes `scored_<name>.json` / `.log` /
`_candidates.xyz` rather than overwriting.

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

  n  cover   E_int(min)     dE_int  found by   pool  contacts  dissolved   wall
-------------------------------------------------------------------------------
  0     0%        -0.03          -       1/1      2      0.00       100%     0%
  1     8%        -6.65      -6.63       3/3      5      0.90        10%     0%
  2    15%       -13.01      -6.36       1/2     14      0.90        35%    25%
```

(This particular sweep is a fast smoke test — 1 ps, 10 frames, two packings
per `n` on average, all well under the current defaults — so its increments
are dominated by sampling noise: they do not settle because the sampling
never did. The 25% wall row is exactly the contamination the wall diagnostic
exists to catch.)

## Reading `report.txt`

Everything the rendered report used to explain in place lives here instead,
so that a report is a page of numbers rather than a page of numbers wrapped in
150 lines of prose that never changes between runs. `dock_report.txt` has the
same sections, minus the ones a constructed chain has no analogue of.

**The columns.**

| column | what it is |
| --- | --- |
| `n` | explicit solvent molecules |
| `cover` | `n` as a percentage of a full first-shell monolayer. Well under 100% is *targeted microsolvation*, where every explicit molecule sits at the continuum boundary |
| `E_int(min)` | the reported number: the lowest `E_int` over the pooled candidates at this `n`, kcal/mol |
| `dE_int` | `E_int(min)` here minus `E_int(min)` at `n − 1` — the column to read, per [the section above](#e_intn-does-not-plateau--read-the-increment) |
| `found by` | how many independent searches reached that minimum, of how many there were. For a sweep those are packings; for a docking chain, refined placements |
| `pool` | distinct minima the pooled search turned up — the *search effort* behind the running minimum above it |
| `contacts`, `dissolved` | solvent molecules in contact with the solute, and the fraction of frames with none, **frame-weighted** over the scored frames — see [Basin occupancy](#basin-occupancy) |
| `wall` | the worst packing's fraction of sampling frames with a nonzero wall energy |

`E_int` is in kcal/mol and absolute energies in eV (ASE's native unit); the
mix is deliberate. A `1/2` in `found by` says the number rests on a single
draw of the arrangement lottery, and the report warns about it.

**What is in the JSON but not in the table.** `E_int(ens)` — the Boltzmann
average over the pool — and `E(cluster)`, its absolute counterpart, are in
`sweep.json` / `dock.json` and in every `scored.json`, and are deliberately
not printed. Showing two averages side by side invites reporting whichever
looks better, which is the same objection this repo already raises against
giving the ensemble average its own `dE_int`; and `E(cluster)` is not
comparable across rows of different `n` — successive rows differ by a whole
solvent molecule — which is precisely what `E_int` exists to fix. The
distinct-minima `mean_contacts` / `dissolved_fraction` are there too; they
describe the search, which `pool` already reports, and the table shows the
frame-weighted pair instead.

**Best geometry at each n** names the file behind each row: `<run>/best.xyz`,
which is also frame 0 of that run's `scored_candidates.xyz`, since the pooled
minimum at an `n` is necessarily its own run's minimum. `weight` is the
candidate's Boltzmann weight within the pooled set — near 1 means the minimum
is effectively the whole ensemble, small means it is one of several
comparable minima. `E_wall/eV` is that sampling frame's wall energy: the one
place the wall diagnostic bites on the reported number itself rather than on
an average. Note `best.xyz` is not named after the scoring `out_name`, so
rescoring the same run directories in a second continuum overwrites it — it
always reflects the most recent scoring pass.

Each is also copied out as `best_n<N>.xyz` beside the report, with
`sweep_n=` / `sweep_E_int_kcal=` / `sweep_packing=` (or the `dock_` forms)
appended to its comment line. One file per `n` rather than one multi-frame
xyz, because the atom count changes with `n` and a viewer that reads a
multi-frame xyz as a trajectory — Avogadro, VMD, most others — takes the
first frame's atom count and applies it to the rest, silently dropping every
`n` after the first. A set of chemically distinct clusters is not a
trajectory, and no single-file XYZ shape expresses that it isn't one.

**Per-packing detail** is one row per packing, its own search alone. These do
not average to the table above and are not meant to: the pooled minimum is
the *lowest* of that column, not its mean. `best` marks a packing within
1 meV of the pooled minimum at its `n` — the same test `found by` counts, so
the number of `*` at an `n` equals its `found by` numerator.

**Basin occupancy** is how many scored frames quenched into each basin,
summed across the packings at each `n`; `seeds` is how many of them visited
it. Five caveats come with reading it, and none is decorative:

1. **Sampling is gas-phase** (`Condition.sample_in_continuum = False`), so
   occupancy describes the gas-phase search, not solvation in the scoring
   continuum. Measured on methanol + 4 water: 3.84–4.39 H-bonds in gas
   against 0.00–0.30 in ALPB(water), no overlap between the two populations.
2. **The 1 meV energy dedupe merges isoenergetic distinct minima**, inflating
   one count. Tolerable for ranking, sharper for counting.
3. **Frames are correlated.** Compare the reported scored-frame spacing
   against a decorrelation time measured on *your* system — 0.55 ps for
   pyrazine + 3 CHCl₃, which is a property of the system rather than of the
   integrator and is not fabricated for you.
4. **These are inherent-structure populations, not basin free energies:** no
   vibrational entropy, no ZPE.
5. **The wall volume (`wall_slack`) sets any occupancy number** and leaves
   `E_int(min)` untouched, since a dissolved molecule contributes ≈ 0 to
   `E_int` regardless of box size. That is why occupancy is a diagnostic here
   and never an energy.

The packings pooled here are independent draws, which is what makes a pooled
frame count a sample rather than a mixture with hand-picked weights. It was
the latter while the packings were stratified over a clustering parameter
chosen by design — one of the reasons that stratification is gone.

**Search convergence** prints the per-packing spread beside `found by`. It is
printed to be seen, not used as an error bar: independent packings are
independent searches rather than repeat measurements of one system, so their
scatter measures the arrangement lottery as much as the sampling noise. A
difference you go on to take between
two sweeps has to be credible against the widest spread in either, and a
double difference accumulates four of them.

**Per-parent detail** (docking only) is one row per parent used to grow to
that `n`. Unlike the sweep's `found by`, several parents landing near one
minimum is *not* independent corroboration: every parent explores the same
shell region with independent random poses, not a differently-arranged
packing.

## Docking: a constructive alternative

`n_sweep.py` explores by thermal sampling and quenches whatever basin the
trajectory happens to visit — which means it can miss one entirely.
`docking.py` finds basins by constructing instead: place a solvent molecule at
a random position and orientation around the relaxed parent cluster and run
BFGS, chained upward (`n = 1`'s survivors become `n = 2`'s starting points).
Measured on pyrazine + 2 chloroform: the MD sweep never finds the
both-nitrogens arrangement (one chloroform H-bonded to each ring nitrogen) in
14 candidates over three packings, even though it is 1.56 kcal/mol *lower*;
docking finds it as its overall minimum, `E_int(2) = −13.00` kcal/mol against
the sweep's pooled `−11.44`.

```bash
python docking.py examples/pyrazine.xyz examples/chloroform.xyz \
  --solvent chcl3 --n 1 2 3 --out pyrazine_dock/ --placements 64
```

Same shape as `n_sweep.py`: writes `dock.json` (matching `sweep.json`'s
`{"params": ..., "runs": [...]}` shape) and `dock_report.txt`, and
`python -m dft_export` and `python -m report` both read a docking output
directory exactly as they read a sweep's. Docking **owns minimum-finding** —
BFGS descending from dozens of independent random poses reliably outfinds one
thermal trajectory at the same job, by construction — which is why the two
generators' numbers are never pooled together: docking would win at every `n`
trivially and erase the informative comparison. The MD sweep keeps two jobs
docking cannot do: it is a non-greedy, independently packed check at each `n`
(docking's chain still grows from one lineage, so a basin no pose in that
lineage lands in is invisible to it), and it is the only source of
[basin occupancy](#basin-occupancy), since a docked minimum was placed, not
visited, and so has no sense of "time spent" in it.

Applicability is limited by the same ~N^2.5 GFN2-xTB gradient cost that limits
the MD sweep, compounded by BFGS step count — comfortable to ~90 atoms,
painful at 180 — and, more fundamentally, docking targets *targeted
microsolvation*: past roughly a third of a monolayer no single minimum
dominates and solvent–solvent cohesion takes over, which is the MD sweep's
regime instead. `run_docking` warns past that fraction, the same convention
the `cover` column uses. Every placement is screened at a loose `fmax` and
only the best are refined at the scorer's tight one: up to `--refine` (default
10) **per parent**, one per distinct screened basin, so the refined set — and
with it the next `n`'s parents and what `dft_export` has to choose from —
carries every basin the screen found rather than the best one several times
over. See `DESIGN.md`'s `docking.py` section for the full measurements (staged
screen-then-refine cost, the placement-count confidence argument, per-parent
detail).

## Packing: the seeds are independent draws

At small `n` the packing *is* the answer. A chloroform hydrogen-bonded to a
pyrazine nitrogen at ~5.7 kcal/mol will not detach, migrate around the ring
and rebind at the far nitrogen within 10 ps of gas-phase Langevin — so if the
packing did not put one molecule at each nitrogen, the trajectory will not
find that geometry. And packmol's draw is badly biased toward putting them
together: measured on pyrazine + 2 CHCl3 over 24 packings, 83% came out
same-face and only 4% opposed.

The seeds used to be **stratified** to work around that: each one constrained
every solvent molecule to its own hemisphere of the shell, with the
hemispheres maximally separated at one end of the range and coincident at the
other, so that a fixed budget of packings covered opposite-faces /
perpendicular / same-face by design rather than by chance. It worked —
opposed arrangements went from 4% to 29% — but it existed to make the MD
sweep better at *finding the lowest minimum*, and
[docking](#docking-a-constructive-alternative) now owns that job and finds
that basin by construction.

For the two jobs the sweep keeps, stratification was working against it. A
non-greedy diversity check wants an independently drawn packing, not one
drawn at a hand-picked clustering value; and pooling frame counts across
packings drawn that way made [basin occupancy](#basin-occupancy) a mixture
with hand-picked weights rather than a sample of anything. So the packings are
independent draws again, and `--seeds` is the number of them at each `n`.
`n = 0` gets one, since every packing of a bare solute is the same packing.
The default is 5: every job the sweep still does scales with packings rather
than steps — `found by` out of 3 can only say 1, 2 or 3, the chance that every
packing at `n = 2` draws the same-face arrangement above is 0.83^k (0.57 at 3
packings, 0.39 at 5), and occupancy's `seeds` column counts packings. Scoring
cost per `n` is `seeds × max_frames`, and 5 × 30 is what 3 × 50 used to be.

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

**Search convergence.** Independent packings are independent *searches*, so
the useful question about a set of them is not how far apart their answers
scattered but how many of them arrived at the same minimum. The report gives,
per `n`, the pooled minimum, `found by`, the spread of the per-packing minima
and the pool size. A minimum several packings reached is one the search finds
reliably; a `1/7` rests entirely on one draw of the arrangement lottery, and
the report warns about it — the fix there is more packings, not more steps.
The spread is printed to be seen rather than propagated: independent packings
are searches rather than repeat measurements of one system, so their scatter
measures the lottery as much as the sampling noise. A single-packing sweep has
none of this evidence and says so in place of the table, rather than leaving
the absence to read as precision.

**Basin occupancy.** The first two diagnostics judge the *search* — did it
converge, did it agree with itself. This one is different: it says how the
trajectory's time was actually spent, frame-weighted and pooled across
packings, rather than how many distinct kinds of basin the search happened to
turn up. See [Basin occupancy](#basin-occupancy) above for what it is, why it
exists now and did not before, and the caveats that come with reading it.

There used to be a fourth, **sampling convergence** — per run, how far into
the trajectory the best candidate was found, and what the last 25% took off
the minimum. It measured the MD sweep's performance at minimum-finding, which
is the job docking took, so it is gone with the stratification that served the
same job.

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
| `n_sweep.py` | the `E_int(n)` sweep: a non-greedy diversity check and the only source of basin occupancy |
| `docking.py` | a second, constructive generator — random placement + BFGS instead of thermal sampling; owns minimum-finding, standalone/either-or with `n_sweep.py` |
| `dft_export.py` | exports deduped, near-minimum candidates from either generator, plus a manifest, for downstream DFT refinement |
| `report.py` | all text rendering, plus the ASE-free numeric helpers; no ASE/tblite at module scope |
| `shell_capacity.py` | monolayer capacity, for choosing `n_solvent` |
| `single_thread.py` | pins the numeric stack to one thread per process — **must be imported before numpy** |

`DESIGN.md` carries the design rationale in full, including the measurements
behind every default; `CLAUDE.md` carries the operating manual and a list of
gotchas worth reading before changing one.

## Gotchas worth knowing up front

- **`import single_thread` first, above every other import**, in any module
  that imports numpy. The OpenMP runtime reads `OMP_NUM_THREADS` once, when it
  loads; setting it afterwards is silently ignored, and getting this wrong
  cost 2.5× on in-process scoring.
- Packing changed at `0.2.0` (stratified over a clustering parameter) and
  again at `0.8.0` (back to independent draws), so sweeps from either side of
  either change are not comparable at `n ≥ 2`. The `version` field in the
  params block is what distinguishes them.
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
