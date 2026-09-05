# Design notes

The measurements behind every default in `CLAUDE.md`, and the record of what
was tried and removed. `CLAUDE.md` is the operating manual an agent needs on
every session; this is the layer beneath it -- read the section named from
`CLAUDE.md`'s "Before you change a default or re-add a feature" index before
touching the thing it names. `README.md` tells the same story at introduction
length; this is where the numbers and the removed alternatives live.

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
the interaction energy converge, and report the diagnostics that say whether
the search and the sampling behind it were sufficient.

## The params block, and what it used to record

`run_sweep` deliberately does *not* assemble a difference of sweeps. One sweep
is one solute in one solvent; a double difference such as
`(A - B)_solvent1 - (A - B)_solvent2` is four sweeps subtracted by hand. The
`Leg` / `delta_delta` machinery that used to do it had no callers and no driver
script, and imposed a shape real sweeps did not have.

What that gives up is the guarantee that both halves of a difference ran under
identical settings, so every sweep now records a **params block**: the whole
`Condition` (via `asdict`, so new fields are picked up automatically) plus the
package version (`report.VERSION`), a timestamp, and the versions of the
libraries the numbers came out of (`report.library_versions`, read from
package metadata so nothing has to be imported). Given `wall_slack` silently
changed meaning at `c429f8a`, this is cheap insurance -- check it before
subtracting two sweeps.

It used to record the git commit too, from `git rev-parse` against the
directory the source lives in. That was removed at 0.6.0: on a cluster the
code is typically a copy without a `.git`, run under a batch environment that
may not even have `git` on `PATH`, so the field came out blank exactly where
two sweeps most needed distinguishing -- and it failed silently, since
`git`'s stderr was discarded and `None` rendered as `-`. `version` is the
pin now, which is why the bump rule in `CLAUDE.md` is not optional.

## `E_int(n)` is what makes "how much explicit solvent?" a measurement

    E_int(n) = E(solute + n solvent) - E(solute) - n E(solvent)

every term relaxed in the same continuum. `E_int(0)` is zero by construction.
It is comparable across n because whole solvent molecules are subtracted off,
and a solvent molecule that optimizes away into the continuum contributes ~0 --
so a dissolved shell lands back on the n = 0 answer. That is what makes
dissolution a usable *null result* rather than a failure mode: if solvent
drifts away, there was no specific interaction to capture.

Departure from `E_int(0)` is exactly "what the continuum was missing".

### It does not plateau, so read the increment

The docs claimed a plateau in n and that was wrong. The reference
`E(solvent)` is one solvent molecule relaxed in the same continuum --
approximately a molecule of bulk liquid -- so the intent was that moving a
molecule from bulk into a bulk-like site in the shell costs nothing, leaving
only the specific sites able to move `E_int`. That cancellation is not clean.
Two terms survive it, both roughly linear in n, neither switching off once the
specific sites are filled:

- **the continuum's per-molecule bias** -- the `bias` column of the binding
  table in CLAUDE.md's "Measurements to not re-derive". -0.9 kcal/mol for
  pyrazine...HCCl3, +6.2 for a water in ALPB(water): not fixed in sign, and a
  property of the solvent/continuum pair rather than of the chemistry being
  measured;
- **solvent-solvent cohesion** from n = 2 on, already in Known limitations
  below, since what `E_int` subtracts is *isolated* solvent molecules.

Add that `E_int` is a potential energy, with no entropic penalty for
condensing a molecule out of the continuum, and that `E_int(min)` is a running
minimum over a configuration space that grows with n, and the curve tends to a
line of nonzero slope rather than to a flat.

So `report.txt` carries a **`dE_int` column** -- `E_int(min)` at this n minus
`E_int(min)` at n - 1, over the **pooled** candidates of every packing at each
n. It is keyed by n alone -- pairing by seed *index* would pair nothing
physical, since seed 0 at n = 2 is not seed 0 at n = 1 with a molecule added
but an independent packing. Convergence is the increment settling to a
**constant**, not to zero.

Only `E_int(min)` gets an increment, and since 0.8.0 it is the only energy the
tables print at all: a second increment on the ensemble average, or the
average beside the minimum, would only invite reporting whichever looks
better.

**The seeds at one n are pooled, not averaged.** They are independent
*searches*, not replicas of a physical system: a mean over them penalizes
searching more widely, diluting the one packing that found a geometry nobody
else did by the ones that missed it. Measured on one sweep at n = 2: six
packings find -12.12 and one finds -11.60, and the mean of the minima is
-12.05 -- a number no geometry has. So the reported number at each n is the
minimum over the pooled
candidates of every packing, and the pool is deduped across packings with the
same energy criterion `ensemble` uses within a run (`report.dedupe_energies`,
1 meV), because seven packings finding one basin would otherwise multiply its
Boltzmann weight by seven. `E_int(ens)` is pooled the same way and kept in the
JSON -- Boltzmann weighting at 298 K (kT = 0.594 kcal/mol) is a soft minimum
rather than a democratic average, and it is continuous in the candidate set
where the minimum is a step function -- but 0.8.0 stopped printing it.

Pooling is only legitimate because the references are sweep-wide --
`score_run_grid` computes them once per distinct (solute, solvent, calculator,
continuum), so every `interaction_eV` in a sweep shares a zero. `pool_by_n`
checks that and **raises** if not: a sweep whose rows were measured against
different zeros is broken, not old.

What a pooled minimum gives up is that it is a running minimum over *search
effort*, and the effort is not constant across n -- one measured sweep's pool
went 42 / 89 / 174 / 247 candidates over n = 1..4, so some of `dE_int`'s slope
is search depth rather than chemistry. That is the honest form of the concern
that would otherwise argue for means; the answer is the `pool` and `found by`
columns, which show the effort rather than averaging it away.

The quantity that does plateau is a **difference at fixed n** between two legs
-- two conformers, two tautomers, bound and free, one solute in two solvents.
Both legs carry n molecules in comparable environments, so the bias and the
cohesion largely cancel. That is also the quantity a cluster-continuum study
reports, which is the other reason one sweep is one leg. `mean_contacts` and
`dissolved_fraction` are convergence indicators for the **search**, not the
**sampling**: both are computed over the pooled *distinct minima*, so they say
how many kinds of basin were found, not how much of the trajectory actually
sat in a contact state. They are in the JSON and no longer in any table, since
`pool` already reports the search; what the `contacts` / `dissolved` columns
show is the frame-weighted `occupancy_*` pair below.

### Basin occupancy: what quenching throws away

Reading `n_sweep.py`'s code settles a question its docs used to leave open:
**it is not producing an ensemble of thermally reasonable geometries.** Its
thermal information is destroyed in two steps. First, the quench --
`ensemble.relax` optimises every selected frame to `fmax = 0.002`, projecting
it onto its basin's minimum and discarding where in the basin it actually sat,
which is the part the MD generated. Second, the dedupe -- `report.dedupe_energies`
used to return kept indices only, so *how many frames landed in each basin*
was never stored anywhere. What survived was a set of distinct
continuum-relaxed local minima with potential energies -- structurally the
same object `docking.py` produces, and docking finds better ones by
construction (measured: -13.00 vs -11.44 kcal/mol at n = 2 on
pyrazine/chloroform, see the `docking.py` section). So the MD sweep was a
basin-hopping search whose proposal move happened to be 10 ps of MD, and a
worse one than random placement + BFGS. `E_int(ens)` does not fill that gap:
it is a Boltzmann average over *quenched potential energies* of a *deduped,
search-effort-dependent* set -- no vibrational entropy, no ZPE, and, because
dedupe threw the counts away, no configurational entropy either.

The fix is not to make MD compete with docking at minimum-finding -- it can't,
by construction. It is to recover the one thing quenching threw away for
free: **which basin the trajectory actually visited, and how often.**
`ensemble.dedupe`'s old justification for discarding frame counts was that
counting them would make the average "an average over how long the trajectory
happened to loiter somewhere" -- true when `dump_interval` was 5 fs, about
100x oversampled against the ~0.55 ps shell decorrelation time above. At
today's defaults -- 20000 x 0.5 fs = 10 ps, dumped every 50 fs = 200 dumps,
`max_frames = 30` selecting down to ~333 fs of scored-frame spacing -- that
ratio is ~1.65x, close to the ~18 effectively independent configurations
the MD-length defaults were chosen to buy. A frame count is no longer pure
autocorrelation, and every energy behind it is already computed, so
`ensemble.dedupe` now returns `(candidate, geometry, members)` groups instead
of representatives alone, and `assemble` sets `n_frames` / `frames` on the
survivor via `dataclasses.replace`. `report.pool_by_n` sums `n_frames` across
every packing that visited a basin into a per-n `basins` list
(`e_int_kcal`, `n_frames`, `frame_share`, `n_seeds_hit`, `weight`,
`n_contacts`), and `format_basin_occupancy` renders the top 5 by frame share
plus an "other" row, with a line contrasting `E_int(min)` against the
frame-share-weighted mean and both pairs of contact statistics -- the gap
between `mean_contacts` (over distinct minima) and `occupancy_mean_contacts`
(frame-weighted) is itself the diagnostic -- it says how far the minima set
over-represents rare tight basins -- and the Basin occupancy section is the
one place both are still printed, since 0.8.0 took the distinct-minima pair
out of the tables.

**Occupancy is quarantined from every `E_int` in this pipeline, on purpose.**
The wall volume (`wall_slack`) sets the translational entropy of a dissociated
solvent molecule and therefore fixes any dissolved/bound *population* --
raise it and an occupancy number moves -- while leaving `E_int(min)`
completely untouched, since a dissolved molecule contributes ~0 to `E_int`
regardless of box size. So occupancy is reported only as a diagnostic, never
folded into an energy: no thermal-average `E_int` is built from it. Reading
the section also means holding five caveats, which live in README's "Reading
report.txt" rather than being restated in every rendered report: sampling is
gas-phase (methanol + 4 water: 3.84-4.39 H-bonds in gas vs 0.00-0.30 in
ALPB(water), no overlap); the 1 meV dedupe merges isoenergetic distinct
minima into one count; frames are correlated, so `scored_frame_spacing_fs`
has to be read against a decorrelation time measured on the system in
question, not assumed; these are inherent-structure populations, with no
vibrational entropy or ZPE; and the wall-volume point above.

There were six until 0.8.0. The sixth was that pooling across *stratified*
packings is a mixture with hand-picked weights rather than a sample, since
`c = (seed + 0.5) / n_seeds` was chosen by design -- which is one of the
reasons the stratification is gone. The packings pooled here are independent
draws now.

A docked candidate has no sampling frame and so no occupancy at all --
`_assemble_dock_n` writes `n_frames` / `frames` as `None`, a real absence
rather than a population of one, the same convention `wall_energy_eV: null`
already uses there. Docking's own dedupe groups constructed *placements*, not
thermal samples; counting them would look like a frame-weighted population
and would not be one, which is exactly what `found_by` already reports
instead.

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
| seeds | 5 | independent packings at each n >= 1; n = 0 gets one, since every packing of a bare solute is the same packing. Every job the sweep still does scales with packings, not steps: `found by` out of 3 can only say 1, 2 or 3; P(every packing same-face at n = 2, pyrazine/chloroform) is 0.83^k = 0.57 at 3 and 0.39 at 5; occupancy's `n_seeds_hit` counts packings. The weighted budget 0.8.0 removed gave the shipped 4-point sweep 1 / 4 / 4 / 3 from `--seeds 3`, so a flat 3 had lost a packing at n = 1 and n = 2 |
| scored frames | 30 per run | `Scoring.max_frames`, was 50. Sized for minimum-finding when the sweep still did that job; for occupancy, frames closer than the decorrelation time recount the same configuration. 30 over 10 ps is 333 fs of spacing, still under 0.55 ps, and 5 x 30 = 3 x 50 relaxations per n, so the two changes are cost-neutral together and trade correlated frames for independent draws |

Expect all of these to grow with the solute. Re-measure the decorrelation time
on your own system rather than carrying 0.55 ps over -- it is the one number
here that is a property of the system rather than of the integrator.

### Packing: independent draws, and why they were once stratified

At small n the packing *is* the answer. A CHCl3 bound to a pyrazine nitrogen
at ~5.7 kcal/mol does not detach, migrate around the ring and rebind at the
far N within 10 ps of gas-phase Langevin, so nothing downstream rescues a bad
draw. And packmol's draw is badly biased: measured on pyrazine + 2 CHCl3 over
24 packings, as the dot product between the two solvent centroid directions,
83% came out same face (> +0.5) and only 4% opposed (< -0.5) -- against a
1.3% site-combinatoric prediction, since `shell_capacity.py` puts pyrazine's
solvent-centre surface at 457 A^2 and a full first shell at ~13 molecules, so
n = 2 is 15% of a monolayer.

**Stratified packing** was the answer to that from 0.2.0 to 0.8.0, and it is
worth recording what it did before someone reinvents it. `pack_solvent` wrote
one packmol `structure` block per molecule rather than one for all n, each
constrained to a hemisphere (`over plane dx dy dz 0.`) around its own
direction; `hemisphere_directions(n, c, rng)` produced those directions with
`c = 0` spreading them as far apart as they go (antipodal, equilateral,
tetrahedral at n = 2/3/4, from a Coulomb relaxation rather than a table) and
`c = 1` collapsing them onto one random focus, the whole set rigidly rotated
at random first so no orientation relative to the solute was assumed.
`run_job_grid` set `c = (seed + 0.5) / n_seeds_at_this_n`. Measured, 24
packings per column:

| packing | mean dot | opposed (< -0.5) | same face (> +0.5) |
| --- | --- | --- | --- |
| unstratified | +0.64 | **4%** | 83% |
| c = 0.00 | -0.13 | **29%** | 21% |
| c = 0.17 | -0.09 | 29% | 8% |
| c = 0.50 | -0.12 | 21% | 8% |
| c = 0.83 | +0.10 | 12% | 17% |
| c = 1.00 | +0.19 | 8% | 38% |

It worked -- opposed went 4% -> 29% -- though less sharply than designed: a
hemisphere is soft (a molecule may sit against the dividing plane and point
almost anywhere from the solute centre), so `c = 0` gives 29% and not ~100%,
and c = 0.00 / 0.17 / 0.50 are indistinguishable at 29 / 29 / 21%.

**It went at 0.8.0 because it was solving the job `docking.py` now owns.**
Raising the n = 2 both-nitrogens hit rate is minimum-finding, and docking
finds that basin by construction. For the two jobs the sweep keeps it was
working against them: a non-greedy diversity check wants an independently
drawn packing, not one drawn at a hand-picked `c`; and pooling frame counts
across packings drawn that way made basin occupancy a mixture with
hand-picked weights rather than a sample of anything (it was caveat 2 of six,
and is now neither).

The **seed budget** went with it. `allocate_seeds` used to spend a fixed total
of `n_seeds * len(n_values)` unevenly -- n = 0 took 1, the rest weighted by
`1 - min(n/capacity, 1)` and handed out by largest remainder over a floor of
2 -- because a packing bought more where stratification paid best. With
independent draws a packing at one n is worth what a packing at any other n
is, so it is `n_seeds` everywhere and 1 at n = 0, where every packing of a
bare solute is the same packing. `--seeds` is a count again, not an average.
`monolayer_capacity` is still computed once per sweep and still lands in the
params block beside `seeds_per_n`, because the `cover` column needs it.

**Because the packings are independent draws, their scatter is still not an
error bar** -- they are independent *searches*, not repeat measurements of one
system. That is why `report.txt` reports **agreement**: `found by` says how
many of an n's packings reached the pooled minimum, and that is the
convergence evidence. Disagreement at small n means the arrangement lottery
has not been won often enough, and the fix is more packings, not more steps.
The spread is still printed, to be seen rather than propagated.

`report.txt` also carries a **`cover` column**, `n / monolayer_capacity` as a
percent, which is what says whether a row is targeted microsolvation or a real
shell. It is read from `params` with no default, so a pre-0.2.0 sweep no
longer re-renders.

### Both halves of a sweep are parallel, and scoring is parallel per candidate

`run_job_grid` fans the MD out across cores and `score_run_grid` does the same
for the scoring, because scoring is the expensive half: at the defaults a
4-point sweep spends a few minutes on MD and the better part of an hour on
candidate optimizations. Both go through `solvate_md.pool_map`, which uses
spawn, so the `if __name__ == "__main__":` guard covers both -- see
`CLAUDE.md`'s Gotchas.

**The granularity is per candidate, not per run directory.** Fanning the
scoring out over run directories gives a default sweep 16 tasks (one packing
at n = 0, then 3 n x 5 seeds) on an 18-core machine, and they are badly
unequal -- n = 0 is a rigid 10-atom molecule, n = 3 a floppy 25-atom cluster
-- so cores idle from the start and the tail is one whole directory long.
Scoring therefore splits into `select_frames` -> `relax` per frame ->
`assemble`, and `score_run_grid` selects every job in the parent, flattens to
one `relax` task per candidate, and assembles afterwards. That is ~500
near-equal tasks with a tail of one
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

`--max-frames` alone sets scoring cost, and that cost is independent of run
length -- a longer trajectory is scored at wider spacing for the same price.
It is 30, down from 50 at 0.9.0: the extra frames bought extra chances at the
lowest quench, which is docking's job now, and for occupancy a frame closer
than the decorrelation time to its neighbour mostly recounts the same
configuration. The 20 relaxations per run that saves pay for `--seeds` going
3 -> 5 at the same cost per n (see the sampling table above). There used to be a `--stride` beside it, selecting every Nth dump
first, which did nothing at all: `max_frames` selects by `linspace` over the
whole trajectory, so it discards whatever thinning `stride` did, and at any
usable setting the cap always bites.

### The two diagnostics that qualify a sweep's numbers

- **Search convergence** -- per n: the pooled minimum, `found by` (how many of
  that n's packings reached it, by the 1 meV dedupe criterion), the spread of
  the per-packing minima, and the pool size. Independent packings are
  independent *searches*, so their **agreement on the minimum** is the
  evidence, not an error bar on a mean: a minimum several packings reached is
  one the search finds reliably, and a `1/7` is a number resting on a single
  draw of the arrangement lottery, which the report warns about. A
  single-packing sweep has no such evidence and says so in place of the table
  rather than leaving the absence to read as precision. A double difference is
  four of these numbers and inherits the weakness of all four.
- **Wall diagnostic** -- the worst wall-active fraction over the sweep's runs,
  with the same 20% warning `run.log`'s footer carries. It says whether any of
  the pooled energies is contaminated by the confinement rather than held
  together by the Hamiltonian.

Both are computed from JSON already on disk, so `python -m report <dir>`
renders them with no MD and no calculator.

There was a third until 0.8.0, **Sampling convergence** -- per run, how far
into the trajectory the best candidate was found and what the last 25% took
off `E_int(min)`, on the argument that `E_int(min)` is a running minimum and
so *when it last fell* is a convergence test. It measured the sweep's
performance at minimum-finding, which is the job docking took, and it went
with the stratification that served the same job. Basin occupancy is what
the sweep reports about its sampling now.

## `docking.py` -- a second generator, beside the MD sweep

The sweep is a **geometry generator** for downstream ab initio refinement, not
a source of final GFN2 numbers. Judged against that goal it has one defect:
at n = 2 on pyrazine/chloroform it never produces the both-nitrogens
arrangement (one chloroform H-bonded to each ring nitrogen), so DFT never
sees it. A missing basin is the one failure downstream refinement cannot
repair -- DFT can re-rank what it is given, but it cannot invent a motif.

Measured, GFN2-xTB/ALPB(chcl3), the MD and docked minima scored against the
same solute/solvent references (pyrazine + chloroform, n = 2):

| structure | E_int (kcal/mol) | how found |
| --- | --- | --- |
| MD pooled minimum, n = 2 | -11.44 | 10 ps Langevin, 14 candidates, 3 packings (stratified, i.e. under the packing 0.8.0 removed -- an independent draw does no better: -11.40 on a shorter run) |
| both-N, docked | **-13.00** | random placement onto the relaxed n = 1 parent, screened, refined |

All 14 MD candidates at n = 2 share one motif -- a single H-bond at 1.90 A,
the second chloroform 3.8-5.6 A away and unbound -- because the trajectory
never visits the other basin, not because the Hamiltonian disfavours it: the
both-N geometry is 1.56 kcal/mol *lower*.

`docking.py` finds it by constructing instead of sampling: place a solvent
molecule at a random position and orientation around the relaxed parent, and
run BFGS. BFGS only descends, so it cannot climb out of the well it lands in
-- which is why it works where a *seeded* MD run would not (`run_one_job`
discards 5 ps of equilibration before recording the first frame, so a seeded
both-N start would already be gone by frame 1). On pyrazine + chloroform,
docking n = 1 -> n = 2 with 3 parents x 64 placements reproduces the
both-N basin as its overall minimum: E_int(2) = -13.00 kcal/mol, both H...N
contacts at 1.94 A, 4 of the 10 refined placements landing within 1 meV of it
(the "found by" column). The n = 1 parent itself came in at -6.65 kcal/mol,
matching the -6.6 single-complex value in the binding table above. Total
wall-clock for both n on 18 cores: 7.3 s -- 256 placements screened, 20
refined.

Both programs report the same *kind* of thing: a continuum-relaxed,
wall-free local minimum, scored identically (`ensemble.relax`, no wall, and
even `n_contacts` / `min_gap_A` computed on the *relaxed* geometry, not the
raw placement). But they are not peers at minimum-finding any more --
**docking owns that job.** The MD sweep's candidates are potential-energy
minima too, reached by quenching 10 ps of MD rather than by construction, and
dozens of independent, unconstrained random poses descending by BFGS reliably
outfind a single thermal trajectory at the same job: the -11.44 vs -13.00
kcal/mol measurement above is the sweep's proposal move (10 ps of gas-phase
Langevin) simply losing to random placement as a basin-hopping search. See
"Basin occupancy" above for why that isn't a reason to drop the MD sweep.

- **Docking** -- random construction, chained upward: n = 1's surviving
  parents become n = 2's starting points. Greedy (the best structure at n
  need not descend from the best at n - 1), which is why `Docking.n_parents`
  carries more than one candidate forward. Wins at minimum-finding by
  construction: BFGS only descends, so trying enough independent poses is
  the whole method.
- **MD sweep** -- demoted to two jobs docking cannot do. First, a
  **non-greedy, independently drawn check**: docking's chain still grows
  from one lineage, so a basin no pose in that lineage lands in is invisible
  to it no matter how many placements are tried, while a freshly packed MD
  run samples a genuinely different region of configuration space. Second,
  the MD sweep is the **only source of basin occupancy** -- a docked minimum
  was placed, not visited, so it has no sense of "time spent" in one basin
  over another, which is exactly what `_assemble_dock_n` writing `n_frames: null`
  is saying.

`scored.json` for a docked run carries `pack_mode: "dock"` (an MD run's says
`"md"`) and every candidate's `wall_energy_eV` is `null` -- a docked
structure has no sampling frame and so no wall energy, a real absence, not a
zero, and `report.py`'s formatters render it as `-`.

**Staged optimisation, measured to cost nothing in quality.** Every
placement is first screened at a loose `screen_fmax` (default 0.05) and only
up to `n_refine` (default 10) **per parent** are re-relaxed at the scorer's
tight `fmax = 0.002`. On the same 64 random placements (same RNG draw),
two-pass screen-then-refine landed at -6.6514 kcal/mol; refining all 64 at
the tight criterion directly landed at -6.6511 -- a 0.0003 kcal/mol
difference, two orders of magnitude under the 1 meV (0.023 kcal/mol) dedupe
tolerance. The two-pass run took 7.3 s combined over two n; refining every
placement tightly took 5.9 s for n = 1 alone.

Which placements get refined changed at 0.9.0. `n_refine` used to be a total
over every parent, taken off the raw screened ranking: ten of 192 at n >= 2,
and since the screened energies were not deduped, those ten could be ten
copies of the best basin. That validation above covered the *minimum* only,
not the diversity of the refined set, and the refined set is what everything
downstream consumes -- the next generation's parents are its deduped top
`n_parents`, so a collapsed set starves the greedy-chain mitigation exactly
where greediness is the documented risk, and `dft_export`'s 3 kcal/mol window
was in practice a cap of ten candidates per n. Now each parent's screened
energies are deduped at `Docking.screen_dedupe_tol_eV` -- 4 meV, ~0.1
kcal/mol, deliberately looser than the 1 meV `DEDUPE_TOL_EV`, since at
`fmax = 0.05` two placements in one basin can still differ by ~0.6 kcal/mol
and the error to prefer is refining a duplicate over dropping a basin -- and
the lowest representative of each screened basin is refined, best-first, up
to `n_refine` per parent. At most 30 tight relaxations at n >= 2 instead of
10, ~4.5 s each over the pool. The screened criterion never reaches a
`scored.json`; the refined candidates are deduped at 1 meV like everything
else, and `n_refined` / the `found by` denominator read the actual count.
Measured on the same `--n 1 2` run before and after, same RNG draw: at n = 2,
30 refined and 19 distinct minima against 10 and 10, the both-N minimum
-13.01 vs -13.00 kcal/mol and found by 2/30 vs 1/10 -- with a second parent
lineage now reaching a both-N basin of its own at -12.97 that the old top-10
never refined -- and `dft_export`'s 3 kcal/mol window admitting 14 candidates
where it was capped at 10. At n = 1, 5 distinct minima of 10 refined against
3, and the minimum moved from -6.6514 to -6.6434 kcal/mol: 0.35 meV, under the
dedupe tolerance, from refining one representative of the best screened basin
instead of six copies of it. Wall-clock 13.1 s vs 7.2 s over both n.
`--freeze-solute` (FixAtoms on the solute during
screening only, off by default) is the other lever, for a large solute whose
soft modes make BFGS crawl without being relevant to where a solvent
molecule binds; refinement is always unconstrained regardless.

**1 - 0.93^K sets the default placement count.** The both-N basin was hit by
4 of 60 random poses onto the relaxed n = 1 parent (7%), so
`1 - 0.93^K` gives 90% confidence at K = 32 and 99% at K = 64 --
`Docking.n_placements` default.

### Applicability: where this stops working

The same ~N^2.5 GFN2 gradient cost that limits the MD sweep limits docking,
compounded by BFGS step count -- so this targets small n on small-to-moderate
solutes, comfortable to ~90 atoms, painful at 180. See the gradient-cost table
this repo already carries for the sweep; the same numbers apply here, since
both programs pay for the identical optimisation.

More fundamentally, **the shell regime belongs to the MD sweep, not to
docking**: past roughly a third of a monolayer (`shell_capacity.
monolayer_capacity`), no single minimum dominates and solvent-solvent
cohesion takes over (already a known limitation of `E_int` at n >= 2 above),
and a greedy chain compounds many sequential conditioned placement decisions
into an arrangement space no longer well sampled by a handful of random
poses. `run_docking` prints a warning once `n` exceeds that fraction of
capacity, the same convention `n_sweep`'s `cover` column uses.

### `dft_export.py`, verified end to end

Reconstructing `E_int` from `dft_export`'s manifest (`energy_eV` plus the
exported reference energies) reproduces `interaction_kcal` to
floating-point precision (< 1e-9 eV). And the solute/solvent reference
energies from an MD sweep and a docking run on the same solute and solvent
matched exactly (-446.915092 / -443.246268 eV), confirming both generators
share one zero -- the invariant `pool_by_n` and `dft_export` both check by
raising when it does not hold.

## Considered and not built

Items gathered here because they were each raised, weighed and rejected at
different points in the pipeline's history -- collected in one place rather
than left scattered where they came up:

- **A solute-free background leg** -- `E(n solvent) - n E(solvent)` in the
  same continuum, which would isolate the bias-plus-cohesion drift directly.
  Not built: the geometries are not comparable to the solvated ones, and a
  difference between two real legs already cancels the same terms without a
  second sweep to maintain.
- **Overlapping the two halves of a sweep** -- starting to score a run as
  soon as its trajectory lands, rather than running all of the MD before any
  of the scoring. Rejected: MD is the cheap half, and a shared pool would let
  a worker that had run MACE then run tblite and hit the OpenMP clash the
  file boundary exists to avoid.
- **Quasi-RRHO free energies** -- not built: GFN2-level thermochemistry is
  triage at best, and DFT supersedes it.
- **RMSD-based dedupe** -- not built: energy dedupe already separates
  distinct minima at the counts docking produces.
- **Pooling docked and swept candidates together** -- not built: docking wins
  at every n by construction, so pooling would let it silently take over the
  headline number and erase the informative comparison between what each
  search actually finds. See `CLAUDE.md`'s invariant that the two are never
  pooled.
- **Stratified packing itself**, in retrospect -- kept from 0.2.0 to 0.8.0,
  removed once `docking.py` took over minimum-finding. Recorded in full above
  ("Packing: independent draws, and why they were once stratified") rather
  than here, since it was built and then removed rather than merely
  considered.

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
the diagnostics were written to catch, and did.

The 0.7.1 - 0.8.0 refactor was verified against a fresh short sweep (pyrazine
+ chloroform, n = 0..2, 1 ps, 10 scored frames per run) and a fresh docking
run, re-run at every step: through 0.7.3 every `energy_eV` and
`interaction_eV` in every `scored.json` was bit-identical and only the
intended keys moved, and at 0.8.0 the docking output stayed byte-identical
while the sweep's changed exactly as the packing change predicts.

Known limitations:

- `E_int` **conflates solute-solvent with solvent-solvent** at n >= 2: two
  chloroforms binding each other counts as solvation. With the continuum's
  per-molecule bias this is why `E_int(n)` has a nonzero asymptotic slope and
  cannot be read for a plateau -- see the `dE_int` discussion above. Both
  terms largely cancel in a difference taken at the same solvent and the same
  n, which is the form to report.
- `n_solvent` is a real choice, not a default to accept. Run
  `shell_capacity.py` first: a count well under a monolayer is targeted
  microsolvation, and that is the regime where *every* explicit molecule sits
  at the continuum boundary. The sweep now prints the same fraction as the
  `cover` column, so a sweep says which regime each of its rows is in.
- **The sweep will miss basins at small n, by design of what it is now.**
  Packmol's independent draw puts both molecules on one face 83% of the time
  at n = 2 on pyrazine/chloroform, so the both-nitrogens arrangement is
  mostly not sampled -- measured at 0.8.0, the sweep reports -11.40 kcal/mol
  where docking reports -13.01 for the same system against the same
  references. That is not a defect to fix in the sweep: `docking.py` owns
  minimum-finding, and the sweep is the independently drawn check beside it.
  Run both. A `found by` of 1/k at small n is asking for more packings, not
  more steps -- and the 0.9.0 default of 5 is the measured form of that
  advice: on a 3 ps smoke sweep at the new defaults, 2 of the 5 packings at
  n = 2 reached the both-N basin (-13.01 kcal/mol, `found by` 2/5) that 3
  packings had missed, which is what 0.83^k = 0.39 rather than 0.57 buys.
- MACE-OFF23 as a generator is **untested** here. It needs a model download,
  and it cannot share a process with tblite (see the file boundary above).
- GFN2 likely **over-binds** the C-H...N contact -- 1.91 A is short against a
  literature 2.2-2.5 A, so absolute magnitudes may be 1.5-2x too strong. Signs
  and orderings should be robust; ratios may not be.
