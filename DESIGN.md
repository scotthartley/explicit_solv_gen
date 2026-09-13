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

### Cropping a *rendered* block is not cropping the record

Being the whole `asdict` is the right rule for what gets **recorded** -- no
`Condition` / `Scoring` / `Docking` field can be added without landing in the
params block, so nothing that shapes a run goes unrecorded by omission. It is
the wrong rule for what gets **printed**, because a field can be live in the
dataclass and provably inert in a given run: grid-mode docking never reads
`n_placements`, random mode never reads `grid_spacing_A`, and no docking run
has a trajectory for `max_frames` to subsample. Before 0.16.0 a rendered
report asserted all three anyway -- a grid-mode `dock_report.txt` printed
`n_placements  64` and `shell_fill  0.5` beside `place_mode  grid`, and a
reader had to know `docking.py`'s dispatch to discount them.

0.16.0's `report.inert_params` answers this at render time only:
`format_report` omits a row when another value already recorded in the same
block says it did nothing, and appends a one-line footnote naming what it
dropped and why. `sweep.json` / `dock.json` are untouched -- `dock_params`
stopped special-casing `max_frames` out of the JSON for exactly this reason,
so the record stays the complete `asdict` and two runs' params blocks stay
diffable key-for-key regardless of what either run's report chose to show.

**The rule is inert *given another recorded value*, never "left at its
default."** Cropping a field merely because it sits at its default was
considered and rejected outright: it is the same mistake that made
`wall_slack`'s silent meaning-change at `c429f8a` dangerous in the first
place. A params block that hides values equal to their default cannot show a
reader that two runs disagree about what a default field *means* -- only
that both happen to hold the same number -- and that is precisely the case
that needs the field visible, not hidden. So a report keeps every field that
is merely at a default and crops only a field another recorded value has
made structurally impossible to have acted.

The same reasoning applies to `run.log`'s header at n = 0: `pack_solvent`
short-circuits before packmol runs, so the whole Packing block and the wall
row of Sampling Hamiltonian describe machinery that did not execute, not
settings that merely took their default value. They are dropped, with a
one-line replacement saying why, on the identical basis.

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
same criterion `ensemble` uses within a run (`report.dedupe_energies`: 5 meV
of energy and 0.15 A of contact descriptor), because seven packings finding
one basin would otherwise multiply its Boltzmann weight by seven. `E_int(ens)` is pooled the same way and kept in the
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

### Binding-site specificity: the basin ladder answers what the increment cannot

The section above proves the increment cannot switch off when the specific
sites fill -- two roughly linear terms survive the cancellation -- and that is
a problem, because "which solvent molecules make a defined interaction, and
which are merely present" is the question `docking.py` exists for. Nothing in
the output answered it: `E_int(min)`, `dE_int` and `pool` are what get
reported, and none of the three can tell a defined contact from a bulk-like
one.

The **spectrum of distinct minima** can. Sort each n's pooled basins by
energy and express them in kT above that n's own minimum. Measured on the
0.14.0 grid run (pyrazine + chloroform, n = 1-4, `--dump-screen` harness of the
sections below, `report.VERSION` 0.14.0 numbers reproduced exactly):

| n | ladder (kT above that n's minimum) | `sites` | `gap/kT` | `dE_int` |
| --- | --- | --- | --- | --- |
| 1 | 0.0 **4.4** 4.4 4.6 4.7 4.7 4.8 | 1 | 4.4 | -6.67 |
| 2 | 0.0 **2.6** 2.6 2.7 2.7 2.9 2.9 ... | 1 | 2.6 | -6.36 |
| 3 | 0.0 0.0 0.1 0.2 0.2 0.2 0.2 0.3 ... | -- | 0.0 | -5.36 |
| 4 | 0.0 0.0 0.0 0.1 0.1 0.2 0.2 0.2 ... | -- | 0.0 | -5.11 |

Pyrazine has two nitrogens, each taking one C-H...N contact, and the
transition sits exactly where they saturate: one basin and then a 4.4 kT cliff
at n = 1, one basin and a 2.6 kT cliff at n = 2, and from n = 3 a flat floor
with no cliff anywhere. The increment column beside it is a smooth decay with
no feature at n = 2 -> 3 at all. **The energy curve structurally cannot see
this transition and the spectrum reads it off directly.**

`pool` cannot see it either, and misleads in the opposite direction: 233
basins at n = 3 reads as a rich landscape demanding search, when it is a flat
floor finely subdivided -- the whole n = 3 pool spans 3.2 kT, which is the
point "'Distinct minima' is a resolution, not a count" makes at length.

**The MD sweep agrees, independently.** Re-reporting the shipped 5-packing
pyrazine/chloroform sweep -- a different generator, independent draws, its own
candidates -- gives `sites` 1 at n = 1 (gap 4.4) and 1 at n = 2 (gap 2.7), and
no cliff at n = 3. Two searches that share no geometry agree on where the
specificity stops.

**What the first rung is, said in the report.** A ladder reading
`0.0 0.0 0.1 ...` was misread on first contact by someone who had not written
it, in both available directions: the leading `0.0` looks like a measured
result rather than the zero the row is defined against, and a repeated `0.0`
looks like the minimum listed twice rather than a second basin. Neither
reading is recoverable from the numbers alone, so `format_basin_spectrum`'s
footer now states both -- rung 0 *is* that n's minimum, and a later `0.0` is a
distinct basin closer than the 0.05 kT one printed decimal resolves. The
second point is the load-bearing one: 0.05 kT is ~1.3 meV against
`DEDUPE_TOL_EV`'s 5 meV window, so two basins that print alike cannot have
been separated by energy at all -- they are distinct because
`contact_descriptor` says so, which is exactly the flat floor the section
exists to show and the reason the count is not a count of species.

**Why not simply the gap to the second basin.** For a solute with several
inequivalent sites of similar affinity, two distinguishable binding sites give
two near-degenerate basins, so the naive gap is ~0 and reports "bulk-like" for
the case that is most specific of all. So `basin_ladder` looks for the
**largest** gap in the low-energy spectrum and reports how many basins sit
below it. Chemically *equivalent* sites need no handling: `contact_descriptor`
folds same-element automorphisms, so pyrazine's two nitrogens are one basin by
construction (see "Geometric basin dedupe").

That was stated as a falsifiable prediction before it was run, and tested on
**2-methylpyrazine + chloroform** (`examples/2-methylpyrazine.xyz`, added for
it): one methyl makes the two nitrogens inequivalent but similar, which is
precisely where the naive gap fails. Predicted: two low rungs 0.2-0.8 kT apart,
then a cliff of several kT, giving `sites = 2`. Measured, n = 1-3, default
random placement:

| n | ladder | `sites` | `gap/kT` |
| --- | --- | --- | --- |
| 1 | 0.0 **0.5** 5.4 | **2** | 4.9 |
| 2 | 0.0 0.1 0.1 0.2 **2.8** 2.8 2.9 3.1 ... | 4 | 2.6 |
| 3 | 0.0 0.1 0.2 0.2 0.2 0.2 0.2 0.2 ... | -- | 0.1 |

The two rungs under the cliff really are the two nitrogens, checked on the
geometries rather than assumed: in `scored_candidates.xyz` the lowest basin
puts the chloroform H 1.89 A from the nitrogen beside the methyl-bearing
carbon, the second puts it 1.90 A from the far nitrogen, and the third -- the
one above the 4.9 kT cliff -- has no nitrogen contact at all (nearest solute
atom 5.27 A, a ring H). So the two inequivalent sites come out 0.5 kT apart
under a 4.9 kT cliff, and a gap-to-the-second-basin metric would have read
0.5 kT and called the most specific case in this file bulk-like. n = 2 finds
four near-degenerate two-molecule arrangements under a 2.6 kT cliff, which is
what one molecule per nitrogen plus a residual splitting looks like, and n = 3
saturates exactly where pyrazine's does.

**The cliff is a property of the solvent, not an artifact of chloroform --
and "strongly bound" is not "specific".** Run in a second continuum,
pyrazine + acetone at n = 1-3 has **no cliff at any n**: `sites` blank
throughout, gaps 0.2 / 0.1 / 0.7 kT, with 34 distinct basins inside 0.7 kT at
n = 1 alone. A cliff was expected there on the grounds that CLAUDE.md's
binding table has pyrazine...acetone at -3.5 kcal/mol in ALPB(acetone), which
is real binding. It is real binding without a defined site, and that is
chemically the right answer rather than a miss: chloroform donates a C-H to
pyrazine's nitrogen lone pair, while acetone is an acceptor like the nitrogen
itself, so its interaction is dispersion-and-dipole and has no directional
anchor to single out one geometry. `found by` says the same thing from the
other side -- 37 of 43 refinements reach chloroform's n = 1 minimum, against 2
of 48 for acetone. **The diagnostic separates binding strength from binding
specificity, which is the distinction it exists to draw**, and the earlier
finding that acetone's eventual n = 1 winner sat 247th by screened energy
("The screen-to-refine handoff") is the same flatness seen through a different
instrument. What is *not* established is that acetone has no specific site at
all -- only that this search, which does find its reported minimum, finds no
gap above it.

**Why a gap has to be wider than the manifold below it.** The first version of
this searched the largest gap among basins within `LADDER_WINDOW_KT = 10.0` of
the minimum, and that is wrong on an MD pool. Docked pools span 1.9-2.8 kT
entire, but the sweep quenches half-dissolved frames too and its n = 3 pool
spans 11.5 kT -- so the largest in-window gap was 2.0 kT sitting at the *118th*
basin, and the column read `sites = 118` for exactly the bulk-like case this
diagnostic exists to call bulk-like. A gap therefore qualifies only if it
exceeds the spread of everything beneath it, which is the content of "these k
basins are one family and everything else is far away". The condition is
trivially true at the first gap, so there is always an answer. The window
survives as the outer of the two guards.

**The knobs, and what each does.** `LADDER_MIN_GAP_KT = 1.0` gates only
whether `sites` is *printed* -- never a verdict, and every cliff measured here
is 2.6-4.9 kT against 0.0-0.7 kT where there is none, an order of magnitude
either side of it, so its exact value is not load-bearing.
`LADDER_WINDOW_KT` is an energy and not a rank: the first draft
capped the search at the first 25 basins, which re-couples the diagnostic to a
rank and breaks on a many-site solute. `LADDER_CLAMP_KT` equals the window on
purpose: a rung printed as `>10` is one no gap could have been measured from.
`LADDER_N = 16` is display only, and
`sites` / `gap/kT` are read off the whole spectrum, so `--ladder` cannot move
them. 16 rather than 6 because the default has to survive a many-site solute:
six pyrazines is twelve nitrogens, so n = 1 shows twelve near-degenerate rungs
before the cliff and a shorter ladder would show only the flat part -- the
"bulk-like" misreading this exists to prevent.

**It is robust to the dedupe's known over-splitting.** `Scoring.fmax = 0.002`
inflates `pool` by 20-45% (see "'Distinct minima' is a resolution"), and the
ladder does not care: recomputing it on the same structures relaxed to
`fmax = 2e-4`, where the pools fall 7 / 151 / 231 / 320 -> 6 / 98 / 128 / 239,
`sites` is identical at every n and `gap/kT` moves by at most 0.16 kT
(4.43 -> 4.38, 2.55 -> 2.54, 0.05 -> 0.21, 0.04 -> 0.18). A count that halves
while the conclusion does not move is the argument for reporting the spectrum
rather than the count.

**No numeric default moved for any of this, and nothing new is persisted.**
Everything is derived from the `e_int_kcal` values every `scored.json` already
carries, so `python -m report <dir>` regenerates the columns and the section
for runs scored by any earlier version, with no rescoring -- unlike
`descriptor` (0.11.0), `n_in_window` (0.13.0) or `screen_cut_kcal` (0.14.0).
The report states the measurement and refuses to grade it: there is no verdict
line and no "saturates at n = 2" claim anywhere in the output.

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
the section also means holding six caveats, which live in README's "Reading
report.txt" rather than being restated in every rendered report: sampling is
gas-phase (methanol + 4 water: 3.84-4.39 H-bonds in gas vs 0.00-0.30 in
ALPB(water), no overlap); the basin criterion has a blind spot of its own,
and moving either of its two tolerances moves every count here (flagged per
basin as `contacts_split` when it provably fired, see below); frames are correlated, so
`scored_frame_spacing_fs` has to be read against a decorrelation time
measured on the system in question, not assumed; these are inherent-structure
populations, with no vibrational entropy or ZPE; the wall-volume point above;
and the modal basin is the same inherent-structure population as the rest,
restated because it is now a named geometry rather than only a row in a
table.

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

**The modal basin is a deliverable now, not just a row in a table.** Through
0.9.0, occupancy stopped at a frame count: the top-5-by-share table said which
basin the shell visited most, but naming a geometry meant grepping every
`scored.json` in the sweep for a matching energy by hand -- the minimum got
`best.xyz`, a Best geometry table row, and `best_n<N>.xyz`; the mode got
nothing of the sort. That asymmetry did not match what the two numbers are
for: `E_int(min)` is the reported energy, but "which geometry did the
trajectory actually spend its time in" is a real question about the system
and deserves the same footing -- a file, not just a statistic. 0.10.0 gives
every pooled `basins` entry an identity (`run`, `seed`, `candidate_index` --
which doubles as the frame index into that run's `scored_candidates.xyz`,
guaranteed equal by `ensemble.assemble`) and writes the pooled `frame_share`
maximum out as `modal_n<N>.xyz`, alongside `best_n<N>.xyz`, unconditionally
for every n of a sweep.

**`n_visits` is the right companion to `frame_share`, not `mean_dwell` alone.**
A frame count already says how much of the trajectory a basin held; what it
cannot say on its own is whether that time was one long loiter or several
independent returns -- and the difference matters, because a proposal move
that keeps re-entering a basin from elsewhere is stronger evidence for that
basin than one that visited it once and got stuck. `basin_visits` answers this
for free: it partitions a basin's scored-frame dump indices into maximal runs
*consecutive in selection order* (not in raw dump index -- `select_frames`
subsamples by `np.linspace`, so scored dumps are not 1 apart in general), and
`n_visits` is how many such runs there are. `mean_dwell = n_frames / n_visits`
is printed beside it rather than instead of it, because the two together
distinguish "12 frames, 1 visit" (one loiter, weak evidence) from "12 frames,
6 visits" (six independent returns, much stronger) -- a distinction a single
number in either direction erases.

**`contacts_split` turns a global caveat into a per-basin flag, for free.**
README's "Reading report.txt" has always admitted that the energy dedupe can
merge isoenergetic distinct minima, "tolerable for ranking, sharper for
counting" -- a caveat that bites harder here, because occupancy's whole job
is counting. The data to *detect* a merge, rather than prevent it, was
already on every candidate: if a basin's members disagree on `n_contacts`,
the test provably fused two structures with different contact patterns into
one count, and `pool_by_n` flags it as `contacts_split`, rendered `!` in
Basin occupancy with a one-line count under the section. Costs nothing,
changes no number, and replaces "this can happen, somewhere" with "it
happened here, N times."

At 0.10.0 this paragraph went on to say that reopening RMSD dedupe to fix the
merges "stays rejected; the case for it has not changed." **It did change,
and the flag is what changed it** -- which is the point of having built it. A
production sweep at n = 3 and 4 came back with **58 basins flagged `!`**:
not a stray flag on a marginal basin but the criterion failing at scale, and
the measurement that opened the next section.

### Geometric basin dedupe: what the energy-only criterion was doing

Through 0.10.0 two candidates were one basin iff their optimised energies
were within 1 meV. The rejection of anything geometric was scoped precisely:
*"energy dedupe already separates distinct minima **at the counts docking
produces**."* Docking produces ~19 distinct minima at n = 2. A five-seed
sweep produces 75-79 at n = 3-4. That is a different regime, and it was
measured rather than argued.

**The energy axis was saturated.** Pooling the 76 candidates of one n = 2
sweep: they span 284 meV with a **median nearest-neighbour gap of 1.08 meV**,
against a 1 meV merge window. Within one run every survivor is pairwise
>1 meV apart *by construction*, so that median is entirely a cross-run
effect -- a window that decides almost nothing. Worse, pooling those 76
candidates into 52 basins merged **24** of them, and a null model in which
**no two runs ever share a basin** -- energies drawn i.i.d. from the pooled
empirical density, kept pairwise >1 meV apart within each run exactly as the
code's own dedupe does -- predicts 20 to 31 merges as its one free parameter
(the kernel bandwidth) runs from 10 meV to 0.5 meV. The observed 24 sits
squarely inside that, and below the two tighter bandwidths. The cross-seed
dedupe was not demonstrably identifying shared basins at all, and
`n_seeds_hit`, pooled `frame_share` and `found_by` all rested on it.

**Both failure modes were real and both were large.** Computing a
permutation- and rigid-motion-invariant contact descriptor (below) for all 76
and comparing it against the energy verdict:

| geom threshold (A) | energy says SAME, geometry says different | energy says DIFFERENT, geometry says same |
| --- | --- | --- |
| 0.15 | **31 of 38 pairs (82%)** | 9 (5 same-run / 4 cross-run) |
| 0.25 | 30 of 38 (79%) | 40 |
| 0.50 | 28 of 38 (71%) | 136 |

Roughly seven of thirty-eight energy-merges were real. The rest fused
structures up to 4.2 A apart.

**A naive RMSD would have been worse than what was there**, which is worth
saying because it is what "RMSD dedupe" usually means. Solvent molecules
occupy a fixed block layout but are chemically interchangeable, so two
structures identical up to swapping solvent 1 and 2 get a large coordinate
RMSD and would be split, where the energy test fused them correctly. Any
geometric criterion here has to be permutation-invariant, and that -- not
RMSD's cost -- is the real work. `ensemble.contact_descriptor` does it by
quotienting the symmetry out of the descriptor rather than searching over
superpositions:

- an `n_mol x n_solute` matrix whose `(m, i)` entry is the distance from
  solute atom `i` to solvent molecule `m`, minimised over `m`'s own atoms --
  which absorbs intra-solvent atom permutations (chloroform's three Cl,
  acetone's methyl rotation) for free;
- each row sorted *within each solute element block*, elements in a fixed
  order, making it invariant under same-element relabelling of the solute.
  For a small rigid aromatic that is exactly its automorphism group, so
  pyrazine's two equivalent nitrogens stay one basin and no automorphism
  search is needed. For a large floppy solute with many same-element atoms it
  is a superset of the true symmetry and can under-split: the documented
  limit;
- distances only, so rigid-body motion is quotiented out with no Kabsch and
  no superposition;
- plus the sorted vector of pairwise solvent-solvent van der Waals gaps, so
  two structures with identical solute contacts but different shell packing
  are still distinguished. Sorted, hence already invariant under solvent
  relabelling.

Comparing two of them asks whether there is an assignment of one structure's
solvent molecules to the other's whose max per-feature deviation is within
tolerance -- a *feasibility* question in descriptor space, not in Cartesian
space, and not the sum-minimisation Hungarian solves. That makes it a
bottleneck (minimax) assignment problem: threshold the pairwise cost matrix
at the tolerance and ask whether the resulting bipartite graph has a perfect
matching, which `report._has_perfect_matching` answers exactly by Kuhn's
augmenting-path algorithm, O(n_mol^3), no scipy needed. Through 0.18.0 this
was instead a brute-force `itertools.permutations` search, capped at
`report.MAX_DESCRIPTOR_MOLECULES = 8` (8! = 40320 comparisons of small
arrays) because it had nowhere else to go past that. Measured head to head on
the same inputs, worst case (no assignment satisfies the tolerance, so a
permutation search exhausts every one of them):

| n_mol | permutation search (worst case) | matching |
| --- | --- | --- |
| 6 | 0.93 ms | 8.8 us |
| 8 | 52.4 ms | 8.4 us |
| 9 | 483.2 ms | 15.4 us |
| 32 | (32! -- not attempted) | 38.1 us |

Identical verdict on 2160 randomized pairs (matching, non-matching and
near-threshold cases, `n_mol` 0-8, four tolerances each) confirms it is the
same predicate, not an approximation of it -- the cap was never forced by an
absent Hungarian solver, it was forced by naming the wrong algorithm for a
minimax question in the first place. `MAX_DESCRIPTOR_MOLECULES` is now 32:
not a factorial wall any more, just a sanity bound comfortably past a full
first shell for the solutes this targets (~27 chloroform on pyrazine, from
`monolayer_capacity`), so a descriptor larger than that is more likely two
unrelated systems mixed than a real run. Cost is negligible regardless, now
doubly so: the greedy dedupe compares each candidate against group
*representatives* only, never pairwise, so a sweep's ~150 candidates against
~50 groups was already milliseconds against an hour of BFGS, and the matching
test does not care whether `n_mol` is 8 or 32.

**Both tolerances were picked from gaps in the data, not guessed.** Over all
2850 pairs of that n = 2 pool the descriptor distances run 0.01-0.12 A for
sixteen pairs, then stop dead until 0.16 A, with the bulk of the
distribution only starting at 0.19 -- a within-basin cluster, an empty band,
and then everything else. `GEOM_TOL_A = 0.15` sits in the empty band. And pairs
the
descriptor calls identical differ in energy by up to **2.07 meV** -- above
the old 1 meV window, which was therefore splitting them -- so
`DEDUPE_TOL_EV` went 1 meV -> 5 meV, covering that with margin while staying
an order of magnitude under the ~29 meV separating the nearest pairs the
descriptor cannot resolve but energy can.

**That the within-basin cluster is tight was also read here as evidence that
`Scoring.fmax = 0.002` is converged enough for the criterion to be stable.
That claim is withdrawn** -- it generalised one n = 2 sweep pool, and
"'Distinct minima' is a resolution, not a count" below now refutes it on
docked pools at n = 1-4: the median residual displacement on re-relaxing to
2e-4 is 0.11-0.13 A, 73-89% of this tolerance. The band above is still where
0.15 comes from and is still the best available anchor; what it does not
license is the further claim about convergence.

So the criterion is `|dE| <= DEDUPE_TOL_EV` **and** `descriptor distance <=
GEOM_TOL_A`. Geometry is the real test; energy stays as a cheap guard against
the descriptor's own lossiness and as the pre-sort that keeps the greedy pass
deterministic. Widening the energy window is what fixes the false splits, the
geometry veto is what fixes the false merges, and both live beside
`EV_TO_KCAL` in `report.py` for the reason that module's comment already
gave -- `ensemble`, `pool_by_n` and `dft_export` must apply the same test by
construction, and `pool_by_n` is handed summaries rather than a `Scoring` to
read a setting off. They are recorded in every summary and every params block
besides, since a criterion change that silently moves basin counts is exactly
the `wall_slack` hazard.

**What does not move: `E_int(min)`, the headline number.** It is `min` over
group representatives, and groups partition the candidates, so the set
minimum is the same however it is partitioned -- invariant under any
regrouping, which is what makes this a fix to the diagnostics rather than a
change to the reported energy. What does move: `pool`, `found_by`, every
`basins` entry, `frame_share`, `n_seeds_hit`, `weight`, `E_int(ens)`,
`modal_n<N>.xyz`, and `dft_export`'s selection. `found_by` and `n_seeds_hit`
falling is the honest outcome given the null-model result above, not a
regression: it is corroboration evidence being corrected downward.

**Measured on the shipped example** -- `--n 0 1 2 3 --seeds 5`,
pyrazine/chloroform, 480 candidates -- by re-pooling that one sweep's own
candidates under both criteria:

| | old (1 meV) | new (5 meV + 0.15 A) |
| --- | --- | --- |
| `E_int(min)`, n = 0..3 | -0.03 / -6.65 / -13.03 / -18.38 | **identical, to 0.00e+00 kcal/mol** |
| `pool` at n = 1 / 2 / 3 | 20 / 41 / 66 | 41 / 114 / 137 |
| basins flagged `!` | 41 of 128 | **0 of 293** |
| basins hit by >1 packing | 71 | 22 |
| `found by`, n = 1 / 2 / 3 | 5 / 2 / 1 | 5 / 2 / 1 |
| `dft_export` structures | 75 | 200 |

`pool` rising rather than falling is the 82% figure showing up at scale: the
criterion is splitting far more than the widened energy window merges.
`contacts_split` going to exactly zero is the strongest single result -- the
detector that motivated the change cannot find a fused basin any more. And
re-running the coincidence null on the new grouping settles the other half:
under it, energy and geometry agreement are independent, so a cross-run pair
merges by chance at `P(|dE| <= 5 meV) x P(dG <= 0.15 A)`, which predicts 5.9 /
4.3 / 0.7 shared basins at n = 1 / 2 / 3 against 30 / 17 / 7 observed -- **4x
to 10x above coincidence**, where the old criterion sat at or below it. The
packings really do re-find each other's basins; the energy-only test simply
could not show it.

`found by` not moving is worth noting rather than glossing: the packings that
reached the pooled minimum reached it geometrically too, so the headline
corroboration column is unaffected even though `n_seeds_hit` across all basins
falls by two thirds. What does cost something downstream is the export --
75 structures to 200, because basins the old criterion fused inside the
3 kcal/mol window are now distinct structures worth their own DFT point.
`--max-per-n` is the lever if that is more than a DFT budget allows.

Three places compared two named candidates with an inline `abs(dE) <= tol`
and so bypassed `dedupe_groups` entirely -- `found_by`, the `best` marker in
the per-packing table, and docking's per-parent table. A two-axis criterion
would have moved the pooling and left all three behind, silently disagreeing
with it, so they now go through `report.same_basin`, and `found_by_seeds`
carries the packings themselves into `format_seed_detail` rather than having
it recompute the test. Docking's screening pass takes the same two-axis test
at its own looser tolerances (`screen_dedupe_tol_eV`, `screen_geom_tol_A`):
a screened geometry is relaxed only to `screen_fmax = 0.05`, so a criterion
tuned for converged minima would shatter one screened basin into a dozen
near-copies and spend every `n_refine` slot on them -- the exact starvation
0.9.0's per-parent refinement budget exists to prevent.

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
  that n's packings reached it, by the same dedupe criterion), the spread of
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
contacts at 1.94 A, 4 of the 10 refined placements landing in the same basin as it
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
difference, four orders of magnitude under the 5 meV (0.115 kcal/mol) dedupe
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
placements are deduped at `Docking.screen_dedupe_tol_eV` /
`Docking.screen_geom_tol_A` -- 10 meV and 0.5 A since 0.11.0, both
deliberately looser than the scorer's `DEDUPE_TOL_EV` / `GEOM_TOL_A`, since
at `fmax = 0.05` two placements in one basin can still differ by ~0.6
kcal/mol and by far more than the scorer's 0.15 A, and the error to prefer is
refining a duplicate over dropping a basin -- and
the lowest representative of each screened basin is refined, best-first, up
to `n_refine` per parent. At most 30 tight relaxations at n >= 2 instead of
10, ~4.5 s each over the pool. The screened criterion never reaches a
`scored.json`; the refined candidates are deduped at the scorer's criterion
like everything else, and `n_refined` / the `found by` denominator read the
actual count.
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

### Systematic placement: coverage instead of confidence

`1 - 0.93^K` buys **confidence**, not **coverage**. A basin captured by 7% of
random poses is 99% certain to be hit at K = 64; one captured by 0.5% is at
27%, and nothing in a random run's output distinguishes "this basin does not
exist" from "we did not draw it". Since the entire reason `docking.py` exists
is that a missing basin is the one thing downstream DFT cannot repair, a
search whose completeness is probabilistic is a weaker foundation than it
needs to be. `Docking.place_mode = "grid"` enumerates instead: positions
crossed with orientations, every pose that clears `tolerance` screened.

The algorithm is otherwise untouched -- scan the solvent around the parent,
optimise, carry the best forward, repeat. Only the scan becomes systematic.
Site detection, per-site coverage reporting, site-restricted placement at
n > 1, a site-additivity ranking and an energy-window parent set were each
considered and dropped: what the run is for is how many molecules it takes to
satisfy the solute's sites, and under that goal the identity of the first
molecule's site does not matter -- only that every site is filled by high
enough n.

**Positions come from a surface, not a volume.** `place_one` draws uniformly
inside an ellipsoidal shell, and `shell_padding`'s own docstring already
concedes that sizing a convex region around an appreciable solute "scatters
them through empty space instead of onto the solute". A Shrake-Rupley surface
is topology-correct where an ellipsoid is not: it puts points in concave
pockets, a macrocycle cavity, and the rim of a stacked aggregate.
`shell_capacity.surface_points` is `sasa` keeping the exposure mask's points
instead of collapsing it to an area -- a deliberately duplicated loop, since
`sasa` feeds the params block's `monolayer_capacity` and the report's `cover`
column, and perturbing a live number to save a dozen lines is a bad trade.
Orientations come from a super-Fibonacci spiral over SO(3) (Alexa, CVPR
2022): quasi-uniform quaternions in about eight lines, no symmetry detection
and no per-solvent table.

**The grid is radial, and one shell is not enough -- measured.** The obvious
region is the solvent-*centre* surface `sasa` measures, at a probe equal to
the solvent's bulk-density sphere radius. That is the contact distance for a
*sphere*, and it is measurably wrong for a directional contact. On pyrazine +
chloroform, the both-nitrogens minimum puts each chloroform centroid **3.17 A**
from the nearest parent atom; that single shell puts centroids at **4.37-4.92
A**, uniformly 1.2-1.8 A too far out, and BFGS from there settles into
whatever shallow dispersion basin it started above. Scanning shells at
fractions of the solvent radius, screening each, on the relaxed n = 1 parent:

| probe | positions | poses kept | screened E_min (eV) | both-N poses |
| --- | --- | --- | --- | --- |
| 1.27 A (0.40 r) | 120 | 475 | -1333.9649 | 13 |
| 1.58 A (0.50 r) | 141 | 1024 | -1333.9607 | 27 |
| 1.90 A (0.60 r) | 157 | 1548 | -1333.9683 | 23 |
| 2.38 A (0.75 r) | 181 | 2117 | -1333.9569 | 1 |
| 3.17 A (1.00 r) | 222 | 2664 | -1333.8801 | **0** |

The sphere-model surface is the only one that finds nothing, and a full n = 1
2 3 run on it alone reproduced the *MD sweep's* -11.44 kcal/mol at n = 2 --
the single-H-bond motif -- rather than docking's -13.00. So
`Docking.grid_probe_fracs` defaults to `(0.4, 0.7, 1.0)`, three shells.
Points are voxel-downsampled (`_voxel_downsample`) per shell, not
over the union -- a `grid_spacing_A` voxel is wider than the gap between
adjacent shells, so pooling first would collapse the radial axis. Voxel
downsampling rather than a greedy minimum-separation filter: O(M) and
deterministic, at the cost of occasionally keeping two points closer than the
spacing across a cell boundary, which for a placement grid is harmless
redundancy.

**What each shell costs, and what the outer one is worth.** Equal
150-pose samples off the relaxed n = 1 parent, screened through the same pool:

| shell | poses kept | mean BFGS steps | throughput |
| --- | --- | --- | --- |
| 0.4 r (1.27 A) | 468 | 27.2 | 24 ms/pose |
| 0.7 r (2.22 A) | 1960 | 16.1 | 16 ms/pose |
| 1.0 r (3.17 A) | 2808 | **1.3** | 6 ms/pose |

The radial axis, not grid mode itself, is where the cost is: a single-shell
`--n 1 2 3` run screened 7728 poses in 28.8 s, three shells 15,542 in 144 s --
2x the poses for 5x the wall-clock, because an inner-shell pose starts in
contact and BFGS does real work on it. The refinement half does not scale at
all: `n_refine` is 10 per parent whatever the mode, so a run does 30 tight
relaxations either way.

That table also put the outer shell in question. It is 54% of the poses and
converges in a **median of one BFGS step** -- those placements never relax
into contact at all, they are accepted by the loose screen sitting exactly
where they were put, which is consistent with the zero both-N hits and the
2 kcal/mol higher screened minimum above. It was kept on the argument that a
bulky orientation may clear `tolerance` only further out, on a solute the
inner shells reject outright -- an argument that was not measured. It has now
been measured, in two solvents.

**The shell question, closed.** Pyrazine, `--n 1 2 3`, grid mode, everything
else at defaults. `E_int(min)` in kcal/mol, then found-by, then `pool`. At one
parent and `--refine 10`, the conditions of the comparison table below:

| solvent | shells | poses | time | n = 1 | n = 2 | n = 3 |
| --- | --- | --- | --- | --- | --- | --- |
| CHCl3 | (0.4, 0.7, 1.0) | 15,542 | 148 s | -6.667 10/10 1 | -13.002 6/10 5 | -18.350 1/10 9 |
| CHCl3 | (0.4, 0.7) | 7,382 | 128 s | identical | identical | identical |
| CHCl3 | (0.5, 0.7) | 8,796 | 157 s | -6.664 10/10 1 | -13.023 9/10 2 | -18.378 2/10 5 |
| acetone | random, 64 | 192 | 13 s | -4.563 1/10 8 | -9.366 1/10 10 | -14.582 1/10 10 |
| acetone | 0.4 alone | 177 | 14 s | -4.566 1/10 9 | -10.101 1/10 10 | -14.743 1/10 8 |
| acetone | 0.7 alone | 3,400 | 44 s | -4.541 1/10 9 | -9.612 1/10 8 | -14.637 1/10 10 |
| acetone | 1.0 alone | 8,448 | 37 s | -4.343 1/10 7 | -9.093 1/10 9 | -14.628 1/10 10 |
| acetone | (0.4, 0.7, 1.0) | 11,673 | 75 s | -4.541 | -9.612 | -14.637 |
| acetone | (0.4, 0.7) | 3,609 | 51 s | identical to default | | |
| acetone | (0.5, 0.7) | 4,104 | 60 s | identical to default | | |

At three parents and `--refine 30` -- `n_parents` is the real default, and 30
is what the comparison below already advises before a `dft_export`:

| solvent | shells | poses | time | n = 1 | n = 2 | n = 3 |
| --- | --- | --- | --- | --- | --- | --- |
| CHCl3 | (0.4, 0.7, 1.0) | 37,385 | 403 s | -6.667 27/30 4 | -13.043 10/90 36 | -18.355 2/90 46 |
| CHCl3 | (0.5, 0.7) | 20,931 | 424 s | -6.665 28/30 3 | -13.037 15/90 31 | -18.360 2/90 45 |
| acetone | (0.4, 0.7, 1.0) | 28,407 | 215 s | -4.541 1/30 26 | -10.129 2/90 82 | -15.462 1/90 85 |
| acetone | (0.4, 0.7) | 8,518 | 150 s | -4.546 4/30 26 | -10.039 1/90 80 | -15.363 1/90 82 |
| acetone | (0.5, 0.7) | 9,925 | 171 s | -4.541 1/30 28 | -10.146 1/90 73 | -15.048 1/90 69 |

Both tables predate the principal-axis alignment described further down, which
slightly changes which poses a grid run generates: the first row re-measured
after it gives -6.6667 / -13.0222 / -18.2082 on 15,148 poses. The n = 3 shift
is the greedy chain following a *lower* n = 2, not a worse search, and it is
the same size as the chain's own scatter below. Shell-against-shell
comparisons hold at that resolution; an absolute number worth quoting should
be re-measured.

**The outer shell is inert.** Remove it and chloroform at one parent is
identical at every n, `pool` and found-by included; at three parents the
default and (0.5, 0.7) differ by at most 0.006 kcal/mol. Run *alone* it is the
worst shell for acetone at every n despite having the most poses. The
structural reason is the median-one-step column above: its poses do not relax
into contact at `screen_fmax`, so they rank below every contact pose and are
refined only when a parent turned up fewer contact basins than `n_refine`.
Dropping it saves 45-70% of the poses and almost none of the wall-clock -- an
inner pose costs ~4x an outer one, so chloroform's 52% pose saving was 14% of
the run.

**0.4 against 0.5 is a wash.** Chloroform at one parent prefers 0.5 (both-N
9/10 against 6/10, and lower at n = 2 and n = 3); at three parents the two are
equal. Acetone is mixed at the 0.1-0.4 kcal/mol level, which is inside the
selection noise below.

**Acetone's spread is selection noise, and the noise is the finding.** Every
acetone run has found-by 1 of N and a `pool` near the cap: many shallow,
near-degenerate basins. The 0.4 r shell alone -- 177 poses -- found -4.566 at
n = 1, lower than any run with 8,000 to 28,000 poses and 30 refinements.
Checked with `same_basin`: that basin *is* in the default grid's pose list
(same parent, same shell), and appears in none of the default run's refined
candidates at `--refine` 10 or 30, while the 64-pose random run and the
(0.4, 0.7) run both reach it. So the refine selection -- a fixed top-k off a
ranking relaxed only to 0.05 eV/A, with screened basins merged at 10 meV /
0.5 A -- loses basins once the screened pool is 50x larger than k, and raising
k alone does not cure it. That is grid mode's first-order weakness and a
bigger effect than any shell set: the screen-to-refine handoff, not the radial
grid, is where the next measurement goes. The greedy chain shows in the same
tables -- chloroform n = 3 is -18.355 at three parents / refine 30 against
-18.404 at three parents / refine 10, because refine 30 found a lower n = 2
(-13.043) whose descendants are not the best n = 3. That is the known
`n_parents` caveat, not something grid mode introduced.

**So `grid_probe_fracs` stays `(0.4, 0.7, 1.0)`.** Nothing measured is better,
and the rule here is not to move a default without a measurement that says to.
What changes is why the outer shell is there: not the unmeasured bulky-orientation
argument, but that it is nearly free, and that the case which would justify it
-- a concave solute whose pockets the inner shells reject outright -- is
exactly the case neither chloroform nor acetone on pyrazine tested.

**A new solvent wants a per-shell survival count first.** The fractions scale
with the solvent's bulk-density sphere radius; the contact distance they are
meant to bracket -- centroid to nearest heavy atom -- does not scale with it.
The 1.0 r shell sits 3.5 A from a pyrazine N for water, which is nearly an
H-bond, 4.4-4.9 A for chloroform and ~5 A for toluene, which is empty space.
The cheap check is the pose count itself -- how many poses clear `tolerance`
per shell, seconds of numpy and no calculator. For water, chloroform, acetone,
methanol, acetonitrile and benzene, against bare pyrazine and against a
grooved pyrazine + 2 CHCl3 parent, the 0.7 r shell never had a position where
all 12 orientations clashed, so the "the outer shell rescues positions the
inner shells reject" case did not fire anywhere; 0.4 r keeps 6-20% of its
poses on the flatter solvents (acetone: 47 of 792) and 1.0 r keeps 100%
everywhere. Run that count on a new solvent before trusting these fractions.

**Grid mode is not frame-independent, so the parent is aligned first.** It was
commented in `dock_at_n` as though it were -- "the placement region *is* the
parent's surface" -- and it is not: `_voxel_downsample` rounds against a
lattice anchored at the lab origin, and the `n_orientations` rotations are
lab-frame. Measured on bare pyrazine, six rigid motions of the input gave
3251-3564 poses against the file frame's 3355, a 6.2% spread, at different
places. Not a correctness bug -- every pose still clears `tolerance`, screened
by the same test -- but a reproducibility caveat for the one mode whose
selling point is coverage. `dock_at_n` now calls
`align_to_principal_axes(parent, n_solute)` before *both* placement modes, as
the random one always did for its axis-aligned ellipsoid; that puts the two
modes' poses in one frame and makes the pose set bit-identical across rigid
motions. What survives is `_principal_frame`'s eigenvector-sign ambiguity:
`eigh` fixes each axis only up to sign, a perturbation as small as a pure
translation can flip one, and the flipped runs give 3348-3363 poses (0.4%)
across the same six motions. Deliberately not fixed -- `_principal_frame` also
sets the frame `pack_solvent` packs into, so pinning the sign convention would
move the starting geometry of every MD sweep to buy 0.4% of the grid.

**The parent is the new solute, literally.** At n = k the surface is computed
on the whole relaxed n = k - 1 complex, with no solute/solvent distinction
anywhere in the placement code. Shrake-Rupley buries the contact patch under
each bound molecule on its own, but a molecule sitting on a surface exposes
more than it buries, so the grid grows with n -- 3351 poses per parent at
n = 1 to 6955 at n = 3 on pyrazine, sublinearly, since surface goes as
volume^(2/3). This does include second-layer positions on top of already-bound
solvent, and a solvent-solvent stack could in principle win a greedy step and
spend a molecule without reaching a new solute site. Restricting the grid to
the solute's own exposed surface would prevent that (`n_solute` is already
threaded through `dock_at_n`, so the distinction is free), but that is
site-restriction in another guise, dropped for the same reason: a mechanism
to save effort on a search that is already affordable. It cost nothing here
-- grid wins at every n below.

**Measured, pyrazine + chloroform, n = 1 2 3, one parent, everything else at
defaults:**

| | random | grid |
| --- | --- | --- |
| E_int(min), n = 1 | -6.6514 | **-6.6667** |
| E_int(min), n = 2 | -12.9971 | **-13.0025** |
| E_int(min), n = 3 | -18.1672 | **-18.3497** |
| both-N at n = 2 | yes, found by 2 of 10 | yes, found by **6 of 10** |
| pool (distinct minima) 1/2/3 | 3 / 9 / 10 | 1 / 5 / 9 |
| placements screened | 192 | 15,542 |
| wall-clock (18 cores) | 13.2 s | 144 s |

Grid is lower at every n and reaches the both-nitrogens basin from three
times as many refined placements. The cost is 11x wall-clock -- still under
three minutes -- and, less obviously, a **narrower** pool: `n_refine` is a
fixed 10 per parent taken off the lowest *screened* representatives, so
against 5236 screened poses instead of 64 those ten all land in the best few
basins. That is the right trade for minimum-finding and the wrong one for
`dft_export`, which wants diversity inside its 3 kcal/mol window; raise
`--refine` when exporting from a grid run. The grid column here predates the
principal-axis alignment, like the shell tables above and with the same
caveat; the random column is unaffected, since that path always aligned.

**`n_parents` still earns its keep under a systematic scan**, which was the
open question -- with random placement a bad parent choice compounds with
placement luck, while with a systematic scan parent choice is the only
remaining source of error. Same runs at `--parents 3`: identical at n = 1 and
n = 2 (the grid's n = 1 pool is a single basin, so there is only one parent to
carry), and -18.4039 against -18.3497 at n = 3, for 320 s against 144 s. It
buys 0.054 kcal/mol, so `n_parents` stays at 3.

**`n_parents` stays unchanged at 3** -- it buys 0.054 kcal/mol under a
systematic scan for 320 s against 144 s, the same margin it bought under
random placement.

**`place_mode` default flipped to `"grid"` at 0.17.0, without a new
measurement.** The numbers above were already the measurement: grid is lower
at every n on the one system tested, reaches the both-nitrogens basin from
three times as many refined placements, and costs ~11x the wall-clock --
still under three minutes at one parent. What changed is not the evidence but
the judgment call sitting on top of it: a missing basin is the failure mode
this whole module exists to avoid, and 11x on a search that already runs in
seconds to minutes is cheap insurance against it, cheaper than it looked
before `n_refine`'s window-based cap (0.13.0, default 400) closed grid mode's
own weak point -- a fixed top-k losing the winning basin to a screened
ranking that does not predict it. That fix was already the live default
before this one, so grid mode no longer trades a narrower pool for its lower
minimum. Random placement stays reachable at `--place-mode random`, for a
cheap first look or to reproduce an older sweep's exact conditions -- older
`dock.json` params blocks recorded `place_mode` explicitly, so nothing
becomes ambiguous retroactively. "Moving a default wants more than one
system's numbers" was the standing bar and is not overridden here; this is a
recorded exception, made on request, and a second solvent's numbers before
trusting `grid_probe_fracs` (above) still applies with the same force now
that grid is the default most runs will actually take.

### The screen-to-refine handoff: a window, not a rank

The staged screen was validated on its *minimum* (above), and separately on
the *diversity* of what it refines (0.9.0's per-parent dedupe). What neither
check asked is whether the screened **ranking** picks the right basins to
refine at all, and under a systematic scan it does not. 0.12.0 already saw
the symptom and mis-diagnosed it, advising "raise `--refine` before a
`dft_export` from a grid run": raising it does not help, because the problem
is not how many are refined but that they are chosen by an energy carrying no
information about where they refine to.

**The diagnostic.** Acetone, n = 1, grid mode, default shells, one parent,
refining *every* screened basin representative instead of the top ten. 2798
poses collapse into 410 screened basins; the best refined basin is the
**247th** of them by screened energy, 1.50 kcal/mol above the screened
minimum, at `E_int(min) = -4.608` against top-10's -4.541. This is the
question the plan posed as "outside top-k, or merged into a neighbour's
group?" and the answer is the first: the basin *is* a representative, it is
simply never reached by a rank cut. Raising the cap does not rescue it --
top-30 and top-50 still return -4.54, and the curve only reaches -4.599 by
top-100.

**Tightening the screen does not rescue it either, and costs more.** The
obvious alternative was to make the screened energy a better predictor by
relaxing further before ranking. Measured on the same 2798 poses:

| `screen_fmax` | screen | refine-all | basins | best `E_int` | winner's rank | top-10 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 (default) | 8.2 s | 21.3 s | 410 | -4.608 | 247 of 410 | -4.541 |
| 0.02 | 30.0 s | 16.7 s | 323 | -4.635 | 283 of 323 | -4.611 |
| 0.01 | 69.4 s | 11.6 s | 230 | -4.625 | 155 of 230 | -4.611 |

A tighter screen does lift what top-10 finds (-4.541 to -4.611) and does
fuse near-duplicate basins (410 to 230), but the winner still sits 60-88% of
the way down the ranking at every tightness, so a rank cut keeps missing it.
And it is the wrong pass to spend on: screening every pose costs 3.7x and
8.5x more, while refining the survivors gets *cheaper* (they start closer to
a minimum), for a net 29.5 -> 46.7 -> 81.0 s. Refining every basin off the
**loose** screen is both the cheapest column here and the only one that
reaches the minimum. `screen_fmax` stays at 0.05.

**So the cut is an energy window and `n_refine` is only its cost cap.** Every
screened basin within `Docking.refine_window_kcal` of *that parent's own*
screened minimum is refined, best-first, up to `n_refine` -- per parent, for
the same reason the dedupe is per parent, so a lineage that screened a little
higher than another still gets its own basins refined. 3.0 kcal/mol is
`dft_export`'s window and about 5 kT, and it covers the measured winner at
every screen tightness above (+1.50, +1.82, +1.17); at `fmax = 0.05` a
screened energy is worth so little that the window's real job is only to drop
what is plainly unbound.

Note what this does *not* change: for a fixed parent, the refined set is now
a strict superset of what the old top-10 refined -- the window admits a
prefix of the same energy-ordered representatives and the cap sits well above
10 -- so `E_int(min)` at that n can only fall or stay. A chain number that
moves the wrong way is the greedy chain reacting to a different (better)
parent, the documented `n_parents` caveat, and not the window losing a basin.
Pyrazine, grid mode, one parent, n = 1 2 3, is exactly that case -- and the
same table settles the cap:

| solvent | policy | n = 1 | n = 2 | n = 3 | pool | wall |
| --- | --- | --- | --- | --- | --- | --- |
| acetone | top-10 (old) | -4.541 | **-10.250** | -15.123 | 8 / 10 / 10 | 76.1 s |
| acetone | window 3.0, cap 100 | -4.599 | -9.812 | -14.912 | 70 / 82 / 80 | 112.9 s |
| acetone | window 3.0, cap 400 | **-4.608** | -10.020 | **-15.678** | 218 / 312 / 324 | 237.6 s |
| chloroform | top-10 (old) | -6.667 | -13.022 | -18.208 | 1 / 3 / 8 | 203.0 s |
| chloroform | window 3.0, cap 100 | -6.667 | -13.022 | -18.370 | 7 / 63 / 69 | 184.9 s |
| chloroform | window 3.0, cap 400 | -6.667 | **-13.028** | **-18.384** | 7 / 151 / 233 | 550.9 s |

At the default cap the change is lower at n = 1 and n = 3 in acetone (0.55
kcal/mol at n = 3) and at n = 2 and n = 3 in chloroform, for 2.7-3.1x the
wall-clock. Acetone's n = 2 goes the other way, and that is the greedy chain,
not the window: the better n = 1 basin is a dead end one step on, which one
parent cannot recover from and `n_parents = 3` exists to damp. The `pool`
column is the other half of the result and the one `dft_export` cares about
-- 8 distinct minima against 218 at n = 1 -- from a search that had already
found them and was throwing them away.

**`n_refine = 400` rather than 100, measured.** At one fixed parent the two
are indistinguishable (n = 1 acetone: -4.599 against -4.608, a 0.009 kcal/mol
gap under the 5 meV dedupe tolerance) and 100 is less than half the
wall-clock, which is what made it the tempting default. Over a chain it is
not: cap 100 lands acetone at n = 3 on **-14.912, worse than the old top-10's
-15.123**, because a cap small enough to bind is still a rank cut in
disguise, re-running the same lottery one level up and handing a worse parent
forward. Cap 400 is at or below the old policy at every n in both solvents.
100 remains the knob (`--refine 100`) for a deliberately cheaper grid run,
with that caveat attached.

**And a binding cap is reported, not left to be inferred.** The failure above
is invisible from the refined output alone -- a run that refined its cap's
worth of basins looks exactly like one that refined every basin it had -- so
`dock_at_n` records how many basins each parent's window admitted
(`n_in_window` in `parent_detail`) beside how many were actually refined. The
Per-parent detail table shows both, marks the row `!` when they differ, and
`report.refine_cap_warning` names the capped parents and says what it costs.
The point is not the cap itself but what a capped run silently becomes: a
rank cut on the screened energy, which is the thing this whole section
measures as not working.

**This is not a grid-only change.** It is easy to assume 64 random
placements collapse into a handful of basins, and they do not: measured per
parent over acetone and chloroform at n = 1, 2 and 3, a random parent yields
**40-63** distinct screened basins, so the old flat top-10 was discarding
about four fifths of them in the default mode too. The cap never binds there
-- a random parent's entire screened spread is 1.5-3.2 kcal/mol, so the
window admits nearly all of them (the two cases where it bit at all were 40
basins to 37 and 62 to 56, both dropping only basins the old top-10 had
never reached). At the real defaults -- `place_mode="random"`,
`n_parents = 3` -- the change is worth *more* here than under the scan it was
diagnosed on:

| solvent | policy | n = 1 | n = 2 | n = 3 | pool | wall |
| --- | --- | --- | --- | --- | --- | --- |
| chloroform | top-10 (old) | -6.651 | -12.997 | -18.167 | 3 / 25 / 26 | 46.0 s |
| chloroform | window 3.0 | -6.651 | **-13.000** | **-18.328** | 17 / 59 / 78 | 107.7 s |
| acetone | top-10 (old) | -4.563 | -9.366 | -14.764 | 8 / 28 / 29 | 50.5 s |
| acetone | window 3.0 | **-4.585** | **-9.643** | **-15.653** | 34 / 162 / 172 | 192.5 s |

Lower or equal at every n in both solvents -- 0.16 and 0.89 kcal/mol at n = 3
-- for 2.3x and 3.8x the wall-clock, with the pool up 3-6x. The old rows
reproduce the random-mode numbers this file already carried (-6.6514 /
-12.9971 / -18.1672), which is the check that the harness behind these tables
is the same pipeline and not a re-implementation of it.

### How deep the cap cut -- and why that is not a safety grade

0.13.0's cap warning says *that* the cap bound and never *where*, which
leaves it with exactly one discharge: re-run at a larger cap. That is the
wrong shape for a diagnostic on a run that costs hours, and the information
was already in hand at selection time and being thrown away -- `dock_at_n`
knows, for every basin it refines, its position in that parent's screened
ordering and its screened offset above that parent's screened floor.

So 0.14.0 keeps that provenance. `dock_at_n` returns one `ScreenOrigin` per
refined candidate instead of a bare parent index (`parent`, `rank`,
`offset_kcal`, plus the parent-wide `n_in_window` and `cut_kcal`), which also
removes the parallel `window_counts` dict that had to stay in lockstep with
it. `parent_detail` gains three fields off that record -- `screen_cut_kcal`,
`best_screen_rank`, `best_screen_offset_kcal` -- and the Per-parent detail
table gains a `cut`, `rank` and `offset` column.

**The grading statistic has to be the offset, not the rank.** "The winner came
from rank 12 of 400" would license "the cap was harmless" only given a
screened-rank-to-refined-energy correlation, and the whole of this section is
the measurement that there isn't one. The offset is the quantity that
transfers between runs, and `cut_kcal` against `refine_window_kcal` says how
much of the window a binding cap left unexplored.

**What that offset does *not* support is a safe/unsafe grade.** The intended
design was three-way: a cap that only ever cut above ~2.0 kcal/mol had
truncated a region no refined winner had been seen in (the three observations
above: +1.50, +1.82, +1.17) and would emit a NOTE rather than a WARNING.
Stated in advance so it could be refuted, and it was. Twenty-eight parent
rows -- pyrazine in chloroform and acetone, random mode, n = 1-3, 32 and 64
placements, the cap never binding so every in-window basin was refined --
put the refined winner's own screened offset at **+0.00 to +2.98 of a 3.0
kcal/mol window**, quartiles 0.55 / 1.12 / 1.64:

| winner's offset | of 28 parents | of the 13 lineages that produced their n's minimum |
| --- | --- | --- |
| above half the window (+1.50) | 11 | 7 |
| above 0.6 (+1.80) | 6 | 4 |
| above 0.7 (+2.10) | 4 | 2 |
| above 0.8 (+2.40) | 3 | 2 |
| above 0.9 (+2.70) | 1 | 1 |

A cut at +2.0 would have been graded "safe" and lost the winner in 5 of the
28. It is the same finding as the section's, one level up: at
`screen_fmax = 0.05` a screened energy says so little about where a basin
refines to that its rank is uninformative and its *offset* is barely better --
a weak concentration toward the floor, with a tail running to the window edge.

Directly, on pyrazine + chloroform, random mode, one parent, 16 placements,
varying only `--refine`:

| `--refine` | n = 1 cut / winner offset | `E_int(1)` | n = 2 cut | `E_int(2)` |
| --- | --- | --- | --- | --- |
| 3 | +1.72 / +2.44 | -6.64 | +0.10 | **-11.34** |
| 12 | +2.53 / +2.44 | **-6.65** | +0.72 | **-13.01** |
| 400 (uncapped) | +2.82 / +2.44 | **-6.65** | +0.97 | **-13.01** |

The n = 2 row is what a shallow cut costs -- 1.67 kcal/mol, the both-nitrogens
basin -- and the n = 1 column is the counter-example to the 2.0 threshold: the
winner sits at +2.44 and only a cut above that keeps it.

So the warning stays unconditional on the cap binding, and what the
measurement bought is a warning that quantifies rather than one that grades:
it names each capped parent with its cut, says what fraction of the window
that left unexplored, and quotes the measured winner spread instead of
implying a safe depth. `report.SCREEN_WINNER_OFFSET_KCAL` carries the range.
One caveat it does not cover: all 28 rows are random mode, where a parent
holds 21-61 in-window representatives. A grid parent holds hundreds to
thousands in the same window, and whether the offsets distribute the same way
there is untested -- the three grid observations that exist all sit under
+1.9.

**And the screening partition itself can now be re-examined offline.** The
screened energies and descriptors were computed and discarded, so every
question about whether the partition is over-splitting -- is 883 screened
basins at n = 3 a real count, or the loose tolerances shattering one basin
into near-copies? -- needed another full screen to ask. `--dump-screen`
writes `screen.json` into every n's run directory: per parent, every screened
energy and contact descriptor, plus the representative / window / cap decision
taken off them. Off by default, read by nothing in the pipeline, and
deliberately *not* a `Docking` field -- it changes nothing about the run, so
it has no business in the params block that says what the run was. Re-running
`report.dedupe_energies` over a dump reproduces that parent's `n_in_window`
and its refined set exactly, which is the check that the dump describes the
run rather than a re-derivation of it.

### "Distinct minima" is a resolution, not a count -- and there is no truth to check it against

The screening tolerances (`Docking.screen_dedupe_tol_eV` 10 meV,
`screen_geom_tol_A` 0.5 A) never got the histogram calibration the scorer's
pair got, and the suspicion they invite is specific: if they shatter one
screened basin into near-copies, then the 883 and 1667 in-window "basins" at
n = 3 and n = 4 are an artifact, `n_refine` is truncating duplicates rather
than candidates, and the fix belongs in the dedupe rather than in `--refine`.

Measured on one grid run with `--dump-screen` (pyrazine + chloroform, one
parent, n = 1-4, 22,906 poses, 977 s), which reproduces this file's existing
grid numbers exactly -- `E_int(min)` -6.667 / -13.028 / -18.384, `pool`
7 / 151 / 233 -- so the harness is the pipeline, not a re-implementation.

**First, the methodological trap, because it is easy to fall into and this
file did.** There is no independent source of truth here. The screened basin
count and the refined `pool` are *the same function*, `dedupe_groups`,
differing only in tolerance -- 10 meV / 0.5 A against 5 meV / 0.15 A. Dividing
one by the other and calling the stricter one ground truth measures nothing
except the tolerance gap. Done that way the "over-count" reads 6.1 / 1.9 / 1.7
/ 1.2 : 1 and looks like it vanishes with n. Re-cut the *same* refined pool at
the *same* tolerance the screen used, and it does not:

| n | refined | distinct at the screen's own 10 meV / 0.5 A | over-count |
| --- | --- | --- | --- |
| 1 | 43 | 4 | **10.8 : 1** |
| 2 | 287 | 48 | **6.0 : 1** |
| 3 | 400 | 75 | **5.3 : 1** |
| 4 | 400 | 164 | **2.4 : 1** |

So the screen really does over-count, by 5-6x in the middle of the range. The
suspicion was right in direction.

**But the tolerances are not the lever -- convergence is.** The table above
holds the tolerance fixed and varies only how far the geometry was relaxed:
400 screened basins at `screen_fmax = 0.05` are 75 basins once the same
structures reach `Scoring.fmax = 0.002`, judged identically. Two poses heading
for one minimum are still more than 0.5 A apart in contact space at the loose
screen, and no threshold recovers that -- widening it merges genuinely
different basins just as fast. This is the same effect "The screen-to-refine
handoff" already measured from the other side (screening at 0.01 instead of
0.05 fuses 410 basins to 230) and rejected on cost: 8.5x the screening pass
for a net 29.5 -> 81.0 s. The over-count is real, its cause is known, and the
cure is already priced and declined.

**And the proposed loosening is contraindicated on its own terms.**
Re-partitioning at each candidate tolerance, asking whether the pose that
actually refined into that n's reported minimum survives as a representative:

| tolerance | n = 1 | n = 2 | n = 3 | n = 4 |
| --- | --- | --- | --- | --- |
| 10 meV / 0.5 A (current) | survives | survives | survives | survives |
| 20 meV / 0.5 A | survives | survives | survives | survives |
| **30 meV / 0.5 A** | **absorbed** | **absorbed** | survives | survives |
| 10 meV / 0.7 A | -- | -- | **absorbed** | **absorbed** |

30 meV -- the value the ~26 meV of within-basin scatter at `screen_fmax = 0.05`
argues for -- absorbs the winning pose at n = 1 and n = 2; 0.7 A absorbs it at
n = 3 and n = 4. Both directions fail, in complementary halves of the range.
Absorption is not proof the minimum would be lost (the absorbing
representative may relax to the same place, which only a re-run settles), but
there is no version of this that is free. **`screen_dedupe_tol_eV` stays at 10
meV and `screen_geom_tol_A` at 0.5 A.**

**Second, and larger: the refined `pool` is a resolution too.** Calibrating it
the way 0.11.0 calibrated `GEOM_TOL_A` -- all pairs of the deduped, fully
converged candidates -- the docked pools have no empty band at n = 3 or n = 4,
where the n = 2 sweep pool that set 0.15 A had a clean one (cluster at
0.01-0.12, gap, bulk from 0.19). And the count slides with the knob:

| n | pool span | median NN energy gap | @0.15 A | @0.20 | @0.30 | @0.50 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.82 kcal/mol | 2.34 meV | 7 | 5 | 4 | 4 |
| 2 | 2.83 | 0.18 meV | 151 | 126 | 93 | 63 |
| 3 | 1.90 | 0.21 meV | 233 | 183 | 134 | 95 |
| 4 | 1.48 | 0.11 meV | 320 | 289 | 239 | 184 |

Two things to read off it. The count is **saturated on the tight side** --
0.05 A gives the identical 233 and 320, because essentially no pair is closer
than 0.15 (30 of 27,028 at n = 3) -- so the pool is not being shattered by a
threshold that is too tight; every member really is that far from every other.
But it is **not stable on the loose side**, and there is no band to put a
threshold in, so 233-vs-134 is a choice rather than a measurement.

The energy column is the blunter point. At n >= 2 the energy axis does
nothing: **100% of nearest-neighbour gaps fall inside the 5 meV window**, so
the whole partition is geometric, and the entire n = 3 pool spans 82 meV --
**3.2 kT at 298 K**, and well inside GFN2's own error bar. These are 233
optimiser endpoints on a flat, glassy surface, not 233 chemically
distinguishable species. n = 1 is the one honest case: 7 minima over 2.82
kcal/mol with a 2.34 meV median gap.

**They are, however, genuinely different placements rather than jitter.** The
obvious worry with a chain that fixes a parent and varies one molecule is that
the inherited molecules relax slightly differently under each placement and
manufacture spurious minima. Decomposing each near-threshold pair's distance
per molecule under the winning assignment says otherwise: at n = 2 and n = 3,
**100%** of the pairs in 0.15-0.30 A differ in exactly one molecule (median
largest deviation 0.22-0.24 A, median second largest 0.015-0.030 A). Only at
n = 4 does jitter appear at all, and it is still a minority (82% one molecule,
18% two or more, median second largest 0.076 A).

**And the stated resolution is finer than the pipeline can deliver.** The
per-molecule decomposition above cannot see this, because "one molecule
differs by 0.22 A" is exactly what residual optimiser displacement on a soft
mode looks like. Measured by re-relaxing all 711 of these candidates down a
cumulative ladder of `fmax` rungs in their own scoring environment (the 2e-3
control takes zero steps and is the identity map, so there is no harness noise
floor to subtract):

| n | median displacement at 5e-4 | p90 | >= 0.15 A | `pool` at 2e-4 |
| --- | --- | --- | --- | --- |
| 1 | 0.047 A | 0.377 | 29% | 7 -> 6 (-14%) |
| 2 | 0.109 | 0.430 | 41% | 151 -> 98 (-35%) |
| 3 | 0.133 | 0.597 | 56% | 233 -> 128 (-45%) |
| 4 | 0.126 | 0.558 | 48% | 320 -> 239 (-25%) |

The median residual displacement at n >= 2 is 73-89% of `GEOM_TOL_A`, about
half of each pool moves further than the tolerance meant to separate distinct
minima, and 71 of 711 candidates were not near a minimum at all -- they fell
into a *different* basin, the deepest dropping 108 meV. The collapse is a
lower bound, since these candidates were already deduped at 0.15 A and so
start mutually that far apart. It is not a greedy-ordering artifact either:
re-deduping the original geometries with energies jittered by the observed
spread gives 147.9 +/- 1.5 / 228.4 +/- 1.9 / 317.7 +/- 1.3 basins at
n = 2/3/4, tens of standard deviations from the observed 98 / 128 / 239.
Curvature corroborates the scale without being able to predict it
per-structure: the softest mode the Hessian can resolve above its own ~4.4e-3
eV/A^2 numerical floor permits 0.21-0.29 A of displacement at `fmax = 0.002`,
and every unresolved mode is softer.

Two remedies were priced. Tightening `Scoring.fmax` to 5e-4 costs 1.5-2.7x on
the scoring optimisation and still does not reach a stable partition (n = 3
slides 134 -> 128 -> 117 across three tight rungs); widening `GEOM_TOL_A` to
cover p90 displacement (~0.5 A) would take n = 3 from 233 basins to ~50 and
destroy the diversity measure it is supposed to report. **Neither default
moved**, and `E_int(min)` never rose under any of it -- it falls by at most
0.014 kcal/mol, monotonically, at every n and every rung.

So the count is not a criterion artifact -- but "distinct minimum" here means
"optimiser endpoint at least 0.15 A away in contact space", never "separated
by a barrier", which nothing in this pipeline checks, and the endpoint itself
is only reproducible to ~0.11-0.13 A (median) or ~0.5 A (p90). **Read `pool`
as a diversity measure at a stated resolution, not as a count of species**,
and
compare it only between runs that share both tolerances -- which is what
recording `dedupe_tol_eV` / `geom_tol_A` in every summary is for. The number
that survives all of this untouched is `E_int(min)`: it is a set minimum
however the set is partitioned.

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
capacity, the same convention `n_sweep`'s `cover` column uses. Since 0.19.0
n >= 9 (`MONOLAYER_WARN_FRACTION` of chloroform's ~27-molecule monolayer on
pyrazine) actually *runs* in both generators, where it used to hit
`MAX_DESCRIPTOR_MOLECULES` in the dedupe -- so this warning is now the only
thing saying past-a-third-of-capacity is a bad idea. It always was a
statement about the chemistry (no single minimum dominates, solvent-solvent
cohesion takes over), never a former implementation limit that happened to
sit near the same n by coincidence.

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
- **Pooling docked and swept candidates together** -- not built: docking wins
  at every n by construction, so pooling would let it silently take over the
  headline number and erase the informative comparison between what each
  search actually finds. See `CLAUDE.md`'s invariant that the two are never
  pooled.
- **RMSD-based dedupe**, in retrospect -- listed here through 0.10.0, on the
  grounds that "energy dedupe already separates distinct minima at the counts
  docking produces," and reversed at 0.11.0 once a five-seed sweep was shown
  to produce four times those counts and the energy axis to be saturated at
  them. The bullet is gone from this list because the thing was built. What
  replaced it is deliberately *not* an RMSD -- a naive one would have been
  worse than the criterion it replaced -- for the reasons under "Geometric
  basin dedupe" above, which is also where the measurements that reversed the
  decision live.
- **Stratified packing itself**, in retrospect -- kept from 0.2.0 to 0.8.0,
  removed once `docking.py` took over minimum-finding. Recorded in full above
  ("Packing: independent draws, and why they were once stratified") rather
  than here, since it was built and then removed rather than merely
  considered.
- **Ranking the modal basin by `n_seeds_hit` instead of pooled `frame_share`**
  -- i.e. by how many independent packings visited it at all, the same
  denominator `found_by` uses, rather than by how much of the pooled
  trajectory time it holds. Not built: `n_seeds_hit` answers "how many
  searches stumbled into this basin," which is corroboration evidence, not
  occupancy -- it is already what `found_by` and Search convergence report,
  and a seed that spends 90% of its 10 ps in one basin should outrank three
  seeds that each glance off three different ones for a few frames apiece.
  `frame_share` is a property of the trajectory; `n_seeds_hit` a property of
  the search; conflating them would double another job into "which basin is
  modal." `n_seeds_hit` stays the tie-break (then energy), for the cases
  `frame_share` alone cannot separate.

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
- **A `pool` is a resolution-relative diversity measure, not a count of
  species, and nothing here checks for a barrier.** See "'Distinct minima' is
  a resolution, not a count": at n >= 2 the whole partition is geometric (100%
  of nearest-neighbour energy gaps sit inside the 5 meV window), a docked n = 3
  pool spans 3.2 kT, and its size slides from 233 to 95 as `geom_tol_A` goes
  0.15 -> 0.5 with no empty band to anchor a choice. It is saturated on the
  tight side and the members really are distinct placements rather than
  jitter, so it is not an artifact -- but `E_int(min)` is the number that is
  invariant to all of it, and two `pool`s are comparable only at identical
  tolerances.
- **The screening tolerances rest on a weaker argument than the scorer's.**
  The screen over-counts 5-6x at n = 2-3 at matched resolution, but the cause
  is `screen_fmax = 0.05`, not the thresholds, and tightening it was already
  measured and rejected on cost. Both proposed loosenings absorb the winning
  pose. So 10 meV / 0.5 A stay on a *negative* result, which is enough not to
  move them and not enough to call them optimal -- one system, one solvent,
  grid mode, one parent. `--dump-screen` plus `--screen-dedupe-tol` /
  `--screen-geom-tol` make re-opening it on another system post-processing
  rather than compute.
- MACE-OFF23 as a generator is **untested** here. It needs a model download,
  and it cannot share a process with tblite (see the file boundary above).
- GFN2 likely **over-binds** the C-H...N contact -- 1.91 A is short against a
  literature 2.2-2.5 A, so absolute magnitudes may be 1.5-2x too strong. Signs
  and orderings should be robust; ratios may not be.
