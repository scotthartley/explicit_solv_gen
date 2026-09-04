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

Conda env from `environment.yml` (packmol, ase, tblite, numpy; MACE via pip).
The `xtb` binary is deliberately *not* in it -- nothing here invokes it, and
its conda package constrains which tblite you get; install it separately if
you want the manual cross-check below. **packmol is installed into the env's
bin and is not on the default PATH**, so either activate the env or run with
it prepended:

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
| `docking.py` | a second, constructive generator: random placement + BFGS instead of thermal sampling. Standalone, either/or with `n_sweep.py`; see below. |
| `dft_export.py` | exports deduped, near-minimum candidates (from either generator) plus a manifest for a downstream DFT single point or reopt. |
| `report.py` | all text rendering, plus the ASE-free numeric helpers (`EV_TO_KCAL`, `boltzmann_weights`, `ensemble_energy`, `dedupe_energies`) that `ensemble` and `docking` re-export. No ASE/tblite at module scope. |
| `shell_capacity.py` | monolayer capacity, for choosing `n_solvent`. |

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
pin now, which is why the bump below is not optional.

### Versioning

`report.VERSION` is that package version. **Bump it, in the same commit, on
any change to the pipeline's numerics or output shapes** -- it is what
distinguishes two sweeps whose `Condition`s are identical but whose code was
not, and the bump is manual, so nothing catches a forgotten one. Doc-only and
refactor-only commits leave it alone.

| version | what changed |
| --- | --- |
| 0.6.0 | the params block of `sweep.json` / `dock.json` loses `git_commit`, and `report.git_commit` is gone. It was blank on every cluster run (no `.git` in the copied tree, or no `git` in the batch environment), so it pinned nothing where it mattered. Subtractive: an older sweep still carries the field and re-renders unchanged, since the params block is rendered key-by-key with nothing reading `git_commit` by name |
| 0.5.0 | the params block of `sweep.json` / `dock.json` gains `ase_version`, `numpy_version` and the calculator's own library (`tblite_version`, or `mace_torch_version` / `torch_version`), from installed package metadata. Additive: nothing reads them, so older sweeps re-render unchanged |
| 0.4.0 | `scored.json` gains `pack_mode` (`"md"` or `"dock"`); every run directory gains `ref_solute.xyz` / `ref_solvent.xyz`, the relaxed reference geometries the run's `E_int` was measured against. `docking.py` (a second, constructive generator) and `dft_export.py` (candidate export for DFT) are new. Existing sweeps re-render unchanged (only the version line moves); `pack_mode` defaults to `"md"` when absent so nothing pre-0.4.0 breaks |
| 0.3.1 | `report.txt` gained a Best geometry at each n section and a `best` marker in the per-packing detail table; the sweep directory gained one `best_n<N>.xyz` per n. `report.py` only, so 0.3.0 sweeps re-render |
| 0.1.0 | first numbered version; `report.VERSION` added and recorded in the params block |
| 0.1.1 | `report.txt` gained the `dE_int` column, so an 0.1.0 report has one fewer column and no increment. The docs' claim that `E_int(n)` plateaus was corrected in the same commit -- see below |
| 0.3.0 | the sweep table is per **n**, not per run: the candidates of every packing at one n are pooled and deduped across packings, and the reported number is the pooled minimum rather than a seed mean. `dE_int` is keyed by n alone; Seed spread becomes Search convergence; the old per-run rows are demoted to a Per-packing detail table. `report.py` only, so 0.2.0 sweeps re-render |
| 0.2.0 | stratified packing: the arrangement of the solvent now varies by design across the seeds at each n, and the seed budget is spread unevenly over n. Changes the packing for **every n >= 2 run**, so sweeps either side of it are not comparable. `report.txt` gained a `cover` column and `metadata.json` two fields |

## Artefacts

Per run directory: `packed.xyz`, `opt.log`, `traj.xyz`, `energies.json`,
`metadata.json`, `best.xyz`, `scored.json` -- and now

| file | written by | contents |
| --- | --- | --- |
| `run.log` | `solvate_md.run_one_job` | header (system, packing, Hamiltonian, pre-MD relaxation, MD settings), a streaming per-dump table, a footer |
| `scored.log` | `ensemble.assemble` | provenance, references, per-candidate table (including BFGS `steps`), result block. Named after `out_name`, so a second continuum gives `scored_acetone.json` / `.log` |
| `scored_candidates.xyz` | `ensemble.assemble` | every deduped candidate, not just the best, as a multi-frame xyz in the same order as `scored.json`'s `candidates` list -- frame *i* is `candidates[i]`. Named after `out_name` like `scored.log` |
| `ref_solute.xyz`, `ref_solvent.xyz` | `ensemble.assemble` | the relaxed reference geometries `E_int` for this run was measured against (`reference_energies` keeps only energies otherwise). Written into every run directory that shares one reference, redundant but cheap. `dft_export.export_dft` reads either copy to reconstruct `E(solute) + n E(solvent)` at the DFT level |
| `report.txt` | `n_sweep.run_sweep` | params block, the per-n `E_int(n)` table over the pooled candidates, a Best geometry at each n section naming the file behind each row, a per-packing detail table under it (with a `best` marker for the packings that reached the pooled minimum), and the sampling diagnostics below |
| `best_n<N>.xyz` | `n_sweep.run_sweep` | one file per n -- the pooled-minimum packing's `best.xyz` at that n, with `sweep_n=` / `sweep_E_int_kcal=` / `sweep_packing=` appended to its comment line. One file per n, not one multi-frame file, because the atom count changes with n and a viewer that reads a multi-frame xyz as a trajectory (Avogadro, VMD, most others) shows only the first frame. The deliverable of the run |

`metadata.json` also carries `pack_clustering` and `pack_directions` -- the
stratification parameter this packing was drawn at and the per-molecule
hemisphere axes it was constrained to, both `null` for an unstratified
packing and for `n < 2`. They are the one thing that deliberately differs
between the seeds at a given n, so they travel with the run rather than being
recoverable only by re-deriving them from the seed. `run.log`'s Packing block
prints both.

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
rather than re-reporting it. The same now applies to the 0.2.0 packing fields:
a `metadata.json` without `pack_directions`, or a `sweep.json` params block
without `monolayer_capacity`, raises rather than rendering a report that
silently omits a column.)

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

The `bias` column is not a curiosity: it is the per-molecule drift that stops
`E_int(n)` from plateauing, so it is worth measuring on a new solvent before
reading a sweep in it.

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
  table above. -0.9 kcal/mol for pyrazine...HCCl3, +6.2 for a water in
  ALPB(water): not fixed in sign, and a property of the solvent/continuum
  pair rather than of the chemistry being measured;
- **solvent-solvent cohesion** from n = 2 on, already in Known limitations
  below, since what `E_int` subtracts is *isolated* solvent molecules.

Add that `E_int` is a potential energy, with no entropic penalty for
condensing a molecule out of the continuum, and that `E_int(min)` is a running
minimum over a configuration space that grows with n, and the curve tends to a
line of nonzero slope rather than to a flat.

So `report.txt` carries a **`dE_int` column** -- `E_int(min)` at this n minus
`E_int(min)` at n - 1, over the **pooled** candidates of every packing at each
n. It is keyed by n alone. It used to pair by seed *index*, which paired
nothing physical: seed 0 at n = 2 is not seed 0 at n = 1 with a molecule
added, it is an independent packing, so the pairing had variance
`var(A) + var(B)` where pairing exists to cancel variance, and under
stratification it did not even hold the arrangement fixed, since
`c = (seed + 0.5) / n_seeds_at_this_n` and the seed count varies with n. It
also silently dropped every row whose index had no counterpart -- with
`allocate_seeds` giving n = 0 one packing, that was all but one row of the
table. Convergence is the increment settling to a **constant**, not to zero.

Only `E_int(min)` gets an increment. A second one on the ensemble average
would only invite reporting whichever of the two looks better.

**The seeds at one n are pooled, not averaged.** They are independent
*searches*, not replicas of a physical system: a mean over them penalizes
searching more widely, and it fights the stratification below head-on, since
that design exists precisely so that one packing in seven finds the
both-nitrogens geometry and the mean then dilutes that discovery by the six
that missed it. Measured on one sweep at n = 2: six packings find -12.12 and
one finds -11.60, and the mean of the minima is -12.05 -- a number no geometry
has. So the reported number at each n is the minimum over the pooled
candidates of every packing, and the pool is deduped across packings with the
same energy criterion `ensemble` uses within a run (`report.dedupe_energies`,
1 meV), because seven packings finding one basin would otherwise multiply its
Boltzmann weight by seven. `E_int(ens)` is pooled the same way and retained:
Boltzmann weighting at 298 K (kT = 0.594 kcal/mol) is a soft minimum rather
than a democratic average, and it is continuous in the candidate set where the
minimum is a step function.

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
`dissolved_fraction` are the remaining honest convergence indicators: the
monolayer capacity bounds them, so they saturate where `E_int` cannot.

A solute-free background leg, `E(n solvent) - n E(solvent)` in the same
continuum, would isolate the bias-plus-cohesion drift directly. Considered and
deliberately not built: the geometries are not comparable to the solvated
ones, and a difference between two real legs already cancels the same terms
without a second sweep to maintain.

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
| seeds | 3 | now the *average* per n, not the count at each: the total is `--seeds x len(--n)` and `allocate_seeds` spends it by coverage. See below |

Expect all of these to grow with the solute. Re-measure the decorrelation time
on your own system rather than carrying 0.55 ps over -- it is the one number
here that is a property of the system rather than of the integrator.

### Stratified packing: spend the seed budget on arrangement diversity

At n = 2 on pyrazine/chloroform the lowest geometry is almost certainly one
chloroform H-bonded to each nitrogen. The sweep finds it easily at n = 3 and
n = 4 and *not* at n = 2, and `report.txt` correctly said STILL FALLING.

The cause was the packing, not the MD. `pack_solvent` wrote one packmol block
for all n molecules, so every molecule was placed independently, and nothing
downstream rescues a bad draw: a CHCl3 bound to N at ~5.7 kcal/mol does not
detach, migrate around the ring and rebind at the far N within 10 ps of
gas-phase Langevin. **At n = 2 the starting geometry effectively is the
answer.** Brute force cannot fix it either -- `shell_capacity.py` puts
pyrazine's solvent-centre surface at 457 A^2, a full first shell at ~13
molecules, so n = 2 is 15% of a monolayer, n = 3 is 23%, n = 4 is 31%. Read as
13 sites, a random packing occupies both N sites 1.3% of the time at n = 2,
3.8% at n = 3, 7.7% at n = 4: ~176 packings for 90% confidence at n = 2. (Note
this also says the n = 3/4 successes are combinatorial rather than a coverage
effect -- at 31% of a monolayer nothing is forcing an even distribution
there.)

So stop drawing every seed from one distribution and stratify them over the
**degree of clustering** instead. `hemisphere_directions(n, c, rng)` returns n
unit vectors, and `pack_solvent` emits one `structure` block per molecule --
same ellipsoid, plus `over plane dx dy dz 0.`, i.e. the hemisphere around that
vector. `c = 0` puts the directions as far apart as they go (antipodal,
equilateral, tetrahedral at n = 2/3/4, from a Coulomb relaxation rather than a
table -- the golden spiral is not a max-min-separation set at tiny n), `c = 1`
collapses them all onto one random focus, and in between each is rotated that
fraction of the way toward it. The whole set is rigidly rotated by a random
rotation first, so no orientation relative to the solute is assumed; no site
is named and no chemistry is used. `run_job_grid` sets
`c = (seed + 0.5) / n_seeds_at_this_n`, so three packings give 0.17 / 0.50 /
0.83, and applies it only for `n_solvent >= 2` (at n = 1 there is no mutual
arrangement, and packmol already randomises the single site).

**This is not an assumption that the solvent spreads out.** The clustered end
is exactly how "both molecules on one face, sharing a Cl...Cl contact" gets
sampled -- a live arrangement at n = 2, and the same solvent-solvent cohesion
blamed above for `E_int`'s nonzero slope.

**It self-attenuates, which is why there is no coverage cutoff.** Two
hemispheres are disjoint (maximally restrictive, exactly where it is needed);
three at 120 deg overlap somewhat; four tetrahedral overlap so heavily a
molecule is barely confined; by n ~ 8 the constraint is vacuous. The fade
covers n = 2..6, the range where the problem is observed to disappear, and it
happens long before any solute approaches a monolayer.

Measured on pyrazine + 2 CHCl3, 24 packings per column, as the dot product
between the two solvent centroid directions -- so `opposed` is roughly the
both-nitrogens arrangement and `same face` its opposite:

| packing | mean dot | opposed (< -0.5) | same face (> +0.5) |
| --- | --- | --- | --- |
| unstratified | +0.64 | **4%** | 83% |
| c = 0.00 | -0.13 | **29%** | 21% |
| c = 0.17 | -0.09 | 29% | 8% |
| c = 0.50 | -0.12 | 21% | 8% |
| c = 0.83 | +0.10 | 12% | 17% |
| c = 1.00 | +0.19 | 8% | 38% |

Two things to read off it. First, **packmol's unstratified draw is far more
biased than the 13-site estimate above** -- 83% same-face against a 1.3%
site-combinatoric prediction for the opposed one -- so the n = 2 failure is
sharper than the combinatorics alone suggest, and the stratification is
correspondingly more valuable: opposed goes 4% -> 29%, and with the 4 packings
n = 2 now gets, the chance of at least one opposed packing goes from ~15% to
~70%. Second, **a hemisphere is soft and the gradient across `c` is gentler
than the idealised description** -- `c = 0` gives 29% opposed, not ~100%,
because a molecule may sit right against the dividing plane and point almost
anywhere from the solute centre. The constraint itself is satisfied every time
(the centroid's projection on its own axis is positive in every packing); it
is just a weaker statement than "on that face". Softness was the point -- it
never crowds packmol and needs no radius tuned against an anisotropic solute
-- but do not read `c` as a hard placement.

#### The seed budget is spread over n, not shared equally

The seeds are not all worth the same. At n = 0 there is no solvent to arrange,
so every packing is the same packing. Near a monolayer there is little freedom
in where the molecules go. The room to arrange, and so the value of another
packing, is largest at small nonzero n -- which is where the problem is. So
`n_sweep.allocate_seeds` spends a **fixed total** of `n_seeds * len(n_values)`
unevenly: n = 0 takes 1, the rest are weighted by `1 - min(n/capacity, 1)` and
handed out by largest remainder over a floor of 2 packings, so every n >= 1
keeps something to agree with it. Capacity comes from
`shell_capacity.monolayer_capacity`, computed once per sweep, and lands in the
params block as
`monolayer_capacity` alongside `seeds_per_n`.

    capacity 13, budget 3 x 4 = 12
      n=0 -> 1   (was 3)      n=2 -> 4   (was 3)
      n=1 -> 4   (was 3)      n=3 -> 3   (was 3)

Same compute, better spent. When the budget cannot meet the floor of 2 --
notably `--seeds 1` -- the split is abandoned rather than fudged and every n
gets `n_seeds`, so a single-packing sweep still reports no agreement and
says so.
`--seeds` is therefore the *average* packings per n now, not the count at
each. Nothing downstream assumes an equal count: `report.txt` aggregates by n
over the pooled candidates and prints how many packings each n got.

**Because the packings differ by design, their scatter is not an error bar.**
Independent seeds used to differ only in packing and velocities; they now
differ deliberately, so their spread measures the arrangement lottery as much
as the sampling noise. That is why `report.txt` reports **agreement** rather
than scatter: `found by` says how many of an n's packings reached the pooled
minimum, and that is the convergence evidence -- searches started from
different arrangements converging on one answer. Disagreement at small n means
the lottery has not been won often enough, and the fix is more packings, not
more steps. The spread is still printed, to be seen rather than to be
propagated.

`report.txt` also gained a **`cover` column**, `n / monolayer_capacity` as a
percent, which is what says whether a row is targeted microsolvation or a real
shell. It is read from `params` with no default, so a pre-0.2.0 sweep no
longer re-renders.

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

Both are computed from the candidate list already in `scored.json`, so
`python -m report <dir>` renders them for anything on disk -- no MD, no
calculator. They separate the two ways an under-sampled sweep fails, which
otherwise look alike in the table: a run still falling on the final frame
(too short) and a run whose minimum came from frame 2 and never moved (stuck
in the basin it was packed into).

## `docking.py` -- a second generator, beside the MD sweep

The sweep is a **geometry generator** for downstream ab initio refinement, not
a source of final GFN2 numbers. Judged against that goal it has one defect:
at n = 2 on pyrazine/chloroform it never produces the both-nitrogens
arrangement (one chloroform H-bonded to each ring nitrogen), so DFT never
sees it. A missing basin is the one failure downstream refinement cannot
repair -- DFT can re-rank what it is given, but it cannot invent a motif.

Measured, GFN2-xTB/ALPB(chcl3), against the same references already in
`s_test/pyrazine_chcl3_n2_seed0/scored.json`:

| structure | E_int (kcal/mol) | how found |
| --- | --- | --- |
| MD pooled minimum, n = 2 | -11.44 | 10 ps Langevin, 14 candidates, 3 packings |
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
raw placement). So neither report has to caveat the other -- they differ
only in how the starting geometry was found, not in what "found" means once
it is optimised:

- **MD sweep** -- thermal exploration. Gas-phase Langevin inside a confining
  wall; basin coverage is set by what the trajectory visits, which is why it
  missed the both-N basin at n = 2 above.
- **Docking** -- random construction, chained upward: n = 1's surviving
  parents become n = 2's starting points. Greedy (the best structure at n
  need not descend from the best at n - 1), which is why `Docking.n_parents`
  carries more than one candidate forward, and why the MD sweep remains
  available as an independent, non-greedy check on the same system.

Run one from the command line, the same shape as `n_sweep.py`:

    python docking.py examples/pyrazine.xyz examples/chloroform.xyz \
      --solvent chcl3 --n 1 2 3 --out pyrazine_dock/ --placements 64

It writes `<out>/dock.json` (`{"params": ..., "runs": [...]}`, matching
`sweep.json`'s shape) and `<out>/dock_report.txt`, plus one
`<label>_n<N>_dock/` run directory and one `best_n<N>.xyz` per requested n.
`python -m report <dir>` regenerates the report from `dock.json` alone, the
same as for a sweep. `n_sweep.py` and `solvate_md.py` are unmodified;
`docking.py` only reads from `solvate_md` (`solute_semi_axes`,
`shell_padding`, `align_to_principal_axes`, `_random_rotation`,
`bulk_molecular_volume`, `solvent_radius`, `pool_map`) and from `ensemble`
(`relax`, `reference_energies`, `solvent_molecule_gaps`) -- it adds no new
Hamiltonian and no new optimiser criterion of its own; `Scoring.fmax` /
`opt_steps` are the ones a candidate is finally relaxed to, so a docked and a
swept minimum are relaxed to the identical criterion and stay comparable.

`scored.json` for a docked run carries `pack_mode: "dock"` (an MD run's says
`"md"`) and every candidate's `wall_energy_eV` is `null` -- a docked
structure has no sampling frame and so no wall energy, a real absence, not a
zero, and `report.py`'s formatters render it as `-`.

**Staged optimisation, measured to cost nothing in quality.** Every
placement is first screened at a loose `screen_fmax` (default 0.05) and only
the best `n_refine` (default 10) are re-relaxed at the scorer's tight
`fmax = 0.002`. On the same 64 random placements (same RNG draw), two-pass
screen-then-refine landed at -6.6514 kcal/mol; refining all 64 at the tight
criterion directly landed at -6.6511 -- a 0.0003 kcal/mol difference, two
orders of magnitude under the 1 meV (0.023 kcal/mol) dedupe tolerance. The
two-pass run took 7.3 s combined over two n; refining every placement tightly
took 5.9 s for n = 1 alone. `--freeze-solute` (FixAtoms on the solute during
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

Not built: quasi-RRHO free energies (GFN2-level thermochemistry is triage at
best; DFT supersedes it), RMSD-based dedupe (energy dedupe already separates
distinct minima at the counts docking produces), and pooling docked and swept
candidates together (docking wins at every n by construction, so pooling
would let it silently take over the headline number and erase the
informative comparison between what each search actually finds).

### `dft_export.py`

    python -m dft_export pyrazine_dock/ --out pyrazine_dock/dft_export
    python -m dft_export pyrazine_chcl3/ --out pyrazine_chcl3/dft_export

Reads either a sweep or a docking output directory -- both write one
`scored.json` per run in the same shape, so this needs no branch on which
generator produced them. Per n: pool every run's candidates, dedupe at the
same 1 meV criterion `pool_by_n` uses, keep everything within `--window-kcal`
(default 3.0, ~5 kT) of that n's minimum. A window rather than a fixed count,
so the exported set adapts to the system; 3 kcal/mol comfortably spans the
1.56 kcal/mol gap between the two pyrazine/chloroform motifs above, which is
the concrete test that it is not too tight. `--max-per-n` is a safety cap
applied after the window.

Writes `manifest.json`, `references/{solute,solvent}.xyz` (the relaxed
geometries every exported `E_int` was measured against), and
`n<N>/cand<i>.xyz`. Verified end to end: reconstructing `E_int` from the
manifest's `energy_eV` and reference energies reproduces `interaction_kcal`
to floating-point precision (< 1e-9 eV), and the solute/solvent reference
energies from an MD sweep and a docking run on the same solute and solvent
matched exactly (-446.915092 / -443.246268 eV), confirming both generators
share one zero.

Raises rather than exporting if the runs found were scored against different
references -- the same invariant `pool_by_n` enforces, for the same reason.

Two caveats worth knowing rather than rediscovering: GFN2 over-binds the
C-H...N contact this pipeline studies (1.91 A against a literature 2.2-2.5),
so a DFT single point on a GFN2 geometry sits partway up a repulsive wall and
re-optimisation is preferable where affordable; and a raw MD frame must never
be exported for a single point, since 298 K thermal strain is worth several
kcal/mol at random and would swamp the signal -- `scored_candidates.xyz`
never contains one, only optimised candidates.

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
  `run_sweep` spawns twice, once per half. `docking.dock_at_n` goes through
  the same `pool_map` (twice per n: screening, then refinement), and
  `docking.main()` is guarded the same way, for the same reason -- killing a
  docking run needs the process-group kill above too, not `pkill -f
  docking.py`, which matches only the parent for the same reason it matches
  only `n_sweep.py`'s.
- Sampling is **gas phase by default**: `Condition.sample_in_continuum` is
  `False`, and scoring applies the continuum regardless. It was called
  `implicit_solvent` and defaulted to `True` -- the opposite of the documented
  design -- until this commit, and the `metadata.json` key changed with it, so
  older run directories carry `implicit_solvent` instead.
- `run_job_grid` takes `seeds_per_condition`, a sequence aligned with
  `conditions`, not an `n_seeds` scalar -- the budget is spread unevenly over
  n, and an int-or-sequence union would only hide which a caller meant. It is
  also where the stratification rule lives, because `c` depends on the seed
  count at that n and nothing else needs to know it. `run_one_job` and
  `pack_solvent` take `clustering` as a required argument, with no default,
  the same as every other packing parameter.
- The packing RNG is `np.random.default_rng(seed + 2_000_000)` -- a third
  independent stream alongside the initial velocities (`seed`) and the
  thermostat (`seed + 1_000_000`), and independent of packmol's own `seed`
  line, which still varies placement *within* the constraints.
- Packmol's `tolerance` (default 2.0) forbids hydrogen-bond contacts at t = 0
  (H...O/N sit at 1.8-2.0 A). Harmless here -- gas MD forms them within a few
  ps -- but it means a packed structure never starts bonded.
- `wall_slack` changed meaning at `c429f8a`. It is now measured in the wall's
  own metric and defaults to 0.25. Saved `Condition`s from before that behave
  differently. It is not just a margin: the wall volume sets the translational
  entropy of a dissociated solvent molecule, so a looser wall makes
  dissociation more favourable.
- **A huge stdout file means tblite 0.4.x.** `tblite-python` 0.4.0 ships
  debug `print`s in `library.get_post_processing_dict`, which runs *twice per
  single point* (`get_bond_orders` and `get_dipole` both call it), so every
  force evaluation dumps the full N x N bond-order matrix -- one sweep
  produced a 50 GB slurm file. 0.5.0, 0.6.0 and 0.7.0 are clean, checked by
  extracting the packages. This pipeline writes nothing of the sort: all real
  output goes to `run.log` / `scored.log` / JSON, and a whole sweep emits a
  couple of hundred bytes to stdout, so a large one is always the
  environment, never the code. `environment.yml` therefore carries a
  `tblite>=0.5` floor -- what used to select 0.4.x was `xtb`, since most
  linux-64 `xtb 6.7.1` builds require `tblite >=0.4.0,<0.5.0a0` and the
  newest caps it at 0.6.x. Which of the two a solver holds back is a
  tie-break, and it differed between machines.
- **The `dftd4<4` pin is gone, and the segfault behind it did not reproduce.**
  dftd4 4.2.0 was crashing GFN2 single points on the cluster -- SIGSEGV in
  libdftd4's OpenMP region, on a bare water molecule, even at
  `OMP_NUM_THREADS=1`, with GFN1/D3 unaffected -- which pinned dftd4 below 4
  and dragged tblite down with it (b36541c). Retested at tblite 0.7.0 with
  dftd4 4.2.0 `gfortran_*_1`: the same reproducer now runs clean on that
  cluster and locally, to identical energies (GFN1 -156.9675059, GFN2
  -137.9677759 eV on H2O). Whether the tblite upgrade or dftd4's `_1` rebuild
  fixed it was not isolated. If it returns, try `ulimit -s unlimited` and
  `OMP_STACKSIZE` first: a SIGSEGV in a Fortran OpenMP region that reproduces
  single-threaded is the signature of stack exhaustion, not necessarily a
  code bug.
- Don't pipe a long background run through `tail` -- it buffers until the pipe
  closes, so no interim output appears no matter what `flush=True` says. Tail
  the run's own `run.log` instead; it is flushed per line.
- `sweep.json` is `{"params": {...}, "runs": [...]}`, not the bare list it was
  before. Nothing reads the old shape any more.
- `best.xyz` is relaxed **in the scoring continuum, with no wall**, so a plain
  `xtb best.xyz --opt` optimizes it on a different surface and keeps moving it.
  Reproduce it with `xtb best.xyz --gfn 2 --alpb <solvent> --sp`; `scored.log`
  prints the exact command. The binary is not in this env (see Environment);
  `conda create -n xtb_check -c conda-forge xtb` when you want it. Two
  further traps, both checked rather than assumed: ASE's `fmax` (largest per-atom force, eV/A) and xtb's gradient norm
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
- Stratified packing improves the odds of the right arrangement; it does not
  guarantee it. At n = 2 the measured chance of at least one opposed packing
  in the 4 the reallocation buys is ~70%, not 100%. A run that still says
  STILL FALLING at small n is asking for more packings, not more steps.
- MACE-OFF23 as a generator is **untested** here. It needs a model download,
  and it cannot share a process with tblite (see the file boundary above).
- GFN2 likely **over-binds** the C-H...N contact -- 1.91 A is short against a
  literature 2.2-2.5 A, so absolute magnitudes may be 1.5-2x too strong. Signs
  and orderings should be robust; ratios may not be.
