# explicit_solv_gen

Explicit-solvent MD for cluster-continuum solvation studies.
Packmol -> ASE -> tblite (GFN2-xTB) or MACE-OFF23, with a confining wall.

`DESIGN.md` carries the measurements behind every default here and the record
of what was tried and removed; `CHANGELOG.md` carries the version table.
`README.md` tells the same design story at introduction length. This file is
the operating manual: environment, artefacts, invariants, gotchas.

## Before you change a default or re-add a feature

DESIGN.md carries the measurements behind every number here and the record of
what was tried and removed. Read the named section before you touch:

- any MD length, seed count, dump interval or `max_frames` -- "How much
  sampling? The defaults are production defaults now" (they are measured
  against a 0.55 ps shell decorrelation time, not guessed)
- per-molecule packing constraints -- "Packing: independent draws, and why
  they were once stratified" (stratified packing existed from 0.2.0 to 0.8.0
  and was removed deliberately; do not reinvent it)
- `E_int` plateau reasoning, or adding a second increment column --
  "`E_int(n)` is what makes 'how much explicit solvent?' a measurement"
- docking's `n_refine` / `n_placements` -- "`docking.py` -- a second
  generator, beside the MD sweep" (`1 - 0.93^K` sets the default; `n_refine`
  is per parent and per distinct screened basin since 0.9.0)
- a solute-free background leg, RMSD dedupe, quasi-RRHO, overlapping the two
  halves of a sweep, or pooling docked and swept candidates -- "Considered
  and not built"

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
      --solvent chcl3 --n 0 1 2 3 --out pyrazine_chcl3/ --seeds 5

`n_sweep.py --help` lists the rest; every flag maps onto a field of
`Condition` or of `Scoring`, and takes its default from there. That is the
entry point -- there is no driver script, and `solvate_md.py` has no
`__main__` smoke test of its own. A fast end-to-end check is the same command
with `--n 2 --seeds 1 --steps 6000 --equilibrate 2000 --dump-interval 20
--max-frames 20`.

## Layout

| file | role |
| --- | --- |
| `single_thread.py` | pins the numeric stack to one thread per process. Must be imported **before numpy**; see its docstring. |
| `solvate_md.py` | packing + MD. The **generator**. |
| `ensemble.py` | rescoring optimized frames in a continuum. The **scorer**. |
| `n_sweep.py` | `E_int(n)` sweep for **one solute in one solvent**: an independently drawn diversity check, and the only source of basin occupancy. |
| `docking.py` | a second, constructive generator: random placement + BFGS instead of thermal sampling. Standalone, either/or with `n_sweep.py`; see below. |
| `dft_export.py` | exports deduped, near-minimum candidates (from either generator) plus a manifest for a downstream DFT single point or reopt. |
| `report.py` | all text rendering, plus the ASE-free numeric helpers (`EV_TO_KCAL`, `boltzmann_weights`, `ensemble_energy`, `dedupe_energies`, `dedupe_groups`) that `ensemble` and `docking` re-export. No ASE/tblite at module scope. |
| `shell_capacity.py` | monolayer capacity, for choosing `n_solvent`. |

Every sweep records a **params block** -- the whole `Condition` (via
`asdict`), `report.VERSION`, a timestamp, and the versions of the libraries
the numbers came out of (`report.library_versions`). This exists because
`run_sweep` deliberately does *not* assemble a difference of sweeps itself:
one sweep is one solute in one solvent, and a double difference such as
`(A - B)_solvent1 - (A - B)_solvent2` is four sweeps subtracted by hand. So
**check the params block before subtracting two sweeps** -- `wall_slack`
silently changed meaning at `c429f8a`, and a params block no longer carries
`git_commit` (removed at 0.6.0; see DESIGN.md's "The params block, and what
it used to record" for why).

### Versioning

`report.VERSION` is that package version. **Bump it, in the same commit, on
any change to the pipeline's numerics or output shapes** -- it is what
distinguishes two sweeps whose `Condition`s are identical but whose code was
not, and the bump is manual, so nothing catches a forgotten one. Doc-only and
refactor-only commits leave it alone.

The full version history is in `CHANGELOG.md`.

## Artefacts

Per run directory: `packed.xyz`, `opt.log`, `traj.xyz`, `energies.json`,
`metadata.json`, `best.xyz`, `scored.json` -- and now

| file | written by | contents |
| --- | --- | --- |
| `run.log` | `solvate_md.run_one_job` | header (system, packing, Hamiltonian, pre-MD relaxation, MD settings), a streaming per-dump table, a footer |
| `scored.log` | `ensemble.assemble` | provenance, references, per-candidate table (including BFGS `steps`), result block. Named after `out_name`, so a second continuum gives `scored_acetone.json` / `.log` |
| `scored_candidates.xyz` | `ensemble.assemble` | every deduped candidate, not just the best, as a multi-frame xyz in the same order as `scored.json`'s `candidates` list -- frame *i* is `candidates[i]`. Named after `out_name` like `scored.log` |
| `ref_solute.xyz`, `ref_solvent.xyz` | `ensemble.assemble` | the relaxed reference geometries `E_int` for this run was measured against (`reference_energies` keeps only energies otherwise). Written into every run directory that shares one reference, redundant but cheap. `dft_export.export_dft` reads either copy to reconstruct `E(solute) + n E(solvent)` at the DFT level |
| `report.txt` | `n_sweep.run_sweep` | params block, the per-n `E_int(n)` table over the pooled candidates, a Best geometry at each n section naming the file behind each row, a per-packing detail table under it (with a `best` marker for the packings that reached the pooled minimum), a Basin occupancy section (see below), and the two diagnostics below. Each table ends with a one-line column key and points at README's "Reading report.txt", which is where the explanatory prose lives -- once, rather than in a docstring, in every rendered report, and here |
| `best_n<N>.xyz` | `n_sweep.run_sweep` | one file per n -- the pooled-minimum packing's `best.xyz` at that n, with `sweep_n=` / `sweep_E_int_kcal=` / `sweep_packing=` appended to its comment line. One file per n, not one multi-frame file, because the atom count changes with n and a viewer that reads a multi-frame xyz as a trajectory (Avogadro, VMD, most others) shows only the first frame. The deliverable of the run |

`run.log` is flushed on every line, so a long run can be followed with
`tail -f` instead of going silent until it exits. **The footer surfaces the
wall diagnostic** -- wall-active fraction and max wall energy -- and warns
above 20%, rather than leaving it buried in a 74 KB `energies.json` array.
The same diagnostic now travels with the numbers it qualifies: a `wall` column
and a **Wall diagnostic** section in `report.txt`, and in `scored.log` a
provenance line plus a per-candidate `E_wall/eV` -- the wall energy of the
sampling frame each candidate came from. `scored.json` carries it as
`sampling_wall` / `n_scored_wall_active`.

**Nothing reads a `scored.json` or a run directory older than the current
scorer.** `run_one_job` always writes both JSON files and the scorer always
writes every field, so a missing one is a broken run rather than an old one,
and reading it fails loudly instead of silently defaulting the field.

Regenerate any of these from the JSON already on disk, no MD and no calculator:

    python -m report /path/to/sweep_or_run_dir/

(Anything scored before `n_opt_steps` and the mandatory wall fields no longer
re-renders, since the readers no longer default a missing field. Rescore it
rather than re-reporting it. The same applies to a `sweep.json` params block
without `monolayer_capacity`: it raises rather than rendering a report that
silently omits the `cover` column.)

`E_int(min)` is the reported number, because it alone is comparable across n,
and at 0.8.0 it is the only energy any table prints. `E_int(ens)` and
`E(cluster)` are still computed and still in `scored.json` / `sweep.json` --
within a run `E_int` and `E(cluster)` differ only by the constant
`E(solute) + n E(solvent)`, and `boltzmann_weights` subtracts the minimum
before exponentiating, so the two ensemble averages carry identical weights by
construction -- but they are no longer rendered. Two averages side by side
invite reporting whichever looks better, which is the same objection DESIGN.md
raises against giving `E_int(ens)` its own `dE_int`; and `E(cluster)`
is not comparable across rows of different n, which is what `E_int` exists to
fix. Absolute energies are in eV (ASE's native unit), `E_int` in kcal/mol; the
mix is deliberate.

An MD candidate in `scored.json` also carries `n_frames` and `frames` -- how
many scored frames quenched into that minimum, and their sampling dump
indices -- and each run's summary carries `scored_frame_spacing_fs`,
`occupancy_mean_contacts`, `occupancy_dissolved_fraction`. See DESIGN.md's
"Basin occupancy: what quenching throws away" for why these exist and what
they are quarantined from. A docked candidate's are `null`: see DESIGN.md's
`docking.py` section for why a constructed minimum has no occupancy to
report.

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

## Generators: `docking.py` and `dft_export.py`

`docking.py` is a second, constructive generator -- random placement + BFGS,
not thermal sampling -- standalone and either/or with `n_sweep.py`. It owns
minimum-finding; the MD sweep is the independently-drawn, non-greedy check
beside it and the only source of basin occupancy. Full rationale, the
both-nitrogens measurement, the staged screen-then-refine cost, and the
applicability limits are in DESIGN.md's `docking.py` section.

    python docking.py examples/pyrazine.xyz examples/chloroform.xyz \
      --solvent chcl3 --n 1 2 3 --out pyrazine_dock/ --placements 64

Writes `<out>/dock.json` (`{"params": ..., "runs": [...]}`, matching
`sweep.json`'s shape) and `<out>/dock_report.txt`, plus one
`<label>_n<N>_dock/` run directory and one `best_n<N>.xyz` per requested n.
`python -m report <dir>` regenerates the report from `dock.json` alone, the
same as for a sweep. `n_sweep.py` and `solvate_md.py` are unmodified;
`docking.py` only reads from them, adding no new Hamiltonian and no new
optimiser criterion of its own -- `Scoring.fmax` / `opt_steps` are the ones a
candidate is finally relaxed to, so a docked and a swept minimum are relaxed
to the identical criterion and stay comparable.

`scored.json` for a docked run carries `pack_mode: "dock"` (an MD run's says
`"md"`) and every candidate's `wall_energy_eV` is `null` -- a real absence,
not a zero, and `report.py`'s formatters render it as `-`.

`dft_export.py` exports deduped, near-minimum candidates from either
generator, plus a manifest, for a downstream DFT single point or reopt:

    python -m dft_export pyrazine_dock/ --out pyrazine_dock/dft_export
    python -m dft_export pyrazine_chcl3/ --out pyrazine_chcl3/dft_export

Reads either a sweep or a docking output directory -- both write one
`scored.json` per run in the same shape, so this needs no branch on which
generator produced them. Per n: pool every run's candidates, dedupe at the
same 1 meV criterion `pool_by_n` uses, keep everything within `--window-kcal`
(default 3.0, ~5 kT) of that n's minimum; `--max-per-n` is a safety cap
applied after the window. Writes `manifest.json`,
`references/{solute,solvent}.xyz` (the relaxed geometries every exported
`E_int` was measured against), and `n<N>/cand<i>.xyz`. Verified end to end
against the manifest and against both generators sharing one reference zero
-- see DESIGN.md's `docking.py` section for the numbers.

Two DFT caveats worth knowing rather than rediscovering: GFN2 over-binds the
C-H...N contact this pipeline studies (1.91 A against a literature 2.2-2.5),
so a DFT single point on a GFN2 geometry sits partway up a repulsive wall and
re-optimisation is preferable where affordable; and a raw MD frame must never
be exported for a single point -- `scored_candidates.xyz` never contains one,
only optimised candidates.

**Two invariants, checked in the code rather than merely documented:**
`pool_by_n` and `dft_export` both **raise** rather than pooling/exporting if
the runs found were scored against different references -- a sweep or export
whose rows share no common zero is broken, not old. And **docked and swept
candidates are never pooled together** in one table: docking wins at every n
by construction, so pooling would let it silently take over the headline
number and erase the comparison between what each search actually finds (see
DESIGN.md's "Considered and not built").

## Gotchas

- Scoring settings live on `Scoring` and **only** on `Scoring`, the same way
  sampling settings live on `Condition`. `run_sweep` takes a
  `condition_kwargs` dict and a `Scoring` and names no field of either, and
  the CLI reads every default off whichever dataclass owns it via
  `dataclass_default`. Internal functions -- `pack_solvent`, `relax`,
  `select_frames` and the rest -- carry no defaults at all, so a caller has to
  say what it means. `pack_solvent` used to default `wall_slack = 1.0` against
  `Condition`'s 0.25, and a since-deleted serial `score_run` defaulted
  `stride = 10, max_frames = 40` against the CLI's 1 and 50.
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
  `conditions`, not an `n_seeds` scalar -- n = 0 gets one packing where every
  other n gets `n_seeds`, and an int-or-sequence union would only hide which
  a caller meant.
- Two independent RNG streams per run: the initial velocities (`seed`) and
  the thermostat (`seed + 1_000_000`), both independent of packmol's own
  `seed` line, which is what varies the placement. There was a third,
  `seed + 2_000_000`, for the stratified hemisphere directions; it went with
  them at 0.8.0.
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
