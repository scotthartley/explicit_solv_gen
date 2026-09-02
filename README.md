# explicit_solv_gen

Explicit-solvent MD for cluster–continuum solvation studies, and a way to
answer the question that method always raises: **how many explicit solvent
molecules do you actually need?**

Packmol packs a small solvent shell around a solute, ASE runs Langevin MD
under a confining wall with tblite (GFN2-xTB) or MACE-OFF23, and the resulting
frames are re-optimised in an ALPB continuum. The output is an interaction
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
construction. If explicit solvent is capturing something the continuum cannot,
`E_int` departs from zero and then plateaus — the departure is exactly *what
the continuum was missing*, and the plateau is the answer to "how much is
enough". If it never departs, the continuum was already sufficient.

Two properties make this work. `E_int` is comparable across `n`, because whole
solvent molecules are subtracted off. And a solvent molecule that optimises
away into the continuum contributes ≈ 0, so a run whose shell dissolves lands
back on the `n = 0` answer rather than on an arbitrary offset. "The shell
dissolved" and "there was no explicit shell" agree, which makes dissolution a
usable null result instead of a failure mode.

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
leg                    n   E_int(ens)   E_int(min)   E(cluster)/eV  contacts  dissolved  uniq
--------------------------------------------------------------------------------------------
pyrazine/chcl3         0         0.02        -0.01     -446.914347      0.00       100%     3
pyrazine/chcl3         1        -5.59        -5.93     -890.403421      0.92         8%    12
pyrazine/chcl3         2        -6.62        -7.00    -1333.694375      0.58        50%    12
pyrazine/chcl3         3       -12.47       -12.77    -1777.194312      1.79         0%    14
```

`E_int` is in kcal/mol and absolute energies in eV (ASE's native unit); the
mix is deliberate. `E_int(ens)` is Boltzmann-averaged over the unique minima,
`E_int(min)` is the lowest. `E(cluster)` is shown beside them to give the
large numbers the small differences came from — it is *not* comparable across
rows of different `n`, which is precisely what `E_int` is for.

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

**Seed spread.** Independent seeds differ only in packing and initial
velocities, so anything they disagree about is sampling error rather than
chemistry. This is the **only error bar the pipeline produces**, and a
single-seed sweep says so in place of the table rather than leaving the
absence to read as precision. A difference between two sweeps has to clear
that spread; a double difference of four inherits it four times.

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
expensive half — fans out **per candidate optimisation** rather than per run
directory, because run directories are badly unequal in cost and would leave
cores idle with a long tail. The two references go into the same pool, at the
front, computed once per distinct (solute, solvent, calculator, continuum) so
that every row of a table is measured against the same zero.

Measured on six run directories — 62 optimisations on 18 cores — as mean wall
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
| `ensemble.py` | re-optimising and scoring frames in a continuum — the **scorer** |
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
- Packmol's `tolerance` (default 2.0 Å) forbids hydrogen-bond contacts at
  *t* = 0, since H···O/N sit at 1.8–2.0 Å. Harmless — gas-phase MD forms them
  within a few ps — but a packed structure never starts bonded.
- Sampling is **gas phase by default** (`Condition.sample_in_continuum =
  False`); scoring applies the continuum regardless.
- `best.xyz` is relaxed in the scoring continuum with **no wall**, so plain
  `xtb best.xyz --opt` optimises it on a different surface. Reproduce it with
  `xtb best.xyz --gfn 2 --alpb <solvent> --sp`; `scored.log` prints the exact
  command.
- ASE's `fmax` (largest per-atom force, eV/Å) and xtb's gradient norm (all
  components, Eh/a₀) are different criteria and are not comparable by eye, so
  both are recorded per candidate.
- Every term of `E_int` must be relaxed *to convergence*, not merely to a
  stationary-ish geometry. The scoring optimiser runs to `fmax = 0.002` eV/Å
  for this reason; at 0.05 a frame is left hanging on whichever soft mode it
  was descending, which on one test case put the reported minimum 0.58 kcal/mol
  above the true one — and that residual does not cancel between solvents or
  between conformers.

## Limitations

- `E_int` **conflates solute–solvent with solvent–solvent** at `n ≥ 2`: two
  solvent molecules binding each other counts as solvation. It largely cancels
  in a difference taken at the same solvent and the same `n`, but it does
  contaminate reading convergence in `n` directly.
- Semi-empirical Hamiltonians can over-bind close contacts, so absolute
  magnitudes may be too strong even where signs and orderings are robust.
  Check your interaction against a higher level of theory before quoting a
  ratio.
- MACE-OFF23 as a generator is supported but untested here; it needs a model
  download and cannot share a process with tblite.

## License

MIT — see [LICENSE](LICENSE).
