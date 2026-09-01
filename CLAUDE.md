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
      ~/opt/miniforge3/envs/solvate_md/bin/python driver.py

## Layout

| file | role |
| --- | --- |
| `solvate_md.py` | packing + MD. The **generator**. |
| `ensemble.py` | rescoring optimised frames in a continuum. The **scorer**. |
| `n_sweep.py` | `E_int(n)` sweep and double-difference assembly. |
| `shell_capacity.py` | monolayer capacity, for choosing `n_solvent`. |

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

## Gotchas

- `run_job_grid` uses spawn, so **every driver script needs an
  `if __name__ == "__main__":` guard** or workers re-run the grid on import
  and fork until the machine dies.
- Packmol's `tolerance` (default 2.0) forbids hydrogen-bond contacts at t = 0
  (H...O/N sit at 1.8-2.0 A). Harmless here -- gas MD forms them within a few
  ps -- but it means a packed structure never starts bonded.
- `wall_slack` changed meaning at `c429f8a`. It is now measured in the wall's
  own metric and defaults to 0.25. Saved `Condition`s from before that behave
  differently. It is not just a margin: the wall volume sets the translational
  entropy of a dissociated solvent molecule, so a looser wall makes
  dissociation more favourable.
- Don't pipe a long background run through `tail` -- it buffers until the pipe
  closes, so no interim output appears no matter what `flush=True` says.
- `resources/` is gitignored. `resources/handoff_md_pipeline_claude_code.md`
  contains a struck-through paragraph saying not to use ALPB with an explicit
  shell. **That advice is wrong** and was corrected in place on 2026-09-01.

## State

Validated: pyrazine, n = 0..3, gas sampling scored in each continuum.
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
