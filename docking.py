"""A second geometry generator: construct microsolvated clusters, don't sample them.

`n_sweep.py` explores by thermal sampling and quenches whatever basin the
trajectory happened to visit. That is the right tool for "what does this
system actually do at 298 K", but as a *generator* for downstream DFT
refinement it has a specific failure: it cannot report a basin the trajectory
never visited, and a missing basin is the one thing no amount of downstream
refinement repairs -- DFT can re-rank the candidates it is handed, but it
cannot invent one.

This module finds basins by constructing instead: place a solvent molecule at
a position and orientation around the (already relaxed) parent cluster --
scanned systematically over the parent's solvent-accessible surface by
default (`place_mode="grid"`), or, at `place_mode="random"`, drawn at random
-- and optimise. BFGS only descends, so it cannot climb out of the well
it lands in -- which is exactly why it works where a *seeded* MD run would
not: `run_one_job` discards 5 ps of equilibration before recording the first
frame, so a seeded arrangement would already be gone by the time anything was
written. Every hit outranks every miss on energy, so a basin sorts to the top
on its own without ever needing to be suspected.

**Docking owns minimum-finding**, and the MD sweep keeps the two jobs docking
cannot do -- an independently drawn, non-greedy check, and basin occupancy,
which a constructed minimum has no sense of at all (`_assemble_dock_n` writes
`n_frames` / `frames` as `None` for exactly that reason). DESIGN.md's
`docking.py` section carries the measurements behind all of that: the
both-nitrogens result on pyrazine + 2 chloroform, the staged
screen-then-refine cost, the `1 - 0.93^K` placement-count argument, and where
this stops working. Not repeated here.

`n_sweep.py` and `solvate_md.py` are unmodified by this; it only reads from
them, adds no Hamiltonian of its own, and relaxes its candidates to
`Scoring.fmax` -- the criterion the scorer uses -- so a docked and a swept
minimum stay comparable.

Run one from the command line:

    python docking.py examples/pyrazine.xyz examples/chloroform.xyz \
      --solvent chcl3 --n 1 2 3 --out pyrazine_grid/
    python docking.py examples/pyrazine.xyz examples/chloroform.xyz \
      --solvent chcl3 --n 1 2 3 --out pyrazine_dock/ --place-mode random

`dock_at_n` fans placements out through `solvate_md.pool_map`, which uses
spawn, so a script calling `run_docking` itself MUST guard the call:

    if __name__ == "__main__":
        run_docking(...)

`main()` below is under such a guard already.
"""

import single_thread  # noqa: F401  -- must precede numpy; see its docstring

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from ase.io import read, write

from ensemble import (
    CONTACT_GAP_A,
    Candidate,
    Scoring,
    contact_descriptor,
    reference_energies,
    relax,
    solvent_molecule_gaps,
    summarise,
)
from report import (
    DEDUPE_TOL_EV,
    EV_TO_KCAL,
    GEOM_TOL_A,
    LADDER_N,
    VERSION,
    dedupe_energies,
    format_report,
    library_versions,
    pool_by_n,
    same_basin,
    timestamp,
    write_best_geometries,
)
from shell_capacity import monolayer_capacity, surface_points
from solvate_md import (
    _random_rotation,
    _vdw_volume,
    align_to_principal_axes,
    bulk_molecular_volume,
    pool_map,
    shell_padding,
    solute_semi_axes,
    solvent_radius,
)


@dataclass
class Docking:
    """Everything that shapes a docking run, owned here and only here.

    `Scoring` still owns the final optimiser criterion (`fmax`, `opt_steps`)
    and the Boltzmann temperature -- docking and the MD sweep relax to the
    identical criterion, so their minima are comparable -- and this
    dataclass owns everything specific to *construction*: how many random
    poses to try, how many to carry forward, and the loose screening pass
    that makes trying dozens of them affordable.
    """

    # Random placements per parent per n, read only under `place_mode =
    # "random"`. 1 - 0.93^K gives 90% confidence of hitting a basin found in
    # 7% of random poses (the measured both-N rate on pyrazine/chloroform) at
    # K = 32 and 99% at K = 64.
    n_placements: int = 64
    # How the poses are generated. "grid" is the default since 0.17.0:
    # positions from the parent's solvent-accessible surface, crossed with
    # `n_orientations` quasi-uniform rotations -- which buys coverage rather
    # than confidence: nothing in a random run's output distinguishes "this
    # basin does not exist" from "we did not draw it", and a missing basin is
    # the one thing downstream DFT cannot repair. DESIGN.md's "Systematic
    # placement" measured grid lower at every n and 3x more likely to hit the
    # both-nitrogens basin, at ~11x the wall-clock of `"random"` -- still
    # reachable at `--place-mode random` for a cheap first look or a rerun of
    # an older sweep's conditions. `n_placements` is ignored in grid mode; a
    # rendered report omits it there rather than asserting a setting that did
    # nothing (`report.inert_params`), though `dock.json`'s own `asdict`
    # still carries it, live or not.
    place_mode: str = "grid"
    # Grid mode only. Surface points are voxel-downsampled to roughly this
    # separation, so each position stands for ~`grid_spacing_A**2` of
    # surface; 2.0 A puts ~220 positions on the outermost shell of pyrazine.
    grid_spacing_A: float = 2.0
    # Grid mode only, and the reason the grid is radial rather than a single
    # surface. Each entry is a probe radius as a fraction of the solvent's
    # bulk-density sphere radius, and each generates its own shell of
    # positions. 1.0 is the solvent-*centre* surface `shell_capacity.sasa`
    # measures -- the right contact distance for a sphere, and measurably the
    # wrong one for a directional contact: a real C-H...N puts the chloroform
    # centroid 3.17 A from the nearest pyrazine atom where that shell puts it
    # at 4.4-4.9 A, and a scan of that shell alone found the both-nitrogens
    # basin in 0 of 2664 poses against 13-27 of a few hundred at 0.4-0.6.
    # The outer shell is no longer argued for but measured, in two solvents,
    # and it is **inert**: dropping it leaves chloroform identical at every n
    # (E_int(min), found-by and pool alike) at one parent and within 0.006
    # kcal/mol at three, and moves acetone only inside its own selection
    # noise. Its poses do not relax into contact at the loose screen, so they
    # rank below every contact pose and are refined only when a parent has
    # fewer contact basins than `n_refine`. It stays because it is nearly
    # free -- it is 45-70% of the poses, but an outer pose costs ~4x less
    # than an inner one, so dropping it saved 14% of the wall-clock -- and
    # because the one case that would justify it, a concave solute whose
    # pockets the inner shells reject outright, is exactly what neither
    # solvent tested. See DESIGN.md's "Systematic placement" for both tables,
    # and for why a new solvent wants a per-shell survival count before these
    # fractions are trusted.
    grid_probe_fracs: tuple = (0.4, 0.7, 1.0)
    # Grid mode only. Quasi-uniform rotations per position, from a
    # super-Fibonacci spiral over SO(3). No symmetry detection and no
    # per-solvent table -- BFGS absorbs the residual misorientation.
    n_orientations: int = 12
    # Top-m deduped minima carried forward as next n's parents. The chain is
    # greedy -- the best structure at n need not descend from the best at
    # n - 1 -- and this is the mitigation.
    n_parents: int = 3
    # Which of a parent's distinct screened basins (see
    # `screen_dedupe_tol_eV`) are re-relaxed at the scorer's tight fmax:
    # every one within `refine_window_kcal` of that parent's own screened
    # minimum, best-first, up to `n_refine` of them. **Both are per parent**,
    # so a parent whose placements all screened a little higher than
    # another's still gets its own basins refined -- that is what keeps the
    # chain's `n_parents` lineages alive into the next generation.
    #
    # The window is the selector and `n_refine` is a cost cap, which is the
    # opposite of how this worked until 0.13.0: `n_refine` was a flat top-10
    # off the screened ranking, and **the screened ranking does not predict
    # the refined one**. Measured, acetone n = 1 grid mode, refining all 410
    # screened basins instead of the top ten: the best refined basin is the
    # 247th by screened energy, 1.50 kcal/mol above the screened minimum, and
    # E_int(min) is -4.608 against top-10's -4.541. Raising the flat cap does
    # not reach it either (top-30 and top-50 both stall at -4.54), and
    # neither does a tighter screen -- at `screen_fmax` 0.02 and 0.01 the
    # winner still sits at rank 283 of 323 and 155 of 230, for 3.7x and 8.5x
    # the screening cost. Hence a window wide enough to cover that scatter
    # rather than a rank cut. See DESIGN.md's "The screen-to-refine handoff".
    #
    # 3.0 kcal/mol is `dft_export`'s window and about 5 kT, and it covers the
    # measured winner at every screen tightness tried (+1.50, +1.82, +1.17).
    # It is deliberately generous: at `screen_fmax = 0.05` a screened
    # geometry's energy is worth little, so this excludes only what is
    # plainly unbound -- which is the whole of its job.
    refine_window_kcal: float = 3.0
    # The cap, raised from 10 with the window above, and only a cost guard:
    # the window is what selects. **This moves the random path too**, worth
    # stating (random was the default when this was measured, and stays
    # reachable at `--place-mode random`) because the opposite is easy to
    # assume: 64 random placements do not collapse into
    # a handful of basins but into 40-63 distinct ones per parent (measured,
    # acetone and chloroform, n = 1 to 3), so the old flat top-10 was
    # refining about a fifth of them there as well. The cap does not bind in
    # random mode; the window does the work, and it is nearly all of them --
    # a random parent's whole screened spread is 1.5-3.2 kcal/mol.
    #
    # Nothing the old cut refined is dropped, at either mode: the window
    # admits a prefix of the same energy-ordered representatives and the cap
    # is well above 10, so the refined set is a strict superset of the old
    # one for a fixed parent, and `E_int(min)` at that n can only fall or
    # stay. A chain number that moves the wrong way is the greedy chain
    # reacting to a better parent -- the `n_parents` caveat -- not a lost
    # basin.
    n_refine: int = 400
    # Two screened placements this close in energy *and* in contact
    # descriptor are one basin for the purpose of choosing what to refine.
    # Both are deliberately looser than the scorer's `report.DEDUPE_TOL_EV` /
    # `report.GEOM_TOL_A`, because a screened geometry is relaxed only to
    # `screen_fmax`: at 0.05 eV/A two placements in the same basin can still
    # differ by up to ~0.6 kcal/mol in energy and by far more than the
    # scorer's 0.15 A in geometry, so a criterion tuned for converged minima
    # would shatter one screened basin into a dozen near-copies and spend
    # every `n_refine` slot on them -- the exact starvation 0.9.0's per-parent
    # refinement budget exists to prevent. Both still sit comfortably inside
    # that scatter, so the pass errs toward refining a duplicate rather than
    # dropping a basin. The screened energies never reach a `scored.json`, so
    # neither does this criterion; the refined candidates are deduped at the
    # scorer's like everything else.
    screen_dedupe_tol_eV: float = 1e-2
    screen_geom_tol_A: float = 0.5
    # Loose first pass so paying for `n_placements` per parent is cheap;
    # DESIGN.md measures 1.3 s vs 4.5 s per candidate at 0.05 vs 0.002 on the
    # pyrazine + 2 chloroform system.
    screen_fmax: float = 0.05
    # FixAtoms on the solute during screening only -- an approximation, off
    # by default. A large aromatic solute's soft modes make BFGS crawl
    # without being relevant to where a solvent molecule binds; refinement is
    # always unconstrained regardless of this flag.
    freeze_solute: bool = False
    # Same meaning as Condition.shell_fill: how full the placement region is
    # at bulk density for n_solvent molecules.
    shell_fill: float = 0.5
    # Minimum interatomic distance at placement, packmol's `tolerance` in the
    # same units and for the same reason -- and the same consequence: it
    # forbids an H-bond distance at t = 0, so BFGS forms the contact itself.
    tolerance: float = 2.0
    solvent: str = "chcl3"
    calculator: str = "gfn2-xtb"
    calculator_kwargs: dict = field(default_factory=dict)


# Above this fraction of a monolayer, docking's combinatorics stop being
# targeted microsolvation: no single minimum dominates, and solvent-solvent
# cohesion (already a known limitation of E_int at n >= 2) takes over. Not a
# hard cutoff -- a warning, matching how `run_sweep` treats `cover`.
MONOLAYER_WARN_FRACTION = 1.0 / 3.0

# Grid placements per parent per n, above which `grid_placements` raises
# rather than screening. Measured at the defaults: pyrazine is 3351 poses per
# parent at n = 1 and 6955 at n = 3 (a 25-atom parent), and the count grows as
# surface, i.e. sublinearly in atom count -- so this leaves better than an
# order of magnitude of headroom on anything docking is applicable to, and
# only catches a typo. `--grid-spacing 0.2` on pyrazine is 237,300 poses per
# parent, a multi-day run entered by accident, and is what this stops.
MAX_GRID_PLACEMENTS = 200_000


def _random_point_in_ellipsoid(semi_axes, rng, max_tries=10000):
    """A point uniform over the volume of the ellipsoid with these semi-axes.

    Rejection sampling against the enclosing box -- about 52% acceptance in
    3D -- rather than a closed-form draw, because it needs nothing beyond
    numpy and the region here is always a modest few hundred cubic Angstrom.
    """
    semi_axes = np.asarray(semi_axes, dtype=float)
    for _ in range(max_tries):
        p = rng.uniform(-1.0, 1.0, size=3)
        if p @ p <= 1.0:
            return p * semi_axes
    raise RuntimeError("could not sample a point inside the shell ellipsoid")


def _voxel_downsample(points, spacing):
    """One representative point per occupied cell of a `spacing` lattice.

    O(M) and deterministic, which is the whole reason it is this and not a
    greedy minimum-separation filter: two points either side of a cell
    boundary can survive closer together than `spacing`, and for a placement
    grid that is harmless redundancy rather than a defect.
    """
    keys = np.round(np.asarray(points, dtype=float) / spacing).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.asarray(points, dtype=float)[np.sort(first)]


def _orientation_quaternions(n):
    """`n` quasi-uniform unit quaternions -- a super-Fibonacci spiral on SO(3).

    Alexa, "Super-Fibonacci Spirals: Fast, Low-Discrepancy Sampling of SO(3)"
    (CVPR 2022). A low-discrepancy sequence in four lines, needing no
    rejection, no symmetry detection and no per-solvent table -- the
    orientational analogue of `_unit_sphere_points`, which is the same trick
    one dimension down.
    """
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    i = np.arange(n) + 0.5
    t = i / n
    d = 2.0 * np.pi * i
    r, R = np.sqrt(t), np.sqrt(1.0 - t)
    alpha, beta = d / phi, d / psi
    return np.c_[r * np.sin(alpha), r * np.cos(alpha),
                 R * np.sin(beta), R * np.cos(beta)]


def _quaternion_to_matrix(q):
    """Rotation matrix from a unit quaternion, `_random_rotation`'s convention."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _pose_if_clear(parent_atoms, parent_positions, solvent_unit, centered,
                   rotation, point, tolerance):
    """`solvent_unit` rotated and centred on `point`, appended to the parent.

    `None` if any interatomic distance to the parent would fall below
    `tolerance` -- the placement test, in packmol's sense and packmol's
    units, and the one place it lives: `place_one` redraws when this returns
    `None` and `grid_placements` skips the pose, so "the identical test" is
    structural rather than two copies that happen to agree. It is also the
    inner-exclusion test, since a point inside the parent fails it too.

    `parent_positions` and `centered` -- the solvent unit's coordinates about
    its own centroid -- are passed in already computed because both callers
    loop over one fixed parent and one fixed solvent molecule.
    """
    coords = centered @ rotation.T + point
    d = np.linalg.norm(
        coords[:, None, :] - parent_positions[None, :, :], axis=2)
    if d.min() < tolerance:
        return None
    placed = solvent_unit.copy()
    placed.set_positions(coords)
    return parent_atoms + placed


def grid_placements(parent_atoms, solvent_unit, docking):
    """Every position x orientation pose of `solvent_unit` around the parent.

    The systematic counterpart of `place_one`, and the reason `place_mode`
    exists: `n_placements` random draws buy *confidence* that a basin found
    in some fraction of poses was hit, not *coverage*, and nothing in the
    output of such a run distinguishes a basin that does not exist from one
    that was never drawn.

    Positions come from `shell_capacity.surface_points`, voxel-downsampled to
    `docking.grid_spacing_A`, rather than from the ellipsoidal volume
    `place_one` draws in, because a surface follows the parent's topology and
    an ellipsoid does not. It is a *set* of surfaces, one per entry of
    `docking.grid_probe_fracs`, because no single one is right: the outermost
    is the solvent-centre surface, which is the contact distance for a sphere
    and about 1.5 A too far out for a directional H-bond. Orientations come
    from `_orientation_quaternions`. Poses whose closest interatomic approach
    to the parent is under `docking.tolerance` are dropped -- literally
    `_pose_if_clear`, the test `place_one` redraws on -- so the returned count
    is already post-rejection, which is what `n_placements_tried` reports.

    Callers pass an already principal-axis-aligned parent, exactly as
    `place_one`'s do: the grid is *not* frame-independent, because
    `_voxel_downsample` rounds against a lattice anchored at the lab origin
    and the orientations are lab-frame, so a rigidly moved input would
    otherwise give a slightly different pose set. See `dock_at_n`.

    **The parent is the whole complex**, with no solute/solvent distinction
    anywhere here: at n = k the surface is computed on the relaxed n = k - 1
    cluster, and Shrake-Rupley buries the contact patch under each bound
    molecule on its own. A bound molecule exposes more surface than it
    buries, so the grid grows slowly with n -- sublinearly, since surface
    goes as volume^(2/3) -- and it includes second-layer positions on top of
    already-bound solvent. That is deliberate rather than overlooked; see
    DESIGN.md's "Systematic placement".
    """
    probe = solvent_radius(docking.solvent, solvent_unit)
    # Downsampled per shell, not over the union: a `grid_spacing_A` voxel is
    # wider than the gap between adjacent shells, so pooling them first would
    # collapse the radial axis this loop exists to add.
    positions = np.concatenate([
        _voxel_downsample(surface_points(parent_atoms, probe * frac),
                          docking.grid_spacing_A)
        for frac in docking.grid_probe_fracs])
    quaternions = _orientation_quaternions(docking.n_orientations)

    n_poses = len(positions) * len(quaternions)
    if n_poses > MAX_GRID_PLACEMENTS:
        raise ValueError(
            f"grid_placements: {len(positions)} surface positions over "
            f"{len(docking.grid_probe_fracs)} shells x "
            f"{docking.n_orientations} orientations = {n_poses} poses per "
            f"parent, over MAX_GRID_PLACEMENTS = {MAX_GRID_PLACEMENTS}. "
            "Raise grid_spacing_A, or lower n_orientations or the number of "
            "grid_probe_fracs (or raise the ceiling deliberately -- this "
            "exists to catch a typo, not to cap a real run).")

    parent_positions = parent_atoms.get_positions()
    unit_positions = solvent_unit.get_positions()
    centered = unit_positions - unit_positions.mean(axis=0)

    placements = []
    for q in quaternions:
        rotation = _quaternion_to_matrix(q)
        for point in positions:
            pose = _pose_if_clear(parent_atoms, parent_positions, solvent_unit,
                                  centered, rotation, point, docking.tolerance)
            if pose is not None:
                placements.append(pose)
    return placements


def place_one(parent_atoms, solvent_unit, region, tolerance, rng, max_tries=2000):
    """One random placement of `solvent_unit` around `parent_atoms`.

    A random point inside the ellipsoidal shell `region` (semi-axes, centred
    at the origin -- callers pass an already solute-centred `parent_atoms`),
    a random orientation from `solvate_md._random_rotation`, redrawn whenever
    `_pose_if_clear` rejects the pair. That one test also keeps solvent off
    the solute -- a point too close simply fails it -- exactly as
    `pack_solvent` relies on packmol's own `tolerance` rather than carving
    out a second region for the purpose.

    Pure numpy, no packmol and no subprocess -- what makes it cheap enough to
    try dozens of poses per parent. Returns the combined `Atoms`.
    """
    parent_positions = parent_atoms.get_positions()
    unit_positions = solvent_unit.get_positions()
    centered = unit_positions - unit_positions.mean(axis=0)

    for _ in range(max_tries):
        # Point first, then rotation: the stream is shared across parents and
        # across n, so swapping the two draws reshuffles every random run.
        point = _random_point_in_ellipsoid(region, rng)
        rotation = _random_rotation(rng)
        pose = _pose_if_clear(parent_atoms, parent_positions, solvent_unit,
                              centered, rotation, point, tolerance)
        if pose is not None:
            return pose

    raise RuntimeError(
        f"place_one: no placement clearing tolerance={tolerance} A in "
        f"{max_tries} tries; the shell region may be too tight for this "
        "many molecules -- raise shell_fill or lower n."
    )


def random_placements(parent_atoms, solvent_unit, n_solute, n_total, docking,
                      rng):
    """`docking.n_placements` independent draws around the parent.

    The historical counterpart of `grid_placements`, and the same shape of
    thing: everything one mode needs to turn one parent into a list of poses,
    so `dock_at_n` dispatches to one call per mode rather than carrying one
    mode's sizing arithmetic inline. What it buys is *confidence* rather than
    coverage -- see `Docking.place_mode`.

    The region is the ellipsoidal shell `place_one` draws in, sized by
    `shell_padding` around the **solute block alone** (`n_solute` leading
    atoms of an already principal-axis-aligned parent) but holding the whole
    complex, at the `n_total` the shell is eventually meant to hold rather
    than at the parent's own count -- so the region a molecule is drawn into
    does not shrink as the chain grows.

    `rng` is passed in, not seeded here: one stream runs across every parent
    and every n of a run, which is what makes a given `seed` reproduce a run
    pose for pose.
    """
    solute_only = parent_atoms[:n_solute]
    semi_axes = solute_semi_axes(solute_only)
    padding = shell_padding(
        semi_axes, _vdw_volume(solute_only), n_total,
        bulk_molecular_volume(docking.solvent, solvent_unit),
        docking.shell_fill,
        min_padding=solvent_radius(docking.solvent, solvent_unit))
    region = semi_axes + padding
    return [place_one(parent_atoms, solvent_unit, region, docking.tolerance,
                      rng)
            for _ in range(docking.n_placements)]


@dataclass
class ScreenOrigin:
    """Where one refined candidate sat in its parent's screened ranking.

    `dock_at_n` used to return a flat list of parent indices beside its
    refined results, with everything else about the selection either discarded
    or carried in a second structure (`window_counts`) that had to stay in
    lockstep with the first. This is that provenance as one record per refined
    candidate, so it survives `run_docking`'s `sorted(zip(refined, origins))`
    by construction rather than by two parallel lists agreeing.

    `rank` and `offset_kcal` describe the candidate: its position among its
    parent's energy-ordered screened representatives (0 is that parent's
    screened minimum) and how far above that minimum it screened.
    `n_in_window` and `cut_kcal` describe the parent's whole selection and are
    identical across its candidates -- how many representatives
    `refine_window_kcal` admitted, and the offset of the last one `n_refine`
    actually let through.

    **`offset_kcal` is the statistic to read, not `rank`.** "The winner came
    from rank 12 of 400" does not license "the cap was harmless": that
    inference needs exactly the screened-rank-to-refined-energy correlation
    DESIGN.md's "The screen-to-refine handoff" measures as absent. What does
    transfer between runs is the offset, so `report.refine_cap_warning` says
    where in energy a binding cap cut and how much of the window that left
    behind. It stops short of grading that depth *safe*, and deliberately: a
    refined winner's own offset has been measured across the whole width of
    the window (`report.SCREEN_WINNER_OFFSET_KCAL`), so there is no depth
    above which truncating is known to cost nothing.
    """

    parent: int
    rank: int
    offset_kcal: float
    n_in_window: int
    cut_kcal: float


def dock_run_dir(out_root, label, n):
    """The run directory for one n of a chain.

    Named in one place because two callers need it before it exists:
    `_assemble_dock_n` creates it to write `scored.json` into, and
    `run_docking` points `dock_at_n`'s optional screen dump at it several
    minutes earlier.
    """
    return Path(out_root) / f"{label}_n{n}_dock"


def dock_at_n(parents, n_solute, solvent_unit, n_total, docking, scoring,
              solvation, calculator, calculator_kwargs, seed, n_workers=None,
              screen_dump=None):
    """Grow every parent by one solvent molecule, screen, and refine.

    Builds the placements -- `len(parents) * docking.n_placements` random
    ones (see `random_placements`), or, at `place_mode="grid"`, every surface
    position x orientation pose that clears the tolerance (see
    `grid_placements`) -- screens all of them at `docking.screen_fmax`, and
    refines a selection *per parent* at the scorer's tight criterion: the
    parent's screened placements are deduped at
    `docking.screen_dedupe_tol_eV` / `docking.screen_geom_tol_A` -- the same
    two-axis basin test the scorer uses, at the looser tolerances a
    loosely-relaxed geometry needs -- and the lowest representative of each
    screened basin is refined, best-first, so the refined set carries every
    distinct basin the screen found rather than the best basin several times
    over.

    **Which of those basins is an energy window, not a rank.** Every
    representative within `docking.refine_window_kcal` of that parent's own
    screened minimum is refined, capped at `docking.n_refine`, because a
    geometry relaxed only to `screen_fmax` carries an energy that does not
    predict where it refines to -- the measurement is in
    `Docking.refine_window_kcal` and in DESIGN.md's "The screen-to-refine
    handoff". Fewer than the cap are refined whenever a parent's placements
    collapsed into fewer screened basins than that, which is the ordinary
    case in random mode.

    Returns `(refined, origins, n_tried)`: `refined` is a list of
    `ensemble.Relaxed` objects, `origins[i]` is the `ScreenOrigin` of
    `refined[i]` -- which parent it grew from and where in that parent's
    screened ranking it was picked, which is what makes a binding cap visible
    downstream -- and `n_tried` is the total number of placements screened
    (not all of which were refined), the denominator the docking report shows
    the search effort against.

    `screen_dump`, when given a path, additionally writes the whole screening
    partition there: per parent, every screened energy and contact descriptor
    plus the representative / window / cap decision taken off them. Off by
    default and read by nothing in this pipeline -- it exists so the partition
    can be re-examined offline (is 883 screened basins at n = 3 a real count,
    or the tolerances over-splitting?) without paying for another GFN2
    gradient. Written compactly rather than at `indent=2` like every other
    JSON here: it is a few MB of bare floats per n, and nobody reads it by
    eye.
    """
    # Both modes place into the parent's principal-axis frame: only the
    # solute block defines where the axes point (a docking parent is the
    # whole complex, and has been through BFGS with nothing constraining its
    # centre of mass), but the whole complex moves with it. The random path
    # needs this, since its ellipsoidal region is axis-aligned. Grid mode
    # does not need it to be *correct* -- its region is the parent's own
    # surface -- but it is not frame-independent either, and used to be
    # commented here as though it were: `_voxel_downsample` rounds against a
    # lattice anchored at the lab origin and the orientations are lab-frame,
    # so six rigid motions of bare pyrazine gave 3251-3564 poses (6.2% spread)
    # against the file frame's 3355, at different places. Aligning first cuts
    # that to 3348-3363, and to *bit-identical* pose sets whenever the two
    # frames agree in sign -- what is left is `_principal_frame`'s
    # eigenvector-sign ambiguity, which flips on a pure translation and is
    # deliberately not fixed (it would move packmol's starting frame, and so
    # every sweep). See DESIGN.md's "Systematic placement".
    aligned = [align_to_principal_axes(parent, n_solute) for parent in parents]
    if docking.place_mode == "grid":
        per_parent = [grid_placements(parent, solvent_unit, docking)
                      for parent in aligned]
    elif docking.place_mode == "random":
        rng = np.random.default_rng(seed)
        per_parent = [random_placements(parent, solvent_unit, n_solute,
                                        n_total, docking, rng)
                      for parent in aligned]
    else:
        raise ValueError(
            f"unknown place_mode {docking.place_mode!r}; expected "
            '"random" or "grid"')

    placements, parent_of = [], []
    for parent_index, poses in enumerate(per_parent):
        placements += poses
        parent_of += [parent_index] * len(poses)

    # The same optimiser as the refinement below and as the scorer, at a
    # looser `fmax` and optionally with the solute held fixed. Nothing it
    # produces reaches a `scored.json`: it exists only to rank placements, so
    # only its geometry and its energy are read back.
    freeze_n = n_solute if docking.freeze_solute else 0
    screen_tasks = [(atoms, calculator, solvation, calculator_kwargs,
                     docking.screen_fmax, scoring.opt_steps, freeze_n)
                    for atoms in placements]
    screened = pool_map(relax, screen_tasks, n_workers)

    # Per parent, so a parent whose placements all screened a little higher
    # than another's still gets its basins refined -- that is what keeps the
    # chain's `n_parents` lineages alive into the next generation. Within a
    # parent, one representative per screened basin, lowest first, and every
    # one of them within `refine_window_kcal` of *that parent's* screened
    # minimum: the screened ranking does not predict the refined one (see
    # `Docking.refine_window_kcal`), so the cut has to be an energy window
    # and not a rank. `n_refine` is only the cap behind it.
    by_parent = {}
    for i, parent_index in enumerate(parent_of):
        by_parent.setdefault(parent_index, []).append(i)
    aps = len(solvent_unit)
    descriptors = [contact_descriptor(r.atoms, n_solute, aps) for r in screened]
    window_eV = docking.refine_window_kcal / EV_TO_KCAL
    # `top` indexes `screened`; `origins[j]` is the provenance of `top[j]`, so
    # the two stay aligned through `pool_map` (which preserves input order)
    # and through `run_docking`'s energy sort. What that provenance is for is
    # seeing the cap bind: once it does, the selection is a rank cut on the
    # screened energy again -- the exact failure the window replaced -- and
    # nothing in the refined output says so.
    top, origins = [], []
    dump_parents = [] if screen_dump is not None else None
    for parent_index in sorted(by_parent):
        members = by_parent[parent_index]
        representatives = dedupe_energies(
            [screened[i].energy_eV for i in members],
            [descriptors[i] for i in members],
            tol_eV=docking.screen_dedupe_tol_eV,
            geom_tol_A=docking.screen_geom_tol_A)
        # `dedupe_energies` returns representatives lowest-energy first, so
        # the window is a prefix and the cap a slice of it.
        floor_eV = screened[members[representatives[0]]].energy_eV
        inside = [r for r in representatives
                  if screened[members[r]].energy_eV - floor_eV <= window_eV]
        admitted = inside[:docking.n_refine]
        # In kcal/mol, and only for the record: the selection above stays in
        # eV so that adding this provenance moves no refined set.
        offsets = [(screened[members[r]].energy_eV - floor_eV) * EV_TO_KCAL
                   for r in admitted]
        top += [members[r] for r in admitted]
        origins += [ScreenOrigin(parent=parent_index, rank=rank,
                                 offset_kcal=offset, n_in_window=len(inside),
                                 cut_kcal=offsets[-1])
                    for rank, offset in enumerate(offsets)]
        if dump_parents is not None:
            # Indices are into this parent's own `members`, so each entry is
            # self-contained: `energies_eV[k]` and `descriptors[k]` are one
            # screened placement, and the three index lists select from them.
            dump_parents.append({
                "parent": parent_index,
                "energies_eV": [screened[i].energy_eV for i in members],
                "descriptors": [descriptors[i] for i in members],
                "representatives": representatives,
                "in_window": inside,
                "refined": admitted,
                "floor_eV": floor_eV,
                "cut_kcal": offsets[-1],
            })

    if dump_parents is not None:
        screen_dump = Path(screen_dump)
        screen_dump.parent.mkdir(parents=True, exist_ok=True)
        screen_dump.write_text(json.dumps({
            "n_solvent": n_total,
            "place_mode": docking.place_mode,
            "screen_fmax": docking.screen_fmax,
            "screen_dedupe_tol_eV": docking.screen_dedupe_tol_eV,
            "screen_geom_tol_A": docking.screen_geom_tol_A,
            "refine_window_kcal": docking.refine_window_kcal,
            "n_refine": docking.n_refine,
            "n_solute": n_solute,
            "atoms_per_solvent": aps,
            "version": VERSION,
            "parents": dump_parents,
        }))

    refine_tasks = [(screened[i].atoms, calculator, solvation,
                     calculator_kwargs, scoring.fmax, scoring.opt_steps, 0)
                    for i in top]
    refined = pool_map(relax, refine_tasks, n_workers)
    return refined, origins, len(placements)


def _assemble_dock_n(out_root, label, n, pairs, references, solvation,
                     docking, scoring, n_tried, n_parents_used, n_solute, aps):
    """Turn one n's refined placements into `scored.json` and its files.

    `pairs` is `[(Relaxed, ScreenOrigin), ...]`, already sorted lowest energy
    first, for every refined placement at this n (not yet deduped). The
    summary itself is built by `ensemble.summarise`, the same function
    `ensemble.assemble` calls, so a docking run and a sweep run cannot drift
    apart in shape or in what a field means -- which is what `dft_export` and
    `report` both rely on when they read either without branching.

    Everything specific to construction is what this adds: the placement
    bookkeeping in `extra`, and the per-parent table that makes the greedy
    chain visible. What it deliberately does *not* add is an occupancy: a
    docking dedupe groups constructed placements, not thermal samples, so
    counting them would look like a frame-weighted population and would not
    be one. `Candidate.n_frames` is `None` for these, which is what makes
    `summarise` report no occupancy at all rather than a plausible number.
    """
    e_solute, e_solvent, ref_solute_atoms, ref_solvent_atoms = references
    n_dir = dock_run_dir(out_root, label, n)
    n_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for i, (result, origin) in enumerate(pairs):
        gaps = solvent_molecule_gaps(result.atoms, n_solute, aps)
        candidates.append(Candidate(
            # No trajectory to index into: this is which refined placement
            # the geometry came from, in energy order.
            frame=i,
            energy_eV=result.energy_eV,
            interaction_eV=result.energy_eV - e_solute - n * e_solvent,
            converged=result.converged,
            fmax=result.fmax,
            n_contacts=int((gaps < CONTACT_GAP_A).sum()),
            n_solvent=n,
            min_gap_A=float(gaps.min()) if len(gaps) else float("nan"),
            # A docked structure has no sampling frame, so no wall energy and
            # no basin occupancy -- real absences, which is exactly what
            # `pack_mode: "dock"` exists to say.
            wall_energy_eV=None,
            gnorm_Eh_bohr=result.gnorm_Eh_bohr,
            n_opt_steps=result.n_opt_steps,
            descriptor=contact_descriptor(result.atoms, n_solute, aps),
            parent=origin.parent,
            n_frames=None,
            frames=None,
        ))

    keep_idx = dedupe_energies([c.energy_eV for c in candidates],
                               [c.descriptor for c in candidates])
    unique = [candidates[i] for i in keep_idx]
    best = candidates[keep_idx[0]]

    write(n_dir / "best.xyz", pairs[keep_idx[0]][0].atoms)
    write(n_dir / "scored_candidates.xyz", [pairs[i][0].atoms for i in keep_idx])
    write(n_dir / "ref_solute.xyz", ref_solute_atoms)
    write(n_dir / "ref_solvent.xyz", ref_solvent_atoms)

    # `same_basin` rather than an inline energy comparison, so this table and
    # `found_by` below ask the identical question `dedupe_energies` just
    # asked above -- energy *and* contact descriptor.
    def is_best(c):
        return same_basin(c.energy_eV, c.descriptor,
                          best.energy_eV, best.descriptor)

    # Grouped off the `ScreenOrigin`s rather than off `Candidate.parent`, so
    # the screening provenance is available here without every candidate in
    # `scored.json` having to carry a pair of fields an MD candidate could
    # only write as null.
    per_parent = {}
    for candidate, (_, origin) in zip(candidates, pairs):
        per_parent.setdefault(origin.parent, []).append((candidate, origin))
    # `n_placements` is how many of this parent's basins were refined;
    # `n_in_window` is how many its window admitted. They differ exactly when
    # `n_refine` capped the selection, which `report.refine_cap_warning`
    # surfaces -- a capped run is choosing by screened rank again, and the
    # refined candidates alone cannot show it. `screen_cut_kcal` is how far up
    # the cap reached before it did, and the two `best_screen_*` fields are
    # where this parent's own winner sat: together they are what discharges a
    # cap warning instead of merely repeating it. See `ScreenOrigin` for why
    # the offset and not the rank is the number to grade on.
    parent_detail = []
    for pi, items in sorted(per_parent.items()):
        # `pairs` is energy-sorted, so `items[0]` is this parent's own best
        # refined candidate; `min` says it rather than relying on that.
        best_candidate, best_origin = min(
            items, key=lambda item: item[0].interaction_eV)
        parent_detail.append({
            "parent": pi,
            "n_placements": len(items),
            "n_in_window": best_origin.n_in_window,
            "screen_cut_kcal": best_origin.cut_kcal,
            "best_screen_rank": best_origin.rank,
            "best_screen_offset_kcal": best_origin.offset_kcal,
            "e_int_min_kcal": best_candidate.interaction_eV * EV_TO_KCAL,
            "best": any(is_best(c) for c, _ in items),
        })

    summary = summarise(
        run_dir=n_dir,
        label=label,
        pack_mode="dock",
        seed=None,
        n_solvent=n,
        candidates=candidates,
        unique=unique,
        references=(e_solute, e_solvent),
        sampling_solvation=None,
        scoring_solvation=solvation,
        calculator=docking.calculator,
        scoring=scoring,
        sampling_wall=None,
        scored_frame_spacing_fs=None,
        extra={
            "n_placements_tried": n_tried,
            "n_refined": len(pairs),
            "n_parents": n_parents_used,
            # The constructive analogue of the sweep's `found by`: refined
            # placements that landed on this n's minimum. Not independent
            # corroboration the way agreeing packings are -- see
            # `report.format_dock_parent_detail`.
            "found_by": sum(1 for c in candidates if is_best(c)),
            "parent_detail": parent_detail,
        },
    )
    (n_dir / "scored.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_docking(solute_path, solvent_path, solvent, n_values, out_root,
                docking=None, scoring=None, n_workers=None, label=None,
                dump_screen=False, ladder_n=LADDER_N):
    """Dock one solute in one solvent, chained upward over n.

    n = 1's parent is the bare relaxed solute (the same reference
    `E_int(0) = 0` anchors); each subsequent n grows every surviving parent
    by one random placement, screens, refines, and keeps the top
    `docking.n_parents` deduped minima as the next generation's parents. The
    chain always walks every integer from 1 to `max(n_values)` -- a later n
    depends on the actual geometry an earlier one settled into, not merely on
    its energy -- but only writes an output directory and a report row for
    the n values actually requested.

    Writes `<out_root>/dock.json` -- `{"params": ..., "runs": [...]}`,
    matching `sweep.json`'s shape -- and `<out_root>/dock_report.txt`, plus
    one `best_n<N>.xyz` per requested n. Returns the list of summaries, one
    per requested n.

    `dump_screen` additionally writes `screen.json` into every n's run
    directory -- **every** n the chain walks, not only the requested ones,
    since the partition at a skipped n is exactly as interesting and the
    screening for it was paid for anyway. It is not a `Docking` field on
    purpose: it changes nothing about the run, so it has no business in the
    params block that says what the run was. See `dock_at_n`.

    `ladder_n` -- how many rungs the report's Basin spectrum section prints --
    is out of the params block for the same reason, and is display only: the
    `sites` and `gap/kT` columns it substantiates are read off the whole
    spectrum and do not move with it.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    n_values = sorted(set(int(n) for n in n_values))
    if not n_values or min(n_values) < 1:
        raise ValueError(
            "docking n values must be >= 1; n = 0 is the bare relaxed "
            "solute reference, computed once rather than swept.")
    label = label or Path(solute_path).stem
    docking = docking or Docking(solvent=solvent)
    scoring = scoring or Scoring()
    solvation = ("alpb", docking.solvent)

    solute = align_to_principal_axes(read(solute_path))
    n_solute = len(solute)
    solvent_unit = read(solvent_path)
    aps = len(solvent_unit)

    _, _, capacity = monolayer_capacity(solute, docking.solvent)
    max_n = max(n_values)
    if max_n > MONOLAYER_WARN_FRACTION * capacity:
        print(
            f"WARNING: n = {max_n} is {100 * max_n / capacity:.0f}% of a "
            f"monolayer (~{capacity:.0f} {docking.solvent} molecules). "
            "Docking targets targeted microsolvation, not a full shell: "
            "no single minimum dominates near a monolayer, and "
            "solvent-solvent cohesion takes over. The MD sweep is the "
            "right tool past roughly a third of capacity -- see "
            "DESIGN.md's Applicability section.")

    start = time.time()
    e_solute, e_solvent, ref_solute_atoms, ref_solvent_atoms = reference_energies(
        solute_path, solvent_path, docking.calculator, solvation,
        docking.calculator_kwargs, scoring.fmax, scoring.opt_steps)
    references = (e_solute, e_solvent, ref_solute_atoms, ref_solvent_atoms)

    parents = [ref_solute_atoms]
    all_n_min_kcal = {0: 0.0}  # E_int(0) = 0 by construction
    summaries = []
    n_tried_total = 0

    for n in range(1, max_n + 1):
        n_parents_used = len(parents)
        refined, origins, n_tried = dock_at_n(
            parents, n_solute, solvent_unit, n, docking, scoring, solvation,
            docking.calculator, docking.calculator_kwargs, seed=n,
            n_workers=n_workers,
            screen_dump=(dock_run_dir(out_root, label, n) / "screen.json"
                         if dump_screen else None))
        n_tried_total += n_tried

        pairs = sorted(zip(refined, origins), key=lambda pair: pair[0].energy_eV)
        summary = _assemble_dock_n(
            out_root, label, n, pairs, references, solvation, docking,
            scoring, n_tried, n_parents_used, n_solute, aps)
        all_n_min_kcal[n] = summary["min_interaction_kcal"]

        keep_idx = dedupe_energies(
            [r.energy_eV for r, _ in pairs],
            [contact_descriptor(r.atoms, n_solute, aps) for r, _ in pairs])
        parents = [pairs[i][0].atoms for i in keep_idx[:docking.n_parents]]

        if n in n_values:
            summaries.append(summary)

    elapsed = time.time() - start
    print(f"docking: {n_tried_total} placements screened, "
          f"{sum(s['n_refined'] for s in summaries)} refined at requested n "
          f"in {elapsed:.1f} s")

    params = dock_params(docking, scoring, n_values, label, capacity,
                         solute_path, solvent_path, all_n_min_kcal)
    (out_root / "dock.json").write_text(
        json.dumps({"params": params, "runs": summaries}, indent=2))
    (out_root / "dock_report.txt").write_text(
        format_report(params, summaries, ladder_n))
    write_best_geometries(out_root, pool_by_n(summaries), "dock")
    return summaries


def dock_params(docking, scoring, n_values, label, capacity, solute_path,
                solvent_path, all_n_min_kcal):
    """Everything needed to reproduce the run, recorded with its results.

    Built from `asdict` of the real `Docking` and `Scoring`, the same
    convention `n_sweep.sweep_params` uses, so a docking run and a sweep are
    reproducible from their own params block by the same rule -- including
    the `library_versions` that say which build of the Hamiltonian ran.
    """
    params = {k: v for k, v in asdict(docking).items()}
    # JSON has no tuple, so `python -m report` re-rendering from dock.json
    # would print a list where the run itself printed a tuple. Normalise at
    # the source, so a report and its re-render stay byte-identical.
    params["grid_probe_fracs"] = list(docking.grid_probe_fracs)
    # `max_frames` describes how a trajectory is subsampled, and docking has
    # no trajectory: `summarise` already writes it as `None` per run. It stays
    # in this block regardless -- the JSON records the complete `asdict` of
    # every dataclass so nothing can be added without appearing here;
    # `report.inert_params` is what crops it (and `place_mode`'s other inert
    # fields) at render time.
    params.update(asdict(scoring))
    params.update({
        "solute_label": label,
        "solute_path": str(solute_path),
        "solvent_path": str(solvent_path),
        "n_values": list(n_values),
        "monolayer_capacity": capacity,
        # The basin criterion the refined candidates were deduped by, recorded
        # for the same reason `n_sweep.sweep_params` records it: it moves
        # `pool` and `found by` without moving `E_int(min)`. The screening
        # pass's looser pair is already here, from `asdict(docking)`.
        "dedupe_tol_eV": DEDUPE_TOL_EV,
        "geom_tol_A": GEOM_TOL_A,
        # Every n walked internally, not just the requested rows -- what lets
        # dE_int be computed for a requested n even when n - 1 was not itself
        # requested.
        "all_n_min_kcal": all_n_min_kcal,
        "version": VERSION,
        "timestamp": timestamp(),
    })
    params.update(library_versions(docking.calculator))
    return params


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Dock solvent onto one solute in one solvent, chained "
                    "upward over n -- a constructive alternative to the MD "
                    "sweep's thermal sampling.",
        epilog="Docking wins at every n by construction (BFGS only "
               "descends), so its numbers are never pooled with an MD "
               "sweep's; read the two reports side by side instead.")
    parser.add_argument("solute", help="solute geometry file")
    parser.add_argument("solvent_geometry", help="one solvent molecule")
    parser.add_argument("--solvent", default="chcl3",
                        help="solvent name; the ALPB key and the bulk-density "
                             "key both (default: %(default)s)")
    parser.add_argument("--n", dest="n_values", type=int, nargs="+",
                        required=True, metavar="N",
                        help="explicit solvent counts to dock, e.g. 1 2 3 "
                             "(n = 0 is the bare relaxed solute and is not "
                             "swept)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--placements", type=int, default=Docking.n_placements,
                        help="random placements tried per parent per n; "
                             "ignored under --place-mode grid "
                             "(default: %(default)s)")
    parser.add_argument("--place-mode", default=Docking.place_mode,
                        choices=("random", "grid"),
                        help="how poses are generated: independent random "
                             "draws, or a systematic scan of the parent's "
                             "solvent-accessible surface x a quasi-uniform "
                             "set of orientations (default: %(default)s)")
    parser.add_argument("--grid-spacing", type=float,
                        default=Docking.grid_spacing_A,
                        help="grid mode: separation of surface positions, A "
                             "(default: %(default)s)")
    parser.add_argument("--orientations", type=int,
                        default=Docking.n_orientations,
                        help="grid mode: orientations per surface position "
                             "(default: %(default)s)")
    parser.add_argument("--grid-probe-fracs", type=float, nargs="+",
                        default=list(Docking.grid_probe_fracs),
                        metavar="F",
                        help="grid mode: one shell of positions per value, "
                             "each a probe radius as a fraction of the "
                             "solvent's sphere radius (default: "
                             "%(default)s)")
    parser.add_argument("--parents", type=int, default=Docking.n_parents,
                        help="deduped minima carried forward as the next n's "
                             "parents (default: %(default)s)")
    parser.add_argument("--refine", type=int, default=Docking.n_refine,
                        help="cap on screened basins re-relaxed at the "
                             "scorer's tight fmax, per parent; the selector "
                             "is --refine-window, and this only bounds its "
                             "cost (default: %(default)s)")
    parser.add_argument("--refine-window", type=float,
                        default=Docking.refine_window_kcal,
                        help="screened basins this far above a parent's own "
                             "screened minimum are refined, kcal/mol; the "
                             "screened ranking does not predict the refined "
                             "one, so this is a window and not a rank cut "
                             "(default: %(default)s)")
    parser.add_argument("--screen-fmax", type=float,
                        default=Docking.screen_fmax,
                        help="loose optimiser convergence for the screening "
                             "pass, eV/A (default: %(default)s)")
    parser.add_argument("--screen-dedupe-tol", type=float,
                        default=Docking.screen_dedupe_tol_eV,
                        help="energy half of the screening basin criterion, "
                             "eV; deliberately looser than the scorer's "
                             "because a screened geometry is only relaxed to "
                             "--screen-fmax (default: %(default)s)")
    parser.add_argument("--screen-geom-tol", type=float,
                        default=Docking.screen_geom_tol_A,
                        help="contact-descriptor half of the screening basin "
                             "criterion, A; splitting one basin here spends "
                             "--refine slots on near-copies of it "
                             "(default: %(default)s)")
    parser.add_argument("--dump-screen", action="store_true",
                        help="also write <run>/screen.json per n: every "
                             "screened energy and descriptor plus the "
                             "window/cap decision, so the screening partition "
                             "can be re-examined without re-running the screen")
    parser.add_argument("--ladder", type=int, default=LADDER_N,
                        help="rungs printed in the report's Basin spectrum "
                             "section. Display only, like --dump-screen: it "
                             "is not a Docking or Scoring field and does not "
                             "reach the params block, and the sites / gap-kT "
                             "columns are read off the whole spectrum either "
                             "way (default: %(default)s)")
    parser.add_argument("--freeze-solute", action="store_true",
                        help="FixAtoms the solute during screening only "
                             "(approximation; refinement is always "
                             "unconstrained)")
    parser.add_argument("--fmax", type=float,
                        default=Scoring.__dataclass_fields__["fmax"].default,
                        help="refinement optimiser convergence, eV/A "
                             "per-atom max force (default: %(default)s)")
    parser.add_argument("--opt-steps", type=int,
                        default=Scoring.__dataclass_fields__["opt_steps"].default,
                        help="max optimiser steps per candidate "
                             "(default: %(default)s)")
    parser.add_argument("--temperature", type=float,
                        default=Scoring.__dataclass_fields__["temperature_K"].default,
                        help="K, for the Boltzmann weights "
                             "(default: %(default)s)")
    parser.add_argument("--calculator", default=Docking.calculator,
                        help="default: %(default)s")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel workers (default: all cores)")
    parser.add_argument("--label", default=None,
                        help="solute name in output directories and the "
                             "table (default: the solute filename stem)")
    parser.add_argument("--export-dft", action="store_true",
                        help="also export deduped, near-minimum candidates "
                             "for DFT refinement to <out>/dft_export/")
    args = parser.parse_args(argv)

    docking = Docking(
        n_placements=args.placements,
        place_mode=args.place_mode,
        grid_spacing_A=args.grid_spacing,
        n_orientations=args.orientations,
        grid_probe_fracs=tuple(args.grid_probe_fracs),
        n_parents=args.parents,
        n_refine=args.refine,
        refine_window_kcal=args.refine_window,
        screen_dedupe_tol_eV=args.screen_dedupe_tol,
        screen_geom_tol_A=args.screen_geom_tol,
        screen_fmax=args.screen_fmax,
        freeze_solute=args.freeze_solute,
        solvent=args.solvent,
        calculator=args.calculator,
    )
    scoring = Scoring(fmax=args.fmax, opt_steps=args.opt_steps,
                      temperature_K=args.temperature)

    run_docking(
        args.solute, args.solvent_geometry, args.solvent, args.n_values,
        args.out, docking=docking, scoring=scoring, n_workers=args.workers,
        label=args.label, dump_screen=args.dump_screen,
        ladder_n=args.ladder,
    )
    print(Path(args.out) / "dock_report.txt")

    if args.export_dft:
        from dft_export import export_dft
        out_dir = Path(args.out) / "dft_export"
        export_dft(args.out, out_dir)
        print(out_dir / "manifest.json")


if __name__ == "__main__":
    main()
