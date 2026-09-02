import single_thread  # noqa: F401  -- must precede numpy; see its docstring

import os
import json
import multiprocessing
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase.constraints import FixCom
from ase.data import vdw_radii
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    Stationary,
    ZeroRotation,
    thermalize_momenta,
)
from ase.optimize import BFGS
from ase.units import fs

from report import RunLogger

_FALLBACK_VDW_RADIUS = 1.7  # Angstrom, used when ase.data.vdw_radii has no entry
_AVOGADRO = 6.02214076e23

# Molar mass (g/mol) and density (g/cm^3) near 298 K, keyed by the same name
# that gets handed to ALPB. One field on `Condition` drives both the packing
# density and the continuum, so the explicit shell and the implicit bulk can
# never silently disagree about which solvent this is.
SOLVENT_BULK = {
    "water": (18.015, 0.997),
    "chcl3": (119.38, 1.489),
    "chloroform": (119.38, 1.489),
    "acetone": (58.08, 0.784),
    "methanol": (32.04, 0.792),
    "acetonitrile": (41.05, 0.786),
    "thf": (72.11, 0.889),
    "toluene": (92.14, 0.867),
    "dmso": (78.13, 1.100),
    "ch2cl2": (84.93, 1.327),
}


def _vdw_radii_array(atoms):
    radii = vdw_radii[atoms.get_atomic_numbers()]
    return np.where(np.isnan(radii), _FALLBACK_VDW_RADIUS, radii)


def _unit_sphere_points(n):
    """`n` roughly equispaced points on the unit sphere (golden spiral).

    Deterministic and needs no relaxation, which is all either caller wants:
    `packing_wall_distance` samples a shell surface with it, and
    `shell_capacity.sasa` a Shrake-Rupley atomic sphere.
    """
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi),
                 np.sin(theta) * np.sin(phi),
                 np.cos(phi)]


def _vdw_volume(atoms):
    """Rough molecular volume from a sum of non-overlapping vdW spheres.

    Ignores overlap between bonded atoms, so it overestimates somewhat, but
    only ever appears as the solute's excluded volume when sizing a shell.
    """
    return float(np.sum((4.0 / 3.0) * np.pi * _vdw_radii_array(atoms) ** 3))


def bulk_molecular_volume(solvent, solvent_atoms=None):
    """Volume per solvent molecule in the neat liquid, in Angstrom^3.

    Taken from the experimental density where we know it, because that is
    the only way to make a packed shell come out at a physically meaningful
    density. Falls back to a vdW-sphere sum for solvents not in the table --
    a decent proxy (within a few percent for water and acetone, ~30% low for
    chloroform), but prefer adding a real density to SOLVENT_BULK.
    """
    key = solvent.lower()
    if key in SOLVENT_BULK:
        molar_mass, density = SOLVENT_BULK[key]
        return molar_mass / density / _AVOGADRO * 1e24
    if solvent_atoms is None:
        raise KeyError(f"No bulk density for {solvent!r}; add it to SOLVENT_BULK.")
    return _vdw_volume(solvent_atoms)


def solvent_radius(solvent, solvent_atoms=None):
    """Radius of a sphere with the solvent's bulk molecular volume."""
    volume = bulk_molecular_volume(solvent, solvent_atoms)
    return (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)


@dataclass
class Condition:
    solute_path: str
    solvent_path: str
    n_solvent: int
    # ALPB solvent name; also the key used for bulk density when sizing the
    # shell. Must match what `solvent_path` actually contains.
    solvent: str = "chcl3"
    # Whether the *sampling* MD runs in the continuum. False -- gas phase --
    # by default: sampling wants whatever actually finds the contact
    # geometries, and ALPB can make a shell dissociate outright (measured:
    # 50-60% wall-active for methanol + 4 water in ALPB(water), against 4-5%
    # in gas phase). The continuum belongs to scoring, which always applies
    # it; see the module docstring of `ensemble.py`. Set True only for a
    # deliberate continuum-sampled run.
    sample_in_continuum: bool = False
    calculator: str = "gfn2-xtb"
    temperature_K: float = 298.0
    # 0.5 fs, not 1.0: X-H stretches have ~9 fs periods and nothing here
    # constrains them (no SHAKE, no hydrogen-mass repartitioning).
    timestep_fs: float = 0.5
    friction: float = 0.01  # ASE inverse time units, ~1 ps^-1
    # ~5 ps, i.e. several Langevin relaxation times at the default friction.
    # Much shorter and the "production" run is still equilibrating.
    n_equilibrate_steps: int = 10000
    n_steps: int = 20000
    # 100 steps = 50 fs. Sized against how fast the shell actually decorrelates
    # rather than against the timestep: the solvent-COM autocorrelation in the
    # solute frame, measured on pyrazine + 3 CHCl3, falls to 1/e in 550 fs, so
    # dumping every 5 fs recorded the same configuration ~100 times over and
    # made `energies.json` large for no extra information. Expect this to grow
    # with the solute -- a hexamer's shell rearranges more slowly still.
    dump_interval: int = 100
    # Fraction of the packing shell filled by solvent at bulk density. Sets
    # how thick the shell is for a given n_solvent -- see `shell_padding`.
    shell_fill: float = 0.5
    # How far past the packed shell a solvent atom may stray before the wall
    # pulls it back, in solvent diameters -- and now measured in the wall's own
    # metric, so it means the same thing for every solute (see
    # `packing_wall_distance`). 0.25 leaves ~1-1.5 A of room to rearrange.
    #
    # Not just a safety margin: the wall volume sets the translational entropy
    # of a dissociated solvent molecule, so loosening it makes dissociation
    # more favourable. The old default of 1.0 meant a different thing under
    # the previous formula and is far too loose under this one.
    wall_slack: float = 0.25
    wall_k: float = 1.0  # eV/Angstrom^2
    # Packmol's global minimum interatomic distance. Left at the conventional
    # 2.0 deliberately, but exposed because it has a chemical consequence that
    # is easy to miss: it forbids hydrogen-bond contacts outright (H...O/N sit
    # at 1.8-2.0 A), so a packed structure cannot start bonded. That is fine
    # here -- gas-phase MD forms the contacts within a few ps from a 2.0 start
    # -- but lower it if you need contact geometries straight out of packing.
    tolerance: float = 2.0
    label: str = "condition"
    calculator_kwargs: dict = field(default_factory=dict)


class SolventShellWall(Calculator):
    """Keep every solvent atom within `max_distance` of the nearest solute atom.

    A finite cluster in vacuum has nothing holding the solvent on the solute,
    so the shell needs a confining potential or it drifts apart -- this is why
    CREST/QCG runs its microsolvation MD under wall potentials too.

    How much work that wall has to do is strongly solvent-dependent, and it is
    worth knowing which regime you are in before trusting a run. Measured with
    GFN2-xTB, binding of one solvent molecule, gas -> ALPB:

        methanol...water   in ALPB(water)    -3.6 -> +2.6 kcal/mol
        pyrazine...HCCl3   in ALPB(chcl3)    -5.7 -> -6.6 kcal/mol
        pyrazine...acetone in ALPB(acetone)  -2.1 -> -3.5 kcal/mol

    ALPB(water) reproduces a water's full hydration free energy, so a water
    that leaves the cluster loses nothing and dissociation is downhill: the
    wall ends up load-bearing and its energy contaminates everything. The
    weaker continua do the opposite -- they stabilise the complex, because the
    polarised contact has a larger dipole than the separated monomers.

    So `wall_energy_eV` in the trajectory output is the diagnostic to watch.
    Rarely nonzero means the shell is self-bound and the wall is a safety net.
    Persistently nonzero means the wall is holding together something the
    Hamiltonian wants to disperse, and the energies are not clean.

    Phrased as a distance to the nearest solute atom rather than as a
    geometric cage, which sidesteps every way a cage can be wrong: it follows
    the solute's real shape instead of a fitted primitive, needs no centre
    (so it cannot drift off the solute when the solvent outweighs it), and
    needs no orientation (so it stays correct as the solute reorients). The
    forces are equal and opposite between each solvent atom and the solute
    atom restraining it, so the wall exerts no net force on the system.

    It acts only on solvent, so the solute is never squeezed by it. The
    switch of which solute atom is nearest makes the force direction
    piecewise-continuous; the magnitude is continuous across a switch, and
    under a thermostat this is no worse than an ordinary neighbour cutoff.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, n_solute, max_distance, k=1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_solute = int(n_solute)
        self.max_distance = float(max_distance)
        self.k = float(k)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = self.atoms.get_positions()
        solute, solvent = positions[: self.n_solute], positions[self.n_solute :]

        forces = np.zeros_like(positions)
        if len(solvent) == 0:
            self.results = {"energy": 0.0, "free_energy": 0.0, "forces": forces}
            return

        offsets = solvent[:, None, :] - solute[None, :, :]
        distances = np.linalg.norm(offsets, axis=2)
        nearest = distances.argmin(axis=1)
        rows = np.arange(len(solvent))
        d = distances[rows, nearest]
        overshoot = np.maximum(d - self.max_distance, 0.0)

        energy = 0.5 * self.k * float(np.sum(overshoot**2))
        pull = -self.k * (overshoot / d)[:, None] * offsets[rows, nearest]
        forces[self.n_solute :] = pull
        np.add.at(forces[: self.n_solute], nearest, -pull)

        self.results = {"energy": energy, "free_energy": energy, "forces": forces}


def get_calculator(name, solvation=None, **kwargs):
    name = name.lower()
    if name in ("gfn2-xtb", "gfn1-xtb"):
        from tblite.ase import TBLite

        method = "GFN2-xTB" if name == "gfn2-xtb" else "GFN1-xTB"
        if solvation is not None:
            kwargs.setdefault("solvation", solvation)
        # Otherwise every SCF cycle is printed, and n_workers of those
        # interleave into unreadable noise.
        kwargs.setdefault("verbosity", 0)
        return TBLite(method=method, **kwargs)
    elif name == "mace-off23":
        from mace.calculators import mace_off

        if solvation is not None:
            raise ValueError(
                "MACE calculators have no implicit-solvation model, so a "
                "cluster-continuum run is not available with them. Leave "
                "sample_in_continuum at False (MACE is a generator; the "
                "scorer is a separate process and can be GFN2) or use a GFN "
                "calculator here too."
            )
        return mace_off(**kwargs)
    else:
        raise ValueError(f"Unknown calculator: {name}")


def _principal_frame(atoms):
    """Centroid and the rotation onto the solute's principal axes.

    Uses the unweighted centroid, not the centre of mass, because that is
    what Packmol's `fixed 0. 0. 0.` actually places at the origin -- Packmol
    gets no masses from an xyz file, so its `centerofmass` keyword collapses
    to the plain centroid. Getting this wrong silently decouples the packing
    region from where the solute really ends up.
    """
    positions = atoms.get_positions()
    centroid = positions.mean(axis=0)
    centered = positions - centroid

    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    order = np.argsort(eigenvalues)[::-1]  # longest spatial extent first
    rotation = eigenvectors[:, order].T
    if np.linalg.det(rotation) < 0:  # keep it a proper rotation, not a reflection
        rotation[2] *= -1.0
    return centroid, rotation


def align_to_principal_axes(atoms):
    """Copy of `atoms` with its centroid at the origin and axes sorted long-first.

    Lets an axis-aligned ellipsoid hug the solute regardless of how the input
    file happened to be oriented, and makes the packing region independent of
    any rigid-body transform applied to the input.
    """
    centroid, rotation = _principal_frame(atoms)
    aligned = atoms.copy()
    aligned.set_positions((atoms.get_positions() - centroid) @ rotation.T)
    return aligned


def solute_semi_axes(atoms):
    """Semi-axes of the vdW envelope of an already-aligned solute."""
    extents = np.abs(atoms.get_positions()) + _vdw_radii_array(atoms)[:, None]
    return extents.max(axis=0)


def shell_padding(semi_axes, v_solute, n_solvent, v_solvent, shell_fill,
                  min_padding=0.0, max_padding=30.0):
    """Shell thickness that puts `n_solvent` molecules against the solute.

    Solves for the thickness `t` at which the ellipsoidal shell between the
    solute and semi-axes `semi_axes + t` is `shell_fill` occupied by solvent
    at bulk liquid density.

    This is the sizing rule, and it is deliberately *not* "pick a box that
    holds N molecules at bulk density". At the molecule counts used for
    microsolvation there are far too few solvent molecules to fill any convex
    region around a solute of this size -- a dozen chloroforms is well under
    half a monolayer on a foldamer hexamer. Sizing a volume from N therefore
    scatters them through empty space instead of onto the solute. Sizing the
    *thickness* from N keeps the region wrapped tightly around the solute at
    any N, and grows smoothly into a genuine full shell as N increases.
    """
    target_volume = n_solvent * v_solvent / shell_fill

    def excess(t):
        ellipsoid = (4.0 / 3.0) * np.pi * float(np.prod(semi_axes + t))
        return ellipsoid - v_solute - target_volume

    if excess(max_padding) < 0.0:
        raise ValueError(
            f"Cannot fit {n_solvent} solvent molecules within {max_padding} A "
            "of the solute; raise max_padding or lower n_solvent."
        )

    low, high = 0.0, max_padding
    for _ in range(80):
        mid = 0.5 * (low + high)
        if excess(mid) < 0.0:
            low = mid
        else:
            high = mid
    return max(0.5 * (low + high), min_padding)


def packing_wall_distance(region, solute_positions, r_solvent, wall_slack,
                          n_samples=4096):
    """Wall radius that just contains the packing region, plus slack.

    The wall measures a solvent atom's distance to the *nearest solute atom*,
    so its radius has to be derived in that same metric. Adding the shell
    `padding` to it instead -- a thickness laid on semi-axes measured from the
    origin, where the semi-axes already include vdW radii -- mixes two
    different reference surfaces, and the mismatch does not scale with solute
    size: for methanol it put the wall 1.75 A outside the region the solvent
    was packed into, so MD's first act was to expand the shell.

    Sampling the region's surface and taking the largest distance to the
    nearest solute atom gives a wall that contains the packed shell exactly,
    and makes `wall_slack` mean the same thing for every solute.
    """
    surface = _unit_sphere_points(n_samples) * np.asarray(region)
    d = np.linalg.norm(surface[:, None, :] - solute_positions[None, :, :], axis=2)
    return float(d.min(axis=1).max() + wall_slack * 2.0 * r_solvent)


@dataclass
class Packing:
    atoms: object
    n_solute: int
    semi_axes: np.ndarray
    padding: float
    wall_distance: float
    atoms_per_solvent: int


def pack_solvent(
    solute_path,
    solvent_path,
    n_solvent,
    out_path,
    solvent,
    shell_fill,
    wall_slack,
    tolerance,
    seed,
):
    """Pack n_solvent copies of solvent_path into a shell around the solute.

    Every packing parameter is required rather than defaulted. `Condition`
    owns these values and `run_one_job` passes all of them, so a default here
    could only ever be a second, silently different opinion -- which is what
    it was: `wall_slack` defaulted to 1.0 where `Condition` says 0.25, a value
    that comment two hundred lines up calls far too loose.

    The solute is aligned to its principal axes and centred on the origin --
    matching where Packmol puts it -- and the solvent goes into an ellipsoidal
    region wrapped around it, thick enough to hold n_solvent molecules in
    contact with the surface (see `shell_padding`). The ellipsoid adapts to
    the solute's shape, so a compact and an extended conformer get the same
    shell thickness rather than boxes of wildly different volume, which is
    what makes their energies comparable.

    Note there is no inner exclusion region: Packmol's `tolerance` already
    keeps solvent off the fixed solute, and it does so following the true
    molecular surface rather than any primitive we could write down.
    """
    solute = align_to_principal_axes(read(solute_path))
    solvent_unit = read(solvent_path)

    semi_axes = solute_semi_axes(solute)
    v_solvent = bulk_molecular_volume(solvent, solvent_unit)
    r_solvent = solvent_radius(solvent, solvent_unit)
    padding = shell_padding(
        semi_axes,
        _vdw_volume(solute),
        n_solvent,
        v_solvent,
        shell_fill=shell_fill,
        min_padding=r_solvent,  # always at least a contact layer
    )
    region = semi_axes + padding
    wall_distance = packing_wall_distance(
        region, solute.get_positions(), r_solvent, wall_slack=wall_slack
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if n_solvent == 0:
        # The continuum-only reference point of an n-sweep. Nothing to pack,
        # and packmol rejects `number 0`, so short-circuit.
        write(out_path, solute)
        return Packing(
            atoms=solute,
            n_solute=len(solute),
            semi_axes=semi_axes,
            padding=padding,
            wall_distance=wall_distance,
            atoms_per_solvent=len(solvent_unit),
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        solute_xyz = tmpdir / "solute.xyz"
        solvent_xyz = tmpdir / "solvent.xyz"
        packed_xyz = tmpdir / "packed.xyz"
        write(solute_xyz, solute)
        write(solvent_xyz, solvent_unit)

        # `inside ellipsoid xc yc zc a b c d` constrains atoms to
        # ((x-xc)/a)^2 + ((y-yc)/b)^2 + ((z-zc)/c)^2 <= d^2, so with d = 1 the
        # a/b/c are the semi-axes directly.
        inp_path = tmpdir / "pack.inp"
        inp_text = f"""\
seed {seed}
tolerance {tolerance}
filetype xyz
output {packed_xyz}

structure {solute_xyz}
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure

structure {solvent_xyz}
  number {n_solvent}
  inside ellipsoid 0. 0. 0. {region[0]:.4f} {region[1]:.4f} {region[2]:.4f} 1.0
end structure
"""
        inp_path.write_text(inp_text)

        with open(inp_path) as stdin:
            result = subprocess.run(
                ["packmol"], stdin=stdin, capture_output=True, text=True
            )
        if not packed_xyz.exists():
            raise RuntimeError(
                f"packmol failed to produce output:\n{result.stdout}\n{result.stderr}"
            )

        packed = read(packed_xyz)

    # Packmol writes its best effort even when it cannot satisfy the
    # constraints, so check rather than trusting the file's existence.
    n_expected = len(solute) + n_solvent * len(solvent_unit)
    if len(packed) != n_expected:
        raise RuntimeError(
            f"packmol wrote {len(packed)} atoms, expected {n_expected}:\n"
            f"{result.stdout}"
        )
    if "Success" not in result.stdout:
        raise RuntimeError(
            "packmol did not converge; the shell is probably too tight. Lower "
            "shell_fill (which thickens the shell) or use fewer solvent "
            f"molecules.\n{result.stdout}"
        )

    write(out_path, packed)
    return Packing(
        atoms=packed,
        n_solute=len(solute),
        semi_axes=semi_axes,
        padding=padding,
        wall_distance=wall_distance,
        atoms_per_solvent=len(solvent_unit),
    )


def run_one_job(condition, seed, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    packing = pack_solvent(
        condition.solute_path,
        condition.solvent_path,
        condition.n_solvent,
        out_dir / "packed.xyz",
        solvent=condition.solvent,
        shell_fill=condition.shell_fill,
        wall_slack=condition.wall_slack,
        tolerance=condition.tolerance,
        seed=seed,
    )
    atoms = packing.atoms

    # Streams `run.log` alongside the JSON, flushing every line so a long run
    # can be followed with `tail -f` instead of going silent until it finishes.
    logger = RunLogger(out_dir / "run.log")

    solvation = ("alpb", condition.solvent) if condition.sample_in_continuum else None
    calc = get_calculator(
        condition.calculator, solvation=solvation, **condition.calculator_kwargs
    )
    wall = SolventShellWall(
        packing.n_solute, packing.wall_distance, k=condition.wall_k
    )
    atoms.calc = SumCalculator([calc, wall])

    # Loose on purpose: this is clash relief before MD, not a minimisation.
    opt = BFGS(atoms, logfile=str(out_dir / "opt.log"))
    opt.run(fmax=0.5, steps=200)
    # Both must be read now, before MD moves the atoms -- `opt.converged()`
    # re-evaluates against the current geometry, so asking later reports on a
    # thermally excited structure instead of on the relaxation.
    relax_converged = bool(opt.converged())
    relax_fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())

    # Built here rather than at the end so the log header and metadata.json
    # are rendered from one dict: the human-readable file cannot then describe
    # the run differently from the machine-readable one beside it.
    metadata = {
        "label": condition.label,
        "seed": seed,
        "solute_path": str(condition.solute_path),
        "solvent_path": str(condition.solvent_path),
        "solvent": condition.solvent,
        "n_solvent": condition.n_solvent,
        "n_atoms": len(atoms),
        # The scorer reads the trajectory back without the Condition that
        # produced it, and needs these to split solute from solvent and
        # solvent into molecules.
        "n_solute": packing.n_solute,
        "atoms_per_solvent": packing.atoms_per_solvent,
        "tolerance": condition.tolerance,
        "calculator": condition.calculator,
        "sample_in_continuum": condition.sample_in_continuum,
        "solvation": list(solvation) if solvation else None,
        "temperature_K": condition.temperature_K,
        "timestep_fs": condition.timestep_fs,
        "friction": condition.friction,
        "n_equilibrate_steps": condition.n_equilibrate_steps,
        "n_steps": condition.n_steps,
        "dump_interval": condition.dump_interval,
        "shell_semi_axes": packing.semi_axes.tolist(),
        "shell_padding": packing.padding,
        "wall_distance": packing.wall_distance,
        "wall_k": condition.wall_k,
        "relax_converged": relax_converged,
        "relax_final_fmax": relax_fmax,
    }
    # shell_fill and wall_slack shape the packing but are not in metadata.json,
    # so hand them to the header separately rather than widening that file.
    logger.header(metadata, extra={"shell_fill": condition.shell_fill,
                                   "wall_slack": condition.wall_slack})

    # Draw velocities reproducibly, then remove the net translation and
    # rotation the draw introduces. Langevin's fixcm handles the translation
    # each step, but nothing removes whole-cluster spin, and a spinning
    # cluster in vacuum flings its own solvent shell off centrifugally.
    thermalize_momenta(atoms, temperature_K=condition.temperature_K,
                       rng=np.random.default_rng(seed))
    Stationary(atoms)
    ZeroRotation(atoms)

    # Langevin's own fixcm=True does not sample NVT strictly, and ASE warns
    # that the error grows for *small* systems -- which is exactly what a
    # microsolvated cluster is. FixCom is the supported replacement.
    atoms.set_constraint(FixCom())

    dyn = Langevin(
        atoms,
        timestep=condition.timestep_fs * fs,
        temperature_K=condition.temperature_K,
        friction=condition.friction,
        fixcm=False,
        # Without an explicit rng ASE falls back to the global np.random,
        # which each spawned worker reseeds from OS entropy -- the runs would
        # not be reproducible from `seed` at all. Offset so the thermostat
        # noise is an independent stream from the initial velocities.
        rng=np.random.default_rng(seed + 1_000_000),
    )

    # Equilibrate before recording, so the trajectory does not open with the
    # transient from Maxwell-Boltzmann velocities relaxing into equipartition.
    dyn.run(condition.n_equilibrate_steps)

    traj_path = out_dir / "traj.xyz"
    # Truncate before the first dump rather than keying `append` off the step
    # number: the step of the first dump is only zero when the equilibration
    # length happens to be a multiple of `dump_interval`, and otherwise every
    # frame would append to whatever a previous run left here.
    traj_path.unlink(missing_ok=True)
    energies = []

    def record():
        step = dyn.get_number_of_steps() - condition.n_equilibrate_steps
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        entry = {
            "step": step,
            "time_fs": step * condition.timestep_fs,
            "potential_energy_eV": epot,
            "kinetic_energy_eV": ekin,
            "total_energy_eV": epot + ekin,
            # ASE's own accessor, which discounts the degrees of freedom
            # removed by FixCom rather than assuming a bare 3N.
            "temperature_K": atoms.get_temperature(),
            # Nonzero means solvent is leaning on the wall rather than
            # being held by the solute: the shell is not self-bound and
            # the confinement is doing real work. Worth watching.
            "wall_energy_eV": float(wall.get_potential_energy(atoms)),
        }
        energies.append(entry)
        # One dict feeds both the JSON and the log, so they cannot drift.
        # Per-dump text I/O is negligible against an SCF.
        logger.md_row(entry)
        write(traj_path, atoms, append=True)

    dyn.attach(record, interval=condition.dump_interval)
    dyn.run(condition.n_steps)

    # Reports the wall-active fraction, and warns when the confinement turns
    # out to have been load-bearing rather than a safety net.
    logger.footer()
    logger.close()

    with open(out_dir / "energies.json", "w") as f:
        json.dump(energies, f, indent=2)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return str(out_dir)


def pool_map(func, arg_tuples, n_workers=None):
    """Call `func(*args)` once per tuple in `arg_tuples`, one process each.

    Both halves of a sweep fan out through here -- the MD jobs below, the
    candidate optimisations in `ensemble` -- because both want the same four
    things and neither wants its own copy of them.

    The pool uses **spawn**, since tblite and torch cannot safely share a
    forked process, so every worker re-imports the calling module. A driver
    script MUST therefore guard its call:

        if __name__ == "__main__":
            run_sweep(...)

    Without the guard each worker re-runs the whole grid on import, forking
    recursively until the machine gives out.

    Workers are capped at the task count. A spawn pool builds every worker up
    front, and one that never receives a task still pays for an interpreter
    and a full numpy/ASE/tblite import -- ~100 MB resident each.

    `chunksize=1` rather than `map`'s default, which hands each worker a
    contiguous block of the input. The tasks here are markedly unequal -- a
    rigid 10-atom molecule against a floppy 25-atom cluster -- and blocking
    them up strands the slow ones together behind a single worker.

    One task, or one worker, runs in this process: spawning an interpreter to
    run a single task costs more than it saves, and an exception arrives with
    a useful traceback rather than a pickled one.
    """
    arg_tuples = list(arg_tuples)
    if not arg_tuples:
        return []
    n_workers = min(n_workers or os.cpu_count(), len(arg_tuples))
    if n_workers == 1:
        return [func(*args) for args in arg_tuples]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        # `starmap` preserves input order, so results come back in the order a
        # serial loop would have produced them and nothing downstream has to
        # sort.
        return pool.starmap(func, arg_tuples, chunksize=1)


def run_job_grid(conditions, n_seeds, out_root, n_workers=None):
    """Run every (condition, seed) pair, one single-threaded worker each.

    Returns the run directories in order, which is also the only place their
    names are built -- the scorer takes them from here rather than rebuilding
    `<label>_seed<n>` for itself.

    One trajectory is inherently sequential, so the parallelism is over jobs
    and its ceiling is the job count; see `pool_map` for the `__main__` guard
    a caller has to provide.
    """
    out_root = Path(out_root)
    jobs = [(condition, seed, out_root / f"{condition.label}_seed{seed}")
            for condition in conditions
            for seed in range(n_seeds)]
    return pool_map(run_one_job, jobs, n_workers)


if __name__ == "__main__":
    examples_dir = Path(__file__).parent / "examples"

    # Smoke test only. Real job lists (e.g. the AAA/BBB x chloroform/acetone
    # grid, and the n_solvent convergence sweep that justifies whatever
    # n_solvent it settles on) belong in a project driver script, not here.
    toy_conditions = [
        Condition(
            solute_path=str(examples_dir / "solute_toy.xyz"),
            solvent_path=str(examples_dir / "water.xyz"),
            n_solvent=4,
            solvent="water",
            calculator="gfn2-xtb",
            temperature_K=298.0,
            timestep_fs=0.5,
            n_equilibrate_steps=200,
            n_steps=1000,
            dump_interval=10,
            label="toy_water",
        ),
    ]

    out_root = Path(__file__).parent / "smoke_test_output"
    results = run_job_grid(toy_conditions, n_seeds=1, out_root=out_root, n_workers=1)
    print("Smoke test jobs written to:")
    for r in results:
        print(f"  {r}")
