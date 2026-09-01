import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import multiprocessing
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase.data import vdw_radii
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from ase.units import fs

_FALLBACK_VDW_RADIUS = 1.7  # Angstrom, used when ase.data.vdw_radii has no entry


def _vdw_volume(atoms):
    """Rough molecular volume from a sum of non-overlapping vdW spheres.

    Ignores overlap between bonded atoms, so this overestimates true
    molecular volume, but that's fine (even desirable) for sizing a
    packing box: it's used only to decide how much space a solvation
    shell of a given molecule count needs.
    """
    radii = vdw_radii[atoms.get_atomic_numbers()]
    radii = np.where(np.isnan(radii), _FALLBACK_VDW_RADIUS, radii)
    return float(np.sum((4.0 / 3.0) * np.pi * radii**3))


@dataclass
class Condition:
    solute_path: str
    solvent_path: str
    n_solvent: int
    calculator: str = "gfn2-xtb"
    temperature_K: float = 298.0
    timestep_fs: float = 1.0
    n_steps: int = 1000
    dump_interval: int = 10
    label: str = "condition"
    calculator_kwargs: dict = field(default_factory=dict)


def get_calculator(name, **kwargs):
    name = name.lower()
    if name in ("gfn2-xtb", "gfn1-xtb"):
        from tblite.ase import TBLite

        method = "GFN2-xTB" if name == "gfn2-xtb" else "GFN1-xTB"
        return TBLite(method=method, **kwargs)
    elif name == "mace-off23":
        from mace.calculators import mace_off

        return mace_off(**kwargs)
    else:
        raise ValueError(f"Unknown calculator: {name}")


def pack_solvent(
    solute_path,
    solvent_path,
    n_solvent,
    out_path,
    packing_fraction=0.4,
    min_padding=2.0,
    tolerance=2.0,
    seed=1,
):
    """Pack n_solvent copies of solvent_path around the fixed solute.

    Box size is derived from the solute's and solvent's estimated vdW
    volumes rather than a fixed padding, so a handful of solvent molecules
    end up in a genuinely tight shell around the solute instead of
    scattered through a box sized for a much larger solvent count.
    `packing_fraction` is the assumed fraction of the box actually filled
    by vdW spheres in a disordered liquid-like packing (~0.4 is a
    generic, not solvent-specific, estimate); lower it to loosen the
    shell, raise it to tighten it. `min_padding` is a floor so the box
    always clears the solute's own extent even when n_solvent is small
    relative to the solute (e.g. a large solute with few solvent
    molecules).
    """
    solute = read(solute_path)
    solvent_unit = read(solvent_path)
    positions = solute.get_positions()

    required_volume = _vdw_volume(solute) + n_solvent * _vdw_volume(solvent_unit)
    box_side = (required_volume / packing_fraction) ** (1.0 / 3.0)

    extent = positions.max(axis=0) - positions.min(axis=0)
    box_side = max(box_side, extent.max() + 2 * min_padding)

    center = (positions.max(axis=0) + positions.min(axis=0)) / 2.0
    lo = center - box_side / 2.0
    hi = center + box_side / 2.0
    box = [*lo, *hi]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        solute_xyz = tmpdir / "solute.xyz"
        solvent_xyz = tmpdir / "solvent.xyz"
        packed_xyz = tmpdir / "packed.xyz"
        write(solute_xyz, solute)
        write(solvent_xyz, solvent_unit)

        inp_path = tmpdir / "pack.inp"
        inp_text = f"""\
seed {seed}
tolerance {tolerance}
filetype xyz
output {packed_xyz}

structure {solute_xyz}
  number 1
  fixed 0. 0. 0. 0. 0. 0.
  centerofmass
end structure

structure {solvent_xyz}
  number {n_solvent}
  inside box {box[0]:.4f} {box[1]:.4f} {box[2]:.4f} {box[3]:.4f} {box[4]:.4f} {box[5]:.4f}
end structure
"""
        inp_path.write_text(inp_text)

        result = subprocess.run(
            ["packmol"],
            stdin=open(inp_path),
            capture_output=True,
            text=True,
        )
        if not packed_xyz.exists():
            raise RuntimeError(
                f"packmol failed to produce output:\n{result.stdout}\n{result.stderr}"
            )

        packed = read(packed_xyz)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(out_path, packed)
    return packed


def run_one_job(condition, seed, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    packed_path = out_dir / "packed.xyz"
    atoms = pack_solvent(
        condition.solute_path,
        condition.solvent_path,
        condition.n_solvent,
        packed_path,
        seed=seed,
    )

    calc = get_calculator(condition.calculator, **condition.calculator_kwargs)
    atoms.calc = calc

    opt = BFGS(atoms, logfile=str(out_dir / "opt.log"))
    opt.run(fmax=0.5, steps=200)

    MaxwellBoltzmannDistribution(atoms, temperature_K=condition.temperature_K, rng=np.random.default_rng(seed))

    dyn = Langevin(
        atoms,
        timestep=condition.timestep_fs * fs,
        temperature_K=condition.temperature_K,
        friction=0.01,
    )

    traj_path = out_dir / "traj.xyz"
    energies = []

    def record():
        step = dyn.get_number_of_steps()
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        energies.append(
            {
                "step": step,
                "potential_energy_eV": epot,
                "kinetic_energy_eV": ekin,
                "total_energy_eV": epot + ekin,
            }
        )
        write(traj_path, atoms, append=step > 0)

    dyn.attach(record, interval=condition.dump_interval)
    dyn.run(condition.n_steps)

    with open(out_dir / "energies.json", "w") as f:
        json.dump(energies, f, indent=2)

    return str(out_dir)


def _run_job_worker(args):
    condition, seed, out_dir = args
    return run_one_job(condition, seed, out_dir)


def run_job_grid(conditions, n_seeds, out_root, n_workers=None):
    n_workers = n_workers or os.cpu_count()
    out_root = Path(out_root)

    jobs = []
    for condition in conditions:
        for seed in range(n_seeds):
            job_dir = out_root / f"{condition.label}_seed{seed}"
            jobs.append((condition, seed, job_dir))

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_run_job_worker, jobs)

    return results


if __name__ == "__main__":
    examples_dir = Path(__file__).parent / "examples"

    toy_conditions = [
        Condition(
            solute_path=str(examples_dir / "solute_toy.xyz"),
            solvent_path=str(examples_dir / "water.xyz"),
            n_solvent=4,
            calculator="gfn2-xtb",
            temperature_K=298.0,
            timestep_fs=1.0,
            n_steps=50,
            dump_interval=5,
            label="toy_water",
        ),
    ]

    out_root = Path(__file__).parent / "smoke_test_output"
    results = run_job_grid(toy_conditions, n_seeds=1, out_root=out_root, n_workers=1)
    print("Smoke test jobs written to:")
    for r in results:
        print(f"  {r}")
