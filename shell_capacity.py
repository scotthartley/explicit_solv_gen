"""How many solvent molecules does one full first shell take?

Sizing helper for choosing `n_solvent`, kept out of solvate_md.py because it
answers a question about a system rather than doing any sampling.

Two regimes, and they want different molecule counts:

  * targeted microsolvation -- fewer molecules than a monolayer, placed to
    cover specific interaction sites, with ALPB describing everything else.
    The count is set by the number of sites, not by surface area.
  * full first shell -- enough molecules to cover the solute, at which point
    the shell itself is the model.

This script reports the monolayer capacity so you know which regime a given
`n_solvent` actually puts you in, and how that differs between conformers.

    python shell_capacity.py solute.xyz --solvent chcl3

Comparing conformers matters: a compact and an extended conformer have
different surface areas, so the same n_solvent is a different fraction of a
monolayer for each. Under cluster-continuum that is not automatically a
problem -- uncovered surface is what ALPB is for -- but it is worth knowing
the size of the asymmetry before reading anything into a conformer energy
difference.
"""

import single_thread  # noqa: F401  -- must precede numpy; see its docstring

import argparse

import numpy as np
from ase.io import read

from solvate_md import _unit_sphere_points, _vdw_radii_array, solvent_radius


def sasa(atoms, probe, n_points=4096):
    """Shrake-Rupley area of the surface traced by the solvent *centre*.

    That is the surface the solvent centres tile, so dividing it by the area
    one molecule occupies gives a monolayer count directly.
    """
    positions = atoms.get_positions()
    radii = _vdw_radii_array(atoms) + probe
    unit = _unit_sphere_points(n_points)

    total = 0.0
    for k in range(len(positions)):
        points = positions[k] + radii[k] * unit
        d = np.linalg.norm(points[:, None, :] - positions[None, :, :], axis=2)
        d[:, k] = np.inf
        exposed = (d > radii).all(axis=1).mean()
        total += exposed * 4.0 * np.pi * radii[k] ** 2
    return total


def monolayer_capacity(atoms, solvent):
    """(area, area per molecule, molecules in a complete first shell)."""
    r = solvent_radius(solvent)
    area = sasa(atoms, r)
    per_molecule = 2.0 * np.sqrt(3.0) * r**2  # hexagonal close packing on a surface
    return area, per_molecule, area / per_molecule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solute", nargs="+", help="solute geometry file(s)")
    parser.add_argument("--solvent", default="chcl3", help="solvent name (ALPB key)")
    parser.add_argument("--n-solvent", type=int, default=None,
                        help="report what fraction of a monolayer this is")
    args = parser.parse_args()

    r = solvent_radius(args.solvent)
    print(f"solvent {args.solvent}: effective radius {r:.2f} A "
          f"(from bulk density)\n")

    for path in args.solute:
        atoms = read(path)
        area, per_molecule, capacity = monolayer_capacity(atoms, args.solvent)
        print(f"{path}")
        print(f"  {len(atoms)} atoms, solvent-centre surface {area:.0f} A^2")
        print(f"  full first shell ~ {capacity:.0f} molecules "
              f"({per_molecule:.1f} A^2 each)")
        if args.n_solvent is not None:
            frac = args.n_solvent / capacity
            regime = "targeted microsolvation" if frac < 0.8 else "full shell"
            print(f"  n_solvent={args.n_solvent} -> {frac * 100:.0f}% of a "
                  f"monolayer ({regime})")
        print()


if __name__ == "__main__":
    main()
