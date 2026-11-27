"""Small representative cells for numerical qualification.

The cells are intentionally small (a few to tens of atoms) so a convergence matrix is
affordable. They stand in for chemistry and defect types, not for a production supercell,
and the manifest says so; their results are never the 1,024-site campaign's results.
Elements, lattices and sizes come from the configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.data import atomic_numbers
from ase.io import write
from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.core.fileio import atomic_write_text

NOT_PRODUCTION = "qualification cell; not the production supercell"


class MatrixSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matrix_element: str
    structure: str = "bcc"
    lattice_A: float
    supercell: tuple[int, int, int] = (3, 3, 3)
    solutes: list[str] = Field(default_factory=list)
    gas: list[str] = Field(default_factory=list)
    interstitial_species: list[str] = Field(default_factory=list)
    rattle_A: float = 0.05
    training_configs: int = 4
    seed: int = 0


def _bulk(spec: MatrixSpec) -> Atoms:
    cell = bulk(spec.matrix_element, spec.structure, a=spec.lattice_A, cubic=True)
    return cell.repeat(spec.supercell)


def _midpoint_interstitial(atoms: Atoms, species: str) -> Atoms:
    out = atoms.copy()
    a = atoms.cell[0][0] / 3  # conventional cell edge is not stored; use the midpoint of an edge
    out.append(species)
    out.positions[-1] = np.array([a / 2, 0.0, a / 4]) % np.diag(atoms.cell)
    return out


def generate(spec: MatrixSpec, out_dir: str | Path) -> dict[str, Any]:
    """Write each cell as extxyz under ``out_dir`` and a manifest describing them."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    base = _bulk(spec)
    cells: dict[str, tuple[str, Atoms]] = {"bulk": ("bulk", base)}

    vacancy = base.copy()
    del vacancy[0]
    cells["vacancy"] = ("vacancy", vacancy)

    # Nearest neighbour of site 0 gives a vacancy-hop endpoint pair and linear images.
    distances = base.get_distances(0, range(1, len(base)), mic=True)
    neighbour = 1 + int(np.argmin(distances))
    initial = base.copy()
    del initial[0]
    final = base.copy()
    final.positions[0] = final.positions[neighbour]
    del final[neighbour]
    for index, fraction in enumerate((0.0, 0.5, 1.0)):
        image = initial.copy()
        shift = final.positions - initial.positions
        image.positions = initial.positions + fraction * shift
        cells[f"neb_image{index}"] = ("neb_image", image)

    for solute in spec.solutes:
        substituted = base.copy()
        substituted.symbols[0] = solute
        cells[f"sub_{solute}"] = ("substitution", substituted)
    for species in spec.interstitial_species:
        cells[f"int_{species}"] = ("interstitial", _midpoint_interstitial(base, species))
    for species in spec.gas:
        box = (
            Atoms(species, positions=[[0, 0, 0]])
            if species in atomic_numbers
            else molecule(species)
        )
        box.set_cell([10.0, 10.0, 10.0])
        box.center()
        box.pbc = True
        cells[f"gas_{species}"] = ("gas", box)

    for index in range(spec.training_configs):
        rattled = base.copy()
        rattled.positions += rng.normal(0.0, spec.rattle_A, rattled.positions.shape)
        strain = 1.0 + rng.uniform(-0.02, 0.02)
        rattled.set_cell(rattled.cell * strain, scale_atoms=True)
        cells[f"train{index}"] = ("bulk", rattled)

    manifest: dict[str, Any] = {"note": NOT_PRODUCTION, "cases": []}
    for name, (kind, atoms) in cells.items():
        path = out / f"{name}.extxyz"
        write(path, atoms)
        manifest["cases"].append(
            {
                "name": name,
                "kind": kind,
                "natoms": len(atoms),
                "structure": str(path),
                "formula": atoms.get_chemical_formula(),
            }
        )
    atomic_write_text(out / "manifest.json", json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    import argparse

    from matsim_agents.core.yamlio import load_yaml_mapping

    parser = argparse.ArgumentParser(
        description="Write qualification cells from a YAML matrix spec"
    )
    parser.add_argument("spec")
    parser.add_argument("out_dir")
    args = parser.parse_args(argv)
    manifest = generate(MatrixSpec.model_validate(load_yaml_mapping(args.spec)), args.out_dir)
    print(json.dumps([(c["name"], c["natoms"]) for c in manifest["cases"]]))
    return 0


__all__ = ["MatrixSpec", "NOT_PRODUCTION", "generate"]


if __name__ == "__main__":
    raise SystemExit(main())
