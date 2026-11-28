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


class AlloySpec(BaseModel):
    """Random substitutional alloy: ``fractions`` is the site fraction of each solute."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,32}$")
    fractions: dict[str, float] = Field(min_length=1)


class MatrixSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matrix_element: str
    structure: str = "bcc"
    lattice_A: float
    supercell: tuple[int, int, int] = (3, 3, 3)
    solutes: list[str] = Field(default_factory=list)
    gas: list[str] = Field(default_factory=list)
    interstitial_species: list[str] = Field(default_factory=list)
    divacancy: bool = False  # first- and second-neighbour divacancies
    dumbbells: list[str] = Field(default_factory=list)  # self-interstitial directions, e.g. "110"
    dumbbell_separation_a: float = Field(default=0.5, gt=0.0)  # pair separation / lattice_A
    vacancy_gas: list[str] = Field(default_factory=list)  # gas atom at a vacancy (complex)
    alloys: list[AlloySpec] = Field(default_factory=list)
    magnetic_elements: list[str] = Field(default_factory=list)  # cells holding any are flagged
    rattle_A: float = 0.05
    training_configs: int = 4
    seed: int = 0


def _bulk(spec: MatrixSpec) -> Atoms:
    cell = bulk(spec.matrix_element, spec.structure, a=spec.lattice_A, cubic=True)
    return cell.repeat(spec.supercell)


def _midpoint_interstitial(atoms: Atoms, species: str, edge_A: float) -> Atoms:
    out = atoms.copy()
    out.append(species)
    out.positions[-1] = np.array([edge_A / 2, 0.0, edge_A / 4]) % np.diag(atoms.cell)
    return out


def _neighbour_shells(atoms: Atoms, count: int) -> list[int]:
    """Index of one atom in each of the first ``count`` neighbour shells of site 0."""
    distances = atoms.get_distances(0, range(1, len(atoms)), mic=True)
    shells: list[int] = []
    seen: list[float] = []
    for position in np.argsort(distances, kind="stable"):
        distance = float(distances[position])
        if all(abs(distance - known) > 1e-3 for known in seen):
            seen.append(distance)
            shells.append(1 + int(position))
        if len(shells) == count:
            break
    return shells


def _dumbbell(atoms: Atoms, direction: str, separation_A: float) -> Atoms:
    """Replace site 0 by two atoms of that species straddling it along ``direction``."""
    vector = np.array([float(c) for c in direction])
    if vector.shape != (3,) or not np.any(vector):
        raise ValueError(f"dumbbell direction {direction!r} must be three digits, e.g. '110'")
    unit = vector / np.linalg.norm(vector)
    out = atoms.copy()
    centre = out.positions[0].copy()
    out.positions[0] = centre - 0.5 * separation_A * unit
    out.append(out.symbols[0])
    out.positions[-1] = centre + 0.5 * separation_A * unit
    return out


def _alloy(atoms: Atoms, spec: AlloySpec, seed: int, index: int) -> Atoms:
    out = atoms.copy()
    rng = np.random.default_rng([seed, 1 + index])
    sites = rng.permutation(len(out))
    start = 0
    for solute, fraction in sorted(spec.fractions.items()):
        count = round(fraction * len(out))
        if count < 1 or start + count > len(out):
            raise ValueError(
                f"alloy {spec.name}: fraction {fraction} of {solute} gives {count} sites"
            )
        for site in sites[start : start + count]:
            out.symbols[int(site)] = solute
        start += count
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
        cells[f"int_{species}"] = (
            "interstitial",
            _midpoint_interstitial(base, species, spec.lattice_A),
        )
    if spec.divacancy:
        for label, partner in zip(("1nn", "2nn"), _neighbour_shells(base, 2), strict=False):
            pair = base.copy()
            del pair[[0, partner]]
            cells[f"divacancy_{label}"] = ("divacancy", pair)
    for direction in spec.dumbbells:
        cells[f"dumbbell_{direction}"] = (
            "dumbbell",
            _dumbbell(base, direction, spec.dumbbell_separation_a * spec.lattice_A),
        )
    for species in spec.vacancy_gas:
        complex_ = base.copy()
        complex_.symbols[0] = species
        cells[f"vac_{species}"] = ("complex", complex_)
    for index, alloy in enumerate(spec.alloys):
        cells[f"alloy_{alloy.name}"] = ("alloy", _alloy(base, alloy, spec.seed, index))
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
    magnetic = set(spec.magnetic_elements)
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
                "magnetic": bool(magnetic & set(atoms.get_chemical_symbols())),
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


__all__ = ["AlloySpec", "MatrixSpec", "NOT_PRODUCTION", "generate"]


if __name__ == "__main__":
    raise SystemExit(main())
