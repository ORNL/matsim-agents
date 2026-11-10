from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from pymatgen.core import Structure


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mesh_for_spacing(structure: Structure, spacing: float) -> tuple[int, int, int]:
    if spacing <= 0:
        raise ValueError("K-point spacing must be positive.")
    reciprocal_lengths = structure.lattice.reciprocal_lattice.abc
    return tuple(max(1, math.ceil(length / spacing - 1e-12)) for length in reciprocal_lengths)


def write_kpoints(
    structure_path: Path,
    kpoints_path: Path,
    metadata_path: Path,
    spacing: float | None,
    gamma_only: bool = False,
) -> dict[str, object]:
    structure = Structure.from_file(structure_path)
    mesh = (1, 1, 1) if gamma_only else mesh_for_spacing(structure, spacing or 0.0)
    title = (
        "Gamma-only molecular mesh"
        if gamma_only
        else f"Gamma-centered mesh; target spacing {spacing:g} 1/A"
    )
    text = f"{title}\n0\nGamma\n{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n"
    kpoints_path.write_text(text, encoding="ascii")
    record: dict[str, object] = {
        "policy": "gamma_only" if gamma_only else "gamma_centered_target_spacing",
        "center": "Gamma",
        "target_spacing_A^-1": None if gamma_only else spacing,
        "mesh_dimensions": list(mesh),
        "source_structure_sha256": sha256(structure_path),
        "kpoints_sha256": sha256(kpoints_path),
        "formula": (
            "N_i=max(1,ceil(|B_i|/spacing)); |B_i| includes 2*pi. "
            "Ceiling keeps reciprocal intervals at or below target spacing."
        ),
    }
    metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="ascii")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("structure", type=Path)
    parser.add_argument("kpoints", type=Path)
    parser.add_argument("metadata", type=Path)
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--spacing", type=float)
    policy.add_argument("--gamma-only", action="store_true")
    args = parser.parse_args()
    record = write_kpoints(
        args.structure,
        args.kpoints,
        args.metadata,
        args.spacing,
        gamma_only=args.gamma_only,
    )
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
