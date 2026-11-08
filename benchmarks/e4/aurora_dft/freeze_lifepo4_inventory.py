"""Freeze the audited, symmetry-distinct 112-atom LiFePO4 candidate set."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import ase
import numpy as np
import spglib
from ase import Atoms
from ase.io import read

from benchmarks.e4.aurora_dft.prepare_stage1 import (
    _lifepo4_antisites,
    _lifepo4_defect_metadata,
    _write_poscar,
    canonical_occupation_hash,
    lattice_site_permutations,
    sha256_file,
)

EXPECTED_COMPOSITION = {"Fe": 16, "Li": 16, "O": 64, "P": 16}
INVENTORY_SIZE = 510


def _validate_geometry(structure: Atoms, *, structure_id: str) -> float:
    if len(structure) != 112:
        raise ValueError(f"{structure_id} has {len(structure)} atoms, expected 112")
    if not np.all(np.isfinite(np.asarray(structure.cell))):
        raise ValueError(f"{structure_id} has non-finite cell vectors")
    if not np.all(np.isfinite(structure.positions)) or structure.get_volume() <= 0:
        raise ValueError(f"{structure_id} has invalid geometry")
    if Counter(structure.get_chemical_symbols()) != EXPECTED_COMPOSITION:
        raise ValueError(f"{structure_id} has an unexpected chemical composition")
    distances = structure.get_all_distances(mic=True)
    minimum = float(np.min(distances[np.triu_indices(len(structure), k=1)]))
    if not np.isfinite(minimum) or minimum < 0.5:
        raise ValueError(f"{structure_id} contains overlapping sites: min distance {minimum}")
    return minimum


def freeze_lifepo4_inventory(parent_path: Path, output: Path) -> dict[str, Any]:
    parent_path = parent_path.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen inventory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    parent_primitive = read(parent_path)
    if len(parent_primitive) != 28:
        raise ValueError(f"expected a 28-atom LiFePO4 parent, found {len(parent_primitive)}")
    parent = parent_primitive.repeat((2, 2, 1))
    permutations = lattice_site_permutations(parent, preserve_species=True)
    candidates = _lifepo4_antisites(parent, permutations, limit=INVENTORY_SIZE)
    if len(candidates) != INVENTORY_SIZE:
        raise ValueError(f"expected {INVENTORY_SIZE} structures, found {len(candidates)}")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        structures_dir = staging / "structures"
        structures_dir.mkdir()
        records: list[dict[str, Any]] = []
        canonical_ids: set[str] = set()
        minimum_distances: list[float] = []
        for index, (canonical_id, structure, pair_count) in enumerate(candidates, start=1):
            structure_id = f"E4-LFP-{index:04d}"
            if canonical_id in canonical_ids:
                raise ValueError(f"duplicate canonical identity: {structure_id}")
            canonical_ids.add(canonical_id)
            minimum_distances.append(_validate_geometry(structure, structure_id=structure_id))
            poscar_path = structures_dir / f"{structure_id}.POSCAR"
            _write_poscar(structure, poscar_path, structure_id)
            staged = read(poscar_path, format="vasp")
            _validate_geometry(staged, structure_id=structure_id)
            if canonical_occupation_hash(structure, permutations) != canonical_id:
                raise ValueError(f"canonical identity changed while staging {structure_id}")

            defect_sites, defect_distances = _lifepo4_defect_metadata(
                parent, structure, poscar_path
            )
            records.append(
                {
                    "calculation_id": structure_id,
                    "material_family": "lifepo4",
                    "atom_count": len(staged),
                    "composition": dict(sorted(Counter(staged.get_chemical_symbols()).items())),
                    "antisite_exchange_pairs": pair_count,
                    "configuration_category": (
                        "pristine" if pair_count == 0 else f"{pair_count}-antisite-pair"
                    ),
                    "initial_relaxed_status": "initial_unrelaxed_candidate",
                    "canonical_structure_id": canonical_id,
                    "structure_sha256": sha256_file(poscar_path),
                    "poscar": f"structures/{poscar_path.name}",
                    "volume_A3": float(staged.get_volume()),
                    "minimum_periodic_interatomic_distance_A": min(
                        _validate_geometry(staged, structure_id=structure_id),
                        minimum_distances[-1],
                    ),
                    "defect_sites": defect_sites,
                    "defect_pair_distances_A": defect_distances,
                }
            )

        manifest_path = staging / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")

        prepare_module = Path(__file__).with_name("prepare_stage1.py")
        configuration = {
            "inventory_id": "e4-lifepo4-112-symmetry-distinct-v1",
            "parent_poscar": str(parent_path),
            "parent_poscar_sha256": sha256_file(parent_path),
            "supercell_matrix": [2, 2, 1],
            "atom_count": 112,
            "composition": EXPECTED_COMPOSITION,
            "inventory_count": INVENTORY_SIZE,
            "selection": (
                "pristine; all symmetry-distinct single-pair configurations ordered by "
                "canonical SHA256; then lexicographic site-combination enumeration of "
                "two-pair and three-pair exchanges, stopping at 510"
            ),
            "symmetry_settings": {
                "preserve_parent_species": True,
                "spglib_symprec_A": 1e-5,
                "site_coordinate_quantization": 1e-7,
            },
            "generator_sha256": {
                Path(__file__).name: sha256_file(Path(__file__)),
                prepare_module.name: sha256_file(prepare_module),
            },
            "runtime_versions": {
                "ase": ase.__version__,
                "numpy": np.__version__,
                "spglib": spglib.__version__,
            },
            "potcar_files_included": False,
            "external_duplicate_audit": "incomplete",
            "execution_authorized": False,
        }
        config_path = staging / "configuration.json"
        config_path.write_text(json.dumps(configuration, indent=2) + "\n")

        category_counts = {
            str(pair_count): sum(
                record["antisite_exchange_pairs"] == pair_count for record in records
            )
            for pair_count in range(4)
        }
        summary = {
            "status": "frozen_candidate_inventory",
            "inventory_id": configuration["inventory_id"],
            "inventory_count": len(records),
            "canonical_identity_count": len(canonical_ids),
            "atom_count_counts": {"112": len(records)},
            "configuration_counts_by_exchange_pairs": category_counts,
            "minimum_interatomic_distance_A": min(minimum_distances),
            "maximum_minimum_interatomic_distance_A": max(minimum_distances),
            "manifest_sha256": sha256_file(manifest_path),
            "configuration_sha256": sha256_file(config_path),
            "execution_authorized": False,
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        if output.exists():
            raise FileExistsError(f"refusing to overwrite frozen inventory: {output}")
        os.rename(staging, output)
        return summary
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_lifepo4_inventory(args.parent, args.output), indent=2))


if __name__ == "__main__":
    main()
