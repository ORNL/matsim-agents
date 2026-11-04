from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / "research/nb-ta-o/source_cifs/COD_1528678_zeta-Nb2O5.cif"
DEFAULT_OUTPUT = ROOT / "research/nb-ta-o/orderings"


def cation_permutations(
    structure: Structure, cation_indices: list[int], symprec: float = 0.15
) -> set[tuple[int, ...]]:
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    cation_coordinates = structure.frac_coords[cation_indices]
    permutations = set()

    for operation in analyzer.get_symmetry_operations(cartesian=False):
        transformed = (
            cation_coordinates @ operation.rotation_matrix.T + operation.translation_vector
        ) % 1
        displacement = transformed[:, None, :] - cation_coordinates[None, :, :]
        displacement -= np.round(displacement)
        distances = np.linalg.norm(displacement @ structure.lattice.matrix, axis=2)
        permutation = tuple(int(index) for index in np.argmin(distances, axis=1))
        if (
            len(set(permutation)) == len(cation_indices)
            and max(distances[i, permutation[i]] for i in range(len(permutation))) < 1e-4
        ):
            permutations.add(permutation)

    if not permutations:
        raise ValueError("No valid parent-symmetry permutations found for cation sites")
    return permutations


def ordering_orbits(site_count: int, substitutions: int, permutations: set[tuple[int, ...]]):
    configurations = {
        frozenset(configuration)
        for configuration in itertools.combinations(range(site_count), substitutions)
    }
    remaining = set(configurations)
    representatives = []

    while remaining:
        representative = min(remaining, key=lambda configuration: tuple(sorted(configuration)))
        orbit = {
            frozenset(permutation[index] for index in representative)
            for permutation in permutations
        }
        representatives.append((representative, orbit))
        remaining -= orbit

    return representatives, configurations


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NbTaO5 orderings from an Nb2O5 CIF")
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    parent = Structure.from_file(args.parent)
    if {element.symbol for element in parent.composition.elements} != {"Nb", "O"}:
        raise ValueError(f"Expected a binary Nb-O parent, got {parent.composition.formula}")
    if any(abs(site.species.num_atoms - 1) > 1e-8 for site in parent):
        raise ValueError("Disordered or partially occupied parent sites are not supported")

    cation_indices = [index for index, site in enumerate(parent) if site.specie.symbol == "Nb"]
    if not cation_indices or len(cation_indices) % 2:
        raise ValueError("Equimolar substitution requires an even, nonzero number of Nb sites")

    analyzer = SpacegroupAnalyzer(parent, symprec=0.15)
    permutations = cation_permutations(parent, cation_indices)
    representatives, configurations = ordering_orbits(
        len(cation_indices), len(cation_indices) // 2, permutations
    )
    covered = set().union(*(orbit for _, orbit in representatives))
    if covered != configurations:
        raise ValueError("Symmetry orbits do not cover every equimolar substitution")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parent_hash = hashlib.sha256(args.parent.read_bytes()).hexdigest()
    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="ascii") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=(
                "ordering_id",
                "formula",
                "atom_count",
                "parent_space_group",
                "ta_parent_site_indices",
                "symmetry_orbit_size",
                "parent_cod_id",
                "parent_sha256",
            ),
        )
        writer.writeheader()

        for ordering_number, (representative, orbit) in enumerate(representatives, start=1):
            ordering = parent.copy()
            ta_parent_indices = sorted(cation_indices[index] for index in representative)
            for index in ta_parent_indices:
                ordering.replace(index, "Ta")

            expected_cations = len(cation_indices) // 2
            if (
                ordering.composition["Nb"] != expected_cations
                or ordering.composition["Ta"] != expected_cations
                or ordering.composition["O"] != parent.composition["O"]
            ):
                raise ValueError(f"Unexpected ordering composition: {ordering.composition.formula}")
            if len(ordering) != len(parent):
                raise ValueError("Ordering changed the parent cell atom count")
            for parent_site, ordering_site in zip(parent, ordering, strict=True):
                if not np.allclose(parent_site.frac_coords, ordering_site.frac_coords, atol=1e-10):
                    raise ValueError("Ordering changed parent fractional coordinates")

            ordering_id = f"NbTaO5_ordering_{ordering_number:02d}"
            label_counts: dict[str, int] = {}
            for _index, site in enumerate(ordering):
                symbol = site.specie.symbol
                label_counts[symbol] = label_counts.get(symbol, 0) + 1
                site.label = f"{symbol}{label_counts[symbol]}"
            ordering.to(args.output_dir / f"{ordering_id}.cif")
            writer.writerow(
                {
                    "ordering_id": ordering_id,
                    "formula": ordering.composition.formula.replace(" ", ""),
                    "atom_count": len(ordering),
                    "parent_space_group": analyzer.get_space_group_symbol(),
                    "ta_parent_site_indices": ";".join(map(str, ta_parent_indices)),
                    "symmetry_orbit_size": len(orbit),
                    "parent_cod_id": "1528678",
                    "parent_sha256": parent_hash,
                }
            )

    print(
        f"Generated {len(representatives)} symmetry-inequivalent orderings "
        f"from {len(configurations)} equimolar substitutions; parent SG "
        f"{analyzer.get_space_group_symbol()} ({len(parent)} atoms)."
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
