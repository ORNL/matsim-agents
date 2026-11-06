from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase.build import bulk, molecule
from prepare_nbtao5_orderings import cation_permutations, ordering_orbits
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "research/nb-ta-o/source_cifs"
ORDERING_DIR = ROOT / "research/nb-ta-o/orderings"
PILOT_DIR = ROOT / "research/nb-ta-o/pilot"
PARENT_PATH = SOURCE_DIR / "COD_1528678_zeta-Nb2O5.cif"
SYMPRECS = (0.01, 0.05, 0.10, 0.15, 0.20, 0.50)
PILOT_SIZE = 5
OXYGEN_MOTIFS = ((2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent_site_mapping(parent: Structure, ordering: Structure) -> tuple[int, ...]:
    displacement = parent.frac_coords[:, None, :] - ordering.frac_coords[None, :, :]
    displacement -= np.round(displacement)
    distances = np.linalg.norm(displacement @ parent.lattice.matrix, axis=2)
    mapping = tuple(int(index) for index in np.argmin(distances, axis=1))
    if (
        len(set(mapping)) != len(parent)
        or max(
            distances[parent_index, ordering_index]
            for parent_index, ordering_index in enumerate(mapping)
        )
        > 1e-4
    ):
        raise ValueError("CIF sites do not map one-to-one onto parent coordinates")
    return mapping


def load_orderings(parent: Structure) -> tuple[list[dict[str, str]], list[Structure]]:
    with (ORDERING_DIR / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    structures = [Structure.from_file(ORDERING_DIR / f"{row['ordering_id']}.cif") for row in rows]
    cation_indices = [
        index for index, site in enumerate(parent) if site.specie.symbol in {"Nb", "Ta"}
    ]
    permutations = cation_permutations(parent, cation_indices)
    orbits, configurations = ordering_orbits(
        len(cation_indices), len(cation_indices) // 2, permutations
    )
    expected = {
        tuple(sorted(cation_indices[index] for index in representative)): len(orbit)
        for representative, orbit in orbits
    }
    seen: set[tuple[int, ...]] = set()
    for row, structure in zip(rows, structures, strict=True):
        ta_indices = tuple(sorted(map(int, row["ta_parent_site_indices"].split(";"))))
        if ta_indices in seen:
            raise ValueError(f"Duplicate enumerated representative: {ta_indices}")
        seen.add(ta_indices)
        if expected.get(ta_indices) != int(row["symmetry_orbit_size"]):
            raise ValueError(f"Orbit mismatch for {row['ordering_id']}")
        if (
            structure.composition["Nb"] != 4
            or structure.composition["Ta"] != 4
            or structure.composition["O"] != 20
            or len(structure) != len(parent)
        ):
            raise ValueError(f"Unexpected composition or atom count in {row['ordering_id']}")
        if not np.allclose(structure.lattice.matrix, parent.lattice.matrix, atol=1e-8):
            raise ValueError(f"Lattice changed in {row['ordering_id']}")
        mapping = parent_site_mapping(parent, structure)
        expected_ta = set(ta_indices)
        for parent_index, structure_index in enumerate(mapping):
            parent_species = parent[parent_index].specie.symbol
            expected_species = "Ta" if parent_index in expected_ta else parent_species
            if structure[structure_index].specie.symbol != expected_species:
                raise ValueError(f"Species/site-index mismatch in {row['ordering_id']}")
        if sum(structure[index].specie.symbol == "Ta" for index in mapping) != len(expected_ta):
            raise ValueError(f"Species/site-index mismatch in {row['ordering_id']}")
    if seen != set(expected) or sum(int(row["symmetry_orbit_size"]) for row in rows) != len(
        configurations
    ):
        raise ValueError("Ordering representatives do not partition all substitutions")
    return rows, structures


def cation_shells(
    parent: Structure, cutoff: float = 8.0
) -> list[tuple[float, list[tuple[int, int, tuple[int, int, int]]]]]:
    cation_indices = {index for index, site in enumerate(parent) if site.specie.symbol == "Nb"}
    centers, points, offsets, distances = parent.get_neighbor_list(cutoff)
    bonds: set[tuple[int, int, tuple[int, int, int]]] = set()
    for center, point, offset, distance in zip(centers, points, offsets, distances, strict=True):
        first, second = int(center), int(point)
        if first not in cation_indices or second not in cation_indices or distance < 1e-8:
            continue
        image = tuple(map(int, offset))
        forward = (first, second, image)
        reverse = (second, first, tuple(-value for value in image))
        bonds.add(min(forward, reverse))
    ordered = sorted(
        (parent.get_distance(first, second, jimage=image), first, second, image)
        for first, second, image in bonds
    )
    shells: list[tuple[float, list[tuple[int, int, tuple[int, int, int]]]]] = []
    for distance, first, second, image in ordered:
        if not shells or abs(distance - shells[-1][0]) > 0.12:
            shells.append((distance, []))
        shells[-1][1].append((first, second, image))
    return shells


def ordering_descriptors(
    rows: list[dict[str, str]],
    structures: list[Structure],
    parent: Structure,
    shells: list[tuple[float, list[tuple[int, int, tuple[int, int, int]]]]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row, structure in zip(rows, structures, strict=True):
        ta_indices = set(map(int, row["ta_parent_site_indices"].split(";")))
        shell_counts = [
            sum(first in ta_indices and second in ta_indices for first, second, _ in edges)
            for _, edges in shells[:6]
        ]
        first_shell_degree = {index: 0 for index in ta_indices}
        for first, second, _ in shells[0][1]:
            if first in ta_indices and second in ta_indices:
                first_shell_degree[first] += 1
                first_shell_degree[second] += 1
        degree_histogram = [
            sum(degree == value for degree in first_shell_degree.values()) for value in range(5)
        ]
        oxygen_motifs = oxygen_centered_motifs(structure)
        ta_oxygen_shells = {
            str(index): sorted(
                round(neighbor.nn_distance, 6)
                for neighbor in parent.get_neighbors(parent[index], 3.0)
                if neighbor.specie.symbol == "O"
            )
            for index in sorted(ta_indices)
        }
        result[row["ordering_id"]] = {
            "ta_ta_pair_counts_by_shell": shell_counts,
            "ta_first_shell_degree_histogram_0_to_4": degree_histogram,
            "oxygen_centered_motif_counts": oxygen_motifs,
            "ta_centered_oxygen_shells_A": ta_oxygen_shells,
        }
    return result


def oxygen_centered_motifs(structure: Structure, cutoff: float = 2.8) -> dict[str, int]:
    oxygen_indices = {index for index, site in enumerate(structure) if site.specie.symbol == "O"}
    neighbor_species: dict[int, list[str]] = {index: [] for index in oxygen_indices}
    centers, points, _, distances = structure.get_neighbor_list(cutoff)
    for center, point, distance in zip(centers, points, distances, strict=True):
        center, point = int(center), int(point)
        if (
            center in oxygen_indices
            and structure[point].specie.symbol in {"Nb", "Ta"}
            and distance < cutoff
        ):
            neighbor_species[center].append(structure[point].specie.symbol)
    counts = Counter(
        (
            len(species),
            sum(symbol == "Ta" for symbol in species),
        )
        for species in neighbor_species.values()
    )
    if set(counts) - set(OXYGEN_MOTIFS):
        raise ValueError(f"Unsupported oxygen-centered coordination motifs: {set(counts)}")
    return {
        f"cn{coordination}_ta{ta_count}": counts.get((coordination, ta_count), 0)
        for coordination, ta_count in OXYGEN_MOTIFS
    }


def select_representatives(
    descriptors: dict[str, dict[str, object]], size: int = PILOT_SIZE
) -> list[str]:
    ordering_ids = sorted(descriptors)
    feature_rows = [
        np.array(
            descriptors[ordering_id]["ta_ta_pair_counts_by_shell"]
            + descriptors[ordering_id]["ta_first_shell_degree_histogram_0_to_4"]
            + list(descriptors[ordering_id]["oxygen_centered_motif_counts"].values()),
            dtype=float,
        )
        for ordering_id in ordering_ids
    ]
    matrix = np.vstack(feature_rows)
    spread = np.ptp(matrix, axis=0)
    active = spread > 0
    scaled = (matrix[:, active] - matrix[:, active].min(axis=0)) / spread[active]
    first_shell_counts = matrix[:, 0]
    least_clustered = min(
        index for index, value in enumerate(first_shell_counts) if value == first_shell_counts.min()
    )
    most_clustered = min(
        index for index, value in enumerate(first_shell_counts) if value == first_shell_counts.max()
    )
    selected = [least_clustered, most_clustered]
    while len(selected) < size:
        remaining = [index for index in range(len(ordering_ids)) if index not in selected]
        distances = {
            index: min(np.linalg.norm(scaled[index] - scaled[chosen]) for chosen in selected)
            for index in remaining
        }
        maximum = max(distances.values())
        selected.append(
            min(index for index, distance in distances.items() if abs(distance - maximum) < 1e-12)
        )
    return [ordering_ids[index] for index in selected]


def minimum_pair_distances(structure: Structure) -> dict[str, float]:
    distances: dict[str, float] = {}
    centers, points, _, neighbor_distances = structure.get_neighbor_list(10.0)
    for center, point, distance in zip(centers, points, neighbor_distances, strict=True):
        if distance <= 1e-8:
            continue
        pair = ":".join(
            sorted((structure[int(center)].specie.symbol, structure[int(point)].specie.symbol))
        )
        distances[pair] = min(distances.get(pair, float("inf")), float(distance))
    return {pair: round(distance, 6) for pair, distance in sorted(distances.items())}


def grouped_structure(structure: Structure, species_order: tuple[str, ...]) -> Structure:
    indices = [
        index
        for symbol in species_order
        for index, site in enumerate(structure)
        if site.specie.symbol == symbol
    ]
    if len(indices) != len(structure):
        raise ValueError(f"Species order does not cover {structure.composition.formula}")
    return Structure(
        structure.lattice,
        [structure[index].species for index in indices],
        [structure[index].frac_coords for index in indices],
    )


def write_elemental_candidates() -> dict[str, dict[str, object]]:
    output_dir = PILOT_DIR / "elemental_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    structures = {
        "Nb-bcc": AseAtomsAdaptor.get_structure(bulk("Nb", "bcc", a=3.300)),
        "Ta-bcc": AseAtomsAdaptor.get_structure(bulk("Ta", "bcc", a=3.306)),
    }
    oxygen = molecule("O2")
    oxygen.set_cell([15.0, 15.0, 15.0])
    oxygen.center()
    oxygen.set_pbc(True)
    structures["O2-triplet"] = AseAtomsAdaptor.get_structure(oxygen)
    entries: dict[str, dict[str, object]] = {}
    for phase_id, structure in structures.items():
        path = output_dir / f"{phase_id}.vasp"
        Poscar(structure, comment=f"{phase_id}; generated reference geometry candidate").write_file(
            path
        )
        entries[phase_id] = {
            "formula": structure.composition.reduced_formula,
            "structure_path": path.relative_to(ROOT).as_posix(),
            "structure_sha256": sha256(path),
            "geometry_source": "ASE candidate geometry; not an approved or calculated reference",
            "status": "geometry_candidate_pending_method_and_PAW_approval",
        }
    manifest = {
        "schema_version": 1,
        "manifest_kind": "planning_inventory_not_executor_input",
        "reference_role": "elemental_endpoints",
        "method_status": "PBE candidate; homogeneous authorized PAW release unresolved",
        "phases": entries,
        "completeness": {"complete": False, "reason": "No VASP qualification or energy exists."},
    }
    path = PILOT_DIR / "elemental_reference_inventory.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    return entries


def main() -> None:
    parent = Structure.from_file(PARENT_PATH)
    parent_hash = sha256(PARENT_PATH)
    rows, structures = load_orderings(parent)
    symmetry = []
    for symprec in SYMPRECS:
        analyzer = SpacegroupAnalyzer(parent, symprec=symprec)
        permutations = cation_permutations(
            parent,
            [index for index, site in enumerate(parent) if site.specie.symbol == "Nb"],
            symprec=symprec,
        )
        symmetry.append(
            {
                "symprec_A": symprec,
                "space_group": analyzer.get_space_group_symbol(),
                "cation_site_permutation_count": len(permutations),
            }
        )
    matcher = StructureMatcher(
        ltol=1e-4,
        stol=1e-4,
        angle_tol=1e-3,
        primitive_cell=False,
        scale=False,
        comparator=ElementComparator(),
        attempt_supercell=True,
    )
    matches = [
        [rows[first]["ordering_id"], rows[second]["ordering_id"]]
        for first, second in itertools.combinations(range(len(structures)), 2)
        if matcher.fit(structures[first], structures[second])
    ]
    shells = cation_shells(parent)
    descriptors = ordering_descriptors(rows, structures, parent, shells)
    selected = select_representatives(descriptors)
    pilot_rank = {ordering_id: rank for rank, ordering_id in enumerate(selected, start=1)}
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for row, structure in zip(rows, structures, strict=True):
        ordering_id = row["ordering_id"]
        cif_path = ORDERING_DIR / f"{ordering_id}.cif"
        poscar_path = PILOT_DIR / "poscars" / f"{ordering_id}.POSCAR"
        poscar_path.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha256(cif_path)
        vasp_structure = grouped_structure(structure, ("Nb", "Ta", "O"))
        comment = f"{ordering_id}; parent COD 1528678; CIF SHA256 {source_hash}"
        Poscar(vasp_structure, comment=comment, sort_structure=False).write_file(poscar_path)
        poscar = Poscar.from_file(poscar_path)
        if poscar.site_symbols != ["Nb", "Ta", "O"] or poscar.natoms != [4, 4, 20]:
            raise ValueError(f"Unexpected POSCAR species order/counts for {ordering_id}")
        poscar_mapping = parent_site_mapping(structure, poscar.structure)
        if any(
            structure[parent_index].specie.symbol != poscar.structure[poscar_index].specie.symbol
            for parent_index, poscar_index in enumerate(poscar_mapping)
        ):
            raise ValueError(f"POSCAR species differ from source CIF for {ordering_id}")
        displacement = structure.frac_coords[:, None, :] - poscar.structure.frac_coords[None, :, :]
        displacement -= np.round(displacement)
        mapping_distances = np.linalg.norm(displacement @ structure.lattice.matrix, axis=2)
        max_site_mapping_error = max(
            mapping_distances[parent_index, poscar_index]
            for parent_index, poscar_index in enumerate(poscar_mapping)
        )
        entry: dict[str, object] = {
            "ordering_id": ordering_id,
            "formula": structure.composition.reduced_formula,
            "atom_count": len(structure),
            "parent_cif": PARENT_PATH.relative_to(ROOT).as_posix(),
            "parent_sha256": parent_hash,
            "ordering_cif": cif_path.relative_to(ROOT).as_posix(),
            "ordering_cif_sha256": sha256(cif_path),
            "ta_parent_site_indices": list(map(int, row["ta_parent_site_indices"].split(";"))),
            "symmetry_orbit_size": int(row["symmetry_orbit_size"]),
            "descriptors": descriptors[ordering_id],
            "minimum_pair_distances_A": minimum_pair_distances(structure),
            "poscar_max_site_mapping_error_A": round(float(max_site_mapping_error), 10),
            "pilot_selected": ordering_id in pilot_rank,
            "pilot_rank": pilot_rank.get(ordering_id),
        }
        entry["poscar"] = {
            "path": poscar_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(poscar_path),
            "species_order": poscar.site_symbols,
            "species_counts": poscar.natoms,
            "conversion": (
                "pymatgen POSCAR; species grouped Nb, Ta, O; cell and coordinates unchanged"
            ),
            "canonical_for_computation": True,
        }
        entries.append(entry)
    if matches:
        raise ValueError(f"Periodic structure matching found duplicate orderings: {matches}")
    manifest = {
        "schema_version": 1,
        "status": "structure_audited_dft_not_run",
        "parent": {
            "path": PARENT_PATH.relative_to(ROOT).as_posix(),
            "sha256": parent_hash,
            "formula": parent.composition.reduced_formula,
            "atom_count": len(parent),
        },
        "enumeration": {
            "raw_half_substitutions": 70,
            "symmetry_inequivalent_orderings": len(rows),
            "symmetry_audit": symmetry,
            "periodic_structure_matcher": {
                "tolerances": {"ltol": 1e-4, "stol": 1e-4, "angle_tol_degrees": 1e-3},
                "primitive_cell": False,
                "scale": False,
                "attempt_supercell": True,
                "species_preserved": True,
                "duplicate_pairs": matches,
            },
        },
        "descriptor_method": {
            "periodic_cation_neighbor_cutoff_A": 8.0,
            "radial_shell_grouping_tolerance_A": 0.12,
            "reported_shells": [round(distance, 6) for distance, _ in shells[:6]],
            "oxygen_centered_neighbor_cutoff_A": 2.8,
            "oxygen_centered_motif_bins": [
                f"cn{coordination}_ta{ta_count}" for coordination, ta_count in OXYGEN_MOTIFS
            ],
            "pilot_selection": (
                "least and most first-shell Ta-Ta clustered, then greedy maximin in "
                "min-max-scaled six-shell pair counts, Ta first-shell degree histogram, "
                "and oxygen-centered Nb/Ta coordination-motif counts; ties by ordering ID"
            ),
            "pilot_size": len(selected),
            "pilot_ids_in_selection_order": selected,
        },
        "poscar_conversion": {
            "tool": "pymatgen",
            "species_order": ["Nb", "Ta", "O"],
            "structure_changes": (
                "species grouping only; lattice and fractional coordinates unchanged"
            ),
            "canonical_computational_format": "POSCAR",
            "cif_role": "source/provenance and independent structure-audit input",
            "vasp_execution": False,
        },
        "orderings": entries,
    }
    path = PILOT_DIR / "ordering_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    elemental = write_elemental_candidates()
    print(f"Audited {len(rows)} orderings; selected {', '.join(selected)}.")
    print(f"Wrote {path.relative_to(ROOT)} and {len(entries)} ordering POSCARs.")
    print(f"Wrote {len(elemental)} elemental structure candidates; no DFT was run.")


if __name__ == "__main__":
    main()
