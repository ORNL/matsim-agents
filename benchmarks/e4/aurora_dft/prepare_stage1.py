"""Generate reproducible, non-submitted E4 Aurora Stage 1 input bundles."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import spglib
from ase import Atoms
from ase.build import bulk
from ase.geometry import find_mic
from ase.io import read, write

from benchmarks.e4.aurora_dft.planning import ALLOCATIONS, build_candidate_inventory

PAW_BY_ELEMENT = {
    "Co": "Co",
    "Cr": "Cr_pv",
    "Fe": "Fe_pv",
    "Hf": "Hf_pv",
    "Li": "Li_sv",
    "Mn": "Mn_pv",
    "Nb": "Nb_pv",
    "Ni": "Ni_pv",
    "O": "O",
    "P": "P",
    "Ta": "Ta_pv",
    "Ti": "Ti_pv",
    "V": "V_pv",
    "Zr": "Zr_sv",
}

FCC_MOMENTS = {"Co": 2.0, "Cr": 3.0, "Fe": 3.0, "Mn": 3.0, "Ni": 1.0}
BCC_SOURCE_POTCAR_SHA256 = "4c2efb63638e6f89bd80129da1c38cf658ccce4a5c6da45b5e18c760659c11e2"
FCC_SOURCE_POTCAR_SHA256 = "c0e6f0e3856534892c239c7a8ec155102fdb9ee703a10e88b94d0c13f7ae4918"
LFP_SOURCE_POTCAR_SHA256 = "f2a05d8ed7e9bfe3976233c793251a2ab96a49a35bc4f349babdc9f708ea4232"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


@lru_cache(maxsize=32)
def _inverse_permutations(permutations: tuple[tuple[int, ...], ...]) -> np.ndarray:
    return np.argsort(np.asarray(permutations, dtype=int), axis=1)


@lru_cache(maxsize=16)
def _geometry_signature(cell: tuple[float, ...], positions: tuple[float, ...]) -> bytes:
    payload = json.dumps(
        {"cell": cell, "fractional_positions": positions},
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).digest()


def _atoms_geometry_signature(atoms: Atoms) -> bytes:
    cell = tuple(np.round(np.asarray(atoms.cell), 8).ravel().tolist())
    positions = tuple(np.round(np.mod(atoms.get_scaled_positions(), 1.0), 8).ravel().tolist())
    return _geometry_signature(cell, positions)


def lattice_site_permutations(atoms: Atoms, *, preserve_species: bool) -> list[list[int]]:
    positions = np.mod(atoms.get_scaled_positions(), 1.0)
    type_numbers = atoms.numbers if preserve_species else np.ones(len(atoms), dtype=int)
    symmetry = spglib.get_symmetry((np.asarray(atoms.cell), positions, type_numbers), symprec=1e-5)
    if symmetry is None:
        raise ValueError("spglib could not determine parent-cell symmetry")

    scale = 10**7

    def key(position: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.mod(np.rint(np.mod(position, 1.0) * scale), scale).astype(int))

    site_by_key = {key(position): index for index, position in enumerate(positions)}
    permutations: list[list[int]] = []
    for rotation, translation in zip(symmetry["rotations"], symmetry["translations"], strict=True):
        permutation = [
            site_by_key.get(key(rotation @ position + translation), -1) for position in positions
        ]
        if all(index >= 0 for index in permutation):
            permutations.append(permutation)
    if not permutations:
        raise ValueError("parent-cell symmetry produced no valid site permutations")
    return permutations


def _canonical_species_hash(
    numbers: np.ndarray, inverse_permutations: np.ndarray, geometry: bytes
) -> str:
    species = np.asarray(numbers, dtype=np.uint8)
    canonical = min(species[row].tobytes() for row in inverse_permutations)
    return sha256_bytes(geometry + canonical)


def canonical_occupation_hash(atoms: Atoms, permutations: list[list[int]]) -> str:
    return _canonical_species_hash(
        atoms.numbers,
        _inverse_permutations(tuple(tuple(row) for row in permutations)),
        _atoms_geometry_signature(atoms),
    )


def _bcc_stage_records(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = [row for row in candidates if row["material_family"] == "bcc_refractory"]
    systems_by_order: dict[int, list[tuple[str, ...]]] = {}
    for order in (4, 5, 6):
        systems_by_order[order] = list(
            dict.fromkeys(
                tuple(row["elements"]) for row in records if len(row["elements"]) == order
            )
        )

    selected: list[dict[str, Any]] = []
    for order, ranks in ((4, (0, 7, 14)), (5, (0, 3, 5))):
        for rank in ranks:
            system = systems_by_order[order][rank]
            selected.append(
                next(
                    row
                    for row in records
                    if tuple(row["elements"]) == system
                    and row["composition_index"] == 1
                    and row["disorder_realization"] == 1
                )
            )
    six_element = systems_by_order[6][0]
    for composition_index in (1, 2):
        selected.append(
            next(
                row
                for row in records
                if tuple(row["elements"]) == six_element
                and row["composition_index"] == composition_index
                and row["disorder_realization"] == 1
            )
        )
    return selected


def _fcc_stage_records(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = [row for row in candidates if row["material_family"] == "fcc_hea"]
    return [
        next(
            row
            for row in records
            if row["composition_index"] == composition_index and row["disorder_realization"] == 1
        )
        for composition_index in range(1, 5)
    ]


def _lattice_occupations(parent: Atoms, composition: dict[str, int], seed: int) -> Atoms:
    if sum(composition.values()) != len(parent):
        raise ValueError("composition does not match the parent lattice site count")
    symbols = [symbol for symbol, count in sorted(composition.items()) for _ in range(count)]
    np.random.default_rng(seed).shuffle(symbols)
    structure = parent.copy()
    structure.set_chemical_symbols(symbols)
    return structure


def _fcc_parent(parent_108: Atoms) -> Atoms:
    lengths = parent_108.cell.lengths()
    if not np.allclose(lengths, lengths[0], atol=1e-6) or len(parent_108) != 108:
        raise ValueError("FCC source must be the archived 108-atom cubic Cantor cell")
    conventional_a = float(lengths[0] / 3.0)
    structure = bulk("Ni", "fcc", a=conventional_a, cubic=True).repeat((4, 4, 2))
    if len(structure) != 128:
        raise AssertionError("FCC 4x4x2 conventional supercell must contain 128 atoms")
    return structure


def _lifepo4_antisites(
    parent_112: Atoms, permutations: list[list[int]], *, limit: int = 4
) -> list[tuple[str, Atoms, int]]:
    if limit < 1:
        raise ValueError("LiFePO4 candidate limit must be positive")
    fe_sites = [i for i, symbol in enumerate(parent_112.get_chemical_symbols()) if symbol == "Fe"]
    li_sites = [i for i, symbol in enumerate(parent_112.get_chemical_symbols()) if symbol == "Li"]
    if len(fe_sites) != 16 or len(li_sites) != 16 or len(parent_112) != 112:
        raise ValueError("LiFePO4 parent must be the 112-atom 2x2x1 supercell")

    configurations: dict[str, tuple[Atoms, int]] = {}
    inverse_permutations = _inverse_permutations(tuple(tuple(row) for row in permutations))
    pristine_hash = _canonical_species_hash(
        parent_112.numbers, inverse_permutations, _atoms_geometry_signature(parent_112)
    )
    configurations[pristine_hash] = (parent_112.copy(), 0)
    geometry = _atoms_geometry_signature(parent_112)
    base_numbers = np.asarray(parent_112.numbers, dtype=np.uint8)

    single_pairs: dict[str, tuple[Atoms, int]] = {}
    for fe_index in fe_sites:
        for li_index in li_sites:
            numbers = base_numbers.copy()
            numbers[fe_index], numbers[li_index] = 3, 26
            canonical = _canonical_species_hash(numbers, inverse_permutations, geometry)
            if canonical not in single_pairs:
                structure = parent_112.copy()
                symbols = structure.get_chemical_symbols()
                symbols[fe_index], symbols[li_index] = "Li", "Fe"
                structure.set_chemical_symbols(symbols)
                single_pairs[canonical] = (structure, 1)
    if len(single_pairs) < 2:
        raise ValueError("fewer than two symmetry-distinct single antisite configurations")
    for canonical, configuration in single_pairs.items():
        configurations.setdefault(canonical, configuration)

    if limit == 4:
        selected = [(pristine_hash, parent_112.copy(), 0)]
        selected.extend(
            (canonical, structure, pair_count)
            for canonical, (structure, pair_count) in sorted(single_pairs.items())[:2]
        )
    else:
        selected = [(pristine_hash, parent_112.copy(), 0)]
        selected.extend(
            (canonical, structure, pair_count)
            for canonical, (structure, pair_count) in sorted(single_pairs.items())[: limit - 1]
        )

    for pair_count in (2, 3):
        if len(selected) >= limit:
            break
        found_enough = False
        for fe_group in itertools.combinations(fe_sites, pair_count):
            for li_group in itertools.combinations(li_sites, pair_count):
                numbers = base_numbers.copy()
                numbers[list(fe_group)] = 3
                numbers[list(li_group)] = 26
                canonical = _canonical_species_hash(numbers, inverse_permutations, geometry)
                if canonical not in configurations:
                    structure = parent_112.copy()
                    symbols = structure.get_chemical_symbols()
                    for index in fe_group:
                        symbols[index] = "Li"
                    for index in li_group:
                        symbols[index] = "Fe"
                    structure.set_chemical_symbols(symbols)
                    configurations[canonical] = (structure, pair_count)
                    selected.append((canonical, structure, pair_count))
                    if len(selected) == limit:
                        found_enough = True
                        break
            if found_enough:
                break
    if len(selected) > limit:
        raise AssertionError("LiFePO4 candidate selection exceeded its limit")
    return selected


def _write_poscar(structure: Atoms, path: Path, label: str) -> None:
    structure.info["comment"] = label
    write(path, structure, format="vasp", direct=True, sort=True, vasp5=True)


def _poscar_species(path: Path) -> list[tuple[str, int]]:
    lines = path.read_text().splitlines()
    return list(zip(lines[5].split(), map(int, lines[6].split()), strict=True))


@lru_cache(maxsize=32)
def _read_paw(path: str) -> bytes:
    return Path(path).read_bytes()


def _potcar_metadata(
    species: list[tuple[str, int]],
    paw_root: Path,
    verified_source_components: dict[str, dict[str, str]],
) -> tuple[dict[str, str], dict[str, str], str]:
    paw_labels: dict[str, str] = {}
    component_hashes: dict[str, str] = {}
    payload = bytearray()
    for symbol, _ in species:
        label = PAW_BY_ELEMENT[symbol]
        path = paw_root / label / "POTCAR"
        if not path.is_file():
            raise FileNotFoundError(f"PAW potential is missing: {path}")
        if label not in verified_source_components:
            raise ValueError(f"{label} is absent from the audited source POTCAR")
        component = _read_paw(str(path.resolve()))
        title = re.search(rb"(?m)^\s*TITEL\s*=\s*PAW_PBE\s+(\S+)\s+\S+\s*$", component)
        if title is None or label != title.group(1).decode("ascii"):
            raise ValueError(f"PAW component identity mismatch: {path}")
        paw_labels[symbol] = label
        component_hashes[symbol] = sha256_bytes(component)
        payload.extend(component)
    return paw_labels, component_hashes, sha256_bytes(bytes(payload))


def _verified_paw_source_components(
    source_potcar: Path, paw_root: Path
) -> dict[str, dict[str, str]]:
    source_bytes = source_potcar.read_bytes()
    components: dict[str, dict[str, str]] = {}
    for section in re.split(rb"End of Dataset\s*", source_bytes):
        if not section.strip():
            continue
        component = section + b"End of Dataset"
        title = re.search(
            rb"(?m)^\s*TITEL\s*=\s*PAW_PBE\s+(\S+)\s+(\S+)\s*$",
            component,
        )
        if title is None:
            raise ValueError(f"unrecognized PAW metadata in {source_potcar}")
        label, release = (value.decode("ascii") for value in title.groups())
        if label in components:
            raise ValueError(f"duplicate PAW component {label} in {source_potcar}")
        path = paw_root / label / "POTCAR"
        if not path.is_file():
            raise FileNotFoundError(f"PAW potential is missing: {path}")
        library_component = _read_paw(str(path.resolve()))
        if component.strip() != library_component.strip():
            raise ValueError(f"PAW data payload mismatch for {label}: {source_potcar} vs {path}")
        components[label] = {
            "title": f"PAW_PBE {label} {release}",
            "source_payload_sha256": sha256_bytes(component.strip()),
            "library_sha256": sha256_bytes(library_component),
            "library_path": str(path.resolve()),
        }
    if not components:
        raise ValueError(f"no PAW components found in {source_potcar}")
    return components


def _lifepo4_defect_metadata(
    parent: Atoms, structure: Atoms, poscar_path: Path
) -> tuple[list[dict[str, Any]], list[float]]:
    parent_symbols = parent.get_chemical_symbols()
    symbols = structure.get_chemical_symbols()
    written = read(poscar_path, format="vasp")
    changed = []
    for parent_site, (before, after) in enumerate(zip(parent_symbols, symbols, strict=True)):
        if before == after:
            continue
        delta = written.positions - structure.positions[parent_site]
        _, distances = find_mic(delta, cell=written.cell, pbc=written.pbc)
        distances = np.asarray(distances)
        written_site = int(np.argmin(distances))
        if distances[written_site] > 1e-5 or written[written_site].symbol != after:
            raise ValueError("could not map a LiFePO4 defect site into the staged POSCAR")
        changed.append(
            {
                "site": written_site,
                "parent_site": parent_site,
                "from": before,
                "to": after,
            }
        )
    fe_to_li = [item["parent_site"] for item in changed if item["from"] == "Fe"]
    li_to_fe = [item["parent_site"] for item in changed if item["from"] == "Li"]
    distances = sorted(
        float(parent.get_distance(fe_site, li_site, mic=True))
        for fe_site in fe_to_li
        for li_site in li_to_fe
    )
    return changed, distances


def _incar(family: str, species: list[tuple[str, int]], phase: str) -> str:
    if family == "bcc_refractory":
        raise ValueError("BCC inputs use the archived START template, not this writer")

    if family == "fcc_hea":
        tags = [
            ("PREC", "Accurate"),
            ("EDIFF", "1.0E-06"),
            ("ALGO", "Normal"),
            ("NELM", "200"),
            ("NELMIN", "4"),
            ("LREAL", ".FALSE."),
            ("LASPH", ".TRUE."),
            ("LORBIT", "11"),
            ("LCHARG", ".FALSE."),
            ("ENCUT", "480"),
            ("ISPIN", "2"),
            ("ISMEAR", "1"),
            ("SIGMA", "0.10"),
            ("KPAR", "8"),
        ]
        if phase == "relax":
            tags.extend(
                [
                    ("IBRION", "2"),
                    ("ISIF", "3"),
                    ("NSW", "200"),
                    ("EDIFFG", "-0.02"),
                    ("LWAVE", ".TRUE."),
                ]
            )
        else:
            tags.extend([("IBRION", "-1"), ("NSW", "0"), ("LWAVE", ".FALSE.")])
        tags.append(
            (
                "MAGMOM",
                " ".join(f"{count}*{FCC_MOMENTS[symbol]:.1f}" for symbol, count in species),
            )
        )
    elif family == "lifepo4":
        if phase == "relax":
            tags = [
                ("ENCUT", "520"),
                ("EDIFF", "1.0E-06"),
                ("PREC", "Accurate"),
                ("ISMEAR", "0"),
                ("SIGMA", "0.05"),
                ("ISPIN", "2"),
                ("IBRION", "2"),
                ("ISIF", "3"),
                ("NSW", "200"),
                ("EDIFFG", "-0.01"),
                ("LWAVE", ".TRUE."),
                ("LCHARG", ".FALSE."),
                ("LORBIT", "11"),
                ("ALGO", "Normal"),
                ("KPAR", "1"),
                ("NCORE", "12"),
            ]
        else:
            tags = [
                ("ENCUT", "520"),
                ("EDIFF", "1.0E-06"),
                ("PREC", "Accurate"),
                ("ISMEAR", "0"),
                ("SIGMA", "0.05"),
                ("ISPIN", "2"),
                ("IBRION", "-1"),
                ("NSW", "0"),
                ("LWAVE", ".FALSE."),
                ("LCHARG", ".FALSE."),
                ("LORBIT", "11"),
                ("ALGO", "Normal"),
                ("KPAR", "1"),
                ("NCORE", "12"),
            ]
        tags.append(
            (
                "MAGMOM",
                " ".join(
                    f"{count}*{5.0 if symbol == 'Fe' else 0.60:.2f}" for symbol, count in species
                ),
            )
        )
    else:
        raise ValueError(f"unknown material family: {family}")
    return "".join(f"{key} = {value}\n" for key, value in tags)


def _bcc_stage_incar(start: str, precision: str, *, restart: bool) -> str:
    if re.search(r"(?im)^\s*EDIFFG\s*=", start):
        raise ValueError("archived BCC START unexpectedly contains EDIFFG")
    incar = start.replace("PRECISION", precision)
    if restart:
        incar = re.sub(r"(?im)^(\s*ISTART\s*=\s*)0\b", r"\g<1>1", incar)
        incar = re.sub(r"(?im)^(\s*ICHARG\s*=\s*)2\b", r"\g<1>1", incar)
    return f"{incar.rstrip()}\nEDIFFG = -0.01\n"


def _case_protocol(family: str, source: dict[str, str]) -> dict[str, Any]:
    protocols: dict[str, dict[str, Any]] = {
        "bcc_refractory": {
            "protocol_id": "e4-bcc-production-low-to-normal-v2",
            "xc": "PBE",
            "precision_stages": ["Low", "Normal"],
            "ionic_convergence": {
                "EDIFF": "1.0E-6 eV",
                "EDIFFG": "-0.01 eV/Angstrom; newly approved production setting, "
                "not recovered from the archived six-element input",
                "maximum_steps_per_stage": 240,
            },
            "incomplete_tag": (
                "The prior A/B and restart runs do not qualify production outputs; "
                "they used the archived START without an explicit EDIFFG."
            ),
            "restart": "Pilot PBS workflow validates checkpoint files and SHA256 manifests.",
            "source": source,
            "qualification_gate": (
                "Enforce the cumulative 240-step ceiling per precision stage, verify "
                "the final force norms against 0.01 eV/Angstrom, and complete "
                "normal-precision relaxations."
            ),
        },
        "fcc_hea": {
            "protocol_id": "e2-cantor-fcc-adapted-128-v1",
            "xc": "PBE",
            "precision_stages": ["Accurate"],
            "incomplete_tag": (
                "Archived 108-atom relaxations repeatedly hit NELM=200; "
                "128-atom SCF is unqualified."
            ),
            "restart": "Requires a Stage 1 soft-stop/restart validation before execution.",
            "source": source,
            "qualification_gate": (
                "Confirm every ionic-step SCF convergence and force acceptance "
                "without changing the archived magnetic initialization."
            ),
        },
        "lifepo4": {
            "protocol_id": "e2-lifepo4-112-derived-v1",
            "xc": "PBE; no Hubbard U",
            "precision_stages": ["Accurate"],
            "incomplete_tag": (
                "112-atom input is derived from the archived 28-atom cell; production "
                "VASP 6.6.1 compatibility with the archived 6.6.0 run remains unverified."
            ),
            "restart": "Requires a Stage 1 soft-stop/restart validation before execution.",
            "source": source,
            "qualification_gate": (
                "Verify PAW provenance and restart behavior; do not interpret as PBE+U "
                "or magnetic ground-state validation."
            ),
        },
    }
    return protocols[family]


def _stage_pbs(output: Path) -> None:
    definitions = (
        ("bcc", "bcc_refractory", "HydraGNN"),
        ("fcc", "fcc_hea", "GenesisHackathonOct26"),
        ("lifepo4", "lifepo4", "GridAI"),
    )
    for short, family, allocation in definitions:
        path = output / f"pbs-dryrun-{short}.pbs"
        path.write_text(
            "\n".join(
                [
                    f"#PBS -N e4-s1-{short}-preflight",
                    f"#PBS -A {allocation}",
                    "#PBS -q capacity",
                    "#PBS -l select=1",
                    "#PBS -l place=scatter",
                    "#PBS -l walltime=08:00:00",
                    "#PBS -l filesystems=home:flare",
                    "#PBS -k doe",
                    "#PBS -j oe",
                    "set -euo pipefail",
                    (
                        f'echo "E4 Stage 1 preflight only: family={family}; '
                        f'allocation={allocation}; nodes=1; walltime=08:00:00"'
                    ),
                    'echo "No VASP executable, mpiexec, or nested qsub is invoked."',
                    "exit 0",
                    "",
                ]
            )
        )


def prepare_stage1(
    *,
    bcc_parent_path: Path,
    fcc_parent_path: Path,
    lifepo4_parent_path: Path,
    bcc_start_path: Path,
    paw_root: Path,
    output: Path,
    full_inventory: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing Stage 1 bundle: {output}")
    bcc_source_potcar = bcc_parent_path.resolve().parent / "POTCAR"
    fcc_source_potcar = fcc_parent_path.resolve().parent / "POTCAR"
    lifepo4_source_potcar = lifepo4_parent_path.resolve().parent / "POTCAR"
    if bcc_source_potcar.is_file() and sha256_file(bcc_source_potcar) != BCC_SOURCE_POTCAR_SHA256:
        raise ValueError("BCC source POTCAR hash differs from the audited E4 A/B input")
    if fcc_source_potcar.is_file() and sha256_file(fcc_source_potcar) != FCC_SOURCE_POTCAR_SHA256:
        raise ValueError("FCC source POTCAR hash differs from the audited E2 input")
    if not lifepo4_source_potcar.is_file():
        raise FileNotFoundError(f"LiFePO4 source POTCAR is missing: {lifepo4_source_potcar}")
    if sha256_file(lifepo4_source_potcar) != LFP_SOURCE_POTCAR_SHA256:
        raise ValueError("LiFePO4 source POTCAR hash differs from the audited reference input")
    paw_sources = {
        "bcc_refractory": bcc_source_potcar,
        "fcc_hea": fcc_source_potcar,
        "lifepo4": lifepo4_source_potcar,
    }
    if not all(path.is_file() for path in paw_sources.values()):
        raise FileNotFoundError("one or more audited source POTCAR files are unavailable")
    verified_components = {
        family: _verified_paw_source_components(source_potcar, paw_root)
        for family, source_potcar in paw_sources.items()
    }
    output.mkdir(parents=True, exist_ok=False)
    candidates = build_candidate_inventory()
    bcc_records = (
        [row for row in candidates if row["material_family"] == "bcc_refractory"]
        if full_inventory
        else _bcc_stage_records(candidates)
    )
    fcc_records = (
        [row for row in candidates if row["material_family"] == "fcc_hea"]
        if full_inventory
        else _fcc_stage_records(candidates)
    )
    bcc_parent = read(bcc_parent_path)
    if len(bcc_parent) != 128:
        raise ValueError("BCC source must contain 128 sites")
    fcc_parent = _fcc_parent(read(fcc_parent_path))
    lifepo4_primitive = read(lifepo4_parent_path)
    if len(lifepo4_primitive) != 28:
        raise ValueError("LiFePO4 source must be the archived 28-atom parent")
    lifepo4_parent = lifepo4_primitive.repeat((2, 2, 1))
    records: list[dict[str, Any]] = []

    families: list[tuple[str, list[tuple[dict[str, Any], Atoms]]]] = []
    families.append(
        (
            "bcc_refractory",
            [
                (
                    candidate,
                    _lattice_occupations(
                        bcc_parent, candidate["composition"], candidate["random_seed"]
                    ),
                )
                for candidate in bcc_records
            ],
        )
    )
    families.append(
        (
            "fcc_hea",
            [
                (
                    candidate,
                    _lattice_occupations(
                        fcc_parent, candidate["composition"], candidate["random_seed"]
                    ),
                )
                for candidate in fcc_records
            ],
        )
    )
    lfp_permutations = lattice_site_permutations(lifepo4_parent, preserve_species=True)
    lfp_cases = _lifepo4_antisites(
        lifepo4_parent, lfp_permutations, limit=510 if full_inventory else 4
    )
    lfp_rows = []
    for index, (canonical, structure, pair_count) in enumerate(lfp_cases, start=1):
        lfp_rows.append(
            (
                {
                    "calculation_id": f"E4-LFP-{index:04d}",
                    "material_family": "lifepo4",
                    "allocation": ALLOCATIONS["lifepo4"],
                    "atom_count": 112,
                    "composition": {"Li": 16, "Fe": 16, "P": 16, "O": 64},
                    "random_seed": None,
                    "antisite_exchange_pairs": pair_count,
                    "canonical_structure_id": canonical,
                    "configuration_category": (
                        "pristine" if pair_count == 0 else f"{pair_count}-antisite-pair"
                    ),
                },
                structure,
            )
        )
    families.append(("lifepo4", lfp_rows))

    source_paths = {
        "bcc_refractory": bcc_parent_path.resolve(),
        "fcc_hea": fcc_parent_path.resolve(),
        "lifepo4": lifepo4_parent_path.resolve(),
    }
    for family, family_rows in families:
        family_dir = output / family
        family_dir.mkdir()
        parent_hash = sha256_file(source_paths[family])
        source = {
            "parent_path": str(source_paths[family]),
            "parent_poscar_sha256": parent_hash,
        }
        source_potcar = paw_sources[family]
        source["parent_potcar_sha256"] = sha256_file(source_potcar)
        protocol = _case_protocol(family, source)
        permutations = lattice_site_permutations(
            bcc_parent
            if family == "bcc_refractory"
            else fcc_parent
            if family == "fcc_hea"
            else lifepo4_parent,
            preserve_species=family == "lifepo4",
        )
        seen: set[str] = set()
        for candidate, structure in family_rows:
            case_id = candidate["calculation_id"]
            case_dir = family_dir / case_id
            case_dir.mkdir()
            poscar_path = case_dir / "POSCAR"
            _write_poscar(structure, poscar_path, case_id)
            species = _poscar_species(poscar_path)
            paw_labels, paw_hashes, potcar_sha = _potcar_metadata(
                species, paw_root, verified_components[family]
            )
            poscar_sha = sha256_file(poscar_path)
            canonical_hash = canonical_occupation_hash(structure, permutations)
            if canonical_hash in seen:
                raise ValueError(f"duplicate Stage 1 structure in {family}: {case_id}")
            seen.add(canonical_hash)
            defect_site_indices: list[dict[str, Any]] | None = None
            defect_separations_A: list[float] | None = None
            if family == "lifepo4":
                defect_site_indices, defect_separations_A = _lifepo4_defect_metadata(
                    lifepo4_parent, structure, poscar_path
                )

            if family == "bcc_refractory":
                shutil.copyfile(bcc_start_path, case_dir / "START")
                (case_dir / "KPOINTS").write_text("Gamma-centred\n0\nGamma\n2 2 2\n0 0 0\n")
                (case_dir / "POTCAR.order").write_text(
                    "\n".join(f"{paw_labels[symbol]} {paw_hashes[symbol]}" for symbol, _ in species)
                    + "\n"
                )
                (case_dir / "POTCAR.spec").write_text(
                    "\n".join(
                        [
                            (
                                f"{symbol} {paw_labels[symbol]} "
                                f"{paw_root.resolve() / paw_labels[symbol] / 'POTCAR'} "
                                f"{paw_hashes[symbol]}"
                            )
                            for symbol, _ in species
                        ]
                        + [f"assembled_potcar_sha256 {potcar_sha}", ""]
                    )
                )
                start = (case_dir / "START").read_text()
                (case_dir / "INCAR.low").write_text(_bcc_stage_incar(start, "Low", restart=False))
                (case_dir / "INCAR.normal").write_text(
                    _bcc_stage_incar(start, "Normal", restart=True)
                )
            else:
                kpoints = (
                    "Gamma-centred\n0\nGamma\n2 2 2\n0 0 0\n"
                    if family == "fcc_hea"
                    else "Automatic Monkhorst-Pack\n0\nMonkhorst-Pack\n2 4 8  0 0 0\n"
                )
                (case_dir / "KPOINTS").write_text(kpoints)
                (case_dir / "INCAR.relax").write_text(_incar(family, species, "relax"))
                (case_dir / "INCAR.static").write_text(_incar(family, species, "static"))
                (case_dir / "POTCAR.order").write_text(
                    "\n".join(f"{paw_labels[symbol]} {paw_hashes[symbol]}" for symbol, _ in species)
                    + "\n"
                )
                (case_dir / "POTCAR.spec").write_text(
                    "\n".join(
                        [
                            (
                                f"{symbol} {paw_labels[symbol]} "
                                f"{paw_root.resolve() / paw_labels[symbol] / 'POTCAR'} "
                                f"{paw_hashes[symbol]}"
                            )
                            for symbol, _ in species
                        ]
                        + [f"assembled_potcar_sha256 {potcar_sha}", ""]
                    )
                )

            manifest_record = {
                **candidate,
                "stage": None if full_inventory else 1,
                "method_signature": (
                    "bcc-pbe-350eV-gamma222-nonspin-v2-ediffg-001"
                    if family == "bcc_refractory"
                    else candidate.get("method_signature")
                ),
                "protocol": protocol,
                "source_parent_sha256": parent_hash,
                "structure_sha256": poscar_sha,
                "canonical_structure_id": canonical_hash,
                "species_order": species,
                "paw_labels": paw_labels,
                "paw_component_sha256": paw_hashes,
                "paw_component_source": {
                    paw_labels[symbol]: verified_components[family][paw_labels[symbol]]
                    for symbol, _ in species
                },
                "paw_root": str(paw_root.resolve()),
                "defect_site_indices": defect_site_indices,
                "defect_separations_A": defect_separations_A,
                "assembled_potcar_sha256": potcar_sha,
                "inputs": {
                    name: sha256_file(case_dir / name)
                    for name in sorted(
                        child.name for child in case_dir.iterdir() if child.is_file()
                    )
                },
                "output_directory": str((output / "outputs" / family / case_id).resolve()),
                "assigned_nodes": None if full_inventory else 1,
                "per_attempt_walltime": None if full_inventory else "08:00:00",
                "assigned_pbs_job": None,
                "status": "candidate_generated" if full_inventory else "staged_not_submitted",
                "execution_authorized": False,
                "external_duplicate_audit": "incomplete",
                "eligibility": "blocked_until_external_archive_audit",
                "provenance": (
                    "BCC 128-site E4 A/B parent; new seeded occupations"
                    if family == "bcc_refractory"
                    else (
                        "Derived 128-atom FCC 4x4x2 supercell from archived "
                        "108-atom Cantor lattice parameter"
                    )
                    if family == "fcc_hea"
                    else (
                        "Archived 28-atom olivine parent repeated 2x2x1; "
                        "symmetry-distinct antisite exchanges"
                    )
                ),
                "case_directory": str(case_dir.resolve()),
            }
            records.append(manifest_record)

    _stage_pbs(output)
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "status": "candidate_inventory_only" if full_inventory else "staged_not_submitted",
        "execution_authorized": False,
        "case_count": len(records),
        "families": {
            family: sum(row["material_family"] == family for row in records)
            for family in ALLOCATIONS
        },
        "lifepo4_antisite_pair_counts": {
            str(pair_count): sum(
                row.get("antisite_exchange_pairs") == pair_count
                for row in records
                if row["material_family"] == "lifepo4"
            )
            for pair_count in range(4)
        },
        "one_attempt_node_hours_max": (
            None
            if full_inventory
            else {
                family: 8 * sum(row["material_family"] == family for row in records)
                for family in ALLOCATIONS
            }
        ),
        "stage1_initial_attempt_node_hours_max": (None if full_inventory else 8 * len(records)),
        "potcar_files_staged": False,
        "external_duplicate_audit": (
            "incomplete; public OSTI structure payloads were not enumerated"
        ),
        "pbs_scripts": "preflight-only; they do not launch VASP",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcc-parent", type=Path, required=True)
    parser.add_argument("--fcc-parent", type=Path, required=True)
    parser.add_argument("--lifepo4-parent", type=Path, required=True)
    parser.add_argument("--bcc-start", type=Path, required=True)
    parser.add_argument("--paw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--full-inventory",
        action="store_true",
        help=(
            "Generate up to 2,040/510/510 BCC/FCC/LiFePO4 candidates, "
            "using only symmetry-distinct LiFePO4 configurations."
        ),
    )
    args = parser.parse_args()
    summary = prepare_stage1(
        bcc_parent_path=args.bcc_parent,
        fcc_parent_path=args.fcc_parent,
        lifepo4_parent_path=args.lifepo4_parent,
        bcc_start_path=args.bcc_start,
        paw_root=args.paw_root,
        output=args.output,
        full_inventory=args.full_inventory,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
