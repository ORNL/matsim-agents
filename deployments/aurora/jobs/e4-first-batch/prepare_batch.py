#!/usr/bin/env python3
"""Freeze the approved 20-case E4 batch into private, restart-ready inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
STAGE1 = REPO / "research/paper/e4/aurora-dft-scaling/stage1-inputs-v2"
LFP_INVENTORY = REPO / "research/paper/e4/aurora-dft-scaling/lifepo4-inventory-v1"
DEFAULT_ROOT = REPO.parent / "e4-first-batch-01"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "batch-manifest.json"
EXECUTABLE = Path("/soft/applications/vasp/vasp.6.6.1/bin/vasp_std")

BCC_IDS = (
    "E4-BCC-0001",
    "E4-BCC-0561",
    "E4-BCC-1121",
    "E4-BCC-1201",
    "E4-BCC-1501",
    "E4-BCC-1701",
    "E4-BCC-1801",
    "E4-BCC-1805",
)
FCC_IDS = ("E4-FCC-0001", "E4-FCC-0006", "E4-FCC-0011", "E4-FCC-0016")
LFP_PAIR_COUNTS = {
    "E4-LFP-0002": 1,
    "E4-LFP-0003": 1,
    "E4-LFP-0010": 2,
    "E4-LFP-0025": 2,
    "E4-LFP-0050": 2,
    "E4-LFP-0100": 2,
    "E4-LFP-0505": 2,
    "E4-LFP-0506": 3,
}
ALLOCATIONS = {
    "bcc": ("HydraGNN", "job-bcc.pbs"),
    "fcc": ("GenesisHackathonOct26", "job-fcc.pbs"),
    "lifepo4": ("GridAI", "job-lifepo4.pbs"),
}
WALLTIME_SECONDS = 24 * 60 * 60
VASP_SHA256 = "3354871ac0b7a812ebe3444dbd3d5bb4176995a5c37c3859978b58b5c7b55eb6"
WEAK_SCALING_IDS = (
    "E4-BCC-0002",
    "E4-BCC-0562",
    "E4-BCC-1122",
    "E4-BCC-1202",
    "E4-BCC-1502",
    "E4-BCC-1702",
    "E4-BCC-1802",
    "E4-BCC-1806",
)
WEAK_SCALING_TEMPLATES = dict(
    zip(WEAK_SCALING_IDS, BCC_IDS, strict=True)
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            result[record["calculation_id"]] = record
    return result


def parse_poscar(path: Path) -> tuple[list[str], list[int]]:
    lines = path.read_text().splitlines()
    if len(lines) < 7:
        raise ValueError(f"{path}: incomplete POSCAR")
    elements = lines[0].split()
    counts = [int(value) for value in lines[6].split()]
    if len(elements) != len(counts) or sum(counts) < 1:
        raise ValueError(f"{path}: invalid species/count lines")
    return elements, counts


def parse_paw_spec(path: Path) -> tuple[dict[str, tuple[str, Path, str]], str | None]:
    paw: dict[str, tuple[str, Path, str]] = {}
    assembled_hash = None
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "assembled_potcar_sha256":
            assembled_hash = fields[-1]
        elif len(fields) >= 4:
            element, label, source_path, expected_hash = fields[:4]
            paw[element] = (label, Path(source_path), expected_hash)
    return paw, assembled_hash


def install_potcar(spec: Path, poscar: Path, destination: Path) -> list[dict[str, str]]:
    paw, expected_assembled = parse_paw_spec(spec)
    elements, _ = parse_poscar(poscar)
    components: list[dict[str, str]] = []
    chunks = []
    for element in elements:
        if element not in paw:
            raise ValueError(f"{spec}: no PAW mapping for POSCAR element {element}")
        label, source, expected = paw[element]
        actual = sha256(source)
        if actual != expected:
            raise ValueError(f"PAW hash mismatch for {element}/{label}: {source}")
        chunks.append(source.read_bytes())
        components.append(
            {
                "element": element,
                "label": label,
                "sha256": actual,
                "source": str(Path("external/vasp6/potcar") / label / "POTCAR"),
            }
        )
    assembled = b"".join(chunks)
    actual_assembled = hashlib.sha256(assembled).hexdigest()
    if expected_assembled and actual_assembled != expected_assembled:
        raise ValueError(f"assembled POTCAR hash mismatch for {spec}")
    destination.write_bytes(assembled)
    return components


def with_ediffg(text: str, expected: str) -> str:
    lines = text.splitlines()
    matches = [
        i for i, line in enumerate(lines) if re.match(r"^\s*EDIFFG\s*=", line, flags=re.IGNORECASE)
    ]
    if len(matches) > 1:
        raise ValueError("INCAR contains multiple active EDIFFG assignments")
    if matches:
        value = re.split(r"[#!]", lines[matches[0]], maxsplit=1)[0].split("=", 1)[1].strip()
        if float(value.replace("D", "E").replace("d", "e")) != float(expected):
            raise ValueError(f"unexpected EDIFFG={value}; expected {expected}")
    else:
        lines.append(f"EDIFFG = {expected}")
    return "\n".join(lines) + "\n"


def incar_values(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        active = re.split(r"[#!]", line, maxsplit=1)[0].strip()
        if "=" in active:
            key, value = active.split("=", 1)
            result[key.strip().upper()] = value.strip()
    return result


def verify_protocol(family: str, case_dir: Path) -> None:
    incar = incar_values((case_dir / "START").read_text())
    expected: dict[str, str]
    mesh: tuple[str, str]
    if family == "bcc":
        expected = {
            "ENCUT": "350",
            "ISMEAR": "1",
            "SIGMA": "0.2",
            "EDIFF": "1.0E-6",
            "EDIFFG": "-0.01",
            "ISPIN": "1",
            "KPAR": "8",
            "IBRION": "2",
            "ISIF": "3",
            "NSW": "240",
        }
        mesh = ("Gamma", "2 2 2")
    elif family == "fcc":
        expected = {
            "ENCUT": "480",
            "ISMEAR": "1",
            "SIGMA": "0.10",
            "EDIFF": "1.0E-06",
            "EDIFFG": "-0.02",
            "ISPIN": "2",
            "KPAR": "8",
            "IBRION": "2",
            "ISIF": "3",
            "NSW": "200",
        }
        mesh = ("Gamma", "2 2 2")
    else:
        expected = {
            "ENCUT": "520",
            "ISMEAR": "0",
            "SIGMA": "0.05",
            "EDIFF": "1.0E-06",
            "EDIFFG": "-0.01",
            "ISPIN": "2",
            "KPAR": "1",
            "NCORE": "12",
            "IBRION": "2",
            "ISIF": "3",
            "NSW": "200",
        }
        mesh = ("Monkhorst-Pack", "2 4 8")
    for key, wanted in expected.items():
        actual = incar.get(key)
        if actual is None:
            raise ValueError(f"{family} START is missing {key}")
        try:
            matches = float(actual.replace("D", "E").replace("d", "e")) == float(wanted)
        except ValueError:
            matches = actual == wanted
        if not matches:
            raise ValueError(f"{family} protocol mismatch: {key}={actual}, expected {wanted}")
    if family == "bcc" and incar.get("PREC") != "PRECISION":
        raise ValueError("BCC Low-to-Normal precision placeholder is missing")
    if family == "fcc" and "MAGMOM" not in incar:
        raise ValueError("FCC magnetic initialization is missing")
    if family == "lifepo4" and "MAGMOM" not in incar:
        raise ValueError("LiFePO4 magnetic initialization is missing")
    kpoints = (case_dir / "KPOINTS").read_text().splitlines()
    if (
        len(kpoints) < 4
        or kpoints[2].strip() != mesh[0]
        or " ".join(kpoints[3].split()[:3]) != mesh[1]
    ):
        raise ValueError(f"{family} KPOINTS mismatch: expected {mesh}")


def set_private_permissions(root: Path) -> None:
    root.chmod(stat.S_IRWXU)
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(stat.S_IRWXU)
        else:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def input_hashes(case_dir: Path) -> dict[str, str]:
    return {name: sha256(case_dir / name) for name in ("START", "POSCAR", "KPOINTS", "POTCAR")}


def prepare_case(
    calc_id: str,
    family: str,
    source_dir: Path,
    poscar_source: Path,
    record: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    allocation, pbs_script = ALLOCATIONS[family]
    case_dir = root / "inputs" / calc_id
    case_dir.mkdir(parents=True)

    shutil.copyfile(poscar_source, case_dir / "POSCAR")
    if family == "bcc":
        start = (source_dir / "START").read_text()
        start = with_ediffg(start, "-0.01")
        # Preserve the frozen Low-to-Normal source protocol as the stage template.
        nsw_max = 240
        ediffg = "-0.01"
        protocol = "bcc-pbe-low-normal-v1-approved-ediffg-0.01"
    else:
        start = with_ediffg(
            (source_dir / "INCAR.relax").read_text(), "-0.02" if family == "fcc" else "-0.01"
        )
        nsw_max = 200
        ediffg = "-0.02" if family == "fcc" else "-0.01"
        protocol = f"{family}-pbe-verified-v1"
    (case_dir / "START").write_text(start)
    shutil.copyfile(source_dir / "KPOINTS", case_dir / "KPOINTS")
    verify_protocol(family, case_dir)
    shutil.copyfile(source_dir / "POTCAR.spec", case_dir / "POTCAR.spec")
    shutil.copyfile(source_dir / "POTCAR.order", case_dir / "POTCAR.order")
    paw_components = install_potcar(
        case_dir / "POTCAR.spec", case_dir / "POSCAR", case_dir / "POTCAR"
    )
    hashes = input_hashes(case_dir)
    (case_dir / "inputs.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    )
    source_hashes = {
        str(path.relative_to(REPO)): sha256(path)
        for path in sorted(source_dir.iterdir())
        if path.is_file()
    }
    if family == "lifepo4":
        source_hashes[str(poscar_source.relative_to(REPO))] = sha256(poscar_source)
    structure_elements, structure_counts = parse_poscar(case_dir / "POSCAR")
    composition = dict(zip(structure_elements, structure_counts, strict=True))
    if sum(structure_counts) != 128 and family != "lifepo4":
        raise ValueError(f"{calc_id} is not a 128-atom configuration")
    if family == "lifepo4" and sum(structure_counts) != 112:
        raise ValueError(f"{calc_id} is not a 112-atom configuration")
    if record.get("atom_count", sum(structure_counts)) != sum(structure_counts):
        raise ValueError(f"{calc_id} atom count does not match frozen record")

    bundle_digest = hashlib.sha256(
        "".join(
            f"{name}:{hashes[name]}\n" for name in ("START", "POSCAR", "KPOINTS", "POTCAR")
        ).encode()
    ).hexdigest()
    metadata = {
        "calculation_id": calc_id,
        "material_family": family,
        "composition": composition,
        "atom_count": sum(structure_counts),
        "allocation": allocation,
        "pbs_script": str(Path("deployments/aurora/jobs/e4-first-batch") / pbs_script),
        "case_dir": os.path.relpath(case_dir.resolve(), REPO),
        "repeat": "repeat-01",
        "walltime_seconds": WALLTIME_SECONDS,
        "nodes": 1,
        "mpi_ranks": 12 if family == "lifepo4" else 8,
        "omp_threads_per_rank": 8,
        "kpar": 1 if family == "lifepo4" else 8,
        "ncore": 12 if family == "lifepo4" else None,
        "stage_nsw_max": nsw_max,
        "ediffg_eV_per_A": ediffg,
        "protocol_id": protocol,
        "structure_id": record.get("canonical_structure_id"),
        "source_poscar": str(poscar_source.relative_to(REPO)),
        "source_poscar_sha256": sha256(poscar_source),
        "prepared_input_bundle_sha256": bundle_digest,
        "antisite_exchange_pairs": record.get("antisite_exchange_pairs"),
        "source_hashes": source_hashes,
        "input_hashes": hashes,
        "paw_components": paw_components,
        "potcar_sha256": hashes["POTCAR"],
        "poscar_sha256": hashes["POSCAR"],
        "kpoints_sha256": hashes["KPOINTS"],
        "start_sha256": hashes["START"],
    }
    (case_dir / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    set_private_permissions(case_dir)
    return metadata


def prepare_weak_scaling_inputs(root: Path, source_manifest: Path, output_manifest: Path) -> list[dict[str, Any]]:
    """Freeze second disorder realizations for the unauthorized weak-scaling plan."""
    import sys

    sys.path.insert(0, str(REPO))
    from benchmarks.e4.aurora_dft.planning import build_candidate_inventory
    from benchmarks.e4.aurora_dft.prepare_stage1 import (
        _lattice_occupations,
        _write_poscar,
        canonical_occupation_hash,
        lattice_site_permutations,
    )
    from ase.io import read

    root = root.resolve()
    source_poscars = REPO / "research/paper/e4/aurora-dft-scaling/priority-scaling/weak-inputs"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty scaling input directory: {root}")
    if source_poscars.exists() and any(source_poscars.iterdir()):
        raise FileExistsError(f"refusing to overwrite frozen scaling POSCARs: {source_poscars}")
    if sha256(EXECUTABLE) != VASP_SHA256:
        raise ValueError(f"VASP executable hash mismatch: {EXECUTABLE}")

    source_manifest_data = json.loads(source_manifest.read_text())
    if source_manifest_data.get("batch_id") != "E4-first-production-batch-01":
        raise ValueError("unexpected source batch manifest")
    if source_manifest_data.get("execution_authorized") is not True:
        raise ValueError("source batch manifest is not the frozen production input set")
    source_manifest_sha256 = sha256(source_manifest)
    stage_records = read_jsonl(STAGE1 / "manifest.jsonl")
    candidates = {
        row["calculation_id"]: row
        for row in build_candidate_inventory()
        if row["material_family"] == "bcc_refractory"
    }
    parent_path = STAGE1 / "bcc_refractory" / "E4-BCC-0001" / "POSCAR"
    parent = read(parent_path, format="vasp")
    permutations = lattice_site_permutations(parent, preserve_species=False)
    known_canonical = {
        stage_records[calc_id]["canonical_structure_id"] for calc_id in BCC_IDS
    }
    known_poscar_hashes = {
        case["input_hashes"]["POSCAR"] for case in source_manifest_data["cases"] if case["material_family"] == "bcc"
    }
    source_poscars.mkdir(parents=True, mode=0o700)
    root.mkdir(parents=True, mode=0o700)
    (root / "inputs").mkdir(mode=0o700)
    records = []
    for calc_id in WEAK_SCALING_IDS:
        candidate = candidates[calc_id]
        structure = _lattice_occupations(parent, candidate["composition"], candidate["random_seed"])
        canonical = canonical_occupation_hash(structure, permutations)
        if canonical in known_canonical:
            raise ValueError(f"{calc_id} duplicates a selected strong-scaling occupation")
        known_canonical.add(canonical)
        source_dir = source_poscars / calc_id
        source_dir.mkdir(mode=0o700)
        poscar_source = source_dir / "POSCAR"
        _write_poscar(structure, poscar_source, calc_id)
        digest = sha256(poscar_source)
        if digest in known_poscar_hashes:
            raise ValueError(f"{calc_id} duplicates a selected POSCAR hash")
        known_poscar_hashes.add(digest)
        records.append(
            prepare_case(
                calc_id,
                "bcc",
                STAGE1 / "bcc_refractory" / WEAK_SCALING_TEMPLATES[calc_id],
                poscar_source,
                {**candidate, "canonical_structure_id": canonical},
                root,
            )
        )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "E4-Aurora-BCC-Weak-v1",
                "execution_authorized": False,
                "authorization": None,
                "source_manifest_sha256": source_manifest_sha256,
                "protocol_id": "bcc-pbe-low-normal-v1-approved-ediffg-0.01",
                "structure_identity": "POSCAR SHA-256 and symmetry-canonical occupation hash",
                "external_duplicate_audit": "incomplete",
                "cases": records,
            },
            indent=2,
        )
        + "\n"
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--weak-scaling", action="store_true")
    args = parser.parse_args()
    if args.weak_scaling:
        output_manifest = (
            REPO
            / "research/paper/e4/aurora-dft-scaling/priority-scaling/weak-input-manifest.json"
        )
        records = prepare_weak_scaling_inputs(args.root, args.manifest, output_manifest)
        print(f"Prepared {len(records)} unauthorized weak-scaling inputs at {args.root.resolve()}")
        print(f"Frozen manifest: {output_manifest}")
        return 0
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty batch directory: {root}")
    if sha256(EXECUTABLE) != VASP_SHA256:
        raise SystemExit(f"VASP executable hash mismatch: {EXECUTABLE}")

    stage_records = read_jsonl(STAGE1 / "manifest.jsonl")
    lfp_records = read_jsonl(LFP_INVENTORY / "manifest.jsonl")
    selections = []
    for family, ids, directory in (
        ("bcc", BCC_IDS, STAGE1 / "bcc_refractory"),
        ("fcc", FCC_IDS, STAGE1 / "fcc_hea"),
    ):
        for calc_id in ids:
            source = directory / calc_id
            record = stage_records[calc_id]
            selections.append((calc_id, family, source, source / "POSCAR", record))
    for calc_id, pair_count in LFP_PAIR_COUNTS.items():
        record = lfp_records[calc_id]
        if record["antisite_exchange_pairs"] != pair_count or record["atom_count"] != 112:
            raise SystemExit(f"frozen LiFePO4 selection mismatch: {calc_id}")
        source_poscar = LFP_INVENTORY / record["poscar"]
        selections.append(
            (calc_id, "lifepo4", STAGE1 / "lifepo4" / "E4-LFP-0002", source_poscar, record)
        )
    if len(selections) != 20 or len({item[0] for item in selections}) != 20:
        raise SystemExit("selection must contain exactly 20 unique structures")
    canonical_ids = [item[4].get("canonical_structure_id") for item in selections]
    if None in canonical_ids or len(set(canonical_ids)) != 20:
        raise SystemExit("selected structures contain missing or duplicate canonical IDs")

    root.mkdir(parents=True, mode=0o700)
    (root / "inputs").mkdir(mode=0o700)
    cases = [
        prepare_case(calc_id, family, source, poscar, record, root)
        for calc_id, family, source, poscar, record in selections
    ]
    total_by_allocation: dict[str, int] = {}
    for case in cases:
        total_by_allocation[case["allocation"]] = (
            total_by_allocation.get(case["allocation"], 0)
            + case["nodes"] * case["walltime_seconds"] // 3600
        )
    if total_by_allocation != {
        "HydraGNN": 192,
        "GenesisHackathonOct26": 96,
        "GridAI": 192,
    }:
        raise SystemExit(f"resource-envelope mismatch: {total_by_allocation}")
    manifest = {
        "schema_version": 1,
        "batch_id": "E4-first-production-batch-01",
        "execution_authorized": False,
        "executable": str(EXECUTABLE),
        "executable_sha256": VASP_SHA256,
        "queue": "capacity",
        "resource_caps_node_hours": total_by_allocation,
        "total_resource_cap_node_hours": 480,
        "cases": cases,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    print(f"Prepared {len(cases)} cases at {root}")
    print(f"Frozen manifest: {manifest_path}")
    print(f"Resource cap: {total_by_allocation}, total 480 node-hours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
