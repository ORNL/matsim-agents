"""Deterministic candidate inventories and disjoint E4 run assignments."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

ELEMENTS_BCC = ("Nb", "Ta", "V", "Hf", "Zr", "Ti")
ELEMENTS_FCC = ("Cr", "Mn", "Fe", "Co", "Ni")
ALLOCATIONS = {
    "bcc_refractory": "HydraGNN",
    "fcc_hea": "GenesisHackathonOct26",
    "lifepo4": "GridAI",
}
NODE_TIERS = (1, 2, 4, 8, 16, 32, 64, 128)
BLOCKED_STATUS = "blocked_pending_archive_audit_and_queue_eligibility"


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        result += digit * factor
        factor /= base
    return result


def _primes(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % prime for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def _quantize_composition(
    fractions: list[float], atoms: int = 128, minimum: int = 4
) -> tuple[int, ...]:
    remaining = atoms - minimum * len(fractions)
    if remaining < 0:
        raise ValueError("minimum atom count exceeds the cell size")
    quotas = [remaining * value for value in fractions]
    counts = [int(value) for value in quotas]
    for index in sorted(range(len(quotas)), key=lambda i: (-(quotas[i] - counts[i]), i))[
        : remaining - sum(counts)
    ]:
        counts[index] += 1
    result = tuple(value + minimum for value in counts)
    if sum(result) != atoms or min(result) < minimum:
        raise AssertionError("composition quantization violated its atom-count invariant")
    return result


def sample_compositions(
    elements: tuple[str, ...], count: int, *, seed: int, atoms: int = 128
) -> list[tuple[int, ...]]:
    """Choose deterministic, space-filling integer compositions on a simplex."""
    if count < 1 or len(elements) < 2:
        raise ValueError("at least two elements and one composition are required")
    minimum = 4
    if atoms < minimum * len(elements):
        raise ValueError("cell is too small to represent every constituent meaningfully")
    bases = _primes(len(elements))
    compositions: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    initial = [[1.0 / len(elements)] * len(elements)]
    initial.extend(
        [
            0.4 if element_index == enriched_index else 0.6 / (len(elements) - 1)
            for element_index in range(len(elements))
        ]
        for enriched_index in range(len(elements))
    )
    for fractions in initial:
        composition = _quantize_composition(fractions, atoms, minimum)
        if composition not in seen and len(compositions) < count:
            seen.add(composition)
            compositions.append(composition)
    index = max(1, seed)
    while len(compositions) < count:
        fractions = [_radical_inverse(index, base) for base in bases]
        total = sum(fractions)
        composition = _quantize_composition([value / total for value in fractions], atoms, minimum)
        if composition not in seen:
            seen.add(composition)
            compositions.append(composition)
        index += 1
        if index > seed + count * 10000:
            raise RuntimeError(f"could not produce {count} distinct compositions for {elements}")
    return compositions


def build_candidate_inventory() -> list[dict[str, Any]]:
    """Describe candidates, without claiming their structures passed archive audit."""
    records: list[dict[str, Any]] = []
    order_counts = {4: 20, 5: 25, 6: 60}
    for subset_index, elements in enumerate(
        subset for order in (4, 5, 6) for subset in itertools.combinations(ELEMENTS_BCC, order)
    ):
        composition_count = order_counts[len(elements)]
        compositions = sample_compositions(elements, composition_count, seed=1000 + subset_index)
        for composition_index, atom_counts in enumerate(compositions, start=1):
            for realization in range(1, 5):
                calculation_id = f"E4-BCC-{len(records) + 1:04d}"
                records.append(
                    {
                        "calculation_id": calculation_id,
                        "material_family": "bcc_refractory",
                        "allocation": ALLOCATIONS["bcc_refractory"],
                        "elements": list(elements),
                        "composition": dict(zip(elements, atom_counts, strict=True)),
                        "atom_count": 128,
                        "parent_structure": "128-site BCC, exact archived lattice pending audit",
                        "composition_index": composition_index,
                        "disorder_realization": realization,
                        "random_seed": 400000 + len(records),
                        "structure_hash": None,
                        "canonical_structure_id": None,
                        "method_signature": "bcc-pbe-350eV-gamma222-nonspin-v1-candidate",
                        "status": BLOCKED_STATUS,
                    }
                )

    fcc_compositions = sample_compositions(ELEMENTS_FCC, 102, seed=8001)
    equiatomic_nearest = (26, 26, 26, 25, 25)
    fcc_compositions[0] = equiatomic_nearest
    if len(set(fcc_compositions)) != 102:
        replacement = sample_compositions(ELEMENTS_FCC, 103, seed=18001)
        fcc_compositions = [equiatomic_nearest] + [
            composition for composition in replacement if composition != equiatomic_nearest
        ][:101]
    for composition_index, atom_counts in enumerate(fcc_compositions, start=1):
        for realization in range(1, 6):
            records.append(
                {
                    "calculation_id": f"E4-FCC-{len(records) - 2040 + 1:04d}",
                    "material_family": "fcc_hea",
                    "allocation": ALLOCATIONS["fcc_hea"],
                    "elements": list(ELEMENTS_FCC),
                    "composition": dict(zip(ELEMENTS_FCC, atom_counts, strict=True)),
                    "atom_count": 128,
                    "parent_structure": "128-site FCC, archived Cantor lattice pending audit",
                    "composition_index": composition_index,
                    "disorder_realization": realization,
                    "random_seed": 500000 + (len(records) - 2040),
                    "structure_hash": None,
                    "canonical_structure_id": None,
                    "method_signature": "fcc-pbe-480eV-gamma222-spin-v1-candidate",
                    "status": BLOCKED_STATUS,
                }
            )

    lifepo4_categories = (
        [("pristine", 0)] + [("antisite", 1)] * 8 + [("antisite", 2)] * 496 + [("antisite", 3)] * 5
    )
    for index, (category, exchange_count) in enumerate(lifepo4_categories, start=1):
        records.append(
            {
                "calculation_id": f"E4-LFP-{index:04d}",
                "material_family": "lifepo4",
                "allocation": ALLOCATIONS["lifepo4"],
                "elements": ["Li", "Fe", "P", "O"],
                "composition": {"Li": 16, "Fe": 16, "P": 16, "O": 64},
                "atom_count": 112,
                "parent_structure": "LiFePO4 olivine 2x2x1; derived from archived 28-atom parent",
                "configuration_category": category,
                "antisite_exchange_pairs": exchange_count,
                "defect_site_indices": None,
                "random_seed": 600000 + index,
                "structure_hash": None,
                "canonical_structure_id": None,
                "method_signature": "lifepo4-pbe-520eV-accurate-spin-v1-candidate",
                "status": BLOCKED_STATUS,
            }
        )
    return records


def build_run_matrix(candidates: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    candidates = candidates if candidates is not None else build_candidate_inventory()
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in candidates:
        by_family.setdefault(record["material_family"], []).append(record)
    specifications = (
        ("bcc_refractory", 4, 510, 2),
        ("fcc_hea", 2, 255, 1),
        ("lifepo4", 2, 255, 1),
    )
    rows: list[dict[str, Any]] = []
    for family, sweep_count, sweep_size, calculations_per_node in specifications:
        records = by_family.get(family, [])
        if len(records) != sweep_count * sweep_size:
            raise ValueError(
                f"{family}: expected {sweep_count * sweep_size} candidates, found {len(records)}"
            )
        for sweep in range(1, sweep_count + 1):
            offset = (sweep - 1) * sweep_size
            cursor = 0
            for nodes in NODE_TIERS:
                count = nodes * calculations_per_node
                assigned = records[offset + cursor : offset + cursor + count]
                if len(assigned) != count:
                    raise ValueError(f"{family} sweep {sweep} does not fill the {nodes}-node tier")
                rows.append(
                    {
                        "run_id": f"e4-aurora-{family}-s{sweep:02d}-n{nodes:03d}",
                        "material_family": family,
                        "allocation": ALLOCATIONS[family],
                        "sweep": sweep,
                        "nodes": nodes,
                        "calculations_per_node": calculations_per_node,
                        "calculation_ids": [item["calculation_id"] for item in assigned],
                        "calculation_count": count,
                        "queue": None,
                        "walltime": None,
                        "status": BLOCKED_STATUS,
                        "authorization": "not_authorized",
                    }
                )
                cursor += count
            if cursor != sweep_size:
                raise AssertionError(
                    f"{family} sweep {sweep} assignment does not total {sweep_size}"
                )
    validate_run_matrix(candidates, rows)
    return rows


def validate_run_matrix(candidates: list[dict[str, Any]], runs: list[dict[str, Any]]) -> None:
    candidate_ids = [record["calculation_id"] for record in candidates]
    assigned_ids = [calculation_id for run in runs for calculation_id in run["calculation_ids"]]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate calculation IDs are not unique")
    if len(assigned_ids) != len(set(assigned_ids)):
        raise ValueError("a candidate is assigned to multiple production runs")
    if set(candidate_ids) != set(assigned_ids):
        raise ValueError("candidate/run assignment is incomplete")
    for record in candidates:
        expected = ALLOCATIONS[record["material_family"]]
        if record["allocation"] != expected:
            raise ValueError(
                f"{record['calculation_id']} has wrong allocation {record['allocation']}"
            )
    for run in runs:
        if run["allocation"] != ALLOCATIONS[run["material_family"]]:
            raise ValueError(f"{run['run_id']} has a family/allocation mismatch")
        if run["calculation_count"] != len(run["calculation_ids"]):
            raise ValueError(f"{run['run_id']} calculation count is inconsistent")
        if run["authorization"] != "not_authorized":
            raise ValueError("planning artifacts must never authorize execution")


def write_manifests(output: Path) -> dict[str, Any]:
    candidates = build_candidate_inventory()
    runs = build_run_matrix(candidates)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "candidate-inventory.jsonl").open("w", encoding="utf-8") as stream:
        for record in candidates:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    with (output / "run-manifests.jsonl").open("w", encoding="utf-8") as stream:
        for run in runs:
            stream.write(json.dumps(run, sort_keys=True) + "\n")
    with (output / "run-manifests.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = [key for key in runs[0] if key != "calculation_ids"] + ["calculation_ids"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            writer.writerow({**run, "calculation_ids": ";".join(run["calculation_ids"])})
    allocation_status_path = output.parent / "allocation-status.json"
    allocation_status = (
        json.loads(allocation_status_path.read_text(encoding="utf-8"))
        if allocation_status_path.is_file()
        else {}
    )
    allocation_rows = allocation_status.get("projects", [])
    allocation_records_verified = len(allocation_rows) == len(ALLOCATIONS) and all(
        row.get("active_status") is True
        and row.get("aurora_authorization") is True
        and row.get("remaining_balance") is not None
        and str(row.get("user_access", "")).startswith("Verified:")
        for row in allocation_rows
    )
    project_queue_authorization_verified = allocation_records_verified and all(
        row.get("project_specific_queue_authorization") is True for row in allocation_rows
    )
    summary = {
        "status": BLOCKED_STATUS,
        "production_authorized": False,
        "candidate_counts": {
            family: sum(record["material_family"] == family for record in candidates)
            for family in ALLOCATIONS
        },
        "run_count": len(runs),
        "candidate_structure_hashes_available": 0,
        "archive_audit_complete": False,
        "allocation_status_verified": allocation_records_verified,
        "project_queue_authorization_verified": project_queue_authorization_verified,
    }
    (output / "inventory-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
