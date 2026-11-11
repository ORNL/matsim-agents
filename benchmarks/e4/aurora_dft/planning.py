"""Deterministic candidate inventories and disjoint E4 run assignments."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

ELEMENTS_BCC = ("Nb", "Ta", "V", "Hf", "Zr", "Ti")
ELEMENTS_FCC = ("Cr", "Mn", "Fe", "Co", "Ni")
ALLOCATIONS = {
    "bcc_refractory": "HydraGNN",
    "fcc_hea": "HydraGNN",
    "lifepo4": "HydraGNN",
    "phosphorene": "HydraGNN",
    "zn_formate_mof": "HydraGNN",
    "nb_ta_o": "HydraGNN",
}
NODE_TIERS = (1, 2, 4, 8, 16, 32, 64, 128)
BLOCKED_STATUS = "blocked_pending_archive_audit_and_resource_authorization"
REPO = Path(__file__).resolve().parents[3]


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_from_case(case: dict[str, Any], family: str, assignment: dict[str, Any]):
    stage_mode = "low-normal" if family == "bcc_refractory" else "single"
    return {
        "material_class": family,
        "calculation_id": case["calculation_id"],
        "source_calculation_id": case.get("source_calculation_id", case["calculation_id"]),
        "composition": case["composition"],
        "number_of_atoms": case["atom_count"],
        "structure_family": {
            "bcc_refractory": "128-site BCC refractory alloy",
            "fcc_hea": "128-site FCC Cantor alloy",
            "lifepo4": "112-atom LiFePO4 olivine antisite supercell",
        }[family],
        "initial_geometry": case["source_poscar"],
        "geometry_source": case.get("structure_source", "frozen E4 input manifest"),
        "defect_ordering_or_disorder": {
            "calculation_id": case["calculation_id"],
            "antisite_exchange_pairs": case.get("antisite_exchange_pairs"),
            "composition_index": case.get("composition_index"),
            "disorder_realization": case.get("disorder_realization"),
        },
        "source_poscar": case["source_poscar"],
        "protocol_id": case["protocol_id"],
        "input_file_hashes": case["input_hashes"],
        "input_hashes": case["input_hashes"],
        "prepared_input_bundle_sha256": case["prepared_input_bundle_sha256"],
        "paw_components": case["paw_components"],
        "potcar_sha256": case["potcar_sha256"],
        "allocation": "HydraGNN",
        "estimated_task_walltime_hours": case["walltime_seconds"] / 3600,
        "case_dir": case["case_dir"],
        "source_poscar_sha256": case["source_poscar_sha256"],
        "atom_count": case["atom_count"],
        "structure_id": case.get("structure_id"),
        "mpi_ranks": case["mpi_ranks"],
        "omp_threads_per_rank": case["omp_threads_per_rank"],
        "threads_per_rank": case["omp_threads_per_rank"],
        "ediffg_eV_A": case["ediffg_eV_per_A"],
        "ediffg_eV_per_A": case["ediffg_eV_per_A"],
        "launch_settings": {
            "stage_mode": stage_mode,
            "stage_nsw_max": case.get("stage_nsw_max", 240 if family == "bcc_refractory" else 200),
            "nsw_per_checkpoint_segment": 30,
            "single_precision": "Accurate",
            "single_prefix": "A",
            "checkpoint_seconds": 3600,
        },
        "input_hashes_validated": True,
        "execution_ready": False,
        "execution_status": "input_hashes_frozen; submission_not_authorized_by_six_class_manifest",
        "scaling_assignments": assignment,
    }


def _strong_tiers(
    calculation_ids: list[str],
    task_hours: float | None,
    *,
    planned_count: int,
    runner_ready: bool,
) -> list[dict[str, Any]]:
    tiers = []
    for nodes in NODE_TIERS:
        count = planned_count
        enough_inputs = len(calculation_ids) >= count
        if nodes > 16 or task_hours is None or nodes > count:
            walltime = node_hours = None
        else:
            waves = (count + nodes - 1) // nodes
            walltime = waves * task_hours
            if walltime > 168:
                # The one-node reference is explicitly segmented, never truncated.
                group_size = max(1, int(96 // task_hours))
                segments = [
                    calculation_ids[i : i + group_size] for i in range(0, count, group_size)
                ]
                segment_hours = [len(segment) * task_hours for segment in segments]
                if max(segment_hours) > 168:
                    walltime = None
                    node_hours = None
                else:
                    walltime = max(segment_hours)
                    node_hours = sum(segment_hours)
                    status = (
                        "input_complete_authorization_required"
                        if runner_ready and enough_inputs
                        else "blocked_input_or_E4_runtime_adapter"
                    )
                    tiers.append(
                        {
                            "nodes": nodes,
                            "queue_eligible": nodes <= 16,
                            "required_unique_calculations": count,
                            "available_unique_calculations": len(calculation_ids),
                            "calculation_ids": calculation_ids[:count] if enough_inputs else [],
                            "segments": [
                                {
                                    "segment": index + 1,
                                    "calculation_ids": ids,
                                    "walltime_cap_hours": hours,
                                }
                                for index, (ids, hours) in enumerate(
                                    zip(segments, segment_hours, strict=True)
                                )
                            ],
                            "walltime_cap_hours": walltime,
                            "node_hours_cap": (
                                node_hours
                                if status == "input_complete_authorization_required"
                                else None
                            ),
                            "planning_cap_if_inputs_complete_node_hours": node_hours,
                            "inputs_ready": status == "input_complete_authorization_required",
                            "status": status,
                        }
                    )
                    continue
            else:
                node_hours = nodes * walltime
        if nodes > 16:
            status = "blocked_capacity_queue_node_limit"
        elif nodes > count:
            status = "blocked_fixed_workload_underfills_node_tier"
        elif task_hours is None:
            status = "blocked_no_runtime_basis"
        elif not enough_inputs:
            status = "blocked_insufficient_unique_inputs"
        elif not runner_ready:
            status = "blocked_E4_runtime_adapter_or_protocol_qualification"
        elif walltime is None:
            status = "blocked_one_node_reference_exceeds_queue_walltime"
        else:
            status = "input_complete_authorization_required"
        tiers.append(
            {
                "nodes": nodes,
                "queue_eligible": nodes <= 16,
                "inputs_ready": status == "input_complete_authorization_required",
                "required_unique_calculations": count,
                "available_unique_calculations": len(calculation_ids),
                "calculation_ids": calculation_ids[:count] if enough_inputs else [],
                "wave_count": (count + nodes - 1) // nodes
                if enough_inputs and nodes <= count
                else 0,
                "walltime_cap_hours": walltime,
                "node_hours_cap": (
                    node_hours if status == "input_complete_authorization_required" else None
                ),
                "planning_cap_if_inputs_complete_node_hours": (
                    node_hours if task_hours is not None and nodes <= count else None
                ),
                "status": status,
            }
        )
    return tiers


def _weak_tiers(
    calculation_ids: list[str],
    task_hours: float | None,
    *,
    planned_calculation_count: int,
    runner_ready: bool,
) -> list[dict[str, Any]]:
    tiers = []
    for nodes in NODE_TIERS:
        required = 2 * nodes
        enough_inputs = len(calculation_ids) >= required
        if nodes > 16:
            status = "blocked_capacity_queue_node_limit"
            walltime = 2 * task_hours if task_hours is not None else None
            node_hours = None
            planning_cap = 2 * task_hours * nodes if task_hours is not None else None
        elif task_hours is None:
            status = "blocked_no_runtime_basis"
            walltime = node_hours = None
            planning_cap = None
        elif not enough_inputs:
            status = "blocked_insufficient_unique_inputs"
            walltime = 2 * task_hours
            node_hours = None
            planning_cap = 2 * task_hours * nodes
        elif not runner_ready:
            status = "blocked_E4_runtime_adapter_or_protocol_qualification"
            walltime = 2 * task_hours
            node_hours = None
            planning_cap = 2 * task_hours * nodes
        elif nodes == 16:
            status = "blocked_separate_16_node_resource_authorization"
            walltime = 2 * task_hours
            node_hours = None
            planning_cap = 2 * task_hours * nodes
        else:
            status = "input_complete_authorization_required"
            walltime = 2 * task_hours
            node_hours = 2 * task_hours * nodes
            planning_cap = 2 * task_hours * nodes
        tiers.append(
            {
                "nodes": nodes,
                "queue_eligible": nodes <= 16,
                "inputs_ready": status == "input_complete_authorization_required",
                "tasks_per_node": 2,
                "required_unique_calculations": required,
                "available_unique_calculations": len(calculation_ids),
                "calculation_ids": calculation_ids[:required] if enough_inputs else [],
                "planned_unique_calculation_count": planned_calculation_count,
                "tasks_per_node_run_sequentially": True,
                "walltime_cap_hours": walltime if enough_inputs else None,
                "planning_walltime_if_queue_available": (
                    2 * task_hours if task_hours is not None else None
                ),
                "node_hours_cap": (
                    node_hours if status == "input_complete_authorization_required" else None
                ),
                "planning_cap_if_inputs_complete_node_hours": planning_cap,
                "status": status,
            }
        )
    return tiers


def build_six_class_workload_manifest(repo: Path = REPO) -> dict[str, Any]:
    """Build a provenance-first E4 matrix; unavailable science remains blocked."""
    repo = repo.resolve()
    batch = json.loads(
        (repo / "deployments/aurora/jobs/e4-first-batch/batch-manifest-hydragnn.json").read_text()
    )
    bcc_root = repo / "research/paper/e4/aurora-dft-scaling/priority-scaling/authorized"
    bcc_strong = json.loads((bcc_root / "strong-manifest.json").read_text())
    bcc_weak = json.loads((bcc_root / "weak-manifest.json").read_text())
    bcc_tasks = {task["calculation_id"]: task for task in bcc_strong["tasks"]}
    bcc_tasks.update({task["calculation_id"]: task for task in bcc_weak["tasks"]})
    bcc_strong_ids = list(bcc_strong["fixed_workload"]["case_poscar_sha256"])
    bcc_weak_ids = list(bcc_weak["available_unique_case_ids"])
    bcc_records = [
        _candidate_from_case(
            {
                **bcc_tasks[calc_id],
                "atom_count": sum(bcc_tasks[calc_id]["composition"].values()),
                "structure_source": "authorized BCC strong/weak scaling manifests",
                "walltime_seconds": 86400,
            },
            "bcc_refractory",
            {"strong": calc_id in bcc_strong_ids, "weak": calc_id in bcc_weak_ids},
        )
        for calc_id in sorted(set(bcc_strong_ids + bcc_weak_ids))
    ]
    batch_cases = {case["calculation_id"]: case for case in batch["cases"]}
    fcc_ids = [
        case["calculation_id"] for case in batch["cases"] if case["material_family"] == "fcc"
    ]
    lfp_ids = [
        case["calculation_id"] for case in batch["cases"] if case["material_family"] == "lifepo4"
    ]

    def scaling_case(case_id: str, prefix: str) -> dict[str, Any]:
        case = dict(batch_cases[case_id])
        case["source_calculation_id"] = case_id
        case["calculation_id"] = f"E4-SCALE-{prefix}-{case_id.rsplit('-', 1)[-1]}"
        case["structure_source"] = f"{case_id} frozen first-batch input; fresh scaling identity"
        return case

    fcc_records = [
        _candidate_from_case(
            scaling_case(calc_id, "FCC"),
            "fcc_hea",
            {"strong": True, "weak": True},
        )
        for calc_id in fcc_ids
    ]
    lfp_records = [
        _candidate_from_case(
            scaling_case(calc_id, "LFP"),
            "lifepo4",
            {"strong": True, "weak": True},
        )
        for calc_id in lfp_ids
    ]

    nbtao_path = repo / "research/nb-ta-o/pilot/vasp_inputs/inputs_manifest.json"
    nbtao_input = json.loads(nbtao_path.read_text())
    nbtao_runtime = json.loads(
        (repo / "research/nb-ta-o/pilot/vasp_runtime_manifest.json").read_text()
    )
    nbtao_pilot_ids = {f"E4-{case_id}" for case_id in nbtao_input["production_pilot_candidates"]}
    nbtao_paws = [
        {"label": label, **item} for label, item in nbtao_runtime["proposed_components"].items()
    ]
    nbtao_records = []
    for case in nbtao_input["cases"]:
        if case["category"] not in {
            "production_candidate",
            "ordering_family_extension_pending_approval",
        }:
            continue
        calc_id = case["case_id"]
        source = f"research/nb-ta-o/pilot/poscars/{calc_id}.POSCAR"
        nbtao_records.append(
            {
                "material_class": "nb_ta_o",
                "calculation_id": f"E4-{calc_id}",
                "composition": case["species_counts"],
                "number_of_atoms": sum(case["species_counts"].values()),
                "structure_family": "28-atom monoclinic NbTaO5; symmetry-distinct Nb/Ta ordering",
                "initial_geometry": source,
                "geometry_source": case["category"],
                "defect_ordering_or_disorder": {
                    "ordering_id": calc_id,
                    "source_parent": "COD 1528678 Nb2O5",
                },
                "protocol_id": case["protocol_id"],
                "input_file_hashes": case["input_hashes"],
                "paw_components": nbtao_paws,
                "allocation": "HydraGNN",
                "estimated_task_walltime_hours": 2.0,
                "case_dir": (
                    f"research/nb-ta-o/pilot/vasp_inputs/ordering-family-pending-approval/{calc_id}"
                ),
                "input_approval_category": case["category"],
                "source_poscar_sha256": case["input_hashes"]["POSCAR"],
                "structure_id": case["input_hashes"]["POSCAR"],
                "execution_ready": False,
                "execution_status": "legacy_inputs_require_E4_case_staging_and_protocol_review",
                "scaling_assignments": {
                    "strong": f"E4-{calc_id}" in nbtao_pilot_ids,
                    "weak": True,
                },
            }
        )

    from ase.io import read

    def seed_records(material_class: str, seed_names: list[str], structure_family: str, issue: str):
        records = []
        for seed_name in seed_names:
            path = repo / "examples/paper_cases/seeds" / seed_name
            atoms = read(path)
            composition: dict[str, int] = {}
            for symbol in atoms.get_chemical_symbols():
                composition[symbol] = composition.get(symbol, 0) + 1
            records.append(
                {
                    "material_class": material_class,
                    "calculation_id": f"E4-{material_class.upper()}-{path.stem.upper()}",
                    "composition": composition,
                    "chemical_formula": atoms.get_chemical_formula(),
                    "number_of_atoms": len(atoms),
                    "cell_lengths_A": [float(value) for value in atoms.cell.lengths()],
                    "periodic_boundary_conditions": [bool(value) for value in atoms.pbc],
                    "structure_family": structure_family,
                    "initial_geometry": str(path.relative_to(repo)),
                    "geometry_source": "existing E3 example seed; not E4-qualified",
                    "defect_ordering_or_disorder": None,
                    "protocol_id": None,
                    "input_file_hashes": {"seed_geometry": _sha256(path)},
                    "paw_components": None,
                    "allocation": "HydraGNN",
                    "estimated_task_walltime_hours": None,
                    "case_dir": None,
                    "source_poscar_sha256": _sha256(path),
                    "structure_id": None,
                    "execution_ready": False,
                    "execution_status": issue,
                    "scaling_assignments": {"strong": False, "weak": False},
                }
            )
        return records

    phos_records = seed_records(
        "phosphorene",
        ["phosphorene_black_P.extxyz", "phosphorene_blue_P.extxyz"],
        "2D phosphorene allotrope seed with vacuum cell",
        "seed_only; retrieve archived E2 structures and freeze a qualified 2D VASP protocol",
    )
    mof_records = seed_records(
        "zn_formate_mof",
        ["zn_formate_alpha.extxyz", "zn_formate_beta.extxyz"],
        "Zn-formate framework placeholder",
        (
            "invalid_seed_stoichiometry_missing_H_and_half_linker_O; "
            "obtain validated CIF and verify bonding"
        ),
    )

    queue = "capacity"
    protocol_data = {
        "bcc_refractory": {
            "id": "bcc-pbe-low-normal-v1-approved-ediffg-0.01",
            "functional": "PBE",
            "encut_eV": 350,
            "kpoints": "Gamma 2x2x2",
            "spin": "non-spin",
            "smearing": "ISMEAR=1 SIGMA=0.20",
            "ediff_eV": 1e-6,
            "ediffg_eV_A": -0.01,
            "stages": ["Low", "Normal"],
            "max_ionic_steps_per_stage": 240,
            "mpi_ranks": 8,
            "threads_per_rank": 8,
            "kpar": 8,
            "paw": ["Nb_pv", "Ta_pv", "V_pv", "Hf_pv", "Zr_sv", "Ti_pv"],
            "task_walltime_hours": 24,
        },
        "fcc_hea": {
            "id": "fcc-pbe-verified-v1",
            "functional": "PBE",
            "encut_eV": 480,
            "kpoints": "Gamma 2x2x2",
            "spin": "ISPIN=2; archived MAGMOM initialization",
            "smearing": "ISMEAR=1 SIGMA=0.10",
            "ediff_eV": 1e-6,
            "ediffg_eV_A": -0.02,
            "stages": ["Accurate relaxation"],
            "max_ionic_steps": 200,
            "mpi_ranks": 8,
            "threads_per_rank": 8,
            "kpar": 8,
            "paw": ["Cr_pv", "Mn_pv", "Fe_pv", "Co", "Ni_pv"],
            "task_walltime_hours": 24,
        },
        "lifepo4": {
            "id": "lifepo4-pbe-verified-v1",
            "functional": "PBE",
            "encut_eV": 520,
            "kpoints": "Monkhorst-Pack 2x4x8",
            "spin": "ISPIN=2; archived FM initialization (Fe 5.0, other species 0.6 muB)",
            "smearing": "ISMEAR=0 SIGMA=0.05",
            "ediff_eV": 1e-6,
            "ediffg_eV_A": -0.01,
            "stages": ["Accurate relaxation"],
            "max_ionic_steps": 200,
            "mpi_ranks": 12,
            "threads_per_rank": 8,
            "kpar": 1,
            "ncore": 12,
            "paw": ["Fe_pv", "Li_sv", "O", "P"],
            "task_walltime_hours": 24,
        },
        "phosphorene": {
            "id": None,
            "status": "not_frozen",
            "references": [
                "E3 Aurora AL example is single-point only",
                "E2 reference geometries are not staged in this checkout",
            ],
            "blocker": (
                "retrieve qualified puckered E2 geometries; freeze 2D vacuum, "
                "k-mesh, strain, and defect protocol"
            ),
        },
        "zn_formate_mof": {
            "id": None,
            "status": "not_frozen",
            "candidate_reference": (
                "E3 example uses PBE/520 eV; seed lacks H and IVDW=12 support is unverified"
            ),
            "blocker": "obtain complete CIF and freeze a validated dispersion protocol",
        },
        "nb_ta_o": {
            "id": "nbtao5-fixed-kmesh-v1",
            "status": "E2_pilot_protocol_not_yet_E4_qualified",
            "functional": "PBE",
            "encut_eV": 600,
            "kpoints": "Gamma-centered mesh targeting 0.15 A^-1; 28-atom ordering uses 4x8x9",
            "spin": "ISPIN=1",
            "smearing": "ISMEAR=0",
            "ediff_eV": 1e-6,
            "ediffg_eV_A": -0.01,
            "relax_and_static": True,
            "task_walltime_hours": 2,
            "paw": ["Nb_pv", "Ta_pv", "O"],
            "blocker": (
                "E2 inputs need E4 staging/runtime qualification, reference review, "
                "and separate authorization"
            ),
        },
    }

    def class_record(
        name: str,
        records: list[dict[str, Any]],
        strong_ids: list[str],
        weak_ids: list[str],
        readiness: str,
        blockers: list[str],
        generation_procedure: str,
        wall_hours: float | None,
        runner_ready: bool,
        input_inventory_count: int,
        planned_strong_count: int = 8,
        planned_weak_count: int = 16,
        budget_basis: str = "24-hour maximum per task; not a runtime prediction",
    ):
        strong = _strong_tiers(
            strong_ids,
            wall_hours,
            planned_count=planned_strong_count,
            runner_ready=runner_ready,
        )
        weak = _weak_tiers(
            weak_ids,
            wall_hours,
            planned_calculation_count=planned_weak_count,
            runner_ready=runner_ready,
        )
        return {
            "material_class": name,
            "readiness": readiness,
            "allocation": "HydraGNN",
            "queue": queue,
            "input_inventory_count": input_inventory_count,
            "generation_procedure": generation_procedure,
            "candidates": records,
            "protocol": protocol_data[name],
            "strong_scaling": {"fixed_calculation_ids": strong_ids, "tiers": strong},
            "weak_scaling": {
                "tasks_per_node": 2,
                "unique_calculation_ids": weak_ids,
                "tiers": weak,
            },
            "blockers": blockers,
            "resource_budget": {
                "task_walltime_basis_hours": wall_hours,
                "budget_basis": budget_basis,
                "strong_requested_node_hours_ceiling": sum(
                    tier["node_hours_cap"] or 0 for tier in strong
                ),
                "weak_requested_node_hours_ceiling": sum(
                    tier["node_hours_cap"] or 0 for tier in weak
                ),
                "strong_planning_cap_if_inputs_complete": sum(
                    tier["planning_cap_if_inputs_complete_node_hours"] or 0 for tier in strong
                ),
                "weak_planning_cap_if_inputs_complete": sum(
                    tier["planning_cap_if_inputs_complete_node_hours"] or 0 for tier in weak
                ),
            },
            "execution_authorized": False,
            "authorization": None,
        }

    fcc_ready_ids = [r["calculation_id"] for r in fcc_records]
    lfp_ready_ids = [r["calculation_id"] for r in lfp_records]
    nbtao_ids = [r["calculation_id"] for r in nbtao_records]
    classes = [
        class_record(
            "bcc_refractory",
            bcc_records,
            bcc_strong_ids,
            bcc_weak_ids,
            "running_existing_BCC_workloads",
            [],
            "Use the frozen 128-site BCC parent; retain composition and disorder realizations",
            24,
            True,
            16,
        ),
        class_record(
            "fcc_hea",
            fcc_records,
            fcc_ready_ids,
            fcc_ready_ids,
            "minimal_preparation_four_frozen_geometries",
            [
                "Four E4 POSCARs: strong tiers above 4 nodes and weak tiers above "
                "2 nodes lack unique structures.",
                "The 510 composition candidates lack audited structure hashes.",
            ],
            (
                "Use audited 128-site FCC Cantor lattice and frozen disorder "
                "realizations; do not sample new compositions for scaling"
            ),
            24,
            True,
            4,
            planned_strong_count=4,
            planned_weak_count=16,
        ),
        class_record(
            "lifepo4",
            lfp_records,
            lfp_ready_ids,
            lfp_ready_ids,
            "minimal_preparation_eight_frozen_geometries",
            [
                "Eight frozen inputs support strong through 8 nodes and weak "
                "through 4; weak-8 needs 16 unique cases."
            ],
            "Use the frozen 112-atom olivine parent and symmetry-distinct Li/Fe antisite pairs",
            24,
            True,
            510,
        ),
        class_record(
            "phosphorene",
            phos_records,
            [],
            [],
            "blocked_missing_qualified_E4_geometry_and_protocol",
            [
                "Two small seeds are staged; E2 geometries remain in an unavailable "
                "private archive.",
                "The E3 example is single-point DFT, not E4 relaxation; no E4 "
                "2D mesh/vacuum protocol is frozen.",
            ],
            (
                "Retrieve qualified puckered E2 geometries; then make "
                "vacuum-preserving supercells, controlled in-plane strains, and supported vacancies"
            ),
            None,
            False,
            2,
        ),
        class_record(
            "zn_formate_mof",
            mof_records,
            [],
            [],
            "blocked_invalid_seed_and_unverified_dispersion",
            [
                "Both seeds are C8O8Zn4 placeholders; four-Zn Zn(HCOO)2 requires C8H8O16Zn4.",
                "PBE-D3(BJ)/IVDW=12 support and an E4 reference/protocol are unverified.",
            ],
            (
                "Replace placeholders with complete CIFs; generate "
                "topology-preserving supercells, breathing modes, and proton orderings"
            ),
            None,
            False,
            2,
        ),
        class_record(
            "nb_ta_o",
            nbtao_records,
            [calc_id for calc_id in nbtao_ids if calc_id in nbtao_pilot_ids],
            nbtao_ids,
            "minimal_preparation_E2_inputs_need_E4_adapter",
            [
                "Fourteen symmetry-distinct NbTaO5 POSCARs are staged; only "
                "ordering-01 has converged DFT.",
                "Inputs need E4 case staging and capacity-launcher/protocol qualification.",
                "Weak-8 requires 16 unique tasks; only 14 orderings are currently available.",
            ],
            (
                "Reuse the 14 symmetry-inequivalent equimolar Nb/Ta orderings "
                "from Nb2O5; preserve charge-balanced NbTaO5 stoichiometry"
            ),
            2,
            False,
            14,
            planned_strong_count=len(nbtao_pilot_ids),
            budget_basis=(
                "2-hour per-task ceiling from the prior E2 qualification workflow; "
                "E4 runtime unmeasured"
            ),
        ),
    ]
    return {
        "schema_version": 1,
        "manifest_id": "E4-Aurora-Six-Class-Workflow-Scaling-v1",
        "execution_authorized": False,
        "platform": "Aurora",
        "scheduler": "PBS Pro",
        "queue": "capacity",
        "allocation": "HydraGNN",
        "runner": "deployments/aurora/jobs/job-e4-scaling.pbs",
        "per_calculation_wrapper": "deployments/aurora/jobs/job-e4-vasp-ab-aurora.pbs",
        "queue_limits": {"maximum_nodes_per_job": 16, "maximum_walltime_hours": 168},
        "output_root_template": (
            "runs/e4-scaling/<experiment>/<run-id>/n<node-count>/<segment-id-or-all>/"
        ),
        "semantics": {
            "scaling_level": (
                "workflow-level independent calculations; not single-VASP solver scaling"
            ),
            "strong_definition": "same fixed calculation IDs at every feasible node tier",
            "weak_definition": "two unique calculations per node, run sequentially",
            "node_tiers": list(NODE_TIERS),
            "initial_node_tiers": [1, 2, 4, 8],
            "large_tiers_require_new_queue_and_allocation_review": [16, 32, 64, 128],
            "default_task_walltime_cap_hours": 24,
            "execution_authorized": False,
            "scheduler_submissions": 0,
        },
        "preserved_source_manifests": {
            "bcc_strong_sha256": _sha256(
                repo / "research/paper/e4/aurora-dft-scaling/priority-scaling/authorized/"
                "strong-manifest.json"
            ),
            "bcc_weak_sha256": _sha256(
                repo / "research/paper/e4/aurora-dft-scaling/priority-scaling/authorized/"
                "weak-manifest.json"
            ),
            "first_batch_original_sha256": _sha256(
                repo / "deployments/aurora/jobs/e4-first-batch/batch-manifest.json"
            ),
            "first_batch_hydragnn_revision_sha256": _sha256(
                repo / "deployments/aurora/jobs/e4-first-batch/batch-manifest-hydragnn.json"
            ),
        },
        "material_classes": classes,
        "authorization_request": {
            "status": "not_authorized",
            "additional_jobs_authorized": False,
            "per_class_tiers_require_separate_review": True,
            "existing_bcc_authorization_applies_only_to_the_frozen_bcc_manifests": True,
            "existing_bcc_authorized_node_hours": 1488,
            "proposed_additional_node_hours_ceiling": sum(
                entry["resource_budget"]["strong_requested_node_hours_ceiling"]
                + entry["resource_budget"]["weak_requested_node_hours_ceiling"]
                for entry in classes
                if entry["material_class"] != "bcc_refractory"
            ),
            "provisional_phosphorene_and_mof_budget_must_be_replaced_after_pilot": True,
        },
    }


def write_six_class_workload_manifest(path: Path, repo: Path = REPO) -> dict[str, Any]:
    manifest = build_six_class_workload_manifest(repo)
    validate_six_class_workload_manifest(manifest)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _validate_six_class_candidate_files(candidate: dict[str, Any], repo: Path = REPO) -> None:
    case_dir = (repo / candidate["case_dir"]).resolve()
    source_poscar = (repo / candidate["source_poscar"]).resolve()
    if not source_poscar.is_file() or _sha256(source_poscar) != candidate["source_poscar_sha256"]:
        raise ValueError(f"{candidate['calculation_id']}: source POSCAR hash mismatch")
    if not case_dir.is_dir():
        raise ValueError(f"{candidate['calculation_id']}: staged case directory is missing")
    for name in ("START", "POSCAR", "KPOINTS", "POTCAR"):
        path = case_dir / name
        expected = candidate["input_file_hashes"].get(name)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"{candidate['calculation_id']}: {name} hash mismatch")
    poscar_lines = (case_dir / "POSCAR").read_text().splitlines()
    elements = poscar_lines[5].split()
    counts = [int(value) for value in poscar_lines[6].split()]
    composition = dict(zip(elements, counts, strict=True))
    if composition != candidate["composition"] or sum(counts) != candidate["number_of_atoms"]:
        raise ValueError(
            f"{candidate['calculation_id']}: POSCAR composition or atom count mismatch"
        )
    assembled = bytearray()
    for component in candidate["paw_components"]:
        paw = (repo / component["source"]).resolve()
        if not paw.is_file() or _sha256(paw) != component["sha256"]:
            raise ValueError(f"{candidate['calculation_id']}: PAW component hash mismatch")
        assembled.extend(paw.read_bytes())
    if hashlib.sha256(assembled).hexdigest() != candidate["potcar_sha256"]:
        raise ValueError(f"{candidate['calculation_id']}: assembled POTCAR hash mismatch")


def validate_six_class_workload_manifest(
    manifest: dict[str, Any], *, repo: Path = REPO, verify_files: bool = True
) -> None:
    expected = {"bcc_refractory", "fcc_hea", "lifepo4", "phosphorene", "zn_formate_mof", "nb_ta_o"}
    classes = manifest.get("material_classes", [])
    names = [entry.get("material_class") for entry in classes]
    if set(names) != expected or len(names) != len(expected):
        raise ValueError("six-class manifest must contain exactly the six E4 material classes")
    if manifest.get("execution_authorized") is not False:
        raise ValueError("six-class manifest must keep execution_authorized=false")
    if manifest.get("authorization_request", {}).get("additional_jobs_authorized") is not False:
        raise ValueError("six-class planning manifest must not authorize execution")
    for entry in classes:
        ids = [candidate.get("calculation_id") for candidate in entry.get("candidates", [])]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{entry['material_class']}: duplicate candidate calculation IDs")
        required_candidate_fields = {
            "material_class",
            "calculation_id",
            "composition",
            "number_of_atoms",
            "structure_family",
            "initial_geometry",
            "geometry_source",
            "defect_ordering_or_disorder",
            "protocol_id",
            "input_file_hashes",
            "paw_components",
            "allocation",
            "estimated_task_walltime_hours",
            "scaling_assignments",
            "execution_status",
        }
        for candidate in entry.get("candidates", []):
            missing = required_candidate_fields - set(candidate)
            if missing:
                raise ValueError(
                    f"{entry['material_class']}/{candidate.get('calculation_id')}: "
                    f"candidate fields missing {sorted(missing)}"
                )
        if verify_files:
            for candidate in entry.get("candidates", []):
                if candidate.get("input_hashes_validated"):
                    _validate_six_class_candidate_files(candidate, repo)
        known = set(ids)
        for experiment in ("strong_scaling", "weak_scaling"):
            plan = entry[experiment]
            for tier in plan["tiers"]:
                if not set(tier["calculation_ids"]) <= known:
                    raise ValueError(
                        f"{entry['material_class']}: tier references unknown calculation IDs"
                    )
                if tier.get("inputs_ready") and not all(
                    candidate.get("input_hashes_validated")
                    for candidate in entry["candidates"]
                    if candidate["calculation_id"] in tier["calculation_ids"]
                ):
                    raise ValueError(
                        f"{entry['material_class']}: input-ready tier has unvalidated inputs"
                    )


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
