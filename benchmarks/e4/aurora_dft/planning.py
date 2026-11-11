"""Deterministic candidate inventories and disjoint E4 run assignments."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
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
SIX_CLASS_INPUTS = "research/paper/e4/aurora-dft-scaling/six-class-inputs-v2/manifest.json"
CAPACITY_MAX_NODES = 16
QUEUE_WALLTIME_HOURS = 168
HYDRAGNN_AVAILABLE_NODE_HOURS = 36614.0
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
        "structure_family": case.get("structure_family")
        or {
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
            **case.get("variation", {}),
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


def _queue_for(nodes: int) -> tuple[str, str]:
    if nodes <= CAPACITY_MAX_NODES:
        return "capacity", "input_complete_authorization_required"
    return "reservation", "input_complete_requires_aurora_reservation"


def _strong_tiers(
    calculation_ids: list[str],
    task_hours: float,
    expected_hours: float,
    *,
    fixed_count: int,
) -> list[dict[str, Any]]:
    work = calculation_ids[:fixed_count]
    waves_per_job = max(1, int(QUEUE_WALLTIME_HOURS // task_hours))
    tiers = []
    for nodes in NODE_TIERS:
        queue, status = _queue_for(nodes)
        if len(work) < fixed_count:
            status = "blocked_insufficient_unique_inputs"
        waves = math.ceil(len(work) / nodes)
        segments = None
        walltime = waves * task_hours
        if walltime > QUEUE_WALLTIME_HOURS:
            group = nodes * waves_per_job
            chunks = [work[i : i + group] for i in range(0, len(work), group)]
            segments = [
                {
                    "segment": index + 1,
                    "calculation_ids": chunk,
                    "walltime_cap_hours": math.ceil(len(chunk) / nodes) * task_hours,
                }
                for index, chunk in enumerate(chunks)
            ]
            walltime = max(segment["walltime_cap_hours"] for segment in segments)
        ready = not status.startswith("blocked")
        tier = {
            "nodes": nodes,
            "queue": queue,
            "queue_eligible": nodes <= CAPACITY_MAX_NODES,
            "inputs_ready": ready,
            "required_unique_calculations": fixed_count,
            "available_unique_calculations": len(calculation_ids),
            "calculation_ids": work if ready else [],
            "wave_count": waves,
            "jobs": len(segments) if segments else 1,
            "walltime_cap_hours": walltime,
            "node_hours_cap": nodes * waves * task_hours if ready else None,
            "expected_node_hours": round(nodes * waves * expected_hours, 1),
            "expected_makespan_hours": round(waves * expected_hours, 1),
            "planning_cap_if_inputs_complete_node_hours": nodes * waves * task_hours,
            "status": status,
        }
        if segments:
            tier["segments"] = segments
        tiers.append(tier)
    return tiers


def _weak_tiers(
    calculation_ids: list[str],
    task_hours: float,
    expected_hours: float,
) -> list[dict[str, Any]]:
    tiers = []
    for nodes in NODE_TIERS:
        required = 2 * nodes
        queue, status = _queue_for(nodes)
        if len(calculation_ids) < required:
            status = "blocked_insufficient_unique_inputs"
        ready = not status.startswith("blocked")
        tiers.append(
            {
                "nodes": nodes,
                "queue": queue,
                "queue_eligible": nodes <= CAPACITY_MAX_NODES,
                "inputs_ready": ready,
                "tasks_per_node": 2,
                "tasks_per_node_run_sequentially": True,
                "required_unique_calculations": required,
                "available_unique_calculations": len(calculation_ids),
                "calculation_ids": calculation_ids[:required] if ready else [],
                "jobs": 1,
                "walltime_cap_hours": 2 * task_hours,
                "node_hours_cap": 2 * task_hours * nodes if ready else None,
                "expected_node_hours": round(2 * expected_hours * nodes, 1),
                "expected_makespan_hours": round(2 * expected_hours, 1),
                "planning_cap_if_inputs_complete_node_hours": 2 * task_hours * nodes,
                "status": status,
            }
        )
    return tiers


def build_six_class_workload_manifest(repo: Path = REPO) -> dict[str, Any]:
    """Build the full six-class E4 production matrix from the staged v2 inputs."""
    repo = repo.resolve()
    inputs_path = repo / SIX_CLASS_INPUTS
    staged = json.loads(inputs_path.read_text())
    settings = staged["class_settings"]
    strong_count = staged["strong_fixed_tasks_per_class"]
    records: dict[str, list[dict[str, Any]]] = {}
    strong_ids: dict[str, list[str]] = {}
    weak_ids: dict[str, list[str]] = {}
    for name, cases in staged["classes"].items():
        ids = [case["calculation_id"] for case in cases]
        strong_ids[name], weak_ids[name] = ids[:strong_count], ids
        fixed = set(strong_ids[name])
        records[name] = [
            _candidate_from_case(
                case, name, {"strong": case["calculation_id"] in fixed, "weak": True}
            )
            for case in cases
        ]

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
            "id": "phosphorene-pbe-2d-strain-fixed-cell-v1",
            "status": "E2_numerics_with_rebuilt_geometry",
            "functional": "PBE (E2 setting; no dispersion for an isolated monolayer)",
            "encut_eV": 520,
            "precision": "Accurate",
            "kpoints": "Gamma 2x3x1 for the 4x5 supercell (E2 8x11x1 unit-cell density)",
            "spin": "ISPIN=2, MAGMOM 0.5 per atom (vacancies leave unpaired electrons)",
            "smearing": "ISMEAR=0 SIGMA=0.05",
            "ediff_eV": 1e-6,
            "ediffg_eV_A": -0.01,
            "relaxation": "ISIF=2 ionic relaxation at fixed strained in-plane cell",
            "workload": (
                "4x5 supercell; pristine, monovacancy, in-plane and interlayer divacancy; "
                "8x8 armchair (-2..+12%) x zigzag (-6..+8%) strain grid"
            ),
            "vacuum_axis_A": 20.0,
            "max_ionic_steps": 200,
            "mpi_ranks": 12,
            "threads_per_rank": 8,
            "kpar": 1,
            "ncore": 12,
            "paw": ["P"],
            "geometry_basis": (
                "bulk black P (Brown & Rundqvist 1965, Cmce); archived E2 P4/P36/P35 have "
                "no P-P bond below 2.5 A (puckering placed on the zigzag axis)"
            ),
        },
        "zn_formate_mof": {
            "id": "znformate-pbe-d3bj-strain-stress-v1",
            "status": "E2_zn_formate_anhydrous_v1_numerics",
            "functional": "PBE-D3(BJ), IVDW=12",
            "encut_eV": 520,
            "precision": "Accurate",
            "kpoints": "E2 Gamma meshes: alpha 3x4x3, Pbcn 4x2x2",
            "spin": "ISPIN=1",
            "smearing": "ISMEAR=0 SIGMA=0.05",
            "ediff_eV": 1e-6,
            "ediffg_eV_A": -0.02,
            "relaxation": "ISIF=2 relaxed-ion stress at fixed strained cell",
            "workload": (
                "alpha and Pbcn frameworks; 8 volumetric (+/-0.5-2%) and 120 Voigt "
                "(6 components x +/- x 0.1-1.0%) strain states each"
            ),
            "max_ionic_steps": 200,
            "mpi_ranks": 12,
            "threads_per_rank": 8,
            "kpar": 12,
            "paw": ["Zn", "H", "C", "O"],
        },
        "nb_ta_o": {
            "id": "nbtao5-fixed-kmesh-v1",
            "status": "E2_protocol_with_E4_parallelization_adapter",
            "functional": "PBE",
            "encut_eV": 600,
            "kpoints": "Gamma-centered mesh targeting 0.15 A^-1; 56-atom cell uses 4x8x5",
            "spin": "ISPIN=1",
            "smearing": "ISMEAR=0 SIGMA=0.05",
            "ediff_eV": 1e-6,
            "ediffg_eV_A": -0.01,
            "relaxation": "ISIF=3, ISYM=0",
            "workload": (
                "256 of 854 symmetry-distinct equimolar Nb/Ta orderings of the 1x1x2 "
                "cell; the 14 doubled E2 orderings first"
            ),
            "max_ionic_steps": 200,
            "mpi_ranks": 12,
            "threads_per_rank": 8,
            "kpar": 12,
            "e4_adapter": "KPAR=12 added, exactly as in the converged E2 ordering-01 run",
            "paw": ["Nb_pv", "Ta_pv", "O"],
        },
    }
    for name, spec in settings.items():
        protocol_data[name]["task_walltime_hours"] = spec["task_cap_hours"]
        protocol_data[name]["expected_task_hours"] = spec["expected_task_hours"]
        protocol_data[name]["runtime_basis"] = spec["runtime_basis"]

    def class_record(
        name: str,
        readiness: str,
        blockers: list[str],
        generation_procedure: str,
    ):
        cap = settings[name]["task_cap_hours"]
        expected = settings[name]["expected_task_hours"]
        strong = _strong_tiers(strong_ids[name], cap, expected, fixed_count=strong_count)
        weak = _weak_tiers(weak_ids[name], cap, expected)

        def total(tiers: list[dict[str, Any]], field: str, capacity_only: bool = False):
            return round(
                sum(
                    tier[field] or 0
                    for tier in tiers
                    if not capacity_only or tier["queue"] == "capacity"
                ),
                1,
            )

        return {
            "material_class": name,
            "readiness": readiness,
            "allocation": "HydraGNN",
            "queue": "capacity (1-16 nodes); Aurora reservation (32-128 nodes)",
            "input_inventory_count": len(records[name]),
            "generation_procedure": generation_procedure,
            "candidates": records[name],
            "protocol": protocol_data[name],
            "strong_scaling": {"fixed_calculation_ids": strong_ids[name], "tiers": strong},
            "weak_scaling": {
                "tasks_per_node": 2,
                "unique_calculation_ids": weak_ids[name],
                "tiers": weak,
            },
            "blockers": blockers,
            "resource_budget": {
                "task_walltime_cap_hours": cap,
                "expected_task_hours": expected,
                "budget_basis": settings[name]["runtime_basis"],
                "strong_requested_node_hours_ceiling": total(strong, "node_hours_cap"),
                "weak_requested_node_hours_ceiling": total(weak, "node_hours_cap"),
                "strong_expected_node_hours": total(strong, "expected_node_hours"),
                "weak_expected_node_hours": total(weak, "expected_node_hours"),
                "capacity_queue_expected_node_hours": total(strong, "expected_node_hours", True)
                + total(weak, "expected_node_hours", True),
                "reservation_expected_node_hours": round(
                    total(strong, "expected_node_hours")
                    + total(weak, "expected_node_hours")
                    - total(strong, "expected_node_hours", True)
                    - total(weak, "expected_node_hours", True),
                    1,
                ),
            },
            "execution_authorized": False,
            "authorization": None,
        }

    classes = [
        class_record(
            "bcc_refractory",
            "inputs_qualified_authorization_required",
            [
                "The 16 authorized inputs are reused unchanged; their 1-8-node "
                "authorized runs remain a separate 8-task workload."
            ],
            (
                "Authorized 16 inputs plus 240 frozen-generator occupations: 22 element "
                "subsets x composition indices 1-11, realization 1, inventory seeds"
            ),
        ),
        class_record(
            "fcc_hea",
            "inputs_qualified_authorization_required",
            ["E4 runtime unmeasured; cap 48 h from E2 108-atom timing."],
            (
                "Frozen stage-1 generator on the E2 Cantor parent: compositions 1-64 x "
                "realizations 1-4; reproduces all 16 previously frozen POSCARs"
            ),
        ),
        class_record(
            "lifepo4",
            "inputs_qualified_authorization_required",
            ["E4 runtime unmeasured; 72-h cap extrapolated from a 28-atom relaxation."],
            "Frozen 8 + qualified 8 + 240 canonical lifepo4-inventory-v1 structures",
        ),
        class_record(
            "phosphorene",
            "inputs_qualified_geometry_rebuilt_authorization_required",
            [
                "No qualified E2 phosphorene geometry exists (archived inputs are "
                "unbonded); inputs are rebuilt from bulk black P with E2 numerics."
            ],
            "4x5 puckered supercell x 4 defect states x 64 in-plane strain states",
        ),
        class_record(
            "zn_formate_mof",
            "inputs_qualified_authorization_required",
            [
                "E2 used the custom VASP 6.6.1 build; E4 uses the ALCF 6.6.1 build "
                "(both include D3(BJ))."
            ],
            "E2-qualified alpha and Pbcn frameworks x 128 strain states each",
        ),
        class_record(
            "nb_ta_o",
            "inputs_qualified_authorization_required",
            ["E2 used the custom VASP 6.6.1 build; E4 uses the ALCF 6.6.1 build."],
            "Symmetry-distinct equimolar orderings of the doubled NbTaO5 cell",
        ),
    ]
    expected_total = sum(
        c["resource_budget"]["strong_expected_node_hours"]
        + c["resource_budget"]["weak_expected_node_hours"]
        for c in classes
    )
    cap_total = sum(
        c["resource_budget"]["strong_requested_node_hours_ceiling"]
        + c["resource_budget"]["weak_requested_node_hours_ceiling"]
        for c in classes
    )
    capacity_expected = sum(
        c["resource_budget"]["capacity_queue_expected_node_hours"] for c in classes
    )
    return {
        "schema_version": 1,
        "manifest_id": "E4-Aurora-Six-Class-Production-Scaling-v3",
        "execution_authorized": False,
        "platform": "Aurora",
        "scheduler": "PBS Pro",
        "queue": "capacity (1-16 nodes); reservation (32-128 nodes)",
        "allocation": "HydraGNN",
        "runner": "deployments/aurora/jobs/job-e4-scaling.pbs",
        "per_calculation_wrapper": "deployments/aurora/jobs/job-e4-vasp-ab-aurora.pbs",
        "queue_limits": {
            "maximum_nodes_per_job": CAPACITY_MAX_NODES,
            "maximum_walltime_hours": QUEUE_WALLTIME_HOURS,
            "capacity_max_running_jobs_per_user": 2,
            "capacity_max_queued_jobs_per_user": 5,
        },
        "scheduler_constraints": {
            "capacity": "1-16 nodes, 168 h, 2 running / 5 queued jobs per user",
            "prod": "routes to small/medium/large; minimum 256 nodes, 12-24 h walltime",
            "debug_scaling": "1-256 nodes but 1 h walltime; unusable for DFT tasks",
            "gap": "no Aurora production queue accepts 17-255-node jobs, so the "
            "32/64/128-node tiers need an ALCF reservation",
        },
        "required_resource_configuration": {
            "tiers_1_to_16_nodes": "capacity queue, HydraGNN, select=<nodes>, walltime=<tier cap>",
            "tiers_32_to_128_nodes": (
                "ALCF reservation of 128 nodes for HydraGNN with jobs up to 168 h "
                "(LiFePO4 weak needs 144 h) and no 2-running-job limit"
            ),
            "allocation": (
                f"HydraGNN has {HYDRAGNN_AVAILABLE_NODE_HOURS:,.0f} node-hours available "
                "(through 2027-01-10); the expected campaign total exceeds it"
            ),
        },
        "output_root_template": (
            "runs/e4-scaling/<experiment>/<run-id>/n<node-count>/<segment-id-or-all>/"
        ),
        "semantics": {
            "scaling_level": (
                "workflow-level independent calculations; not single-VASP solver scaling"
            ),
            "strong_definition": (
                f"same {strong_count} fixed calculation IDs at every node tier 1-128"
            ),
            "weak_definition": "two unique calculations per node, run sequentially",
            "node_tiers": list(NODE_TIERS),
            "segmentation": (
                "strong tiers longer than 168 h run as wave-aligned segments; the "
                "analyzer sums segment makespans"
            ),
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
            "first_batch_hydragnn_revision_sha256": _sha256(
                repo / "deployments/aurora/jobs/e4-first-batch/batch-manifest-hydragnn.json"
            ),
            "six_class_inputs_sha256": _sha256(inputs_path),
        },
        "material_classes": classes,
        "authorization_request": {
            "status": "not_authorized",
            "additional_jobs_authorized": False,
            "existing_bcc_authorization_applies_only_to_the_frozen_bcc_manifests": True,
            "existing_bcc_authorized_node_hours": 1488,
            "expected_node_hours": round(expected_total, 1),
            "node_hours_ceiling": round(cap_total, 1),
            "capacity_queue_expected_node_hours": round(capacity_expected, 1),
            "reservation_expected_node_hours": round(expected_total - capacity_expected, 1),
            "hydragnn_available_node_hours": HYDRAGNN_AVAILABLE_NODE_HOURS,
            "additional_allocation_needed_expected_node_hours": round(
                max(0.0, expected_total - HYDRAGNN_AVAILABLE_NODE_HOURS), 1
            ),
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
