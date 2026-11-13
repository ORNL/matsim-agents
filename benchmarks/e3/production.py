"""Restartable E3-MLIP production planning and Perlmutter execution."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from benchmarks.e3.mlip_exploration import (
    E3_MLIP_CHECKPOINT_SHA256,
    E3_MLIP_MODEL,
    E3_STRATEGIES,
    _verified_checkpoint,
    _verified_checkpoint_sha256,
)

REPO = Path(__file__).resolve().parents[2]
OBJECTIVES = REPO / "benchmarks" / "e3" / "scientific-objectives-v1.yaml"
GPU_HOUR_CAP = 2_000
GPU_COUNT_PER_NODE = 4
ARRAY_ELEMENTS = 36
WALLTIME_SECONDS_PER_ELEMENT = 13 * 3600 + 53 * 60
GPU_SECONDS_PER_ELEMENT = GPU_COUNT_PER_NODE * WALLTIME_SECONDS_PER_ELEMENT
MODEL_NAME = "MACE-OMAT-0 medium"
OPTIMIZER_SETTINGS = {
    "optimizer": "FIRE",
    "maxiter": 500,
    "maxstep_A": 0.05,
    "fmax_eV_A": 0.02,
    "relax_cell": True,
    "stress_tolerance_eV_per_A3": 0.01,
    "max_cell_step_strain": 0.05,
    "max_cell_total_strain": 0.1,
    "require_native_stress": True,
    "phosphorene_cell_mask": [True, True, False, False, False, True],
    "mof_connectivity_guard": True,
}
REPLICATE_SEEDS = (20261111, 20261112, 20261113)
MATERIALS: dict[str, dict[str, Any]] = {
    "bcc_refractory": {
        "formula_families": {
            "bcc-equiatomic-quaternary": ["NbTaVZr"],
            "bcc-equiatomic-quinary": ["NbTaVHfZr"],
            "bcc-equiatomic-senary": ["NbTaVHfZrTi"],
            "bcc-substitutional": ["Nb2TaVHfZrTi", "NbTa2VHfZrTi"],
        },
        "tiers": [4, 8, 16, 32, 64],
        "tier_unit": "formula_units",
        "quota_per_tier": 1000,
        "generators": ["native_lattice_occupations"],
        "operations": ["chemical_disorder", "substitutional_ordering", "bounded_strain"],
    },
    "fcc_cantor": {
        "formula_families": {
            "fcc-cantor-equiatomic": ["CrMnFeCoNi"],
            "fcc-cantor-offstoichiometric": ["Cr2MnFeCoNi", "CrMn2FeCoNi", "CrMnFe2CoNi"],
        },
        "tiers": [4, 8, 16, 32, 64],
        "tier_unit": "formula_units",
        "quota_per_tier": 1000,
        "generators": ["native_lattice_occupations"],
        "operations": ["chemical_disorder", "substitutional_ordering", "bounded_strain"],
    },
    "lifepo4": {
        "formula_families": {
            "olivine-pristine": ["LiFePO4"],
            "olivine-antisite-pair": ["LiFePO4"],
            "olivine-bounded-strain": ["LiFePO4"],
        },
        "tiers": [4, 8, 16, 32],
        "tier_unit": "formula_units",
        "quota_per_tier": 1250,
        "generators": ["ase_seed_variants"],
        "operations": ["antisite_pair", "bounded_strain", "symmetry_distinct_ordering"],
    },
    "phosphorene": {
        "formula_families": {
            "black-phosphorene": ["P"],
            "blue-phosphorene": ["P"],
        },
        "tiers": [4, 16, 36, 64, 144],
        "tier_unit": "atoms",
        "quota_per_tier": 1000,
        "generators": ["2d_template"],
        "operations": ["allotrope", "in_plane_strain", "symmetry_distinct_distortion"],
    },
    "nb_ta_o": {
        "formula_families": {
            "niobium-pentoxide": ["Nb2O5"],
            "tantalum-pentoxide": ["Ta2O5"],
            "mixed-pentoxide": ["NbTaO5"],
            "niobium-dioxide": ["NbO2"],
            "tantalum-dioxide": ["TaO2"],
            "mixed-tetroxide": ["NbTaO4"],
            "niobium-oxygen-vacancy-charge-compensated": ["Nb2O4"],
            "tantalum-oxygen-vacancy-charge-compensated": ["Ta2O4"],
        },
        "tiers": [2, 4, 8, 16, 32],
        "tier_unit": "formula_units",
        "quota_per_tier": 1000,
        "generators": [
            "aflow_prototype",
            "pyxtal",
            "aflow_vacancy",
            "pyxtal_vacancy",
        ],
        "operations": [
            "prototype",
            "space_group",
            "symmetry_distinct_configuration",
            "bounded_strain",
            "oxygen_vacancy_charge_compensated",
        ],
    },
    "zn_formate_mof": {
        "formula_families": {
            "cod-7205949-framework": ["ZnC2H2O4"],
            "cod-7205949-framework-distortion": ["ZnC2H2O4"],
        },
        "tiers": [24, 48, 96, 192],
        "tier_unit": "formula_units",
        "quota_per_tier": 1250,
        "generators": ["cod_7205949"],
        "operations": [
            "connectivity_preserving_distortion",
            "bounded_strain",
            "proton_orientation",
        ],
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def reserve_gpu_hours(
    state_path: Path,
    reservation_id: str,
    *,
    reserved_gpu_seconds: int,
    cap_gpu_seconds: int = GPU_HOUR_CAP * 3600,
) -> dict[str, int]:
    """Atomically reserve an allocation against the persistent campaign cap."""
    if not reservation_id or reserved_gpu_seconds <= 0 or cap_gpu_seconds <= 0:
        raise ValueError("resource reservations require an ID and positive GPU-second limits")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_suffix(f"{state_path.suffix}.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["cap_gpu_seconds"] != cap_gpu_seconds:
                raise ValueError("resource ledger cap differs from the authorized cap")
        else:
            state = {
                "schema_version": 1,
                "cap_gpu_seconds": cap_gpu_seconds,
                "spent_gpu_seconds": 0,
                "reservations": {},
            }
        reservations = state["reservations"]
        if reservation_id in reservations:
            raise RuntimeError(f"duplicate Slurm allocation reservation: {reservation_id}")
        reserved_total = sum(int(item["reserved_gpu_seconds"]) for item in reservations.values())
        remaining = cap_gpu_seconds - int(state["spent_gpu_seconds"]) - reserved_total
        if reserved_gpu_seconds > remaining:
            raise RuntimeError(
                f"allocation needs {reserved_gpu_seconds} GPU-seconds but only "
                f"{max(remaining, 0)} remain in the campaign authorization"
            )
        reservations[reservation_id] = {
            "reserved_gpu_seconds": reserved_gpu_seconds,
        }
        _atomic_json(state_path, state)
        return {
            "cap_gpu_seconds": cap_gpu_seconds,
            "spent_gpu_seconds": int(state["spent_gpu_seconds"]),
            "reserved_gpu_seconds": reserved_total + reserved_gpu_seconds,
        }


def release_gpu_hours(
    state_path: Path, reservation_id: str, *, actual_gpu_seconds: int
) -> dict[str, int]:
    """Replace a reservation with the allocation's actual GPU-time charge."""
    if actual_gpu_seconds < 0:
        raise ValueError("actual GPU seconds cannot be negative")
    lock_path = state_path.with_suffix(f"{state_path.suffix}.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not state_path.is_file():
            raise FileNotFoundError(f"resource ledger does not exist: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        try:
            reservation = state["reservations"].pop(reservation_id)
        except KeyError as exc:
            raise RuntimeError(f"no active resource reservation: {reservation_id}") from exc
        charged = min(actual_gpu_seconds, int(reservation["reserved_gpu_seconds"]))
        state["spent_gpu_seconds"] += charged
        _atomic_json(state_path, state)
        reserved_total = sum(
            int(item["reserved_gpu_seconds"]) for item in state["reservations"].values()
        )
        return {
            "cap_gpu_seconds": int(state["cap_gpu_seconds"]),
            "spent_gpu_seconds": int(state["spent_gpu_seconds"]),
            "reserved_gpu_seconds": reserved_total,
        }


def _split_even(total: int, parts: int, offset: int = 0) -> list[int]:
    quotient, remainder = divmod(total, parts)
    return [quotient + int((index - offset) % parts < remainder) for index in range(parts)]


def _replicate_tier_quotas(spec: dict[str, Any]) -> list[dict[int, int]]:
    seeds_count = len(REPLICATE_SEEDS)
    quotas = [dict.fromkeys(spec["tiers"], 0) for _ in range(seeds_count)]
    totals = [0] * seeds_count
    for tier in spec["tiers"]:
        quotient, remainder = divmod(spec["quota_per_tier"], seeds_count)
        for seed_index in range(seeds_count):
            quotas[seed_index][tier] = quotient
            totals[seed_index] += quotient
        for _ in range(remainder):
            target = min(range(seeds_count), key=lambda index: (totals[index], index))
            quotas[target][tier] += 1
            totals[target] += 1
    expected = _split_even(spec["quota_per_tier"] * len(spec["tiers"]), seeds_count)
    if sorted(totals) != sorted(expected):
        raise AssertionError(f"unbalanced replicate quotas: {totals}, expected {expected}")
    return quotas


def _action_space(material: str) -> list[dict[str, Any]]:
    spec = MATERIALS[material]
    actions = []
    symmetry_bands = [
        ("any-compatible", []),
    ]
    if material == "nb_ta_o":
        symmetry_bands = [
            ("low-symmetry-sg1-15", list(range(1, 16))),
            ("orthorhombic-sg16-74", list(range(16, 75))),
            ("high-symmetry-sg75-230", list(range(75, 231))),
        ]
    for family, formulas in spec["formula_families"].items():
        for formula in formulas:
            for tier in spec["tiers"]:
                for generator in spec["generators"]:
                    for symmetry_name, space_groups in symmetry_bands:
                        for operation in spec["operations"]:
                            if material == "nb_ta_o" and family.endswith(
                                "oxygen-vacancy-charge-compensated"
                            ):
                                if operation not in {
                                    "oxygen_vacancy_charge_compensated",
                                    "bounded_strain",
                                }:
                                    continue
                                if generator not in {"aflow_vacancy", "pyxtal_vacancy"}:
                                    continue
                            elif material == "nb_ta_o" and (
                                operation == "oxygen_vacancy_charge_compensated"
                                or generator in {"aflow_vacancy", "pyxtal_vacancy"}
                            ):
                                continue
                            generator_family = family
                            defect = None
                            strain = None
                            if material in {"bcc_refractory", "fcc_cantor"}:
                                generator_family = (
                                    "equiatomic_ordering"
                                    if "equiatomic" in family
                                    else "controlled_substitution"
                                )
                                if operation == "bounded_strain":
                                    strain = 0.02
                            elif material == "lifepo4":
                                generator_family = {
                                    "olivine-pristine": "olivine",
                                    "olivine-antisite-pair": "antisite_pair",
                                    "olivine-bounded-strain": "olivine_strain",
                                }[family]
                                if generator_family == "antisite_pair":
                                    defect = "antisite_pair"
                                if generator_family == "olivine_strain":
                                    strain = 0.01
                            elif material == "phosphorene":
                                generator_family = (
                                    "black_phosphorene"
                                    if family == "black-phosphorene"
                                    else "blue_phosphorene"
                                )
                                if operation == "in_plane_strain":
                                    strain = 0.02
                                elif operation == "symmetry_distinct_distortion":
                                    defect = "reconstruction"
                            elif material == "nb_ta_o":
                                is_oxygen_defect = "oxygen-vacancy" in family
                                generator_family = (
                                    "oxygen_defect" if is_oxygen_defect else "oxide_prototype"
                                )
                                if is_oxygen_defect:
                                    defect = "oxygen_defect"
                                if operation == "bounded_strain":
                                    strain = 0.015
                                if is_oxygen_defect and operation not in {
                                    "oxygen_vacancy_charge_compensated",
                                    "bounded_strain",
                                }:
                                    continue
                            elif material == "zn_formate_mof":
                                generator_family = (
                                    "strain"
                                    if operation == "bounded_strain"
                                    else "bounded_perturbation"
                                )
                                strain = 0.005
                            action = {
                                "material": material,
                                "family": generator_family,
                                "family_label": family,
                                "formula": formula,
                                "size_tier": tier,
                                "generator": generator,
                                "defect": defect,
                                "operation": operation,
                                "strain": strain,
                                "space_group_family": symmetry_name,
                                "space_groups": space_groups,
                                "prototype_ids": [],
                            }
                            if material == "nb_ta_o" and "oxygen-vacancy" in family:
                                action["charge_compensation"] = (
                                    "charge-neutral M2O4; formal M(IV) with O(II-)"
                                )
                            elif material == "nb_ta_o" and formula == "NbTaO4":
                                action["charge_compensation"] = (
                                    "charge-neutral Nb(IV)/Ta(IV) with O(II-)"
                                )
                            action["action_id"] = hashlib.sha256(
                                _canonical_json(action).encode()
                            ).hexdigest()[:16]
                            actions.append(action)
    return actions


def build_campaign_manifest() -> dict[str, Any]:
    """Build the frozen 360-round schedule; structure choices remain adaptive."""
    objectives = yaml.safe_load(OBJECTIVES.read_text(encoding="utf-8"))
    rounds = []
    class_summaries = {}
    for material, spec in MATERIALS.items():
        replicate_quotas = _replicate_tier_quotas(spec)
        class_summaries[material] = {
            "unique_target_per_strategy": 5000,
            "tier_unit": spec["tier_unit"],
            "tiers": spec["tiers"],
            "tier_quotas_per_strategy": {
                str(tier): spec["quota_per_tier"] for tier in spec["tiers"]
            },
            "replicate_tier_quotas": [
                {str(tier): quota[tier] for tier in spec["tiers"]} for quota in replicate_quotas
            ],
        }
        control_families = (range(0, 7), range(7, 14), range(14, 20))
        for strategy in E3_STRATEGIES:
            for seed_index, seed in enumerate(REPLICATE_SEEDS):
                replicate_total = sum(replicate_quotas[seed_index].values())
                round_targets = _split_even(replicate_total, 10)
                tier_rounds = {
                    tier: _split_even(
                        replicate_quotas[seed_index][tier],
                        10,
                        offset=seed_index,
                    )
                    for tier in spec["tiers"]
                }
                minimum_by_tier_rounds = {
                    tier: _split_even(
                        math.ceil(replicate_quotas[seed_index][tier] * 0.10),
                        10,
                        offset=seed_index,
                    )
                    for tier in spec["tiers"]
                }
                family_ids = list(control_families[seed_index])
                family_rounds = _split_even(len(family_ids), 10)
                family_cursor = 0
                for round_index in range(1, 11):
                    controls = family_ids[
                        family_cursor : family_cursor + family_rounds[round_index - 1]
                    ]
                    family_cursor += len(controls)
                    tier_minima = {
                        str(tier): min(
                            minimum_by_tier_rounds[tier][round_index - 1],
                            tier_rounds[tier][round_index - 1],
                        )
                        for tier in spec["tiers"]
                    }
                    main_target = round_targets[round_index - 1]
                    control_target = len(controls) * len(spec["tiers"])
                    rounds.append(
                        {
                            "material": material,
                            "strategy": strategy,
                            "seed": seed,
                            "round": round_index,
                            "status": "planned",
                            "main_target": main_target,
                            "finite_size_control_families": controls,
                            "finite_size_control_target": control_target,
                            "logical_target": main_target + control_target,
                            "minimum_main_count_by_tier": tier_minima,
                            "replicate_unique_target": replicate_total,
                            "replicate_tier_targets": {
                                str(tier): replicate_quotas[seed_index][tier]
                                for tier in spec["tiers"]
                            },
                            "action_space_material": material,
                        }
                    )
                if family_cursor != len(family_ids):
                    raise AssertionError("finite-size controls were not assigned to rounds")

    manifest = {
        "schema_version": 1,
        "campaign_id": "e3-mlip-production-v1",
        "status": "planned_not_authorized",
        "model": MODEL_NAME,
        "mace_model": E3_MLIP_MODEL,
        "checkpoint_sha256": E3_MLIP_CHECKPOINT_SHA256,
        "optimizer": OPTIMIZER_SETTINGS,
        "unique_configurations_per_material_strategy": 5000,
        "finite_size_controls_per_strategy": 560,
        "logical_optimizations_per_strategy": 30560,
        "total_logical_optimizations": 61120,
        "seeds": list(REPLICATE_SEEDS),
        "rounds_per_replicate": 10,
        "strategies": list(E3_STRATEGIES),
        "material_specs": class_summaries,
        "action_spaces": {material: _action_space(material) for material in MATERIALS},
        "llm_guided_configuration": objectives["llm_guided_configuration"],
        "rounds": rounds,
        "scientific_objectives_sha256": _sha256_file(OBJECTIVES),
        "adaptation": {
            "evidence": "strategy-local MACE geometry-optimization outcomes only",
            "llm_calls_per_round": 1,
            "fallback_on_invalid_llm_proposal": False,
            "minimum_size_tier_coverage_fraction": 0.10,
            "e3_dft_inputs": False,
        },
        "execution": {
            "scheduler": "Slurm",
            "worker": "one independent MACE process per assigned GPU",
            "restart": (
                "completed tasks are skipped; interrupted attempts remain archived and are retried"
            ),
            "round_barrier": True,
            "maximum_nodes_per_array_task": 20,
            "gpus_per_node": 4,
            "resource_authorization_required": True,
            "job_submission_performed": False,
        },
        "resource_request": {
            "account": "m5216",
            "billing_account": "m5216_g",
            "qos": "regular",
            "billing_qos": "gpu_regular",
            "array_elements": ARRAY_ELEMENTS,
            "nodes_per_array_element": 1,
            "gpus_per_node": GPU_COUNT_PER_NODE,
            "walltime_seconds_per_array_element": WALLTIME_SECONDS_PER_ELEMENT,
            "maximum_scheduled_gpu_hours": (ARRAY_ELEMENTS * GPU_SECONDS_PER_ELEMENT / 3600),
            "maximum_scheduled_gpu_node_hours": (ARRAY_ELEMENTS * WALLTIME_SECONDS_PER_ELEMENT)
            / 3600,
            "requested_gpu_hour_cap": GPU_HOUR_CAP,
            "gpu_hours_at_cap": GPU_HOUR_CAP,
            "average_gpu_seconds_per_logical_optimization_at_cap": (GPU_HOUR_CAP * 3600 / 61120),
            "basis": (
                "36 streams x 1 GPU node x 4 GPUs x 13h53m; persistent allocation "
                "ledger caps all submissions and restarts at 2,000 GPU-hours"
            ),
            "measured_throughput": False,
        },
    }
    validate_campaign_manifest(manifest)
    return manifest


def validate_campaign_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("optimizer") != OPTIMIZER_SETTINGS:
        raise ValueError("production manifest optimizer does not match the current protocol")
    if len(manifest["rounds"]) != 360:
        raise ValueError(f"expected 360 adaptive rounds, got {len(manifest['rounds'])}")
    if set(manifest["strategies"]) != set(E3_STRATEGIES):
        raise ValueError("production manifest must contain exactly the paired strategies")
    keys = {
        (row["material"], row["strategy"], row["seed"], row["round"]) for row in manifest["rounds"]
    }
    if len(keys) != 360:
        raise ValueError("production manifest contains duplicate or missing round identities")
    if sum(row["main_target"] for row in manifest["rounds"]) != 60000:
        raise ValueError("main target does not equal 60,000 unique logical configurations")
    if sum(row["finite_size_control_target"] for row in manifest["rounds"]) != 1120:
        raise ValueError("finite-size controls do not equal 1,120 logical optimizations")
    if sum(row["logical_target"] for row in manifest["rounds"]) != 61120:
        raise ValueError("total logical optimization target is not 61,120")
    resources = manifest["resource_request"]
    if resources["requested_gpu_hour_cap"] != GPU_HOUR_CAP:
        raise ValueError("resource cap does not match the authorized 2,000 GPU-hours")
    if resources["maximum_scheduled_gpu_hours"] > GPU_HOUR_CAP:
        raise ValueError("a single production array can exceed the authorized GPU-hour cap")
    for row in manifest["rounds"]:
        if sum(row["minimum_main_count_by_tier"].values()) > row["main_target"]:
            raise ValueError(f"tier minima exceed round target: {row}")


def write_campaign_manifest(output: Path, *, overwrite: bool = False) -> dict[str, Any]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite production manifest: {output}")
    manifest = build_campaign_manifest()
    _atomic_json(output, manifest)
    return manifest


def _history_for_round(
    run_root: Path, material: str, strategy: str, seed: int, round_index: int
) -> list[dict[str, Any]]:
    history = []
    for index in range(1, round_index):
        path = (
            run_root
            / material
            / strategy
            / f"seed-{seed}"
            / f"round-{index:02d}"
            / "round-summary.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"adaptive round {index} is not complete: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            raise RuntimeError(f"cannot adapt from incomplete prior round: {path}")
        history.extend(summary.get("action_outcomes", []))
    return history


def _action_dimensions(action: dict[str, Any]) -> set[str]:
    dimensions = {
        f"material:{action['material']}",
        f"tier:{action['size_tier']}",
        f"family:{action.get('family_label', action['family'])}",
        f"formula:{action['formula']}",
        f"generator:{action['generator']}",
        f"operation:{action.get('operation')}",
        f"symmetry:{action.get('space_group_family', 'unrestricted')}",
    }
    if action.get("defect") is not None:
        dimensions.add(f"defect:{action['defect']}")
    return dimensions


def _finite_capacity_group(action: dict[str, Any]) -> tuple[str, str] | None:
    if (
        action.get("material") == "lifepo4"
        and action.get("family_label") == "olivine-pristine"
        and action.get("family") == "olivine"
        and action.get("defect") is None
        and action.get("strain") is None
    ):
        return "lifepo4-pristine-parent", str(action["size_tier"])
    if (
        action.get("material") == "phosphorene"
        and action.get("family_label") in {"black-phosphorene", "blue-phosphorene"}
        and action.get("family") in {"black_phosphorene", "black", "blue_phosphorene", "blue"}
        and action.get("defect") is None
        and action.get("strain") is None
    ):
        return f"phosphorene-pristine-{action['family_label']}", str(action["size_tier"])
    return None


def _finite_capacity_remaining(
    action_space: list[dict[str, Any]], history: list[dict[str, Any]]
) -> dict[tuple[str, str], int]:
    action_by_id = {action["action_id"]: action for action in action_space}
    capacity_groups = {
        group for action in action_space if (group := _finite_capacity_group(action)) is not None
    }
    used: dict[tuple[str, str], int] = defaultdict(int)
    for record in history:
        action = action_by_id.get(record.get("action_id"), record.get("action", {}))
        group = _finite_capacity_group(action)
        if group is not None:
            used[group] += int(record.get("attempted", record.get("count", 0)))
    return {group: max(0, 1 - used[group]) for group in capacity_groups}


def _rule_allocation(
    action_space: list[dict[str, Any]],
    history: list[dict[str, Any]],
    target: int,
    minimum_by_tier: dict[str, int],
) -> list[dict[str, Any]]:
    allowed = {action["action_id"]: action for action in action_space}
    covered: dict[str, int] = defaultdict(int)
    attempted: dict[str, int] = defaultdict(int)
    converged: dict[str, int] = defaultdict(int)
    for record in history:
        action = allowed.get(record["action_id"], record.get("action", {}))
        if not action:
            continue
        for dimension in _action_dimensions(action):
            covered[dimension] += int(record.get("count", record.get("attempted", 0)))
        attempted[action["action_id"]] += int(record.get("attempted", 0))
        converged[action["action_id"]] += int(record.get("converged", 0))
    counts: dict[str, int] = defaultdict(int)
    finite_remaining = _finite_capacity_remaining(action_space, history)
    finite_allocated: dict[tuple[str, str], int] = defaultdict(int)

    def available(action: dict[str, Any]) -> bool:
        group = _finite_capacity_group(action)
        return group is None or finite_allocated[group] < finite_remaining[group]

    def allocate_one(action: dict[str, Any]) -> None:
        counts[action["action_id"]] += 1
        group = _finite_capacity_group(action)
        if group is not None:
            finite_allocated[group] += 1
        for dimension in _action_dimensions(action):
            covered[dimension] += 1

    def score(action: dict[str, Any]) -> tuple[float, str]:
        undercovered = sum(1 / (1 + covered[dimension]) for dimension in _action_dimensions(action))
        n = attempted[action["action_id"]]
        rate = (converged[action["action_id"]] + 1) / (n + 2)
        return undercovered + 0.25 * rate, action["action_id"]

    for tier_text, count in sorted(minimum_by_tier.items(), key=lambda item: int(item[0])):
        tier_actions = [
            action
            for action in action_space
            if str(action["size_tier"]) == tier_text and available(action)
        ]
        if not tier_actions and count:
            raise ValueError(f"no allowed generation actions for required tier {tier_text}")
        for allocation_index in range(count):
            if not tier_actions:
                raise ValueError(
                    f"finite action capacity cannot satisfy the tier {tier_text} minimum"
                )
            chosen = max(
                tier_actions, key=lambda action: (score(action)[0], -int(action["action_id"], 16))
            )
            allocate_one(chosen)
            tier_actions = [action for action in tier_actions if available(action)]
            if allocation_index < count - 1 and not tier_actions:
                raise ValueError(
                    f"finite action capacity cannot satisfy the tier {tier_text} minimum"
                )
    allocated = sum(counts.values())
    if allocated > target:
        raise ValueError("required tier coverage exceeds the round budget")
    for _ in range(target - allocated):
        eligible = [action for action in action_space if available(action)]
        if not eligible:
            raise ValueError("finite action capacities cannot satisfy the round target")
        chosen = max(eligible, key=lambda action: (score(action)[0], -int(action["action_id"], 16)))
        allocate_one(chosen)
    return [
        {"action": allowed[action_id], "count": count}
        for action_id, count in sorted(counts.items())
    ]


def _compact_action_catalog(material: str, action_space: list[dict[str, Any]]) -> dict[str, Any]:
    dimension_keys = {
        "m": "material_class",
        "c": "formula",
        "t": "size_tier",
        "f": "family_label",
        "g": "generator",
        "p": "perturbation",
        "s": "space_group_family",
    }
    values: dict[str, list[Any]] = {key: [] for key in dimension_keys}
    action_values = []
    for action in action_space:
        dimensions = {
            "m": material,
            "c": action["formula"],
            "t": action["size_tier"],
            "f": action.get("family_label", action["family"]),
            "g": action["generator"],
            "p": {
                key: action.get(key)
                for key in ("operation", "defect", "strain", "charge_compensation")
                if action.get(key) is not None
            },
            "s": action.get("space_group_family", "unrestricted"),
        }
        codes = {}
        for key, value in dimensions.items():
            if value not in values[key]:
                values[key].append(value)
            codes[key] = values[key].index(value)
        action_values.append((action["action_id"], codes))
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"

    def code(number: int) -> str:
        if number < len(alphabet):
            return alphabet[number]
        return f"{alphabet[number // len(alphabet)]}{alphabet[number % len(alphabet)]}"

    return {
        "dimensions": {
            key: {code(index): value for index, value in enumerate(dimension_values)}
            for key, dimension_values in values.items()
        },
        "allowed": [
            f"{action_id} " + " ".join(f"{key}{code(index)}" for key, index in codes.items())
            for action_id, codes in action_values
        ],
    }


def _compact_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "evidence_id",
        "action_id",
        "size_tier",
        "family",
        "generator",
        "attempted",
        "converged",
        "nonconverged",
        "failed",
        "mean_energy_eV_per_atom",
        "distinct_final_geometry_hashes",
        "mean_rms_atom_displacement_A",
        "runtime_seconds",
        "gpu_hours",
    )
    return [{key: item[key] for key in fields if key in item} for item in history]


def _llm_allocation(
    chat: Any,
    material: str,
    action_space: list[dict[str, Any]],
    history: list[dict[str, Any]],
    target: int,
    minimum_by_tier: dict[str, int],
) -> tuple[list[dict[str, Any]], list[str], str]:
    valid_evidence = {"campaign-design"} | {
        item["evidence_id"] for item in history if item.get("evidence_id")
    }
    finite_remaining = _finite_capacity_remaining(action_space, history)
    available_actions = [
        action
        for action in action_space
        if (group := _finite_capacity_group(action)) is None or finite_remaining[group] > 0
    ]
    prompt_catalog = _compact_action_catalog(material, available_actions)
    finite_capacity_by_action = {
        action["action_id"]: finite_remaining[group]
        for action in available_actions
        if (group := _finite_capacity_group(action)) is not None
    }
    prompt_history = _compact_history(history)
    prompt = (
        "Choose E3-MLIP crystallographic generation allocations for this round. "
        "Use only the allow-listed actions and only the strategy-local MACE geometry "
        "optimization history below. There are no DFT labels, DFT references, or "
        "independent validation data. Return JSON with exactly keys allocations, "
        "cited_evidence_ids, rationale. allocations is a list of {action_id,count}; "
        "counts must be positive integers, IDs unique, sum exactly to the round target, "
        "and each minimum tier count must be met. Cite campaign-design and/or valid "
        "prior evidence IDs. Do not select outside the list. The compact hierarchical "
        "catalog maps dimension codes to values; allowed rows are action_id followed "
        "by material class(m), composition(c), tier(t), structural family(f), generator(g), "
        "perturbation(p), and symmetry family(s) codes. These rows are the complete "
        "authoritative allowlist.\n"
        f"Material class: {material}\n"
        f"Round target: {target}\n"
        f"Minimum counts by size tier: {json.dumps(minimum_by_tier, sort_keys=True)}\n"
        f"Prior strategy-local outcomes: {json.dumps(prompt_history, sort_keys=True)}\n"
        f"Remaining finite action counts: "
        f"{json.dumps(finite_capacity_by_action, sort_keys=True)}\n"
        f"Hierarchical action catalog: {json.dumps(prompt_catalog, separators=(',', ':'))}\n"
    )
    response = chat.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM round proposal contains no JSON object")
    proposal = json.loads(content[start : end + 1])
    if set(proposal) != {"allocations", "cited_evidence_ids", "rationale"}:
        raise ValueError("LLM allocation must contain exactly the required fields")
    action_map = {action["action_id"]: action for action in action_space}
    allocations = proposal["allocations"]
    if not isinstance(allocations, list):
        raise ValueError("LLM allocations must be a list")
    selected = []
    seen = set()
    tier_counts: dict[str, int] = defaultdict(int)
    finite_allocated: dict[tuple[str, str], int] = defaultdict(int)
    total = 0
    for allocation in allocations:
        if not isinstance(allocation, dict) or set(allocation) != {"action_id", "count"}:
            raise ValueError("each LLM allocation must contain action_id and count")
        action_id, count = allocation["action_id"], allocation["count"]
        if (
            not isinstance(action_id, str)
            or action_id not in action_map
            or action_id in seen
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise ValueError("LLM allocation contains an invalid or repeated action")
        seen.add(action_id)
        action = action_map[action_id]
        group = _finite_capacity_group(action)
        if group is not None and finite_allocated[group] + count > finite_remaining[group]:
            raise ValueError("LLM allocation exceeds a finite generation-action capacity")
        if group is not None:
            finite_allocated[group] += count
        tier_counts[str(action["size_tier"])] += count
        total += count
        selected.append({"action": action, "count": count})
    if total != target:
        raise ValueError(f"LLM allocation totals {total}, expected {target}")
    if any(tier_counts[tier] < count for tier, count in minimum_by_tier.items()):
        raise ValueError("LLM allocation does not meet the minimum size-tier coverage")
    citations = proposal["cited_evidence_ids"]
    if (
        not isinstance(citations, list)
        or not citations
        or not all(isinstance(item, str) for item in citations)
        or not set(citations) <= valid_evidence
    ):
        raise ValueError("LLM must cite valid campaign-design or strategy-local evidence")
    rationale = proposal["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("LLM rationale must be non-empty")
    return selected, citations, rationale


def _chat_model() -> Any:
    from matsim_agents.backends.llm.provider import get_chat_model

    configuration = yaml.safe_load(OBJECTIVES.read_text(encoding="utf-8"))[
        "llm_guided_configuration"
    ]
    return get_chat_model(
        provider=configuration["provider"],
        model=configuration["model"],
        temperature=configuration["temperature"],
        max_tokens=configuration["request_limits"]["max_tokens"],
        mag_provider=configuration["mag_provider"],
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _structure_valid(path: Path, expected_formula: str) -> tuple[int, list[str]]:
    from collections import Counter

    from ase.io import read

    from matsim_agents.discovery.composition import parse_composition

    atoms = read(path)
    if len(atoms) == 0 or not atoms.pbc.any():
        raise ValueError("generated structure must be non-empty and periodic")
    if not atoms.cell.rank == 3 and expected_formula != "P":
        raise ValueError("three-dimensional material has a singular cell")
    if not all(map(math.isfinite, atoms.positions.ravel())):
        raise ValueError("generated positions contain non-finite values")
    if expected_formula != "P" and atoms.get_volume() <= 0:
        raise ValueError("generated structure has a non-positive cell volume")
    expected = parse_composition(expected_formula)
    if expected is None:
        raise ValueError(f"generated formula is not parseable: {expected_formula}")
    observed = Counter(atoms.get_chemical_symbols())
    observed_gcd = math.gcd(*observed.values())
    observed_reduced = {element: count // observed_gcd for element, count in observed.items()}
    if observed_reduced != expected.elements:
        raise ValueError(
            f"generated composition {dict(observed)} does not reduce to {expected_formula}"
        )
    return len(atoms), sorted(set(atoms.get_chemical_symbols()))


def _geometry_sha256(path: Path) -> str:
    from ase.io import read

    atoms = read(path)
    scaled = atoms.get_scaled_positions(wrap=True)
    sites = sorted(
        (
            int(number),
            *[round(float(value), 7) for value in position],
        )
        for number, position in zip(atoms.numbers, scaled, strict=True)
    )
    payload = {
        "numbers": [site[0] for site in sites],
        "scaled_positions": [site[1:] for site in sites],
        "cell": [[round(float(value), 7) for value in row] for row in atoms.cell.array],
        "pbc": [bool(value) for value in atoms.pbc],
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _materialize_round(
    manifest: dict[str, Any],
    round_spec: dict[str, Any],
    run_root: Path,
    history: list[dict[str, Any]],
    chat: Any | None,
) -> tuple[Path, Path, dict[str, Any]]:
    from ase.io import read

    from benchmarks.e3.structure_generation import (
        generate_structures,
        make_structure_matcher,
        replicate_structure,
        structures_symmetry_equivalent,
    )

    material = round_spec["material"]
    strategy = round_spec["strategy"]
    seed = int(round_spec["seed"])
    round_index = int(round_spec["round"])
    round_dir = run_root / material / strategy / f"seed-{seed}" / f"round-{round_index:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    round_manifest_path = round_dir / "round-manifest.json"
    if round_manifest_path.is_file():
        existing = json.loads(round_manifest_path.read_text(encoding="utf-8"))
        if existing.get("optimizer") != OPTIMIZER_SETTINGS:
            raise ValueError("existing round uses a different optimization protocol")
        if existing.get("status") in {"generation_complete", "complete"}:
            tasks_path = Path(existing["tasks_path"])
            return round_manifest_path, tasks_path, existing

    action_space = manifest["action_spaces"][material]
    target = int(round_spec["main_target"])
    minimum_by_tier = round_spec["minimum_main_count_by_tier"]
    if round_manifest_path.is_file():
        existing = json.loads(round_manifest_path.read_text(encoding="utf-8"))
        allocations = existing["allocations"]
        selection = existing["selection"]
    else:
        if strategy == "rule_based_adaptation":
            allocations = _rule_allocation(action_space, history, target, minimum_by_tier)
            selection = {
                "policy": (
                    "least-covered tier/family/formula/generator/operation with "
                    "smoothed convergence"
                ),
                "cited_evidence_ids": [
                    "campaign-design",
                    *[row["evidence_id"] for row in history],
                ],
                "rationale": "Deterministic greedy selection; tie-breaks are stable action IDs.",
            }
        else:
            if chat is None:
                raise ValueError("LLM-guided round requires a configured chat model")
            allocations, citations, rationale = _llm_allocation(
                chat, material, action_space, history, target, minimum_by_tier
            )
            selection = {
                "policy": "evidence-grounded LLM allocation",
                "model_id": getattr(chat, "model_id", None),
                "cited_evidence_ids": citations,
                "rationale": rationale,
            }
        _atomic_json(
            round_manifest_path,
            {
                **round_spec,
                "status": "generating",
                "optimizer": OPTIMIZER_SETTINGS,
                "selection": selection,
                "allocations": allocations,
            },
        )

    selected_actions = []
    candidate_records = []
    registry_path = run_root / material / strategy / f"seed-{seed}" / "candidate-registry.jsonl"
    known_hashes = set()
    symmetry_index: dict[tuple[str, int, int], list[Any]] = defaultdict(list)
    symmetry_matcher = make_structure_matcher()
    if registry_path.is_file():
        for line in registry_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prior = json.loads(line)
                known_hashes.add(prior["structure_sha256"])
                if prior.get("candidate_kind") == "unique_configuration":
                    prior_atoms = read(prior["structure_path"])
                    key = (
                        str(prior["formula"]),
                        int(prior["action"]["size_tier"]),
                        int(prior["num_atoms"]),
                    )
                    symmetry_index[key].append(prior_atoms)
    for allocation in allocations:
        action_template = dict(allocation["action"])
        count = int(allocation["count"])
        action_seed = int(
            hashlib.sha256(
                f"{seed}:{round_index}:{action_template['action_id']}".encode()
            ).hexdigest()[:8],
            16,
        )
        action_id = action_template["action_id"]
        accepted = 0
        generated_total = 0
        for batch_index in range(100):
            remaining = count - accepted
            if remaining == 0:
                break
            action = {
                **action_template,
                "random_seed": int(
                    hashlib.sha256(f"{action_seed}:{batch_index}".encode()).hexdigest()[:8], 16
                ),
                "count": remaining,
            }
            records = generate_structures(
                material,
                action,
                round_dir / "generated" / action_id / f"batch-{batch_index:03d}",
            )
            generated_total += len(records)
            for record in records:
                structure_path = Path(record["structure_path"]).resolve()
                if not structure_path.is_file():
                    raise FileNotFoundError(
                        f"generator returned missing structure: {structure_path}"
                    )
                file_digest = _sha256_file(structure_path)
                if file_digest != record["structure_sha256"]:
                    raise ValueError(f"structure hash mismatch: {structure_path}")
                num_atoms, species = _structure_valid(structure_path, record["formula"])
                if num_atoms != int(record["num_atoms"]):
                    raise ValueError(f"atom count mismatch for {structure_path}")
                digest = _geometry_sha256(structure_path)
                if digest in known_hashes:
                    continue
                atoms = read(structure_path)
                symmetry_key = (
                    str(record["formula"]),
                    int(action_template["size_tier"]),
                    num_atoms,
                )
                if any(
                    structures_symmetry_equivalent(atoms, prior, symmetry_matcher)
                    for prior in symmetry_index[symmetry_key]
                ):
                    continue
                known_hashes.add(digest)
                symmetry_index[symmetry_key].append(atoms)
                candidate_id = hashlib.sha256(
                    f"{material}:{strategy}:{seed}:{round_index}:{digest}".encode()
                ).hexdigest()[:24]
                row = {
                    "candidate_id": candidate_id,
                    "material": material,
                    "strategy": strategy,
                    "seed": seed,
                    "round": round_index,
                    "candidate_kind": "unique_configuration",
                    "structure_path": str(structure_path),
                    "structure_sha256": digest,
                    "num_atoms": num_atoms,
                    "species": species,
                    "formula": record["formula"],
                    "formula_units": record["formula_units"],
                    "generation": record["generation"],
                    "structure_file_sha256": file_digest,
                    "action_id": action_id,
                    "action": action,
                    "status": "pending",
                }
                candidate_records.append(row)
                accepted += 1
            if batch_index == 99 and accepted < count:
                raise RuntimeError(
                    f"could not fill the {count}-structure allocation for {material}/"
                    f"{action_id}; {accepted} unique configurations accepted after "
                    f"{generated_total} generation attempts"
                )
        selected_actions.append(
            {
                "action": action_template,
                "requested": count,
                "generated": generated_total,
                "accepted": accepted,
            }
        )
        if accepted != count:
            raise RuntimeError(
                f"generation yielded only {accepted}/{count} globally unique structures "
                f"for action {action['action_id']}; round remains incomplete"
            )

    control_records = []
    for family_id in round_spec["finite_size_control_families"]:
        tiers = MATERIALS[material]["tiers"]
        base_tier = tiers[0]
        base_actions = [
            action
            for action in action_space
            if action["size_tier"] == base_tier
            and action["generator"] == MATERIALS[material]["generators"][0]
        ]
        base_action = dict(base_actions[0])
        base_action.update(
            action_id=f"control-{family_id:02d}-tier-{base_tier}",
            random_seed=int(
                hashlib.sha256(f"size-control:{material}:{family_id}".encode()).hexdigest()[:8],
                16,
            ),
            count=1,
            control_family_id=family_id,
        )
        generated = generate_structures(
            material,
            base_action,
            round_dir / "controls" / base_action["action_id"],
        )
        if len(generated) != 1:
            raise ValueError(f"size-control generator did not return one structure: {base_action}")
        base_record = generated[0]
        for tier in tiers:
            control_action = dict(base_action)
            control_action.update(
                action_id=f"control-{family_id:02d}-tier-{tier}",
                size_tier=tier,
            )
            record = replicate_structure(
                material,
                base_record,
                control_action,
                round_dir / "controls" / control_action["action_id"],
                family_id,
            )
            if tier == base_tier:
                base_record = record
            path = Path(record["structure_path"]).resolve()
            file_digest = _sha256_file(path)
            if file_digest != record["structure_sha256"]:
                raise ValueError(f"size-control structure hash mismatch: {path}")
            digest = _geometry_sha256(path)
            num_atoms, species = _structure_valid(path, record["formula"])
            control_records.append(
                {
                    "candidate_id": hashlib.sha256(
                        f"control:{material}:{family_id}:{tier}:{digest}".encode()
                    ).hexdigest()[:24],
                    "material": material,
                    "strategy": strategy,
                    "seed": seed,
                    "round": round_index,
                    "candidate_kind": "finite_size_control",
                    "control_family_id": family_id,
                    "structure_path": str(path),
                    "structure_sha256": digest,
                    "num_atoms": num_atoms,
                    "species": species,
                    "formula": record["formula"],
                    "formula_units": record["formula_units"],
                    "generation": record["generation"],
                    "structure_file_sha256": file_digest,
                    "action_id": control_action["action_id"],
                    "action": control_action,
                    "status": "pending",
                }
            )

    expected_main = target
    if len(candidate_records) != expected_main:
        raise RuntimeError(
            f"round yielded {len(candidate_records)} unique candidates, expected {expected_main}"
        )
    tasks = candidate_records + control_records
    task_path = round_dir / "tasks.jsonl"
    _write_jsonl(task_path, tasks)
    if candidate_records:
        with registry_path.open("a", encoding="utf-8") as stream:
            for record in candidate_records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    round_manifest = {
        **round_spec,
        "status": "generation_complete",
        "model": MODEL_NAME,
        "mace_model": E3_MLIP_MODEL,
        "mace_model_path": str(_verified_checkpoint()[0]),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "optimizer": OPTIMIZER_SETTINGS,
        "selection": selection,
        "allocations": allocations,
        "selected_actions": selected_actions,
        "unique_candidates": len(candidate_records),
        "finite_size_controls": len(control_records),
        "tasks_path": str(task_path),
        "task_count": len(tasks),
        "structure_generation_failures": [],
    }
    _atomic_json(round_manifest_path, round_manifest)
    return round_manifest_path, task_path, round_manifest


def _task_result(attempt_dir: Path, task: dict[str, Any]) -> Path | None:
    for attempt in sorted(attempt_dir.glob("attempt-*"), reverse=True):
        result_path = attempt / "result.json"
        if not result_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            result.get("structure_sha256") == task["structure_sha256"]
            and result.get("terminal")
            and result.get("optimizer") == OPTIMIZER_SETTINGS
        ):
            return result_path
    return None


def _space_group_number(atoms: Any) -> int | None:
    try:
        import spglib
    except ImportError:
        return None
    dataset = spglib.get_symmetry_dataset(
        (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers),
        symprec=1e-2,
    )
    return int(dataset.number) if dataset is not None else None


def _execute_task(task: dict[str, Any], task_root: Path, calculators: Any) -> dict[str, Any]:
    import numpy as np
    from ase.geometry import find_mic
    from ase.io import read, write

    from matsim_agents.backends.mlip.relaxation import (
        RelaxStructureInput,
        relax_with_mlip,
    )

    task_dir = task_root / task["candidate_id"]
    task_dir.mkdir(parents=True, exist_ok=True)
    existing = _task_result(task_dir, task)
    if existing is not None:
        return json.loads(existing.read_text(encoding="utf-8"))
    attempts = sorted(task_dir.glob("attempt-*"))
    attempt_index = len(attempts) + 1
    attempt_dir = task_dir / f"attempt-{attempt_index:04d}"
    attempt_dir.mkdir()
    started = time.monotonic()
    wall_started = time.time()
    initial_energy = None
    initial_max_force = None
    result_payload: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": task["candidate_id"],
        "structure_sha256": task["structure_sha256"],
        "candidate_kind": task["candidate_kind"],
        "model": MODEL_NAME,
        "mace_model": E3_MLIP_MODEL,
        "checkpoint_sha256": E3_MLIP_CHECKPOINT_SHA256,
        "optimizer": OPTIMIZER_SETTINGS,
        "initial_structure_path": task["structure_path"],
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_procid": os.environ.get("SLURM_PROCID"),
        "device": "cuda:0" if os.environ.get("SLURM_JOB_ID") else "unverified",
        "started_at_unix": wall_started,
        "terminal": True,
    }
    try:
        if os.environ.get("SLURM_JOB_ID"):
            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("each Slurm worker must see exactly one allocated CUDA GPU")
            result_payload["gpu_name"] = torch.cuda.get_device_name(0)
            result_payload["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            result_payload["cuda_version"] = torch.version.cuda
        initial = read(task["structure_path"])
        initial.calc = None
        for key in ("energy", "forces", "stress", "free_energy"):
            initial.info.pop(key, None)
            initial.arrays.pop(key, None)
        clean_path = attempt_dir / "initial.extxyz"
        write(clean_path, initial, format="extxyz")
        checkpoint_path, checkpoint_sha256 = _verified_checkpoint()
        if checkpoint_sha256 != E3_MLIP_CHECKPOINT_SHA256:
            raise ValueError("relaxation checkpoint does not match the frozen E3-MLIP model")
        result_payload["mace_model_path"] = str(checkpoint_path)
        relax_args = RelaxStructureInput(
            structure_path=str(clean_path),
            output_dir=str(attempt_dir),
            mlip_backend="mace",
            mace_family="checkpoint",
            mace_model=str(checkpoint_path),
            optimizer=OPTIMIZER_SETTINGS["optimizer"],
            maxiter=OPTIMIZER_SETTINGS["maxiter"],
            maxstep=OPTIMIZER_SETTINGS["maxstep_A"],
            fmax=OPTIMIZER_SETTINGS["fmax_eV_A"],
            relax_cell=OPTIMIZER_SETTINGS["relax_cell"],
            cell_mask=(
                tuple(OPTIMIZER_SETTINGS["phosphorene_cell_mask"])
                if task["material"] == "phosphorene"
                else None
            ),
            stress_tolerance_eV_per_A3=OPTIMIZER_SETTINGS["stress_tolerance_eV_per_A3"],
            max_cell_step_strain=OPTIMIZER_SETTINGS["max_cell_step_strain"],
            max_cell_total_strain=OPTIMIZER_SETTINGS["max_cell_total_strain"],
            require_native_stress=OPTIMIZER_SETTINGS["require_native_stress"],
            mlp_device="cuda",
        )
        calculator_record = calculators.get(relax_args)
        calculator = calculator_record.calculator
        initial.calc = calculator
        initial_energy = float(initial.get_potential_energy())
        initial_forces = initial.get_forces()
        initial_max_force = float((initial_forces**2).sum(axis=1).max() ** 0.5)
        initial_forces_path = attempt_dir / "initial_forces.npy"
        np.save(initial_forces_path, initial_forces, allow_pickle=False)
        result_payload["initial_energy_eV"] = initial_energy
        result_payload["initial_max_force_eV_per_A"] = initial_max_force
        result_payload["initial_forces_path"] = str(initial_forces_path)
        result_payload["initial_forces_sha256"] = _sha256_file(initial_forces_path)
        result_payload["initial_space_group"] = _space_group_number(initial)
        result_payload["initial_force_evaluated"] = True
        step_guard = None
        if task["material"] == "zn_formate_mof":
            from benchmarks.e3.structure_generation import _mof_connectivity_signature

            initial_connectivity = _mof_connectivity_signature(initial)

            def step_guard(atoms):
                if _mof_connectivity_signature(atoms) != initial_connectivity:
                    return "mof_connectivity_changed"
                return None

        relaxed = relax_with_mlip(relax_args, calculators=calculators, step_guard=step_guard)
        result_payload.update(
            status="converged" if relaxed.converged else "nonconverged",
            final_energy_eV=relaxed.final_energy_eV,
            final_max_force_eV_per_A=relaxed.final_max_force_eV_per_A,
            steps=relaxed.num_steps,
            converged=relaxed.converged,
            optimized_structure_path=relaxed.optimized_structure_path,
            trajectory_path=relaxed.trajectory_path,
            optimizer_log_path=relaxed.log_csv_path,
            termination_reason=relaxed.termination_reason,
        )
        final_atoms = read(relaxed.optimized_structure_path)
        final_atoms.calc = calculator
        final_forces = final_atoms.get_forces()
        final_stress = np.asarray(final_atoms.get_stress(), dtype=float)
        if not np.isfinite(final_stress).all():
            raise ValueError("Final native stress is non-finite")
        result_payload["final_stress_eV_per_A3"] = final_stress.tolist()
        result_payload["cell_mask"] = list(relax_args.cell_mask) if relax_args.cell_mask else None
        final_forces_path = attempt_dir / "final_forces.npy"
        np.save(final_forces_path, final_forces, allow_pickle=False)
        final_path = Path(relaxed.optimized_structure_path)
        result_payload["final_structure_file_sha256"] = _sha256_file(final_path)
        result_payload["final_structure_sha256"] = _geometry_sha256(final_path)
        result_payload["final_forces_path"] = str(final_forces_path)
        result_payload["final_forces_sha256"] = _sha256_file(final_forces_path)
        displacement_vectors, _ = find_mic(
            final_atoms.positions - initial.positions,
            cell=initial.cell,
            pbc=initial.pbc,
        )
        displacement_lengths = np.linalg.norm(displacement_vectors, axis=1)
        result_payload["rms_atom_displacement_A"] = float(np.sqrt(np.mean(displacement_lengths**2)))
        result_payload["max_atom_displacement_A"] = float(displacement_lengths.max())
        result_payload["final_num_atoms"] = len(final_atoms)
        result_payload["species"] = sorted(set(final_atoms.get_chemical_symbols()))
        result_payload["final_space_group"] = _space_group_number(final_atoms)
        result_payload["symmetry_changed"] = (
            result_payload["initial_space_group"] != result_payload["final_space_group"]
        )
    except Exception as exc:
        result_payload.update(
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
            initial_energy_eV=initial_energy,
            initial_max_force_eV_per_A=initial_max_force,
            initial_force_evaluated=initial_energy is not None,
            termination_reason="backend_or_runtime_error",
            trajectory_path=str(attempt_dir / "optimization.traj"),
            optimizer_log_path=str(attempt_dir / "optimization.csv"),
        )
    finally:
        elapsed = time.monotonic() - started
        result_payload["runtime_seconds"] = elapsed
        result_payload["gpu_hours"] = elapsed / 3600
        result_payload["gpu_node_hours"] = elapsed / (3600 * 4)
        result_payload["finished_at_unix"] = time.time()
        result_payload["attempt"] = attempt_index
        result_path = attempt_dir / "result.json"
        _atomic_json(result_path, result_payload)
    return result_payload


def run_worker(
    tasks_path: Path, output_root: Path, *, rank: int, world_size: int
) -> dict[str, int]:
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("worker rank/world-size are invalid")
    _, checkpoint_sha = _verified_checkpoint()
    if checkpoint_sha != E3_MLIP_CHECKPOINT_SHA256:
        raise ValueError("worker checkpoint does not match the frozen E3-MLIP model")
    from matsim_agents.backends.mlip.relaxation import CalculatorCache

    tasks = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    calculators = CalculatorCache()
    counts: dict[str, int] = defaultdict(int)
    task_root = output_root / "tasks"
    for index, task in enumerate(tasks):
        if index % world_size != rank:
            continue
        if _task_result(task_root / task["candidate_id"], task) is not None:
            counts["resumed"] += 1
            continue
        result = _execute_task(task, task_root, calculators)
        counts[result["status"]] += 1
        counts["attempted"] += 1
    return dict(counts)


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _round_action_outcomes(tasks: list[dict[str, Any]], task_root: Path) -> list[dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for task in tasks:
        latest = _task_result(task_root / task["candidate_id"], task)
        if latest is None:
            raise RuntimeError(f"optimization result is missing for {task['candidate_id']}")
        result = json.loads(latest.read_text(encoding="utf-8"))
        action_id = task["action_id"]
        aggregate = aggregates.setdefault(
            action_id,
            {
                "action_id": action_id,
                "action": task["action"],
                "candidate_kind": task["candidate_kind"],
                "formula": task["formula"],
                "formula_units": task["formula_units"],
                "size_tier": task["action"]["size_tier"],
                "family": task["action"]["family"],
                "generator": task["action"]["generator"],
                "attempted": 0,
                "converged": 0,
                "nonconverged": 0,
                "failed": 0,
                "runtime_seconds": 0.0,
                "gpu_hours": 0.0,
                "energies_eV_per_atom": [],
                "final_structure_hashes": set(),
                "displacements_A": [],
                "transformed_structures": 0,
            },
        )
        aggregate["attempted"] += int(result["status"] in {"converged", "nonconverged", "error"})
        aggregate["converged"] += int(result["status"] == "converged")
        aggregate["nonconverged"] += int(result["status"] == "nonconverged")
        aggregate["failed"] += int(result["status"] == "error")
        aggregate["runtime_seconds"] += float(result.get("runtime_seconds", 0))
        aggregate["gpu_hours"] += float(result.get("gpu_hours", 0))
        if result.get("final_energy_eV") is not None and task.get("num_atoms"):
            aggregate["energies_eV_per_atom"].append(
                float(result["final_energy_eV"]) / int(task["num_atoms"])
            )
        if result.get("final_structure_sha256"):
            aggregate["final_structure_hashes"].add(result["final_structure_sha256"])
        if result.get("rms_atom_displacement_A") is not None:
            displacement = float(result["rms_atom_displacement_A"])
            aggregate["displacements_A"].append(displacement)
            aggregate["transformed_structures"] += int(displacement >= 0.1)
    for aggregate in aggregates.values():
        energies = aggregate.pop("energies_eV_per_atom")
        displacements = aggregate.pop("displacements_A")
        final_hashes = aggregate.pop("final_structure_hashes")
        aggregate["mean_energy_eV_per_atom"] = sum(energies) / len(energies) if energies else None
        aggregate["minimum_energy_eV_per_atom"] = min(energies) if energies else None
        aggregate["distinct_final_geometry_hashes"] = len(final_hashes)
        aggregate["mean_rms_atom_displacement_A"] = (
            sum(displacements) / len(displacements) if displacements else None
        )
        aggregate["evidence_id"] = hashlib.sha256(_canonical_json(aggregate).encode()).hexdigest()
    return list(aggregates.values())


def _distinct_symmetry_structures(
    tasks: list[dict[str, Any]],
    task_results: list[dict[str, Any]],
    *,
    relaxed: bool,
) -> dict[str, int]:
    from ase.io import read
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor

    matcher = StructureMatcher(
        primitive_cell=True,
        attempt_supercell=False,
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
    )
    representatives: dict[str, list[Any]] = defaultdict(list)
    for task, result in zip(tasks, task_results, strict=True):
        if task["candidate_kind"] != "unique_configuration" or (
            relaxed and result["status"] not in {"converged", "nonconverged"}
        ):
            continue
        structure_path = result["optimized_structure_path"] if relaxed else task["structure_path"]
        atoms = read(structure_path)
        structure = AseAtomsAdaptor.get_structure(atoms)
        formula = task["formula"]
        if not any(matcher.fit(existing, structure) for existing in representatives[formula]):
            representatives[formula].append(structure)
    return {formula: len(structures) for formula, structures in sorted(representatives.items())}


def _distinct_relaxed_minima(
    tasks: list[dict[str, Any]], task_results: list[dict[str, Any]]
) -> dict[str, int]:
    return _distinct_symmetry_structures(tasks, task_results, relaxed=True)


def cluster_campaign_minima(output_root: Path) -> dict[str, Any]:
    """Cluster relaxed minima across complete streams, outside active acquisition."""
    streams: dict[tuple[str, str, int], tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for tasks_path in sorted(output_root.glob("**/tasks.jsonl")):
        round_dir = tasks_path.parent
        tasks = _read_tasks(tasks_path)
        for task in tasks:
            result_path = _task_result(round_dir / "tasks" / task["candidate_id"], task)
            if result_path is None:
                raise RuntimeError(
                    f"cannot cluster an incomplete campaign; missing terminal result "
                    f"for {task['candidate_id']}"
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            key = (task["material"], task["strategy"], int(task["seed"]))
            stream_tasks, stream_results = streams.setdefault(key, ([], []))
            stream_tasks.append(task)
            stream_results.append(result)
    if not streams:
        raise ValueError(f"no production task manifests found under {output_root}")

    clustered = {}
    for (material, strategy, seed), (tasks, results) in sorted(streams.items()):
        controls = [task for task in tasks if task["candidate_kind"] == "finite_size_control"]
        clustered[f"{material}/{strategy}/seed-{seed}"] = {
            "task_count": len(tasks),
            "finite_size_controls": {
                "logical_relaxation_count": len(controls),
                "replicated_supercell_families": len(
                    {int(task["control_family_id"]) for task in controls}
                ),
            },
            "distinct_starting_structures_by_formula": _distinct_symmetry_structures(
                tasks, results, relaxed=False
            ),
            "distinct_relaxed_minima_by_formula": _distinct_relaxed_minima(tasks, results),
        }
    return {
        "campaign": "E3-MLIP",
        "clustering": "composition-aware StructureMatcher over all relaxed unique configurations",
        "streams": clustered,
        "stream_count": len(clustered),
    }


def _find_round(
    manifest: dict[str, Any], material: str, strategy: str, seed: int, round_index: int
) -> dict[str, Any]:
    matches = [
        row
        for row in manifest["rounds"]
        if row["material"] == material
        and row["strategy"] == strategy
        and int(row["seed"]) == seed
        and int(row["round"]) == round_index
    ]
    if len(matches) != 1:
        raise ValueError("round identity is missing or ambiguous in the production manifest")
    return matches[0]


def _execute_round_tasks(
    tasks_path: Path,
    round_dir: Path,
    worker_count: int,
    workers_per_node: int,
    worker_python: str,
) -> None:
    if worker_count < 1 or worker_count % workers_per_node:
        raise ValueError("worker count must be a positive multiple of workers per node")
    if worker_count == 1:
        run_worker(tasks_path, round_dir, rank=0, world_size=1)
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("multi-node workers must run inside an authorized Slurm allocation")
    command = [
        "srun",
        "--nodes",
        str(worker_count // workers_per_node),
        "--ntasks",
        str(worker_count),
        "--ntasks-per-node",
        str(workers_per_node),
        "--gpus-per-task",
        "1",
        "--cpu-bind",
        "cores",
        worker_python,
        "-m",
        "benchmarks.e3.production",
        "worker",
        "--tasks",
        str(tasks_path),
        "--output-root",
        str(round_dir),
    ]
    subprocess.run(command, check=True)


def run_replicate(
    manifest_path: Path,
    output_root: Path,
    material: str,
    strategy: str,
    seed: int,
    *,
    worker_count: int | None = None,
    worker_python: str | None = None,
    workers_per_node: int = 4,
) -> dict[str, Any]:
    """Run all ten adaptive rounds for one material/strategy/paired-seed stream."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_campaign_manifest(manifest)
    if strategy not in E3_STRATEGIES or material not in MATERIALS:
        raise ValueError("unknown E3 material or strategy")
    if seed not in REPLICATE_SEEDS:
        raise ValueError(f"seed {seed} is not in the frozen paired seed set")
    stream_dir = output_root / material / strategy / f"seed-{seed}"
    stream_dir.mkdir(parents=True, exist_ok=True)
    lock_path = stream_dir / ".run.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"replicate is already running: {stream_dir}") from exc
        if worker_count is None:
            worker_count = int(os.environ.get("MATSIM_E3_MLIP_WORKERS", "16"))
        worker_python = worker_python or os.environ.get(
            "MATSIM_E3_MLIP_WORKER_PYTHON", sys.executable
        )
        if worker_count > 1:
            _verified_checkpoint_sha256()
        chat = _chat_model() if strategy == "llm_guided_adaptation" else None
        for round_index in range(1, 11):
            spec = _find_round(manifest, material, strategy, seed, round_index)
            round_dir = stream_dir / f"round-{round_index:02d}"
            summary_path = round_dir / "round-summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("status") == "complete":
                    previous_manifest = round_dir / "round-manifest.json"
                    if not previous_manifest.is_file():
                        raise ValueError(f"completed round has no protocol manifest: {round_dir}")
                    previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
                    if previous.get("optimizer") != OPTIMIZER_SETTINGS:
                        raise ValueError("completed round uses a different optimization protocol")
                    continue
            history = _history_for_round(output_root, material, strategy, seed, round_index)
            round_manifest_path, tasks_path, round_manifest = _materialize_round(
                manifest, spec, output_root, history, chat
            )
            _execute_round_tasks(
                tasks_path,
                round_dir,
                worker_count,
                workers_per_node,
                worker_python,
            )
            tasks = _read_tasks(tasks_path)
            task_root = round_dir / "tasks"
            action_outcomes = _round_action_outcomes(tasks, task_root)
            tier_accounting: dict[str, dict[str, Any]] = {}
            for outcome in action_outcomes:
                tier = str(outcome["size_tier"])
                entry = tier_accounting.setdefault(
                    tier,
                    {
                        "attempted": 0,
                        "converged": 0,
                        "nonconverged": 0,
                        "failed": 0,
                        "runtime_seconds": 0.0,
                        "gpu_hours": 0.0,
                    },
                )
                for key in ("attempted", "converged", "nonconverged", "failed"):
                    entry[key] += int(outcome[key])
                for key in ("runtime_seconds", "gpu_hours"):
                    entry[key] += float(outcome[key])
            task_results = [
                json.loads(_task_result(task_root / task["candidate_id"], task).read_text())
                for task in tasks
            ]
            geometry_hashes_by_tier: dict[str, set[str]] = defaultdict(set)
            for task, result in zip(tasks, task_results, strict=True):
                final_hash = result.get("final_structure_sha256")
                if final_hash:
                    geometry_hashes_by_tier[str(task["action"]["size_tier"])].add(final_hash)
            for tier, entry in tier_accounting.items():
                entry["distinct_final_geometry_hashes"] = len(geometry_hashes_by_tier[tier])
            summary = {
                **spec,
                "status": "complete",
                "task_count": len(tasks),
                "unique_candidates_attempted": sum(
                    task["candidate_kind"] == "unique_configuration" for task in tasks
                ),
                "finite_size_controls_attempted": sum(
                    task["candidate_kind"] == "finite_size_control" for task in tasks
                ),
                "outcomes": dict(Counter(result["status"] for result in task_results)),
                "costs": {
                    "runtime_seconds": sum(
                        float(result.get("runtime_seconds", 0)) for result in task_results
                    ),
                    "gpu_hours": sum(float(result.get("gpu_hours", 0)) for result in task_results),
                    "gpu_node_hours": sum(
                        float(result.get("gpu_node_hours", 0)) for result in task_results
                    ),
                },
                "accounting_by_size_tier": tier_accounting,
                "distinct_relaxed_geometry_hashes": len(
                    {
                        result["final_structure_sha256"]
                        for result in task_results
                        if result.get("final_structure_sha256")
                    }
                ),
                "action_outcomes": action_outcomes,
                "selection": round_manifest["selection"],
                "tasks_sha256": _sha256_file(tasks_path),
            }
            _atomic_json(summary_path, summary)
            round_manifest["status"] = "complete"
            round_manifest["round_summary"] = str(summary_path)
            _atomic_json(round_manifest_path, round_manifest)
        return {
            "material": material,
            "strategy": strategy,
            "seed": seed,
            "status": "complete",
            "rounds": 10,
            "unique_target": 5000,
            "output_dir": str(stream_dir),
        }


def _worker_command(args: argparse.Namespace) -> int:
    rank = args.rank
    world_size = args.world_size
    if rank is None:
        rank = int(os.environ.get("SLURM_PROCID", "0"))
    if world_size is None:
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    result = run_worker(args.tasks, args.output_root, rank=rank, world_size=world_size)
    print(json.dumps(result, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="write the complete 360-round adaptive campaign manifest"
    )
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--overwrite", action="store_true")
    run = commands.add_parser("run-replicate")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--material", choices=sorted(MATERIALS), required=True)
    run.add_argument("--strategy", choices=E3_STRATEGIES, required=True)
    run.add_argument("--seed", type=int, choices=REPLICATE_SEEDS, required=True)
    run.add_argument("--workers", type=int)
    run.add_argument("--workers-per-node", type=int, default=4)
    run.add_argument("--worker-python")
    worker = commands.add_parser("worker")
    worker.add_argument("--tasks", type=Path, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--rank", type=int)
    worker.add_argument("--world-size", type=int)
    cluster = commands.add_parser(
        "cluster-minima", help="cluster relaxed minima after campaign acquisition is complete"
    )
    cluster.add_argument("--output-root", type=Path, required=True)
    cluster.add_argument("--output", type=Path, required=True)
    reserve = commands.add_parser(
        "budget-reserve", help="reserve Slurm GPU time against the campaign authorization"
    )
    reserve.add_argument("--state", type=Path, required=True)
    reserve.add_argument("--reservation-id", required=True)
    reserve.add_argument("--reserved-gpu-seconds", type=int, required=True)
    reserve.add_argument("--cap-gpu-seconds", type=int, default=GPU_HOUR_CAP * 3600)
    release = commands.add_parser(
        "budget-release", help="record actual GPU time and release unused reservation"
    )
    release.add_argument("--state", type=Path, required=True)
    release.add_argument("--reservation-id", required=True)
    release.add_argument("--actual-gpu-seconds", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        manifest = write_campaign_manifest(args.output, overwrite=args.overwrite)
        print(
            json.dumps(
                {
                    "path": str(args.output.resolve()),
                    "rounds": len(manifest["rounds"]),
                    "total_logical_optimizations": manifest["total_logical_optimizations"],
                    "job_submission_performed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "worker":
        return _worker_command(args)
    if args.command == "cluster-minima":
        result = cluster_campaign_minima(args.output_root)
        _atomic_json(args.output, result)
        print(json.dumps({"path": str(args.output.resolve()), "streams": result["stream_count"]}))
        return 0
    if args.command == "budget-reserve":
        result = reserve_gpu_hours(
            args.state,
            args.reservation_id,
            reserved_gpu_seconds=args.reserved_gpu_seconds,
            cap_gpu_seconds=args.cap_gpu_seconds,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "budget-release":
        result = release_gpu_hours(
            args.state,
            args.reservation_id,
            actual_gpu_seconds=args.actual_gpu_seconds,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    result = run_replicate(
        args.manifest,
        args.output_root,
        args.material,
        args.strategy,
        args.seed,
        worker_count=args.workers,
        worker_python=args.worker_python,
        workers_per_node=args.workers_per_node,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
