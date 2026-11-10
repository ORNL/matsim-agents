"""Dry-run planning and analysis helpers for the E4 campaign."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from matsim_agents.execution.allocation import Allocation
from matsim_agents.execution.placement import ResourcePlan
from matsim_agents.execution.presets import get_preset

ROOT = Path(__file__).resolve().parents[2]
FACILITIES = {
    "aurora": ("pbs", "aurora"),
    "frontier": ("slurm", "frontier"),
    "perlmutter": ("slurm", "perlmutter-gpu"),
}


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("E4 config must be an object")
    return value


def resource_counts(
    nodes: int, config: dict[str, Any], *, consistency: bool = False
) -> dict[str, int]:
    if nodes < 3:
        raise ValueError("complete E4 placement needs at least one node per service pool")
    if consistency:
        counts = {"llm": 1, "mlip": 1, "dft": nodes - 2}
    else:
        llm = max(1, math.ceil(nodes / 32))
        mlip = max(1, math.ceil(nodes / 16))
        counts = {"llm": llm, "mlip": mlip, "dft": nodes - llm - mlip}
    if counts["dft"] < 1 or sum(counts.values()) != nodes:
        raise ValueError(f"resource profile does not fit {nodes} nodes: {counts}")
    return counts


def resolve_resource_profile(
    facility: str, nodes: int, config: dict[str, Any], *, consistency: bool = False
) -> dict[str, Any]:
    if facility not in FACILITIES:
        raise ValueError(f"unsupported facility: {facility}")
    scheduler, preset = FACILITIES[facility]
    geometry = get_preset(preset).geometry
    counts = resource_counts(nodes, config, consistency=consistency)
    llm_servers = [
        {
            "name": f"e4-llm-{i:02d}",
            "kind": "managed-vllm",
            "model": "E4_MODEL_PENDING",
            "nodes": 1,
            "devices": "all",
            "tensor_parallel_size": geometry.devices_per_node,
        }
        for i in range(counts["llm"])
    ]
    resources = {
        "preset": preset,
        "pools": {
            "llm": {"nodes": counts["llm"], "devices": "all", "servers": llm_servers},
            "mlip": {"nodes": counts["mlip"], "devices": "all", "max_concurrent": counts["mlip"]},
            "dft": {
                "nodes": "rest",
                "devices": "all",
                "nodes_per_job": 1,
                "ranks_per_device": 1,
                "threads_per_rank": max(1, geometry.cores_per_node // geometry.devices_per_node),
                "max_concurrent": counts["dft"],
            },
        },
    }
    plan = ResourcePlan.model_validate(resources)
    allocation = Allocation(
        scheduler=scheduler, nodes=tuple(f"e4-{facility}-{i:04d}" for i in range(nodes))
    )
    placement = plan.resolve(allocation)
    if placement.idle or any(pool.device_count == 0 for pool in placement.pools.values()):
        raise ValueError(f"resource placement leaves an empty/idle pool: {placement.warnings}")
    return {
        "counts": counts,
        "logical_devices_per_node": geometry.devices_per_node,
        "physical_gpu_packages_per_node": geometry.devices_per_node / geometry.devices_per_gpu,
        "device_unit": geometry.device_kind,
        "logical_devices_total": nodes * geometry.devices_per_node,
        "physical_gpu_packages_total": nodes * geometry.devices_per_node / geometry.devices_per_gpu,
        "idle_devices": 0,
        "placement": placement.model_dump(mode="json"),
    }


def build_run_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scaling = config["scaling"]
    for experiment in ("strong", "weak"):
        for nodes in scaling["node_points"]:
            profile = resolve_resource_profile("aurora", nodes, config)
            for repetition in range(1, scaling["repetitions_per_point"] + 1):
                rows.append(
                    {
                        "run_id": f"e4-{experiment}-n{nodes}-r{repetition:02d}",
                        "experiment": experiment,
                        "facility": "aurora",
                        "nodes": nodes,
                        "repetition": repetition,
                        "workload_tasks": scaling["strong_workload_task_count"]
                        if experiment == "strong"
                        else None,
                        "tasks_per_node": scaling["weak_workload_tasks_per_node"]
                        if experiment == "weak"
                        else None,
                        "llm_nodes": profile["counts"]["llm"],
                        "mlip_nodes": profile["counts"]["mlip"],
                        "dft_nodes": profile["counts"]["dft"],
                        "device_unit": profile["device_unit"],
                        "logical_devices_total": profile["logical_devices_total"],
                        "physical_gpu_packages_total": profile["physical_gpu_packages_total"],
                        "resource_placement_json": json.dumps(profile["placement"], sort_keys=True),
                        "authorization": "not_authorized",
                        "status": "blocked_pending_gate_2_and_workload_sizing",
                    }
                )
    consistency = config["consistency"]
    for facility in consistency["facilities"]:
        nodes = consistency["proposed_nodes_per_facility"]
        profile = resolve_resource_profile(facility, nodes, config, consistency=True)
        for repetition in range(1, consistency["repetitions_per_facility"] + 1):
            rows.append(
                {
                    "run_id": f"e4-consistency-{facility}-n{nodes}-r{repetition:02d}",
                    "experiment": "consistency",
                    "facility": facility,
                    "nodes": nodes,
                    "repetition": repetition,
                    "workload_tasks": 1,
                    "tasks_per_node": None,
                    "llm_nodes": profile["counts"]["llm"],
                    "mlip_nodes": profile["counts"]["mlip"],
                    "dft_nodes": profile["counts"]["dft"],
                    "device_unit": profile["device_unit"],
                    "logical_devices_total": profile["logical_devices_total"],
                    "physical_gpu_packages_total": profile["physical_gpu_packages_total"],
                    "resource_placement_json": json.dumps(profile["placement"], sort_keys=True),
                    "authorization": "not_authorized",
                    "status": "blocked_pending_gate_1_and_gate_3",
                }
            )
    ids = [row["run_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("E4 run IDs are not unique")
    return rows


def validate_measurement(record: dict[str, Any]) -> list[str]:
    required = {
        "run_id",
        "experiment",
        "facility",
        "nodes",
        "repetition",
        "workload_hash",
        "attempted_tasks",
        "valid_tasks",
        "failed_tasks",
        "application_makespan_seconds",
        "allocation_seconds",
        "source_commit",
        "science_config_sha256",
        "resource_config_sha256",
    }
    missing = required - set(record)
    errors = [f"missing field {field}" for field in sorted(missing)]
    if missing:
        return errors
    for field in ("nodes", "repetition", "attempted_tasks", "valid_tasks", "failed_tasks"):
        if not isinstance(record[field], int) or record[field] < 0:
            errors.append(f"{field} must be a nonnegative integer")
    for field in ("application_makespan_seconds", "allocation_seconds"):
        if (
            not isinstance(record[field], (int, float))
            or not math.isfinite(record[field])
            or record[field] <= 0
        ):
            errors.append(f"{field} must be finite and positive")
    if record["valid_tasks"] + record["failed_tasks"] > record["attempted_tasks"]:
        errors.append("valid_tasks + failed_tasks exceeds attempted_tasks")
    if record["valid_tasks"] > record["attempted_tasks"]:
        errors.append("valid_tasks exceeds attempted_tasks")
    return errors


def analyze_scaling(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, record in enumerate(records):
        errors = validate_measurement(record)
        if errors:
            raise ValueError(f"measurement {index}: {'; '.join(errors)}")
    results: list[dict[str, Any]] = []
    for experiment in ("strong", "weak"):
        subset = [r for r in records if r["experiment"] == experiment]
        if not subset:
            continue
        if experiment == "strong":
            if len({r["workload_hash"] for r in subset}) != 1:
                raise ValueError("strong-scaling workload hashes differ")
            if len({r["attempted_tasks"] for r in subset}) != 1:
                raise ValueError("strong-scaling task counts differ")
        else:
            per_node = {r["attempted_tasks"] / r["nodes"] for r in subset}
            if len(per_node) != 1:
                raise ValueError("weak-scaling attempted tasks per node are not constant")
        baseline = None
        if experiment == "strong":
            baseline_rows = [
                r for r in subset if r["nodes"] == 8 and r["valid_tasks"] == r["attempted_tasks"]
            ]
            if baseline_rows:
                baseline = statistics.median(
                    r["application_makespan_seconds"] for r in baseline_rows
                )
        else:
            baseline_rows = [
                r for r in subset if r["nodes"] == 8 and r["application_makespan_seconds"] > 0
            ]
            if baseline_rows:
                baseline = statistics.median(
                    r["valid_tasks"] / r["application_makespan_seconds"] / r["nodes"]
                    for r in baseline_rows
                )
        for nodes in sorted({r["nodes"] for r in subset}):
            point = [r for r in subset if r["nodes"] == nodes]
            valid = [
                r
                for r in point
                if r["valid_tasks"] == r["attempted_tasks"] and r["failed_tasks"] == 0
            ]
            times = [r["application_makespan_seconds"] for r in valid]
            median_time = statistics.median(times) if times else None
            throughput = (
                statistics.median(
                    r["valid_tasks"] / r["application_makespan_seconds"] for r in point
                )
                if point
                else None
            )
            speedup = (
                baseline / median_time
                if experiment == "strong" and baseline and median_time
                else None
            )
            efficiency = speedup / (nodes / 8) if speedup is not None else None
            weak_efficiency = (
                (throughput / nodes) / baseline if experiment == "weak" and baseline else None
            )
            allocation_node_hours = sum(r["nodes"] * r["allocation_seconds"] / 3600 for r in point)
            results.append(
                {
                    "experiment": experiment,
                    "nodes": nodes,
                    "repetitions": len(point),
                    "fully_valid_repetitions": len(valid),
                    "median_makespan_seconds": median_time,
                    "median_valid_task_throughput_per_second": throughput,
                    "speedup": speedup,
                    "efficiency": efficiency,
                    "weak_efficiency": weak_efficiency,
                    "attempted_tasks": sum(r["attempted_tasks"] for r in point),
                    "valid_tasks": sum(r["valid_tasks"] for r in point),
                    "failed_tasks": sum(r["failed_tasks"] for r in point),
                    "allocated_node_hours": allocation_node_hours,
                }
            )
    return results
