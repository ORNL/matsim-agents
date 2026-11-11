"""Freeze, validate and dispatch unauthorized E4 BCC scaling plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
PROTOCOL = "bcc-pbe-low-normal-v1-approved-ediffg-0.01"
QUEUE_WALLTIME_HOURS = 168
TASK_WALLTIME_HOURS = 24
STRONG_IDS = (
    "E4-BCC-0001",
    "E4-BCC-0561",
    "E4-BCC-1121",
    "E4-BCC-1201",
    "E4-BCC-1501",
    "E4-BCC-1701",
    "E4-BCC-1801",
    "E4-BCC-1805",
)
WEAK_PAIRS = (
    ("E4-BCC-0001", "E4-BCC-0002"),
    ("E4-BCC-0561", "E4-BCC-0562"),
    ("E4-BCC-1121", "E4-BCC-1122"),
    ("E4-BCC-1201", "E4-BCC-1202"),
    ("E4-BCC-1501", "E4-BCC-1502"),
    ("E4-BCC-1701", "E4-BCC-1702"),
    ("E4-BCC-1801", "E4-BCC-1802"),
    ("E4-BCC-1805", "E4-BCC-1806"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_converged_final_status(status: str | None) -> bool:
    return bool(re.fullmatch(r"converged:[A-Z]\d+", status or ""))


def task_record(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: case[key]
        for key in (
            "calculation_id",
            "case_dir",
            "source_poscar",
            "source_poscar_sha256",
            "composition",
            "structure_id",
            "protocol_id",
            "input_hashes",
            "prepared_input_bundle_sha256",
            "paw_components",
            "potcar_sha256",
            "mpi_ranks",
            "omp_threads_per_rank",
            "ediffg_eV_per_A",
        )
    }


def validate_task(task: dict[str, Any]) -> None:
    protocol_id = task.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError(f"{task.get('calculation_id')}: protocol_id is required")
    bcc_task = protocol_id == PROTOCOL
    ranks = task.get("mpi_ranks")
    threads = task.get("omp_threads_per_rank")
    if bcc_task and (ranks != 8 or threads != 8):
        raise ValueError(f"{task.get('calculation_id')}: unexpected MPI/OpenMP layout")
    if not isinstance(ranks, int) or ranks < 1 or not isinstance(threads, int) or threads < 1:
        raise ValueError(f"{task.get('calculation_id')}: invalid MPI/OpenMP layout")
    case_dir = (REPO / task["case_dir"]).resolve()
    source_poscar = (REPO / task["source_poscar"]).resolve()
    if not source_poscar.is_file() or sha256(source_poscar) != task["source_poscar_sha256"]:
        raise ValueError(f"{task['calculation_id']}: frozen POSCAR hash mismatch")
    composition = dict(
        zip(
            source_poscar.read_text().splitlines()[5].split(),
            map(int, source_poscar.read_text().splitlines()[6].split()),
            strict=True,
        )
    )
    atom_count = task.get("atom_count", sum(composition.values()))
    if composition != task["composition"] or sum(composition.values()) != atom_count:
        raise ValueError(f"{task['calculation_id']}: composition/atom count mismatch")
    if bcc_task and atom_count != 128:
        raise ValueError(f"{task['calculation_id']}: BCC scaling inputs must have 128 atoms")
    for name in ("START", "POSCAR", "KPOINTS", "POTCAR"):
        path = case_dir / name
        if not path.is_file() or sha256(path) != task["input_hashes"][name]:
            raise ValueError(f"{task['calculation_id']}: input hash mismatch: {name}")
    if sha256(case_dir / "POSCAR") != task["source_poscar_sha256"]:
        raise ValueError(f"{task['calculation_id']}: staged POSCAR differs from source")
    bundle = "".join(
        f"{name}:{task['input_hashes'][name]}\n"
        for name in ("START", "POSCAR", "KPOINTS", "POTCAR")
    )
    if hashlib.sha256(bundle.encode()).hexdigest() != task["prepared_input_bundle_sha256"]:
        raise ValueError(f"{task['calculation_id']}: input bundle digest mismatch")
    assembled = bytearray()
    for component in task["paw_components"]:
        path = (REPO / component["source"]).resolve()
        if not path.is_file() or sha256(path) != component["sha256"]:
            raise ValueError(f"{task['calculation_id']}: PAW hash mismatch: {component['label']}")
        assembled.extend(path.read_bytes())
    if hashlib.sha256(assembled).hexdigest() != task["potcar_sha256"]:
        raise ValueError(f"{task['calculation_id']}: assembled PAW identity mismatch")


def strong_tiers(ids: list[str]) -> list[dict[str, Any]]:
    tiers = []
    for nodes, hours in ((1, 96), (2, 96), (4, 48), (8, 24)):
        waves = math.ceil(len(ids) / nodes)
        tier = {
            "nodes": nodes,
            "tasks": len(ids),
            "waves": waves,
            "walltime_cap_hours": hours,
            "node_hours_cap": nodes * hours * (2 if nodes == 1 else 1),
            "queue_walltime_limit_hours": QUEUE_WALLTIME_HOURS,
            "status": "planning_only_dispatcher_ready_authorization_required",
        }
        if nodes == 1:
            tier["segments"] = [
                {
                    "segment_id": "segment-1",
                    "task_ids": ids[:4],
                    "walltime_cap_hours": 96,
                    "node_hours_cap": 96,
                },
                {
                    "segment_id": "segment-2",
                    "task_ids": ids[4:],
                    "walltime_cap_hours": 96,
                    "node_hours_cap": 96,
                },
            ]
            tier["walltime_cap_hours"] = 96
            tier["segments_per_tier"] = 2
        else:
            tier["segments"] = [{"segment_id": "all", "task_ids": ids}]
        tiers.append(tier)
    return tiers


def weak_tiers(pairs: list[list[str]]) -> list[dict[str, Any]]:
    return [
        {
            "nodes": nodes,
            "required_unique_tasks": 2 * nodes,
            "available_unique_tasks": 2 * nodes,
            "node_assignments": pairs[:nodes],
            "walltime_cap_hours": 48,
            "node_hours_cap": 48 * nodes,
            "status": "planning_only_dispatcher_ready_authorization_required",
        }
        for nodes in (1, 2, 4, 8)
    ]


def freeze_manifests(source_manifest: Path, weak_inputs_manifest: Path, output_dir: Path) -> None:
    source = json.loads(source_manifest.read_text())
    weak_inputs = json.loads(weak_inputs_manifest.read_text())
    source_sha = sha256(source_manifest)
    if weak_inputs.get("source_manifest_sha256") != source_sha:
        raise ValueError("weak inputs were not prepared from this source manifest")
    cases = {case["calculation_id"]: case for case in source["cases"]}
    cases.update({case["calculation_id"]: case for case in weak_inputs["cases"]})
    task_by_id = {calc_id: task_record(case) for calc_id, case in cases.items()}
    if not set(STRONG_IDS) <= task_by_id.keys():
        raise ValueError("fixed strong workload is incomplete")
    if not {calc_id for pair in WEAK_PAIRS for calc_id in pair} <= task_by_id.keys():
        raise ValueError("weak workload does not contain 16 unique frozen cases")
    all_tasks = [task_by_id[calc_id] for pair in WEAK_PAIRS for calc_id in pair]
    hashes = [task["source_poscar_sha256"] for task in all_tasks]
    if len(hashes) != len(set(hashes)):
        raise ValueError("weak workload contains duplicate POSCAR hashes")
    for task in all_tasks:
        validate_task(task)
    output_dir.mkdir(parents=True, exist_ok=True)
    shared = {
        "schema_version": 1,
        "queue": "capacity",
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": source_sha,
            "batch_authorization_applies_to_scaling": False,
        },
        "protocol_id": PROTOCOL,
        "task_layout": {
            "nodes_per_task": 1,
            "mpi_ranks_per_task": 8,
            "omp_threads_per_rank": 8,
            "task_walltime_cap_hours": TASK_WALLTIME_HOURS,
            "output_root": "runs/e4-scaling/<experiment>/<run-id>/<tier>/<segment>/<task-id>/",
            "runner": "deployments/aurora/jobs/job-e4-scaling.pbs",
        },
        "tasks": [task_by_id[calc_id] for calc_id in STRONG_IDS],
        "larger_tiers": {
            "nodes": [16, 32, 64, 128],
            "status": "planned_separate_authorization_required",
        },
        "execution_authorized": False,
        "authorization": None,
    }
    strong = {
        **shared,
        "campaign_id": "E4-Aurora-BCC-Strong-v1",
        "experiment": "strong",
        "run_id_prefix": "E4-SCALE-STRONG-v1",
        "fixed_workload": {
            "required_unique_tasks": 8,
            "unique_key": "POSCAR SHA-256",
            "case_poscar_sha256": {
                t["calculation_id"]: t["source_poscar_sha256"] for t in shared["tasks"]
            },
        },
        "tiers": strong_tiers(list(STRONG_IDS)),
        "aggregate_node_hours_cap": 768,
    }
    weak = {
        **shared,
        "campaign_id": "E4-Aurora-BCC-Weak-v1",
        "experiment": "weak",
        "run_id_prefix": "E4-SCALE-WEAK-v1",
        "tasks": [task_by_id[calc_id] for pair in WEAK_PAIRS for calc_id in pair],
        "available_unique_case_ids": [calc_id for pair in WEAK_PAIRS for calc_id in pair],
        "additional_case_ids": [pair[1] for pair in WEAK_PAIRS],
        "additional_input_manifest": str(weak_inputs_manifest),
        "tiers": weak_tiers([list(pair) for pair in WEAK_PAIRS]),
        "aggregate_node_hours_cap": 720,
    }
    for manifest, name in ((strong, "strong-manifest.json"), (weak, "weak-manifest.json")):
        manifest["resource_request"] = {
            "maximum_nodes_per_job": 8,
            "maximum_walltime_hours_per_job": 96 if manifest["experiment"] == "strong" else 48,
            "total_node_hours_cap": manifest["aggregate_node_hours_cap"],
            "submission": "not authorized; qsub must not be run",
        }
        (output_dir / name).write_text(json.dumps(manifest, indent=2) + "\n")


def validate_manifest(path: Path) -> dict[str, Any]:
    return validate_manifest_data(json.loads(path.read_text()))


def validate_manifest_data(manifest: dict[str, Any]) -> dict[str, Any]:
    authorized = manifest.get("execution_authorized")
    authorization = manifest.get("authorization")
    if not isinstance(authorized, bool):
        raise ValueError("execution_authorized must be explicitly true or false")
    if authorized and not (isinstance(authorization, dict) and authorization.get("id")):
        raise ValueError("authorized manifest must carry a separate authorization ID")
    if not authorized and authorization is not None:
        raise ValueError("unauthorized manifest must not carry an authorization record")
    task_ids = [task["calculation_id"] for task in manifest["tasks"]]
    hashes = [task["source_poscar_sha256"] for task in manifest["tasks"]]
    if len(task_ids) != len(set(task_ids)) or len(hashes) != len(set(hashes)):
        raise ValueError("task IDs and POSCAR hashes must be unique")
    protocol_id = manifest.get("protocol_id")
    if any(task.get("protocol_id") != protocol_id for task in manifest["tasks"]):
        raise ValueError("all scaling tasks must use the manifest's frozen protocol")
    for task in manifest["tasks"]:
        validate_task(task)
    fixed_task_ids = manifest.get("fixed_task_ids", STRONG_IDS)
    for tier in manifest["tiers"]:
        if tier["walltime_cap_hours"] > QUEUE_WALLTIME_HOURS:
            raise ValueError("tier exceeds the capacity queue walltime")
        if manifest["experiment"] == "weak":
            assigned = [task_id for node in tier["node_assignments"] for task_id in node]
            if len(assigned) != tier["required_unique_tasks"] or len(set(assigned)) != len(
                assigned
            ):
                raise ValueError("weak tier task assignment is incomplete or duplicated")
        else:
            assigned = [task_id for segment in tier["segments"] for task_id in segment["task_ids"]]
            if sorted(assigned) != sorted(fixed_task_ids) or len(assigned) != len(set(assigned)):
                raise ValueError("strong tier does not dispatch the fixed workload exactly once")
            if any(
                segment.get("walltime_cap_hours", 0) > QUEUE_WALLTIME_HOURS
                for segment in tier["segments"]
                if len(tier["segments"]) > 1
            ):
                raise ValueError("strong segment exceeds the queue walltime")
    return manifest


def dispatch_plan(
    manifest: dict[str, Any], tier_nodes: int, segment_id: str | None
) -> list[list[str]]:
    tier = next((item for item in manifest["tiers"] if item["nodes"] == tier_nodes), None)
    if tier is None:
        raise ValueError(f"no {tier_nodes}-node tier")
    if manifest["experiment"] == "weak":
        assignments = tier["node_assignments"]
        return [
            [node[index] for node in assignments if index < len(node)]
            for index in range(max(map(len, assignments)))
        ]
    segments = tier["segments"]
    if len(segments) > 1:
        selected = next(
            (segment for segment in segments if segment["segment_id"] == segment_id), None
        )
        if selected is None:
            raise ValueError("the one-node strong tier requires segment-1 or segment-2")
    else:
        if segment_id not in (None, "all"):
            raise ValueError("this tier uses segment-id=all")
        selected = segments[0]
    ids = selected["task_ids"]
    return [ids[index : index + tier_nodes] for index in range(0, len(ids), tier_nodes)]


def dispatch(
    manifest_path: Path,
    nodes: int,
    segment_id: str | None,
    dry_run: bool,
    *,
    manifest_override: dict[str, Any] | None = None,
) -> int:
    allocation_started = time.monotonic()
    manifest = (
        validate_manifest(manifest_path)
        if manifest_override is None
        else validate_manifest_data(manifest_override)
    )
    waves = dispatch_plan(manifest, nodes, segment_id)
    if dry_run:
        print(
            json.dumps(
                {
                    "experiment": manifest["experiment"],
                    "material_class": manifest.get("material_class"),
                    "nodes": nodes,
                    "segment_id": segment_id,
                    "waves": waves,
                    "resource_request": manifest.get("resource_request"),
                    "execution_authorized": False,
                },
                indent=2,
            )
        )
        return 0
    authorization = manifest.get("authorization") or {}
    if manifest["execution_authorized"] is not True or os.environ.get(
        "E4_SCALING_AUTHORIZATION"
    ) != authorization.get("id"):
        raise PermissionError("separate E4 scaling authorization is required")
    if not os.environ.get("PBS_JOBID") or not os.environ.get("PBS_NODEFILE"):
        raise RuntimeError("dispatch must run inside its PBS allocation")
    hosts = list(dict.fromkeys(Path(os.environ["PBS_NODEFILE"]).read_text().split()))
    if len(hosts) != nodes:
        raise RuntimeError(f"PBS allocation has {len(hosts)} unique nodes; tier requires {nodes}")
    run_id = os.environ.get("E4_SCALING_RUN_ID", "")
    if not re.fullmatch(re.escape(manifest["run_id_prefix"]) + r"-[A-Za-z0-9._-]+", run_id):
        raise ValueError("E4_SCALING_RUN_ID must use the frozen campaign prefix")
    tier_name = f"n{nodes:02d}"
    segment_name = segment_id or "all"
    root = REPO / "runs" / "e4-scaling" / manifest["experiment"] / run_id / tier_name / segment_name
    root.mkdir(parents=True, exist_ok=False)
    task_map = {task["calculation_id"]: task for task in manifest["tasks"]}
    log_path = root / "dispatch.jsonl"
    wrapper = REPO / "deployments/aurora/jobs/job-e4-vasp-ab-aurora.pbs"
    failed = False
    all_results: list[dict[str, Any]] = []

    def run_task(calc_id: str, host: str) -> dict[str, Any]:
        task = task_map[calc_id]
        task_root = root / calc_id
        launch = task.get("launch_settings", {})
        task_walltime_hours = float(task.get("estimated_task_walltime_hours", TASK_WALLTIME_HOURS))
        task_walltime_seconds = int(task_walltime_hours * 3600)
        stage_mode = launch.get("stage_mode", "low-normal")
        stage_nsw_max = int(launch.get("stage_nsw_max", 240 if stage_mode == "low-normal" else 200))
        env = os.environ.copy()
        env.update(
            {
                "REPO": str(REPO),
                "CASE_DIR": str((REPO / task["case_dir"]).resolve()),
                "VARIANT": "alcf",
                "REPEAT": "repeat-01",
                "E4_CALCULATION_ID": calc_id,
                "E4_STAGE_MODE": stage_mode,
                "E4_STAGE_NSW_MAX": str(stage_nsw_max),
                "E4_NSW": str(int(launch.get("nsw_per_checkpoint_segment", 30))),
                "E4_EDIFFG": str(task.get("ediffg_eV_per_A", task.get("ediffg_eV_A", "-0.01"))),
                "E4_MPI_RANKS": str(task["mpi_ranks"]),
                "E4_OMP_THREADS": str(task["omp_threads_per_rank"]),
                "E4_REQUESTED_NODES": "1",
                "E4_ACCOUNTING_NODES": "1",
                "E4_REQUESTED_WALLTIME_SECONDS": str(task_walltime_seconds),
                "E4_SOFT_STOP_SECONDS": str(
                    max(60, task_walltime_seconds - min(3600, task_walltime_seconds // 10))
                ),
                "E4_CHECKPOINT_SECONDS": str(int(launch.get("checkpoint_seconds", 3600))),
                "E4_OUTPUT_ROOT": str(task_root),
                "MATSIM_ASSIGNED_NODES": host,
            }
        )
        if stage_mode == "single":
            env["E4_SINGLE_PRECISION"] = str(launch.get("single_precision", "Accurate"))
            env["E4_SINGLE_PREFIX"] = str(launch.get("single_prefix", "A"))
        started = datetime.now(UTC).isoformat()
        start_monotonic = time.monotonic()
        with (
            (root / f"{calc_id}.stdout").open("w") as stdout,
            (root / f"{calc_id}.stderr").open("w") as stderr,
        ):
            process = subprocess.run(["bash", str(wrapper)], env=env, stdout=stdout, stderr=stderr)
        ended = datetime.now(UTC).isoformat()
        execution_seconds = time.monotonic() - start_monotonic
        final_status = None
        stage_records = []
        run_record = task_root / "run-record.log"
        if run_record.is_file():
            run_text = run_record.read_text(errors="replace")
            exits = re.findall(
                r"\bEXIT\s+job=\S+\s+calculation_id=\S+\s+status=(-?\d+)\s+final=(\S+)",
                run_text,
            )
            if exits:
                final_status = exits[-1][1]
            stage_records = re.findall(
                r"\bend\s+tag=\S+\s+rc=(-?\d+)\s+ionic_steps=(\d+)\s+"
                r"max_force_eV_per_A=([^\s]+)\s+seconds=(\d+)\s+status=(\S+)",
                run_text,
            )
        converged = process.returncode == 0 and is_converged_final_status(final_status)
        return {
            "event": "finished",
            "time": ended,
            "started_at": started,
            "calculation_id": calc_id,
            "node": host,
            "wrapper_exit_status": process.returncode,
            "final_status": final_status,
            "task_status": "converged" if converged else "failed_or_incomplete",
            "execution_seconds": round(execution_seconds, 3),
            "total_ionic_steps": sum(int(record[1]) for record in stage_records),
            "final_max_force_eV_per_A": float(stage_records[-1][2]) if stage_records else None,
            "stage_statuses": [record[4] for record in stage_records],
        }

    dispatch_started = time.monotonic()
    for wave_index, wave in enumerate(waves):
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = {}
            for node_index, calc_id in enumerate(wave):
                task_root = root / calc_id
                if task_root.exists():
                    raise FileExistsError(f"refusing to reuse task output root: {task_root}")
                task_root.parent.mkdir(parents=True, exist_ok=True)
                host = hosts[node_index]
                with log_path.open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "event": "started",
                                "time": datetime.now(UTC).isoformat(),
                                "calculation_id": calc_id,
                                "node": host,
                                "wave": wave_index + 1,
                            }
                        )
                        + "\n"
                    )
                futures[executor.submit(run_task, calc_id, host)] = calc_id
            for future in as_completed(futures):
                result = future.result()
                result["wave"] = wave_index + 1
                all_results.append(result)
                failed = failed or result["task_status"] != "converged"
                with log_path.open("a") as stream:
                    stream.write(json.dumps(result) + "\n")
        if failed:
            break
        print(f"wave {wave_index + 1}/{len(waves)} completed in {time.monotonic() - started:.1f}s")
    application_seconds = round(time.monotonic() - dispatch_started, 3)
    tier = next(item for item in manifest["tiers"] if item["nodes"] == nodes)
    if manifest["experiment"] == "strong":
        planned_ids = list(manifest.get("fixed_task_ids", STRONG_IDS))
        segment_count = len(tier["segments"])
    else:
        planned_ids = [task_id for node in tier["node_assignments"] for task_id in node]
        segment_count = 1
    workload_hash = hashlib.sha256(
        "\n".join(
            f"{calc_id}:{task_map[calc_id]['source_poscar_sha256']}"
            for calc_id in sorted(planned_ids)
        ).encode()
    ).hexdigest()
    protocol_hash = hashlib.sha256(
        json.dumps(
            {
                "protocol_id": manifest["protocol_id"],
                "tasks": [task_map[i].get("launch_settings", {}) for i in planned_ids],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    resource_hash = hashlib.sha256(
        json.dumps(
            {
                "nodes": nodes,
                "tasks": [
                    {
                        "id": calc_id,
                        "mpi_ranks": task_map[calc_id]["mpi_ranks"],
                        "threads": task_map[calc_id]["omp_threads_per_rank"],
                    }
                    for calc_id in planned_ids
                ],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()
    valid_ids = [
        result["calculation_id"] for result in all_results if result["task_status"] == "converged"
    ]
    completed = set(valid_ids)
    failed_ids = [
        result["calculation_id"] for result in all_results if result["task_status"] != "converged"
    ]
    task_seconds = [result["execution_seconds"] for result in all_results]
    measurement = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment": manifest["experiment"],
        "material_class": manifest.get("material_class", "bcc_refractory"),
        "facility": "aurora",
        "nodes": nodes,
        "repetition": int(os.environ.get("E4_SCALING_REPETITION", "1")),
        "segment_id": segment_id or "all",
        "segment_count": segment_count,
        "segment_group_id": f"{run_id}-{tier_name}" if segment_count > 1 else None,
        "workload_hash": workload_hash,
        "planned_calculation_ids": planned_ids,
        "completed_calculation_ids": sorted(completed),
        "failed_calculation_ids": failed_ids,
        "planned_tasks": len(planned_ids),
        "attempted_tasks": len(all_results),
        "valid_tasks": len(valid_ids),
        "failed_tasks": len(failed_ids),
        "total_ionic_steps": sum(result["total_ionic_steps"] for result in all_results),
        "task_execution_seconds": task_seconds,
        "application_makespan_seconds": application_seconds,
        "allocation_seconds": round(time.monotonic() - allocation_started, 3),
        "allocated_node_hours": round(nodes * (time.monotonic() - allocation_started) / 3600, 6),
        "source_commit": source_commit or None,
        "science_config_sha256": protocol_hash,
        "resource_config_sha256": resource_hash,
    }
    with (root / "measurements.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(measurement, sort_keys=True) + "\n")
    return 1 if failed else 0


def dispatch_six_class(
    manifest_path: Path,
    material_class: str,
    experiment: str,
    nodes: int,
    segment_id: str | None,
    dry_run: bool,
) -> int:
    from benchmarks.e4.aurora_dft.planning import validate_six_class_workload_manifest

    manifest = json.loads(manifest_path.read_text())
    validate_six_class_workload_manifest(manifest)
    class_record = next(
        (item for item in manifest["material_classes"] if item["material_class"] == material_class),
        None,
    )
    if class_record is None:
        raise ValueError(f"unknown E4 material class: {material_class}")
    section = "strong_scaling" if experiment == "strong" else "weak_scaling"
    plans = class_record[section]["tiers"]
    tier = next((item for item in plans if item["nodes"] == nodes), None)
    if tier is None:
        raise ValueError(f"no {nodes}-node {experiment} plan for {material_class}")
    if not tier.get("inputs_ready", False):
        print(
            json.dumps(
                {
                    "material_class": material_class,
                    "experiment": experiment,
                    "nodes": nodes,
                    "status": tier["status"],
                    "queue_eligible": tier.get("queue_eligible"),
                    "required_unique_calculations": tier.get(
                        "required_unique_calculations", tier.get("required_unique_calculations")
                    ),
                    "available_unique_calculations": tier.get("available_unique_calculations"),
                    "walltime_cap_hours": tier.get("walltime_cap_hours"),
                    "node_hours_cap": tier.get("node_hours_cap"),
                    "planning_cap_if_inputs_complete_node_hours": tier.get(
                        "planning_cap_if_inputs_complete_node_hours"
                    ),
                    "execution_authorized": False,
                },
                indent=2,
            )
        )
        if dry_run:
            return 0
        raise PermissionError("selected material/tier is blocked and not executable")

    selected_ids = tier["calculation_ids"]
    task_map = {
        candidate["calculation_id"]: {
            **candidate,
            "input_hashes": candidate["input_file_hashes"],
            "omp_threads_per_rank": candidate["omp_threads_per_rank"],
        }
        for candidate in class_record["candidates"]
    }
    tasks = [task_map[calc_id] for calc_id in selected_ids]
    if experiment == "weak":
        node_assignments = [
            selected_ids[index : index + 2] for index in range(0, len(selected_ids), 2)
        ]
        tiers = [
            {
                "nodes": nodes,
                "node_assignments": node_assignments,
                "required_unique_tasks": len(selected_ids),
                "walltime_cap_hours": tier["walltime_cap_hours"],
            }
        ]
        fixed_ids = None
    else:
        segments = tier.get("segments")
        if segments:
            segments = [
                {
                    "segment_id": f"segment-{segment['segment']}",
                    "task_ids": segment["calculation_ids"],
                    "walltime_cap_hours": segment["walltime_cap_hours"],
                }
                for segment in segments
            ]
        else:
            segments = [
                {
                    "segment_id": "all",
                    "task_ids": selected_ids,
                    "walltime_cap_hours": tier["walltime_cap_hours"],
                }
            ]
        tiers = [
            {
                "nodes": nodes,
                "segments": segments,
                "walltime_cap_hours": tier["walltime_cap_hours"],
            }
        ]
        fixed_ids = class_record["strong_scaling"]["fixed_calculation_ids"]
    auth = class_record.get("authorization")
    normalized = {
        "schema_version": 1,
        "campaign_id": f"E4-Aurora-{material_class}-{experiment}-v1",
        "experiment": experiment,
        "material_class": material_class,
        "run_id_prefix": f"E4-SIX-{material_class}-{experiment}-v1",
        "protocol_id": class_record["protocol"]["id"],
        "queue": "capacity",
        "execution_authorized": manifest["execution_authorized"]
        and class_record["execution_authorized"],
        "authorization": auth,
        "resource_request": {
            "maximum_nodes_per_job": nodes,
            "maximum_walltime_hours_per_job": tier["walltime_cap_hours"],
            "total_node_hours_cap": tier["node_hours_cap"],
        },
        "fixed_task_ids": fixed_ids,
        "tasks": tasks,
        "tiers": tiers,
    }
    return dispatch(
        manifest_path,
        nodes,
        segment_id,
        dry_run,
        manifest_override=normalized,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_six = sub.add_parser("plan-six-class")
    plan_six.add_argument(
        "--output",
        type=Path,
        default=REPO / "research/paper/e4/aurora-dft-scaling/six-class-workflow-v3.json",
    )
    validate_six = sub.add_parser("validate-six-class")
    validate_six.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=REPO / "research/paper/e4/aurora-dft-scaling/six-class-workflow-v3.json",
    )
    dispatch_six = sub.add_parser("dispatch-six-class")
    dispatch_six.add_argument("--manifest", type=Path, required=True)
    dispatch_six.add_argument("--material-class", required=True)
    dispatch_six.add_argument("--experiment", choices=("strong", "weak"), required=True)
    dispatch_six.add_argument("--nodes", type=int, required=True)
    dispatch_six.add_argument("--segment-id")
    dispatch_six.add_argument("--dry-run", action="store_true")
    freeze = sub.add_parser("freeze-manifests")
    freeze.add_argument("--source-manifest", type=Path, required=True)
    freeze.add_argument("--weak-inputs-manifest", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    run = sub.add_parser("dispatch")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--nodes", type=int, required=True)
    run.add_argument("--segment-id")
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan-six-class":
        from benchmarks.e4.aurora_dft.planning import write_six_class_workload_manifest

        manifest = write_six_class_workload_manifest(args.output)
        print(
            f"wrote six-class plan {args.output}; execution_authorized=false; "
            f"scheduler_submissions={manifest['semantics']['scheduler_submissions']}"
        )
        return 0
    if args.command == "validate-six-class":
        from benchmarks.e4.aurora_dft.planning import (
            validate_six_class_workload_manifest,
        )

        manifest = json.loads(args.manifest.read_text())
        validate_six_class_workload_manifest(manifest)
        print(
            f"validated {len(manifest['material_classes'])} material classes; "
            "execution_authorized=false"
        )
        return 0
    if args.command == "dispatch-six-class":
        return dispatch_six_class(
            args.manifest,
            args.material_class,
            args.experiment,
            args.nodes,
            args.segment_id,
            args.dry_run,
        )
    if args.command == "freeze-manifests":
        freeze_manifests(args.source_manifest, args.weak_inputs_manifest, args.output_dir)
        return 0
    if args.command == "validate":
        manifest = validate_manifest(args.manifest)
        print(
            f"validated {manifest['experiment']} manifest; "
            f"execution_authorized={manifest['execution_authorized']}"
        )
        return 0
    return dispatch(args.manifest, args.nodes, args.segment_id, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
