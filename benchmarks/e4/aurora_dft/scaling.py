"""Freeze, validate and dispatch unauthorized E4 BCC scaling plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
PROTOCOL = "bcc-pbe-low-normal-v1-approved-ediffg-0.01"
QUEUE_WALLTIME_HOURS = 168
TASK_WALLTIME_HOURS = 24
STRONG_IDS = (
    "E4-BCC-0001", "E4-BCC-0561", "E4-BCC-1121", "E4-BCC-1201",
    "E4-BCC-1501", "E4-BCC-1701", "E4-BCC-1801", "E4-BCC-1805",
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


def task_record(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: case[key]
        for key in (
            "calculation_id", "case_dir", "source_poscar", "source_poscar_sha256",
            "composition", "structure_id", "protocol_id", "input_hashes",
            "prepared_input_bundle_sha256", "paw_components", "potcar_sha256",
            "mpi_ranks", "omp_threads_per_rank", "ediffg_eV_per_A",
        )
    }


def validate_task(task: dict[str, Any]) -> None:
    if task.get("protocol_id") != PROTOCOL:
        raise ValueError(f"{task.get('calculation_id')}: unexpected protocol")
    if task.get("mpi_ranks") != 8 or task.get("omp_threads_per_rank") != 8:
        raise ValueError(f"{task.get('calculation_id')}: unexpected MPI/OpenMP layout")
    case_dir = (REPO / task["case_dir"]).resolve()
    source_poscar = (REPO / task["source_poscar"]).resolve()
    if not source_poscar.is_file() or sha256(source_poscar) != task["source_poscar_sha256"]:
        raise ValueError(f"{task['calculation_id']}: frozen POSCAR hash mismatch")
    composition = dict(zip(source_poscar.read_text().splitlines()[5].split(),
                           map(int, source_poscar.read_text().splitlines()[6].split()), strict=True))
    if composition != task["composition"] or sum(composition.values()) != 128:
        raise ValueError(f"{task['calculation_id']}: composition/atom count mismatch")
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
                {"segment_id": "segment-1", "task_ids": ids[:4], "walltime_cap_hours": 96,
                 "node_hours_cap": 96},
                {"segment_id": "segment-2", "task_ids": ids[4:], "walltime_cap_hours": 96,
                 "node_hours_cap": 96},
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
        "source_manifest": {"path": str(source_manifest), "sha256": source_sha,
                            "batch_authorization_applies_to_scaling": False},
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
        "larger_tiers": {"nodes": [16, 32, 64, 128], "status": "planned_separate_authorization_required"},
        "execution_authorized": False,
        "authorization": None,
    }
    strong = {
        **shared,
        "campaign_id": "E4-Aurora-BCC-Strong-v1",
        "experiment": "strong",
        "run_id_prefix": "E4-SCALE-STRONG-v1",
        "fixed_workload": {"required_unique_tasks": 8, "unique_key": "POSCAR SHA-256",
                           "case_poscar_sha256": {t["calculation_id"]: t["source_poscar_sha256"]
                                                   for t in shared["tasks"]}},
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
    manifest = json.loads(path.read_text())
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
    for task in manifest["tasks"]:
        validate_task(task)
    for tier in manifest["tiers"]:
        if tier["walltime_cap_hours"] > QUEUE_WALLTIME_HOURS:
            raise ValueError("tier exceeds the capacity queue walltime")
        if manifest["experiment"] == "weak":
            assigned = [task_id for node in tier["node_assignments"] for task_id in node]
            if len(assigned) != tier["required_unique_tasks"] or len(set(assigned)) != len(assigned):
                raise ValueError("weak tier task assignment is incomplete or duplicated")
        else:
            assigned = [task_id for segment in tier["segments"] for task_id in segment["task_ids"]]
            if sorted(assigned) != sorted(STRONG_IDS) or len(assigned) != len(set(assigned)):
                raise ValueError("strong tier does not dispatch the fixed eight tasks exactly once")
            if any(segment.get("walltime_cap_hours", 0) > QUEUE_WALLTIME_HOURS
                   for segment in tier["segments"] if len(tier["segments"]) > 1):
                raise ValueError("strong segment exceeds the queue walltime")
    return manifest


def dispatch_plan(manifest: dict[str, Any], tier_nodes: int, segment_id: str | None) -> list[list[str]]:
    tier = next((item for item in manifest["tiers"] if item["nodes"] == tier_nodes), None)
    if tier is None:
        raise ValueError(f"no {tier_nodes}-node tier")
    if manifest["experiment"] == "weak":
        assignments = tier["node_assignments"]
        return [[node[index] for node in assignments if index < len(node)]
                for index in range(max(map(len, assignments)))]
    segments = tier["segments"]
    if len(segments) > 1:
        selected = next((segment for segment in segments if segment["segment_id"] == segment_id), None)
        if selected is None:
            raise ValueError("the one-node strong tier requires segment-1 or segment-2")
    else:
        if segment_id not in (None, "all"):
            raise ValueError("this tier uses segment-id=all")
        selected = segments[0]
    ids = selected["task_ids"]
    return [ids[index:index + tier_nodes] for index in range(0, len(ids), tier_nodes)]


def dispatch(manifest_path: Path, nodes: int, segment_id: str | None, dry_run: bool) -> int:
    manifest = validate_manifest(manifest_path)
    waves = dispatch_plan(manifest, nodes, segment_id)
    if dry_run:
        print(json.dumps({"experiment": manifest["experiment"], "nodes": nodes,
                          "segment_id": segment_id, "waves": waves,
                          "execution_authorized": False}, indent=2))
        return 0
    authorization = manifest.get("authorization") or {}
    if manifest["execution_authorized"] is not True or os.environ.get("E4_SCALING_AUTHORIZATION") != authorization.get("id"):
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

    def run_task(calc_id: str, host: str) -> dict[str, Any]:
        task = task_map[calc_id]
        task_root = root / calc_id
        env = os.environ.copy()
        env.update({
            "REPO": str(REPO),
            "CASE_DIR": str((REPO / task["case_dir"]).resolve()),
            "VARIANT": "alcf",
            "REPEAT": "repeat-01",
            "E4_CALCULATION_ID": calc_id,
            "E4_STAGE_MODE": "low-normal",
            "E4_STAGE_NSW_MAX": "240",
            "E4_EDIFFG": "-0.01",
            "E4_MPI_RANKS": "8",
            "E4_OMP_THREADS": "8",
            "E4_REQUESTED_NODES": "1",
            "E4_ACCOUNTING_NODES": "1",
            "E4_REQUESTED_WALLTIME_SECONDS": str(TASK_WALLTIME_HOURS * 3600),
            "E4_SOFT_STOP_SECONDS": str(TASK_WALLTIME_HOURS * 3600 - 3600),
            "E4_OUTPUT_ROOT": str(task_root),
            "MATSIM_ASSIGNED_NODES": host,
        })
        started = datetime.now(timezone.utc).isoformat()
        start_monotonic = time.monotonic()
        with (root / f"{calc_id}.stdout").open("w") as stdout, (root / f"{calc_id}.stderr").open("w") as stderr:
            process = subprocess.run(["bash", str(wrapper)], env=env, stdout=stdout, stderr=stderr)
        ended = datetime.now(timezone.utc).isoformat()
        execution_seconds = time.monotonic() - start_monotonic
        final_status = None
        run_record = task_root / "run-record.log"
        if run_record.is_file():
            exits = re.findall(r"\bEXIT\s+job=\S+\s+calculation_id=\S+\s+status=(-?\d+)\s+final=(\S+)",
                               run_record.read_text(errors="replace"))
            if exits:
                final_status = exits[-1][1]
        converged = process.returncode == 0 and bool(
            re.fullmatch(r"converged:N\d+", final_status or "")
        )
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
        }

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
                    stream.write(json.dumps({"event": "started", "time": datetime.now(timezone.utc).isoformat(),
                                             "calculation_id": calc_id, "node": host,
                                             "wave": wave_index + 1}) + "\n")
                futures[executor.submit(run_task, calc_id, host)] = calc_id
            for future in as_completed(futures):
                result = future.result()
                result["wave"] = wave_index + 1
                failed = failed or result["task_status"] != "converged"
                with log_path.open("a") as stream:
                    stream.write(json.dumps(result) + "\n")
        if failed:
            break
        print(f"wave {wave_index + 1}/{len(waves)} completed in {time.monotonic() - started:.1f}s")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
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
    if args.command == "freeze-manifests":
        freeze_manifests(args.source_manifest, args.weak_inputs_manifest, args.output_dir)
        return 0
    if args.command == "validate":
        manifest = validate_manifest(args.manifest)
        print(f"validated {manifest['experiment']} manifest; execution_authorized=false")
        return 0
    return dispatch(args.manifest, args.nodes, args.segment_id, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())