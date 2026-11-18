"""Roll authorized E4 jobs through PBS as successful jobs release queue slots.

The controller runs outside PBS, uses the existing E4 scaling dispatcher, and
never submits a job whose dependency failed or whose node-hour reservation would
exceed the frozen allocation cap.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CAMPAIGN = REPO / "research/paper/e4/aurora-dft-scaling/six-class-workflow-v4.json"
DEFAULT_PLAN = REPO / "research/paper/e4/aurora-dft-scaling/rolling-regular-v1.json"
DEFAULT_WORK_ROOT = REPO / "runs/e4-scaling/rolling-regular-v1"
PBS_SCRIPT = REPO / "deployments/aurora/jobs/job-e4-scaling.pbs"
JOB_LIMIT = 5
CLASS_ORDER = (
    "bcc_refractory",
    "fcc_hea",
    "lifepo4",
    "phosphorene",
    "zn_formate_mof",
    "nb_ta_o",
)
CLASS_SHORT = {
    "bcc_refractory": "bcc",
    "fcc_hea": "fcc",
    "lifepo4": "lfp",
    "phosphorene": "phos",
    "zn_formate_mof": "zn",
    "nb_ta_o": "nb",
}
AUTHORIZATION_ID = "E4-SIX-USER-AUTH-20261111-ALL-CLASSES"
# Existing job IDs are authoritative roots for the in-flight tier chains.
EXISTING_JOBS = {
    "9069490.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov": "success",
    "9077296.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov": "success",
    "9077297.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov": "success",
    "9088099.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov": "success",
    "9090503.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov": "pending",
    "9092466.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov": "pending",
    "9092467.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov": "pending",
    "9092468.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov": "pending",
    "9092469.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov": "pending",
}
EXISTING_TIER_JOBS = {
    ("fcc_hea", "weak", 1): "9090503.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
    ("lifepo4", "weak", 1): "9092466.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
    ("phosphorene", "weak", 1): "9077296.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
    ("phosphorene", "weak", 2): "9092467.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
    ("zn_formate_mof", "weak", 1): "9077297.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
    ("zn_formate_mof", "weak", 2): "9092468.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
    ("nb_ta_o", "weak", 1): "9088099.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
    ("nb_ta_o", "weak", 2): "9092469.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_qstat_job_ids(text: str, username: str) -> set[str]:
    jobs = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] == username and re.match(r"^\d+\.aurora-pbs-", fields[0]):
            jobs.add(fields[0])
    return jobs


def parse_qstat_full(text: str) -> tuple[str | None, int | None]:
    state_match = re.search(r"(?m)^\s*job_state\s*=\s*(\w+)\s*$", text)
    exit_match = re.search(r"(?m)^\s*Exit_status\s*=\s*(-?\d+)\s*$", text)
    return (
        state_match.group(1) if state_match else None,
        int(exit_match.group(1)) if exit_match else None,
    )


def _dispatch_task(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        **candidate,
        "input_hashes": candidate["input_file_hashes"],
        "omp_threads_per_rank": candidate["omp_threads_per_rank"],
    }


def _normalized_manifest(
    campaign: dict[str, Any],
    class_record: dict[str, Any],
    experiment: str,
    tier: dict[str, Any],
    *,
    run_id_prefix: str,
    job_cap_node_hours: int,
    job_walltime_hours: int,
) -> dict[str, Any]:
    candidate_map = {row["calculation_id"]: row for row in class_record["candidates"]}
    if experiment == "weak":
        ids = tier["calculation_ids"]
        tasks = [_dispatch_task(candidate_map[calc_id]) for calc_id in ids]
        segments = None
        fixed_ids = None
        node_assignments = [ids[index : index + 2] for index in range(0, len(ids), 2)]
        normalized_tier = {
            "nodes": tier["nodes"],
            "node_assignments": node_assignments,
            "required_unique_tasks": len(ids),
            "walltime_cap_hours": job_walltime_hours,
        }
    else:
        fixed_ids = class_record["strong_scaling"]["fixed_calculation_ids"]
        tasks = [_dispatch_task(candidate_map[calc_id]) for calc_id in fixed_ids]
        source_segments = tier.get("segments") or [
            {
                "segment": 1,
                "calculation_ids": fixed_ids,
                "walltime_cap_hours": tier["walltime_cap_hours"],
            }
        ]
        segments = [
            {
                "segment_id": f"segment-{segment['segment']}",
                "task_ids": segment["calculation_ids"],
                "walltime_cap_hours": min(
                    168,
                    segment["walltime_cap_hours"]
                    + (1 if segment["walltime_cap_hours"] < 168 else 0),
                ),
            }
            for segment in source_segments
        ]
        normalized_tier = {
            "nodes": tier["nodes"],
            "segments": segments,
            "walltime_cap_hours": max(segment["walltime_cap_hours"] for segment in segments),
        }
        node_assignments = None
    auth = {
        "id": AUTHORIZATION_ID,
        "scope": (
            "user-authorized regular-queue rolling E4 campaign within the frozen node-hour cap"
        ),
    }
    return {
        "schema_version": 1,
        "campaign_id": (
            f"E4-rolling-{class_record['material_class']}-{experiment}-n{tier['nodes']:02d}"
        ),
        "experiment": experiment,
        "material_class": class_record["material_class"],
        "run_id_prefix": run_id_prefix,
        "protocol_id": class_record["protocol"]["id"],
        "queue": "capacity",
        "execution_authorized": True,
        "authorization": auth,
        "resource_request": {
            "maximum_nodes_per_job": tier["nodes"],
            "maximum_walltime_hours_per_job": job_walltime_hours,
            "total_node_hours_cap": job_cap_node_hours,
        },
        "fixed_task_ids": fixed_ids,
        "tasks": tasks,
        "tiers": [normalized_tier],
    }


def build_rolling_plan(
    campaign_path: Path,
    *,
    available_node_hours: int,
    existing_reserved_node_hours: int,
    output_plan: Path = DEFAULT_PLAN,
    work_root: Path = DEFAULT_WORK_ROOT,
) -> dict[str, Any]:
    campaign = json.loads(campaign_path.read_text())
    classes = {row["material_class"]: row for row in campaign["material_classes"]}
    work_root.mkdir(parents=True, exist_ok=True)
    manifest_root = work_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    key_by_tier: dict[tuple[str, str, int], str] = {}
    tier_nodes = (1, 2, 4, 8, 16)
    for nodes in tier_nodes:
        for name in CLASS_ORDER:
            record = classes[name]
            tier = next(item for item in record["weak_scaling"]["tiers"] if item["nodes"] == nodes)
            if (name, "weak", nodes) in EXISTING_TIER_JOBS:
                key_by_tier[(name, "weak", nodes)] = (
                    f"pbs:{EXISTING_TIER_JOBS[(name, 'weak', nodes)]}"
                )
                continue
            task_wall = int(tier["walltime_cap_hours"])
            job_wall = min(168, task_wall + 1)
            cap = nodes * job_wall
            job_key = f"{name}-weak-n{nodes:02d}"
            previous = key_by_tier.get((name, "weak", nodes // 2)) if nodes > 1 else None
            after_pbs = []
            after_keys = []
            if previous:
                (after_pbs if previous.startswith("pbs:") else after_keys).append(
                    previous[4:] if previous.startswith("pbs:") else previous
                )
            elif name == "bcc_refractory":
                after_pbs = ["9069490.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov"]
            prefix = f"E4-ROLL-{name}-weak-n{nodes:02d}-v1"
            manifest = _normalized_manifest(
                campaign,
                record,
                "weak",
                tier,
                run_id_prefix=prefix,
                job_cap_node_hours=cap,
                job_walltime_hours=job_wall,
            )
            manifest_path = manifest_root / f"{job_key}.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            jobs.append(
                {
                    "job_key": job_key,
                    "material_class": name,
                    "experiment": "weak",
                    "nodes": nodes,
                    "manifest": str(manifest_path),
                    "manifest_sha256": _sha256(manifest_path),
                    "run_id_prefix": prefix,
                    "job_name": f"e4r-{CLASS_SHORT[name]}-w{nodes}",
                    "walltime_hours": job_wall,
                    "node_hours_cap": cap,
                    "after_pbs_job_ids": after_pbs,
                    "after_job_keys": after_keys,
                }
            )
            key_by_tier[(name, "weak", nodes)] = job_key
    for name in CLASS_ORDER:
        record = classes[name]
        tier = next(item for item in record["strong_scaling"]["tiers"] if item["nodes"] == 16)
        weak_parent = key_by_tier[(name, "weak", 4)]
        source_segments = tier.get("segments") or [
            {
                "segment": 1,
                "calculation_ids": record["strong_scaling"]["fixed_calculation_ids"],
                "walltime_cap_hours": tier["walltime_cap_hours"],
            }
        ]
        previous_key = weak_parent
        prefix = f"E4-ROLL-{name}-strong-n16-v1"
        run_id = f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        for segment in source_segments:
            index = segment["segment"]
            source_wall = int(segment["walltime_cap_hours"])
            job_wall = min(168, source_wall + (1 if source_wall < 168 else 0))
            cap = 16 * job_wall
            job_key = f"{name}-strong-n16-segment-{index}"
            manifest = _normalized_manifest(
                campaign,
                record,
                "strong",
                tier,
                run_id_prefix=prefix,
                job_cap_node_hours=cap,
                job_walltime_hours=job_wall,
            )
            manifest_path = manifest_root / f"{job_key}.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            jobs.append(
                {
                    "job_key": job_key,
                    "material_class": name,
                    "experiment": "strong",
                    "nodes": 16,
                    "segment_id": f"segment-{index}",
                    "manifest": str(manifest_path),
                    "manifest_sha256": _sha256(manifest_path),
                    "run_id_prefix": prefix,
                    "run_id": run_id,
                    "job_name": f"e4r-{CLASS_SHORT[name]}-s16-{index}",
                    "walltime_hours": job_wall,
                    "node_hours_cap": cap,
                    "after_pbs_job_ids": [],
                    "after_job_keys": [previous_key],
                }
            )
            previous_key = job_key
    reserved = existing_reserved_node_hours + sum(job["node_hours_cap"] for job in jobs)
    if reserved > available_node_hours:
        raise ValueError(
            f"rolling plan reserves {reserved} node-hours including in-flight jobs, "
            f"above available balance {available_node_hours}"
        )
    plan = {
        "schema_version": 1,
        "execution_authorized": True,
        "authorization_id": AUTHORIZATION_ID,
        "source_campaign": str(campaign_path),
        "source_campaign_sha256": _sha256(campaign_path),
        "allocation": "HydraGNN",
        "queue": "capacity",
        "max_total_user_jobs": JOB_LIMIT,
        "poll_interval_seconds": 60,
        "available_node_hours_snapshot": available_node_hours,
        "existing_inflight_node_hours_reserved": existing_reserved_node_hours,
        "rolling_jobs_maximum_node_hours": sum(job["node_hours_cap"] for job in jobs),
        "combined_reserved_node_hours": reserved,
        "jobs": jobs,
    }
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    output_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def _job_status(job_id: str) -> tuple[str, int | None]:
    result = subprocess.run(["qstat", "-xf", job_id], capture_output=True, text=True, check=False)
    if result.returncode:
        return "unknown", None
    state = re.search(r"(?m)^\s*job_state\s*=\s*(\w+)\s*$", result.stdout)
    exit_status = re.search(r"(?m)^\s*Exit_status\s*=\s*(-?\d+)\s*$", result.stdout)
    return (
        state.group(1) if state else "unknown",
        int(exit_status.group(1)) if exit_status else None,
    )


def _current_user_job_count(username: str) -> int:
    result = subprocess.run(["qstat", "-u", username], capture_output=True, text=True, check=True)
    return sum(
        1
        for line in result.stdout.splitlines()
        if len(line.split()) >= 3
        and line.split()[1] == username
        and re.match(r"^\d+\.aurora-pbs-", line.split()[0])
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _run_controller_loop(plan_path: Path, state_path: Path, *, once: bool = False) -> int:
    plan = json.loads(plan_path.read_text())
    if plan.get("execution_authorized") is not True:
        raise PermissionError("rolling queue plan is not authorized")
    plan_hash = _sha256(plan_path)
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"plan_sha256": plan_hash, "submitted": {}, "blocked": {}, "completed": {}}
    )
    if state.get("plan_sha256") != plan_hash:
        raise ValueError("state file belongs to a different rolling queue plan")
    jobs = {job["job_key"]: job for job in plan["jobs"]}
    print(
        f"rolling E4 submitter active: {len(jobs)} planned jobs, "
        f"PBS user-job limit={plan['max_total_user_jobs']}, "
        f"node-hour cap={plan['combined_reserved_node_hours']}",
        flush=True,
    )
    queue_full_reported = False
    while True:
        changed = False
        job_count = _current_user_job_count(os.environ.get("USER", ""))
        status_cache: dict[str, tuple[str, int | None]] = {}
        for job_key in state["completed"]:
            status_cache["key:" + job_key] = ("F", 0)
        for job_key in state["blocked"]:
            status_cache["key:" + job_key] = ("F", 1)
        for job_key, info in state["submitted"].items():
            status_cache["key:" + job_key] = _job_status(info["pbs_job_id"])
        for job in plan["jobs"]:
            for job_id in job["after_pbs_job_ids"]:
                if job_id not in status_cache:
                    status_cache[job_id] = _job_status(job_id)
        for job in plan["jobs"]:
            key = job["job_key"]
            if key in state["submitted"] or key in state["blocked"]:
                continue
            dependencies = ["key:" + dep for dep in job["after_job_keys"]] + job[
                "after_pbs_job_ids"
            ]
            dep_states = [status_cache.get(dep, ("unknown", None)) for dep in dependencies]
            if any(
                state_name == "F" and exit_code not in (0, None)
                for state_name, exit_code in dep_states
            ):
                state["blocked"][key] = {"reason": "a prerequisite job failed", "time": time.time()}
                changed = True
                continue
            if any(state_name != "F" or exit_code != 0 for state_name, exit_code in dep_states):
                continue
            if job_count >= int(plan["max_total_user_jobs"]):
                if not queue_full_reported:
                    print(
                        f"PBS user-job limit reached ({job_count}); waiting for a job to finish",
                        flush=True,
                    )
                    queue_full_reported = True
                break
            env_values = {
                "E4_SCALING_MANIFEST": job["manifest"],
                "E4_SCALING_NODES": str(job["nodes"]),
                "E4_SCALING_AUTHORIZATION": plan["authorization_id"],
                "E4_SCALING_RUN_ID": job.get(
                    "run_id",
                    f"{job['run_id_prefix']}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
                ),
            }
            if job.get("segment_id"):
                env_values["E4_SCALING_SEGMENT"] = job["segment_id"]
            walltime = f"{job['walltime_hours']}:00:00"
            command = [
                "qsub",
                "-A",
                plan["allocation"],
                "-q",
                plan["queue"],
                "-l",
                f"select={job['nodes']}",
                "-l",
                f"walltime={walltime}",
                "-N",
                job["job_name"],
                "-v",
                ",".join(f"{name}={value}" for name, value in env_values.items()),
                str(PBS_SCRIPT),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode:
                state["last_submit_error"] = {
                    "job_key": key,
                    "stderr": result.stderr.strip(),
                    "time": time.time(),
                }
                changed = True
                print(f"submission failed for {key}: {result.stderr.strip()}", flush=True)
                break
            pbs_job_id = result.stdout.strip().split()[0]
            state["submitted"][key] = {
                "pbs_job_id": pbs_job_id,
                "submitted_at_utc": datetime.now(UTC).isoformat(),
                "maximum_node_hours": job["node_hours_cap"],
                "material_class": job["material_class"],
                "experiment": job["experiment"],
                "nodes": job["nodes"],
            }
            job_count += 1
            queue_full_reported = False
            print(f"submitted {key} as {pbs_job_id}", flush=True)
            changed = True
            _atomic_json(state_path, state)
        for job_key, info in list(state["submitted"].items()):
            state_name, exit_code = status_cache.get("key:" + job_key, ("unknown", None))
            if state_name == "F" and exit_code == 0:
                state["completed"].setdefault(
                    job_key,
                    {
                        "pbs_job_id": info["pbs_job_id"],
                        "completed_at_utc": datetime.now(UTC).isoformat(),
                    },
                )
                changed = True
        if changed:
            _atomic_json(state_path, state)
        if once:
            return 0
        if len(state["submitted"]) + len(state["blocked"]) >= len(jobs):
            print("rolling E4 submitter queue exhausted", flush=True)
            return 0
        time.sleep(int(plan.get("poll_interval_seconds", 60)))


def run_controller(plan_path: Path, state_path: Path, *, once: bool = False) -> int:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"rolling submitter already active: {lock_path}") from exc
        return _run_controller_loop(plan_path, state_path, once=once)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-plan")
    build.add_argument("--available-node-hours", type=int, required=True)
    build.add_argument("--existing-inflight-node-hours", type=int, required=True)
    build.add_argument("--output", type=Path, default=DEFAULT_PLAN)
    build.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    run.add_argument("--state", type=Path, default=DEFAULT_WORK_ROOT / "state.json")
    run.add_argument("--once", action="store_true", help="make one queue pass and exit")
    args = parser.parse_args(argv)
    if args.command == "build-plan":
        plan = build_rolling_plan(
            DEFAULT_CAMPAIGN,
            available_node_hours=args.available_node_hours,
            existing_reserved_node_hours=args.existing_inflight_node_hours,
            output_plan=args.output,
            work_root=args.work_root,
        )
        print(
            f"planned {len(plan['jobs'])} future jobs; reserved cap "
            f"{plan['combined_reserved_node_hours']} / "
            f"{plan['available_node_hours_snapshot']} node-hours"
        )
        return 0
    return run_controller(args.plan, args.state, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
