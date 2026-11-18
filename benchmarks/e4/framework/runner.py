"""Run one E4 orchestration measurement on the production execution path.

Flow: discover the allocation, resolve the production ``ResourcePlan``, freeze
the workload manifest and run identity, dispatch node-group tasks through
``run_dft_batch`` and device-slot tasks through ``MLIPWorkerPool``, then write
a measurement record that ``benchmarks/e4/planning.validate_measurement`` accepts.

Resume reuses the run directory: the identity must match, tasks whose final
attempt is recorded are skipped, and attempts interrupted by a crash are
recorded as consumed (non-terminal) attempts before work continues.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.e4.framework.execution import (
    ControlledStepExecutor,
    DFTLabelExecutor,
    RetryingTaskBackend,
    device_controlled_task,
    device_task_outcome,
    mlip_relax_outcome,
    retry_decision,
    step_launcher,
)
from benchmarks.e4.framework.records import (
    TERMINAL_FIELD,
    AttemptRecorder,
    read_records,
    task_histories,
)
from benchmarks.e4.framework.workload import (
    TaskSpec,
    Workload,
    build_workload,
    canonical_json,
    sha256_text,
)
from matsim_agents.execution.allocation import Allocation, discover_allocation
from matsim_agents.execution.ledger import software_versions
from matsim_agents.execution.placement import Placement, ResourcePlan
from matsim_agents.execution.recovery import RECOVERY_POLICY_VERSION

REPO = Path(__file__).resolve().parents[3]
FACILITIES = ("local", "aurora", "frontier", "perlmutter")
POLICIES = ("concurrent", "staged")
DEVICE_POOLS = ("mlip", "driver")
SCHEMA_VERSION = 1


class RunIdentityError(RuntimeError):
    """A resumed run does not match the frozen identity of the original run."""


def _git(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args], cwd=REPO, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def environment_record(facility: str) -> dict[str, Any]:
    """Code version, software environment, and scheduler identifiers of this process."""

    scheduler = {
        key: os.environ[key]
        for key in (
            "PBS_JOBID",
            "PBS_JOBNAME",
            "PBS_QUEUE",
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_JOB_PARTITION",
            "SLURM_JOB_QOS",
            "SLURM_JOB_NUM_NODES",
        )
        if os.environ.get(key)
    }
    return {
        "facility": facility,
        "hostname": socket.gethostname(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "software": software_versions("ase", "numpy", "torch", "pydantic"),
        "scheduler": scheduler,
        "modules": os.environ.get("LOADEDMODULES"),
        "executables": {name: shutil.which(name) for name in ("mpiexec", "srun", "pw.x")},
        "recorded_utc": datetime.now(UTC).isoformat(),
    }


def local_allocation(nodes: int) -> Allocation:
    """A synthetic local allocation whose node groups all run on this host."""

    return Allocation(
        scheduler="local", nodes=tuple(f"local-{index:04d}" for index in range(nodes))
    )


def resource_plan(config: Mapping[str, Any], facility: str) -> ResourcePlan:
    plans = config.get("resources") or {}
    if facility not in plans:
        raise ValueError(f"config has no resources for facility {facility!r}")
    return ResourcePlan.model_validate(plans[facility])


def _pool_for(task: TaskSpec, placement: Placement, *, local_device_slots: bool = False) -> str:
    if task.pool == "driver" or (task.pool == "mlip" and local_device_slots):
        return task.pool
    if task.pool not in placement.pools:
        raise ValueError(f"{task.task_id}: placement has no pool {task.pool!r}")
    return task.pool


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


class BenchmarkRun:
    """One E4 run directory: identity, manifest, attempt log, and summary."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        facility: str,
        experiment: str,
        repetition: int,
        output_root: str | Path,
        nodes: int | None = None,
        allocation: Allocation | None = None,
        policy: str = "concurrent",
        classes: Sequence[str] | None = None,
        run_id: str | None = None,
        resume: bool = False,
        local_device_slots: int | None = None,
        executors: Mapping[str, Any] | None = None,
    ) -> None:
        self.t_origin = time.monotonic()
        self.wall_origin = time.time()
        start_epoch = os.environ.get("E4_ALLOCATION_START_EPOCH")
        self.allocation_start = float(start_epoch) if start_epoch else self.wall_origin
        if facility not in FACILITIES:
            raise ValueError(f"facility must be one of {FACILITIES}")
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.config = dict(config)
        self.facility = facility
        self.experiment = experiment
        self.repetition = repetition
        self.policy = policy
        self.local_device_slots = local_device_slots
        self.executors = dict(executors or {})

        placement_started = time.monotonic()
        if allocation is None:
            allocation = (
                local_allocation(nodes or 1) if facility == "local" else discover_allocation()
            )
        if nodes is not None and allocation.node_count != nodes:
            raise ValueError(
                f"allocation has {allocation.node_count} nodes but the run requests {nodes}"
            )
        self.allocation = allocation
        self.plan = resource_plan(config, facility)
        self.placement = self.plan.resolve(allocation)
        self.placement_seconds = time.monotonic() - placement_started

        self.workload: Workload = build_workload(
            config,
            experiment=experiment,
            nodes=allocation.node_count,
            root=REPO,
            classes=classes,
        )
        workload_id = self.workload.workload_id
        suffix = f"-{policy}" if policy == "staged" else ""
        if classes is not None:
            suffix += "-only-" + "+".join(sorted(classes))
        default_id = (
            f"{workload_id}-{facility}-{experiment}-n{allocation.node_count:04d}"
            f"-r{repetition:02d}{suffix}"
        )
        self.run_id = run_id or default_id
        # Scaling analysis groups by this label, so subsets and policies never mix.
        self.group_label = workload_id + suffix
        self.run_dir = Path(output_root) / self.run_id
        self.resumed = resume
        self.identity = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "workload_id": workload_id,
            "facility": facility,
            "experiment": experiment,
            "repetition": repetition,
            "nodes": allocation.node_count,
            "policy": policy,
            "workload_hash": self.workload.workload_hash,
            "config_sha256": self.workload.config_sha256,
            "resource_plan_sha256": sha256_text(canonical_json(self.plan.model_dump(mode="json"))),
            "recovery_policy": RECOVERY_POLICY_VERSION,
        }
        self._open_run_directory()
        self.recorder = AttemptRecorder(
            self.run_dir / "attempts.jsonl",
            clock_origin=self.t_origin,
            segment=self._segment_index(),
        )
        self.init_seconds = time.monotonic() - self.t_origin

    def _open_run_directory(self) -> None:
        identity_path = self.run_dir / "identity.json"
        if self.resumed:
            if not identity_path.is_file():
                raise RunIdentityError(f"no run to resume at {self.run_dir}")
            saved = json.loads(identity_path.read_text(encoding="utf-8"))
            changed = sorted(key for key in self.identity if saved.get(key) != self.identity[key])
            if changed:
                raise RunIdentityError(f"resume rejected: identity fields changed: {changed}")
        else:
            if self.run_dir.exists():
                raise FileExistsError(
                    f"{self.run_dir} exists; pass resume=True or choose another run ID"
                )
            self.run_dir.mkdir(parents=True)
            self._write_json("identity.json", self.identity)
            self._write_json("manifest.json", self.workload.manifest())
            self._write_json("resolved_config.json", self.config)
        self._write_json(
            f"placement-{self._segment_index():02d}.json",
            self.placement.model_dump(mode="json"),
        )
        self._write_json(
            f"environment-{self._segment_index():02d}.json", environment_record(self.facility)
        )

    def _segment_index(self) -> int:
        if not hasattr(self, "_segment"):
            existing = sorted(self.run_dir.glob("environment-*.json"))
            self._segment = len(existing) + 1
        return self._segment

    def _write_json(self, name: str, payload: Any) -> None:
        path = self.run_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _reconcile(self) -> dict[str, Any]:
        """Record interrupted attempts and device results written before a crash."""

        records = read_records(self.recorder.path)
        recorded = {(r["task_id"], int(r["attempt"])) for r in records}
        tasks = {task.task_id: task for task in self.workload.tasks}
        for task_dir in sorted((self.run_dir / "tasks").glob("*")):
            task = tasks.get(task_dir.name)
            if task is None:
                continue
            for attempt_dir in sorted(task_dir.glob("attempt-*")):
                attempt = int(attempt_dir.name.split("-")[1])
                if (task.task_id, attempt) in recorded:
                    continue
                device_result = attempt_dir / "device_result.json"
                if task.pool in DEVICE_POOLS and device_result.is_file():
                    payload = json.loads(device_result.read_text(encoding="utf-8"))
                    self.recorder.append(
                        self._device_record(task, attempt_dir, payload, error=None, reconciled=True)
                    )
                else:
                    self.recorder.append(
                        {
                            "record": "attempt",
                            "task_id": task.task_id,
                            "task_class": task.task_class,
                            "kind": task.kind,
                            "pool": task.pool,
                            "attempt": attempt,
                            "status": "interrupted",
                            "failure_class": "interrupted",
                            "outcome": {"executed": False, "qualified": False},
                            "retry_scheduled": False,
                            "reconciled": True,
                            TERMINAL_FIELD: False,
                        }
                    )
        return task_histories(read_records(self.recorder.path))

    def _node_group_executor(self, kind: str):
        if kind in self.executors:
            return self.executors[kind]
        if kind == "controlled":
            return ControlledStepExecutor(
                step_launcher(self.facility), local=self.placement.launcher == "local"
            )
        if kind == "dft":
            backends = self.config.get("dft_backends") or {}
            if self.facility not in backends:
                raise ValueError(f"config has no dft_backends entry for {self.facility!r}")
            return DFTLabelExecutor(backends[self.facility])
        raise ValueError(f"task kind {kind!r} does not run on node groups")

    def _run_node_group(self, tasks: list[TaskSpec], histories) -> None:
        from ase import Atoms

        from matsim_agents.backends.dft.base import DFTJobSpec
        from matsim_agents.backends.dft.runner import run_dft_batch

        pool = self.placement.pool("dft")
        for kind in sorted({task.kind for task in tasks}):
            subset = [task for task in tasks if task.kind == kind]
            executor = self._node_group_executor(kind)
            backend = RetryingTaskBackend(
                executor,
                {task.task_id: task for task in subset},
                self.recorder,
                run_dir=self.run_dir,
                histories=histories,
                timeout_sec=getattr(executor, "timeout_sec", None)
                or self.config.get("timeout_sec"),
                backend=getattr(executor, "backend_name", "controlled"),
            )
            specs = [
                DFTJobSpec(
                    job_id=task.task_id,
                    atoms=Atoms(),
                    work_dir=str(self.run_dir / "tasks" / task.task_id),
                )
                for task in subset
            ]
            run_dft_batch(specs, backend, pool)

    def _device_pool(self, pool_name: str):
        from matsim_agents.backends.mlip.pool import MLIPWorkerPool, WorkerSlot

        if pool_name == "driver":
            return MLIPWorkerPool()
        if self.local_device_slots:
            return MLIPWorkerPool(
                [WorkerSlot(device=index, env={}) for index in range(self.local_device_slots)]
            )
        return MLIPWorkerPool.from_placement(self.placement)

    def _device_record(
        self,
        task: TaskSpec,
        attempt_dir: Path,
        value: Mapping[str, Any] | None,
        *,
        error: BaseException | None,
        reconciled: bool = False,
        retries: Mapping[str, int] | None = None,
        eligible_epoch: float | None = None,
    ) -> dict[str, Any]:
        if task.kind == "controlled":
            outcome = device_task_outcome(task, attempt_dir, value, error)
        elif error is not None:
            outcome = device_task_outcome(task, attempt_dir, None, error)
        else:
            from matsim_agents.backends.mlip.results import RelaxationResult

            outcome = mlip_relax_outcome(
                RelaxationResult.model_validate(value["result"]),
                float(task.params.get("fmax", 0.02)),
            )
        start = value.get("start_epoch") if value else None
        end = value.get("end_epoch") if value else None
        failure, retry, action = retry_decision(outcome, retries or {}, None)
        return {
            "record": "attempt",
            "task_id": task.task_id,
            "task_class": task.task_class,
            "kind": task.kind,
            "pool": task.pool,
            "attempt": int(attempt_dir.name.split("-")[1]),
            "hosts": [socket.gethostname()],
            "resources": {"nodes": 1, "devices": 1 if task.pool == "mlip" else 0},
            "start_offset_seconds": None if start is None else round(start - self.wall_origin, 6),
            "end_offset_seconds": None if end is None else round(end - self.wall_origin, 6),
            "wall_seconds": None if start is None or end is None else round(end - start, 6),
            "queued_seconds": None
            if start is None or eligible_epoch is None
            else round(max(0.0, start - eligible_epoch), 6),
            "outcome": {key: item for key, item in outcome.items() if key != "status"},
            "status": outcome["status"],
            "failure_class": None if failure is None else failure.value,
            "recovery_action": action,
            "recovery_policy": RECOVERY_POLICY_VERSION,
            "retry_scheduled": retry,
            "reconciled": reconciled,
            TERMINAL_FIELD: not retry,
        }

    def _run_device_pool(self, pool_name: str, tasks: list[TaskSpec], histories) -> None:
        """Dispatch to idle device slots; retryable failures go into the next round."""

        state = {
            task.task_id: (
                histories[task.task_id].attempts if task.task_id in histories else 0,
                dict(histories[task.task_id].retries_by_failure)
                if task.task_id in histories
                else {},
            )
            for task in tasks
        }
        pool = self._device_pool(pool_name)
        try:
            pending = list(tasks)
            while pending:
                items, dirs = [], []
                for task in pending:
                    attempt = state[task.task_id][0] + 1
                    attempt_dir = self.run_dir / "tasks" / task.task_id / f"attempt-{attempt:02d}"
                    attempt_dir.mkdir(parents=True, exist_ok=False)
                    dirs.append(attempt_dir)
                    args = {**task.to_json(), "work_dir": str(attempt_dir), "attempt": attempt}
                    items.append((task.kind, json.dumps(args)))
                eligible = time.time()
                outcomes = pool.map(device_task_entry, items)
                retry_next = []
                for task, attempt_dir, outcome in zip(pending, dirs, outcomes, strict=True):
                    attempts, retries = state[task.task_id]
                    record = self._device_record(
                        task,
                        attempt_dir,
                        outcome.value,
                        error=outcome.error,
                        retries=retries,
                        eligible_epoch=eligible,
                    )
                    self.recorder.append(record)
                    if record["retry_scheduled"]:
                        failure = record["failure_class"]
                        retries[failure] = retries.get(failure, 0) + 1
                        retry_next.append(task)
                    state[task.task_id] = (attempts + 1, retries)
                pending = retry_next
        finally:
            pool.close()

    def _dispatch(self, pending: list[TaskSpec], histories) -> None:
        groups: dict[str, list[TaskSpec]] = {}
        for task in pending:
            pool_name = _pool_for(
                task, self.placement, local_device_slots=bool(self.local_device_slots)
            )
            groups.setdefault(pool_name, []).append(task)

        def run_group(pool_name: str, tasks: list[TaskSpec]) -> None:
            if pool_name in DEVICE_POOLS:
                self._run_device_pool(pool_name, tasks, histories)
            elif pool_name == "dft":
                self._run_node_group(tasks, histories)
            else:
                raise ValueError(f"no dispatcher for pool {pool_name!r}")

        if self.policy == "staged":
            order = [spec["name"] for spec in self.config["task_classes"]]
            for name in order:
                for pool_name, tasks in groups.items():
                    subset = [task for task in tasks if task.task_class == name]
                    if subset:
                        run_group(pool_name, subset)
            return
        errors: list[BaseException] = []

        def guarded(pool_name: str, tasks: list[TaskSpec]) -> None:
            try:
                run_group(pool_name, tasks)
            except BaseException as exc:  # noqa: BLE001 - surfaced after all pools finish
                errors.append(exc)

        threads = [
            threading.Thread(target=guarded, args=item, name=f"e4-{item[0]}")
            for item in groups.items()
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]

    def execute(self) -> dict[str, Any]:
        histories = self._reconcile()
        pending = [
            task
            for task in self.workload.tasks
            if task.task_id not in histories or histories[task.task_id].terminal is None
        ]
        skipped = self.workload.planned_tasks - len(pending)
        dispatch_start = time.monotonic()
        self._dispatch(pending, histories)
        dispatch_end = time.monotonic()
        finalize_start = time.monotonic()
        summary = self._summarize(
            dispatch_seconds=dispatch_end - dispatch_start,
            dispatch_start=dispatch_start,
            skipped=skipped,
        )
        self._write_json(f"summary-{self._segment_index():02d}.json", summary)
        finalize_seconds = time.monotonic() - finalize_start
        summary["framework_finalization_seconds"] = round(finalize_seconds, 6)
        measurement = self._measurement(summary)
        with (self.run_dir / "measurements.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(measurement, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._write_json(f"summary-{self._segment_index():02d}.json", summary)
        return summary

    def _summarize(
        self, *, dispatch_seconds: float, dispatch_start: float, skipped: int
    ) -> dict[str, Any]:
        records = read_records(self.recorder.path)
        attempts = [r for r in records if r.get("record") == "attempt"]
        histories = task_histories(attempts)
        terminal = {task_id: h.terminal for task_id, h in histories.items() if h.terminal}
        this_segment = [
            r
            for r in attempts
            if r.get("segment") == self._segment_index() and not r.get("reconciled")
        ]
        outcome = lambda r, key: bool(r.get("outcome", {}).get(key))  # noqa: E731
        executed = {t for t, r in terminal.items() if outcome(r, "executed")}
        converged = {t for t, r in terminal.items() if outcome(r, "converged")}
        qualified = {t for t, r in terminal.items() if outcome(r, "qualified")}
        failed = set(terminal) - qualified
        retried = {t for t, h in histories.items() if h.attempts > 1}
        geometry = self.placement.geometry
        nodes = self.allocation.node_count
        allocation_seconds = time.time() - self.allocation_start
        busy_device_seconds = sum(
            (r.get("wall_seconds") or 0.0) * ((r.get("resources") or {}).get("devices") or 0)
            for r in this_segment
        )
        busy_node_seconds = sum(
            (r.get("wall_seconds") or 0.0) * ((r.get("resources") or {}).get("nodes") or 0)
            for r in this_segment
            if r.get("pool") == "dft"
        )
        device_capacity_seconds = dispatch_seconds * nodes * geometry.devices_per_node
        dft_nodes = sum(
            launch.node_count
            for launch in (
                self.placement.pools["dft"].launches if "dft" in self.placement.pools else []
            )
        )
        launch_overheads = [
            r["wall_seconds"] - r["outcome"]["payload_seconds"]
            for r in this_segment
            if r.get("wall_seconds") is not None
            and r.get("outcome", {}).get("payload_seconds") is not None
        ]
        qualified_now = sum(
            1 for r in this_segment if r.get(TERMINAL_FIELD) and outcome(r, "qualified")
        )
        executed_now = sum(
            1 for r in this_segment if r.get(TERMINAL_FIELD) and outcome(r, "executed")
        )
        by_class: dict[str, dict[str, int]] = {}
        for task in self.workload.tasks:
            row = by_class.setdefault(
                task.task_class, {"planned": 0, "qualified": 0, "failed": 0, "attempts": 0}
            )
            row["planned"] += 1
            row["qualified"] += task.task_id in qualified
            row["failed"] += task.task_id in failed
            row["attempts"] += histories[task.task_id].attempts if task.task_id in histories else 0
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "segment": self._segment_index(),
            "resumed": self.resumed,
            "identity": self.identity,
            "nodes": nodes,
            "placement": {
                "scheduler": self.placement.scheduler,
                "launcher": self.placement.launcher,
                "dft_node_groups": len(self.placement.pools["dft"].launches)
                if "dft" in self.placement.pools
                else 0,
                "mlip_devices": self.placement.pools["mlip"].device_count
                if "mlip" in self.placement.pools
                else 0,
                "device_kind": geometry.device_kind,
                "devices_per_node": geometry.devices_per_node,
            },
            "counts": {
                "planned_tasks": self.workload.planned_tasks,
                "skipped_already_complete": skipped,
                "attempted_tasks": len(histories),
                "executed_tasks": len(executed),
                "converged_tasks": len(converged),
                "qualified_tasks": len(qualified),
                "failed_tasks": len(failed),
                "retried_tasks": len(retried),
                "unfinished_tasks": self.workload.planned_tasks - len(terminal),
                "total_attempts": len(attempts),
                "failed_attempts": sum(1 for r in attempts if not outcome(r, "qualified")),
                "interrupted_attempts": sum(
                    1 for r in attempts if r.get("status") == "interrupted"
                ),
                "failures_by_class": dict(
                    Counter(r["failure_class"] for r in attempts if r.get("failure_class"))
                ),
            },
            "by_task_class": by_class,
            "timing": {
                "framework_initialization_seconds": round(self.init_seconds, 6),
                "placement_seconds": round(self.placement_seconds, 6),
                "application_makespan_seconds": round(dispatch_seconds, 6),
                "allocation_seconds": round(allocation_seconds, 6),
                "checkpoint_provenance_seconds": round(self.recorder.overhead_seconds, 6),
                "checkpoint_records": self.recorder.records_written,
                "queue_wait_seconds": _stats([r.get("queued_seconds") for r in this_segment]),
                "launch_overhead_seconds": _stats(launch_overheads),
                "attempt_wall_seconds": _stats([r.get("wall_seconds") for r in this_segment]),
                "recovery_seconds": round(
                    sum(
                        r.get("wall_seconds") or 0.0
                        for r in this_segment
                        if not outcome(r, "qualified")
                    ),
                    6,
                ),
            },
            "throughput": {
                "completed_tasks_per_second": executed_now / dispatch_seconds
                if dispatch_seconds > 0
                else None,
                "qualified_results_per_second": qualified_now / dispatch_seconds
                if dispatch_seconds > 0
                else None,
                "qualified_results_per_node_second": qualified_now / dispatch_seconds / nodes
                if dispatch_seconds > 0
                else None,
            },
            "resources": {
                "node_hours": nodes * allocation_seconds / 3600,
                "logical_device_hours": nodes
                * geometry.devices_per_node
                * allocation_seconds
                / 3600,
                "physical_gpu_hours": nodes
                * geometry.devices_per_node
                / geometry.devices_per_gpu
                * allocation_seconds
                / 3600,
                "busy_device_hours": busy_device_seconds / 3600,
                "busy_node_hours": busy_node_seconds / 3600,
                "idle_device_hours": max(0.0, device_capacity_seconds - busy_device_seconds) / 3600,
                "idle_dft_node_hours": max(0.0, dispatch_seconds * dft_nodes - busy_node_seconds)
                / 3600,
                "utilization_device_weighted": busy_device_seconds
                / (dispatch_seconds * nodes * geometry.devices_per_node)
                if dispatch_seconds > 0 and geometry.devices_per_node
                else None,
                "utilization_dft_node_weighted": busy_node_seconds / (dispatch_seconds * dft_nodes)
                if dispatch_seconds > 0 and dft_nodes
                else None,
            },
            "task_execution_seconds": [
                r["wall_seconds"]
                for r in this_segment
                if r.get(TERMINAL_FIELD) and r.get("wall_seconds") is not None
            ],
        }

    def _measurement(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        counts, timing = summary["counts"], summary["timing"]
        return {
            "schema_version": SCHEMA_VERSION,
            "benchmark": "e4-framework",
            "run_id": self.run_id,
            "segment": summary["segment"],
            "resumed": self.resumed,
            "experiment": self.experiment,
            "facility": self.facility,
            "policy": self.policy,
            "material_class": self.group_label,
            "nodes": self.allocation.node_count,
            "repetition": self.repetition,
            "workload_hash": self.workload.workload_hash,
            "planned_tasks": counts["planned_tasks"],
            "attempted_tasks": counts["attempted_tasks"],
            "executed_tasks": counts["executed_tasks"],
            "converged_tasks": counts["converged_tasks"],
            "valid_tasks": counts["qualified_tasks"],
            "failed_tasks": counts["failed_tasks"],
            "retried_tasks": counts["retried_tasks"],
            "total_attempts": counts["total_attempts"],
            "application_makespan_seconds": timing["application_makespan_seconds"],
            "allocation_seconds": timing["allocation_seconds"],
            "framework_initialization_seconds": timing["framework_initialization_seconds"],
            "framework_finalization_seconds": summary.get("framework_finalization_seconds"),
            "placement_seconds": timing["placement_seconds"],
            "checkpoint_provenance_seconds": timing["checkpoint_provenance_seconds"],
            "median_queue_wait_seconds": timing["queue_wait_seconds"]["median"],
            "median_launch_overhead_seconds": timing["launch_overhead_seconds"]["median"],
            "p95_launch_overhead_seconds": timing["launch_overhead_seconds"]["p95"],
            "recovery_seconds": timing["recovery_seconds"],
            "node_hours": summary["resources"]["node_hours"],
            "logical_device_hours": summary["resources"]["logical_device_hours"],
            "physical_gpu_hours": summary["resources"]["physical_gpu_hours"],
            "busy_device_hours": summary["resources"]["busy_device_hours"],
            "busy_dft_node_hours": summary["resources"]["busy_node_hours"],
            "idle_device_hours": summary["resources"]["idle_device_hours"],
            "idle_dft_node_hours": summary["resources"]["idle_dft_node_hours"],
            "utilization_device_weighted": summary["resources"]["utilization_device_weighted"],
            "utilization_dft_node_weighted": summary["resources"]["utilization_dft_node_weighted"],
            "failed_attempts": counts["failed_attempts"],
            "interrupted_attempts": counts["interrupted_attempts"],
            "failures_by_class": counts["failures_by_class"],
            "task_classes": sorted(summary["by_task_class"]),
            "by_task_class": summary["by_task_class"],
            "task_execution_seconds": summary["task_execution_seconds"],
            "source_commit": self._commit(),
            "science_config_sha256": self.workload.config_sha256,
            "resource_config_sha256": self.identity["resource_plan_sha256"],
        }

    @staticmethod
    def _commit() -> str | None:
        return _git("rev-parse", "HEAD")


def device_task_entry(item: tuple[str, str]) -> dict[str, Any]:
    """Device-slot worker entry: run one task and persist its result before returning."""

    kind, args_json = item
    args = json.loads(args_json)
    work_dir = Path(args["work_dir"])
    start = time.time()
    if kind == "controlled":
        value = device_controlled_task(args_json)
    elif kind == "mlip-relax":
        from matsim_agents.backends.mlip.pool import relax_task
        from matsim_agents.backends.mlip.relaxation import RelaxStructureInput

        relax_input = RelaxStructureInput.model_validate(
            {
                **args["params"],
                "structure_path": args["structure_path"],
                "output_dir": str(work_dir),
            }
        )
        value = {"result": relax_task(relax_input.model_dump_json())}
    else:
        raise ValueError(f"task kind {kind!r} does not run on device slots")
    value = {**value, "start_epoch": start, "end_epoch": time.time()}
    temporary = work_dir / "device_result.json.tmp"
    temporary.write_text(json.dumps(value, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(work_dir / "device_result.json")
    return value


__all__ = [
    "BenchmarkRun",
    "RunIdentityError",
    "device_task_entry",
    "environment_record",
    "local_allocation",
    "resource_plan",
]
