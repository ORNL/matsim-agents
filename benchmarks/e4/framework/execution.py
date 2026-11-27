"""Execution adapters that plug E4 tasks into the production dispatchers.

Node-group tasks run through :func:`matsim_agents.backends.dft.runner.run_dft_batch`
via :class:`RetryingTaskBackend`, which implements the production ``DFTBackend``
protocol. Each attempt is classified and retried with the production recovery
policy (:mod:`matsim_agents.execution.recovery`) on the same node group.

Device-slot tasks run through :class:`matsim_agents.backends.mlip.pool.MLIPWorkerPool`.
As in production, device-pool tasks are not retried.

Qualification reuses the production acceptance rules: a DFT label must be
converged with finite, correctly shaped forces at the unchanged geometry
(``active_learning.labelling._audit_force_errors``); an MLIP relaxation must be
converged with finite energy and a maximum force at or below ``fmax``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from benchmarks.e4.framework.payload import expected_digest
from benchmarks.e4.framework.records import TERMINAL_FIELD, AttemptRecorder, TaskHistory
from benchmarks.e4.framework.workload import TaskSpec, sha256_file
from matsim_agents.backends.dft.base import DFTJobSpec, DFTResult
from matsim_agents.execution.placement import ProcessLaunch
from matsim_agents.execution.recovery import (
    DEFAULT_POLICY,
    WALLTIME_RETURN_CODE,
    FailureClass,
    adjusted_settings,
    can_retry,
    classify_failure,
)

REPO = Path(__file__).resolve().parents[3]
PAYLOAD = Path(__file__).resolve().with_name("payload.py")
LOCAL_STEP = REPO / "deployments/common/e4-task-step-local.sh"
FACILITY_STEPS = {
    "aurora": REPO / "deployments/aurora/launchers/_e4-task-step-aurora.sh",
    "frontier": REPO / "deployments/frontier/launchers/_e4-task-step-frontier.sh",
    "perlmutter": REPO / "deployments/perlmutter/launchers/_e4-task-step-perlmutter.sh",
}
RECOVERY_POLICY_VERSION = DEFAULT_POLICY.version
_STDERR_TAIL = 2000


def step_launcher(facility: str) -> Path:
    """The step script that launches a controlled task on its assigned nodes."""

    if facility == "local":
        return LOCAL_STEP
    try:
        return FACILITY_STEPS[facility]
    except KeyError:
        raise ValueError(f"unknown facility {facility!r}") from None


class AttemptExecutor(Protocol):
    """Runs one attempt of one task on one launch and returns its outcome."""

    def __call__(
        self,
        task: TaskSpec,
        launch: ProcessLaunch,
        *,
        attempt: int,
        work_dir: Path,
        settings: dict[str, Any],
        timeout_sec: float | None,
    ) -> dict[str, Any]: ...


def _tail(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(errors="replace")[-_STDERR_TAIL:]


class ControlledStepExecutor:
    """Launch the deterministic payload through a facility step script."""

    def __init__(self, launcher: Path, *, local: bool, python: str = sys.executable) -> None:
        self.launcher = Path(launcher)
        self.local = local
        self.python = python

    def __call__(
        self,
        task: TaskSpec,
        launch: ProcessLaunch,
        *,
        attempt: int,
        work_dir: Path,
        settings: dict[str, Any],
        timeout_sec: float | None,
    ) -> dict[str, Any]:
        # The launcher may change directory; relative paths would then miss the attempt.
        work_dir = Path(work_dir).resolve()
        command = [
            "bash",
            str(self.launcher),
            str(work_dir),
            *launch.wrapper_args(),
            "--",
            self.python,
            str(PAYLOAD),
            "--task-id",
            task.task_id,
            "--work-dir",
            str(work_dir),
            "--seconds",
            str(task.seconds),
            "--artifact-bytes",
            str(task.artifact_bytes),
            "--seed",
            str(task.seed),
            "--attempt",
            str(attempt),
            "--fail-attempts",
            str(task.fail_attempts),
            "--failure-mode",
            task.failure_mode,
        ]
        env = launch.environment(local=self.local)
        stdout, stderr = work_dir / "step.out", work_dir / "step.err"
        try:
            with stdout.open("w") as out, stderr.open("w") as err:
                completed = subprocess.run(
                    command, env=env, stdout=out, stderr=err, timeout=timeout_sec, check=False
                )
            return_code = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            return_code, timed_out = WALLTIME_RETURN_CODE, True
        return self._outcome(task, attempt, work_dir, return_code, timed_out, stderr)

    @staticmethod
    def _outcome(
        task: TaskSpec,
        attempt: int,
        work_dir: Path,
        return_code: int,
        timed_out: bool,
        stderr: Path,
    ) -> dict[str, Any]:
        result_path = work_dir / "result.json"
        result = json.loads(result_path.read_text()) if result_path.is_file() else None
        outcome: dict[str, Any] = {
            "return_code": return_code,
            "executed": return_code == 0,
            "payload_seconds": None if result is None else result.get("seconds"),
            "output_sha256": None if result is None else result.get("sha256"),
            "scf_converged": None,
            "steps": None,
        }
        if timed_out:
            reason = "task timed out"
        elif return_code != 0:
            reason = f"step exited with code {return_code}: {_tail(stderr).strip()}"
        elif result is None:
            reason = "no stage result written"
        elif not math.isfinite(float(result["metric"])):
            reason = "non-finite metric"
        else:
            reason = None
        converged = reason is None
        checks = {
            "result_written": result is not None,
            "identity_matches": bool(
                result and result["task_id"] == task.task_id and result["attempt"] == attempt
            ),
            "digest_matches": bool(
                result
                and result["sha256"]
                == expected_digest(task.task_id, task.seed, task.artifact_bytes)
            ),
            "metric_finite": bool(result and math.isfinite(float(result["metric"]))),
            "artifact_matches": _artifact_matches(task, work_dir / "artifact.bin"),
        }
        outcome.update(
            converged=converged,
            qualified=converged and all(checks.values()),
            qualification_checks=checks,
            status="done" if converged else "failed",
            failure_reason=reason,
        )
        return outcome


def _artifact_matches(task: TaskSpec, artifact: Path) -> bool:
    """The on-disk artifact exists and hashes to the deterministic digest."""

    return artifact.is_file() and sha256_file(artifact) == expected_digest(
        task.task_id, task.seed, task.artifact_bytes
    )


def _forces_ok(result: DFTResult, n_atoms: int) -> bool:
    if result.forces_eV_per_A is None:
        return False
    import numpy as np

    forces = np.asarray(result.forces_eV_per_A, dtype=float)
    return forces.shape == (n_atoms, 3) and bool(np.isfinite(forces).all())


def _max_force(result: DFTResult) -> float | None:
    if result.forces_eV_per_A is None:
        return None
    import numpy as np

    forces = np.asarray(result.forces_eV_per_A, dtype=float)
    if forces.size == 0 or not np.isfinite(forces).all():
        return None
    return float(np.linalg.norm(forces, axis=1).max())


def _file_hashes(work_dir: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: sha256_file(work_dir / name) for name in names if (work_dir / name).is_file()}


def _geometry_unchanged(source, result: DFTResult) -> bool:
    import numpy as np

    labelled = result.final_atoms
    tolerance = 1e-5 if result.backend == "qe" else 1e-8
    return bool(
        labelled is not None
        and np.array_equal(source.numbers, labelled.numbers)
        and np.allclose(source.cell, labelled.cell, rtol=0, atol=tolerance)
        and np.allclose(source.positions, labelled.positions, rtol=0, atol=tolerance)
    )


def _vasp_overrides(settings: Mapping[str, Any]) -> dict[str, str]:
    keys = {"nelm": "NELM", "algo": "ALGO"}
    return {keys[key]: str(value) for key, value in settings.items() if key in keys}


class DFTLabelExecutor:
    """Run one production DFT single point and apply the AL label acceptance rule."""

    def __init__(
        self, dft_config: Mapping[str, Any], backend_factory: Callable[[Any], Any] | None = None
    ) -> None:
        from matsim_agents.backends.dft.config import DFTConfig

        self.config = DFTConfig.model_validate(dict(dft_config))
        # Backends launch from the task directory, so repo-relative paths must be absolute.
        repo = Path(__file__).resolve().parents[3]
        block = getattr(self.config, self.config.backend)
        for name, value in block:
            if isinstance(value, Path) and not value.is_absolute():
                setattr(block, name, repo / value)
        if backend_factory is None:
            from matsim_agents.backends.dft.base import make_backend

            backend_factory = make_backend
        self.backend_factory = backend_factory

    @property
    def backend_name(self) -> str:
        return self.config.backend

    @property
    def timeout_sec(self) -> float:
        block = self.config.vasp if self.config.backend == "vasp" else self.config.qe
        return float(block.timeout_sec)

    def _backend(self, settings: Mapping[str, Any], timeout_sec: float | None):
        data = self.config.model_dump()
        block = data[self.config.backend]
        if timeout_sec is not None:
            block["timeout_sec"] = int(timeout_sec)
        if self.config.backend == "qe":
            if "mixing_beta" in settings:
                block["extra_electrons"] = {
                    **block.get("extra_electrons", {}),
                    "mixing_beta": settings["mixing_beta"],
                }
            for key, value in settings.get("extra_electrons", {}).items():
                block.setdefault("extra_electrons", {})[key] = value
        return self.backend_factory(type(self.config).model_validate(data))

    def __call__(
        self,
        task: TaskSpec,
        launch: ProcessLaunch,
        *,
        attempt: int,
        work_dir: Path,
        settings: dict[str, Any],
        timeout_sec: float | None,
    ) -> dict[str, Any]:
        from ase.io import read

        atoms = read(task.structure_path)
        if sha256_file(task.structure_path) != task.structure_sha256:
            raise ValueError(f"{task.task_id}: input structure hash changed")
        backend = self._backend(settings, timeout_sec)
        extra = _vasp_overrides(settings) if self.config.backend == "vasp" else None
        spec = DFTJobSpec(
            job_id=task.task_id, atoms=atoms, work_dir=str(work_dir), extra=extra, launch=launch
        )
        result = backend.run_one(spec)
        energy = result.energy_eV
        energy_ok = energy is not None and math.isfinite(float(energy))
        checks = {
            "converged": bool(result.converged),
            "return_code_zero": result.return_code == 0,
            "energy_finite": energy_ok,
            "forces_finite_and_shaped": _forces_ok(result, len(atoms)),
            "geometry_unchanged": _geometry_unchanged(atoms, result),
        }
        converged = bool(result.converged and result.return_code == 0)
        reason = None
        if result.return_code == WALLTIME_RETURN_CODE or "timed out" in (result.notes or ""):
            reason = "timed out"
        elif not converged:
            reason = result.notes or "did not converge"
        elif not energy_ok:
            reason = "non-finite energy"
        return {
            "return_code": result.return_code,
            "executed": result.return_code == 0,
            "converged": converged,
            "qualified": all(checks.values()),
            "qualification_checks": checks,
            "status": "done" if converged and energy_ok else "failed",
            "scf_converged": bool(result.converged),
            "steps": None,
            "failure_reason": reason,
            "energy_eV": float(energy) if energy_ok else None,
            "energy_eV_per_atom": float(energy) / len(atoms) if energy_ok else None,
            "max_force_eV_per_A": _max_force(result),
            "backend_wall_seconds": result.wall_time_sec,
            "input_sha256": _file_hashes(work_dir, ("pw.in", "INCAR", "KPOINTS", "POSCAR")),
            "output_sha256": _file_hashes(work_dir, ("pw.out", "OUTCAR", "vasprun.xml")) or None,
        }


def retry_decision(
    outcome: Mapping[str, Any], retries: Mapping[str, int], timeout_sec: float | None
) -> tuple[Any, bool, str | None]:
    """``(failure_class, retry, action)`` for one attempt under the production policy."""

    if outcome["qualified"]:
        return None, False, None
    failure = classify_failure(dict(outcome))
    rule = DEFAULT_POLICY.rule(failure)
    used = retries.get(failure.value, 0)
    retry = can_retry(failure, timeout_sec=timeout_sec) and used < rule.max_retries
    return failure, retry, rule.action


class RetryingTaskBackend:
    """A production ``DFTBackend`` that runs E4 tasks with recorded, policy-driven retries."""

    def __init__(
        self,
        executor: AttemptExecutor,
        tasks: Mapping[str, TaskSpec],
        recorder: AttemptRecorder,
        *,
        run_dir: Path,
        histories: Mapping[str, TaskHistory] | None = None,
        timeout_sec: float | None = None,
        settings: Mapping[str, Any] | None = None,
        backend: str = "controlled",
        dispatch_origin: float | None = None,
    ) -> None:
        self.name = f"e4-{backend}"
        self.backend = backend
        self.executor = executor
        self.tasks = dict(tasks)
        self.recorder = recorder
        self.run_dir = Path(run_dir)
        self.histories = dict(histories or {})
        self.timeout_sec = timeout_sec
        self.settings = dict(settings or {})
        self.dispatch_origin = time.monotonic() if dispatch_origin is None else dispatch_origin

    def _attempt_dir(self, task_id: str, attempt: int) -> Path:
        path = self.run_dir / "tasks" / task_id / f"attempt-{attempt:02d}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def run_one(self, spec: DFTJobSpec) -> DFTResult:
        task = self.tasks[spec.job_id]
        launch = spec.require_launch()
        history = self.histories.get(task.task_id, TaskHistory())
        retries = dict(history.retries_by_failure)
        attempt = history.attempts
        settings, timeout = self._resumed_settings(history)
        queued_seconds = self.recorder.offset() - self.recorder.offset(self.dispatch_origin)
        while True:
            attempt += 1
            work_dir = self._attempt_dir(task.task_id, attempt)
            started = time.monotonic()
            try:
                outcome = self.executor(
                    task,
                    launch,
                    attempt=attempt,
                    work_dir=work_dir,
                    settings=settings,
                    timeout_sec=timeout,
                )
            except Exception as exc:  # noqa: BLE001 - a broken attempt is a recorded failure
                outcome = {
                    "return_code": -1,
                    "executed": False,
                    "converged": False,
                    "qualified": False,
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure_reason": str(exc),
                }
            ended = time.monotonic()
            failure, retry, rule_action = retry_decision(outcome, retries, timeout)
            record = {
                "record": "attempt",
                "task_id": task.task_id,
                "task_class": task.task_class,
                "kind": task.kind,
                "pool": task.pool,
                "attempt": attempt,
                "hosts": list(launch.hosts),
                "resources": launch.allocation(),
                "start_offset_seconds": self.recorder.offset(started),
                "end_offset_seconds": self.recorder.offset(ended),
                "wall_seconds": round(ended - started, 6),
                "queued_seconds": round(queued_seconds, 6)
                if attempt == history.attempts + 1
                else 0.0,
                "outcome": {key: value for key, value in outcome.items() if key != "status"},
                "status": outcome["status"],
                "failure_class": None if failure is None else failure.value,
                "recovery_action": rule_action,
                "recovery_policy": RECOVERY_POLICY_VERSION,
                "retry_scheduled": retry,
                "settings": settings,
                "timeout_sec": timeout,
                TERMINAL_FIELD: not retry,
            }
            self.recorder.append(record)
            if not retry:
                return self._result(task, work_dir, outcome, ended - started)
            retries[failure.value] = retries.get(failure.value, 0) + 1
            settings, timeout = adjusted_settings(failure, self.backend, settings, timeout)

    def _resumed_settings(self, history: TaskHistory) -> tuple[dict[str, Any], float | None]:
        """Settings and timeout for the next attempt, replaying recorded recovery."""

        record = history.last_settings_record
        if record is None:
            return dict(self.settings), self.timeout_sec
        settings = dict(record.get("settings") or {})
        timeout = record.get("timeout_sec")
        if record.get("retry_scheduled") and record.get("failure_class"):
            settings, timeout = adjusted_settings(
                FailureClass(record["failure_class"]), self.backend, settings, timeout
            )
        return settings, timeout

    def _result(
        self, task: TaskSpec, work_dir: Path, outcome: Mapping[str, Any], seconds: float
    ) -> DFTResult:
        return DFTResult(
            backend=self.name,
            work_dir=str(work_dir),
            return_code=int(outcome.get("return_code", -1)),
            converged=bool(outcome["qualified"]),
            energy_eV=outcome.get("energy_eV"),
            forces_eV_per_A=None,
            stress_eV_per_A3=None,
            n_atoms=None,
            wall_time_sec=seconds,
            final_atoms=None,
            notes=outcome.get("failure_reason"),
        )


def device_controlled_task(payload_json: str) -> dict[str, Any]:
    """Worker-process entry point: run one controlled payload on the worker's device."""

    from benchmarks.e4.framework import payload

    args = json.loads(payload_json)
    work_dir = Path(args["work_dir"])
    started = time.monotonic()
    return_code = payload.main(
        [
            "--task-id",
            args["task_id"],
            "--work-dir",
            str(work_dir),
            "--seconds",
            str(args["seconds"]),
            "--artifact-bytes",
            str(args["artifact_bytes"]),
            "--seed",
            str(args["seed"]),
            "--attempt",
            str(args.get("attempt", 1)),
            "--fail-attempts",
            str(args["fail_attempts"]),
            "--failure-mode",
            args["failure_mode"],
        ]
    )
    return {
        "return_code": return_code,
        "seconds": time.monotonic() - started,
        "pid": os.getpid(),
        "mask": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "ZE_AFFINITY_MASK")
            if os.environ.get(key) is not None
        },
    }


def device_task_outcome(
    task: TaskSpec, work_dir: Path, value: Mapping[str, Any] | None, error: BaseException | None
) -> dict[str, Any]:
    """Qualify a device-slot controlled task exactly like a node-group one."""

    if error is not None:
        return {
            "return_code": -1,
            "executed": False,
            "converged": False,
            "qualified": False,
            "status": "failed",
            "failure_type": type(error).__name__,
            "failure_reason": str(error),
        }
    attempt = int(work_dir.name.split("-")[1])
    outcome = ControlledStepExecutor._outcome(
        task, attempt, work_dir, int(value["return_code"]), False, work_dir / "missing.err"
    )
    outcome["device_mask"] = value.get("mask")
    return outcome


def mlip_relax_outcome(result: Any, fmax: float) -> dict[str, Any]:
    """Qualify an MLIP relaxation with the production convergence rule."""

    energy = float(result.final_energy_eV)
    force = float(result.final_max_force_eV_per_A)
    checks = {
        "converged": bool(result.converged),
        "energy_finite": math.isfinite(energy),
        "max_force_finite": math.isfinite(force),
        "max_force_within_fmax": math.isfinite(force) and force <= fmax,
    }
    converged = bool(result.converged)
    structure = Path(result.optimized_structure_path)
    n_atoms = None
    if structure.is_file():
        from ase.io import read

        n_atoms = len(read(structure))
    return {
        "return_code": 0,
        "executed": True,
        "converged": converged,
        "qualified": all(checks.values()),
        "qualification_checks": checks,
        "status": "done" if converged else "failed",
        "failure_reason": None if converged else (result.termination_reason or "not converged"),
        "energy_eV": energy if math.isfinite(energy) else None,
        "max_force_eV_per_A": force if math.isfinite(force) else None,
        "energy_eV_per_atom": energy / n_atoms if n_atoms and math.isfinite(energy) else None,
        "steps": result.num_steps,
        "output_sha256": sha256_file(structure) if structure.is_file() else None,
    }


__all__ = [
    "AttemptExecutor",
    "ControlledStepExecutor",
    "DFTLabelExecutor",
    "RetryingTaskBackend",
    "device_controlled_task",
    "device_task_outcome",
    "mlip_relax_outcome",
    "retry_decision",
    "step_launcher",
]
