"""Execute an external LAMMPS on a staged run directory.

The command is ``[*launcher, executable, -in, in.lammps, -log, log.lammps]`` with ``cwd=run_dir``
and no shell. Process success needs exit code 0, ``Total wall time`` in the log and the end
marker printed by the generated script; it is not scientific success.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from matsim_agents.backends.external.lammps.inputs import INPUT_NAME, LOG_NAME
from matsim_agents.execution.slurm_jobs import JobSpec, SlurmJobError, submit_job

STDOUT_NAME = "stdout.txt"
STDERR_NAME = "stderr.txt"
RUN_RECORD = "run_record.json"
WALL_MARKER = "Total wall time"
END_MARKER = "MATSIM_LAMMPS_DONE"
LOG_TAIL = 2000


@dataclass(frozen=True)
class RunOutcome:
    mode: str
    argv: tuple[str, ...]
    cwd: str
    returncode: int | None
    timed_out: bool
    end_marker_seen: bool
    wall_seconds: float
    started_at: str
    job_id: str | None = None
    submitted: bool | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def process_succeeded(self) -> bool:
        return self.returncode == 0 and self.end_marker_seen and not self.timed_out

    def as_dict(self) -> dict:
        return {
            "tool": "lammps",
            "mode": self.mode,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "end_marker_seen": self.end_marker_seen,
            "wall_seconds": self.wall_seconds,
            "started_at": self.started_at,
            "job_id": self.job_id,
            "submitted": self.submitted,
            "process_succeeded": self.process_succeeded,
            "meaning": "process-level outcome only; not scientific success",
        }


def command(executable: Path, launcher: Sequence[str] = ()) -> tuple[str, ...]:
    return (*launcher, str(executable), "-in", INPUT_NAME, "-log", LOG_NAME)


def _markers_seen(run_dir: Path, stdout: str) -> bool:
    log = run_dir / LOG_NAME
    text = log.read_text(errors="replace") if log.is_file() else stdout
    return WALL_MARKER in text and END_MARKER in text


def write_run_record(run_dir: Path, outcome: RunOutcome) -> Path:
    path = run_dir / RUN_RECORD
    path.write_text(json.dumps(outcome.as_dict(), indent=2, sort_keys=True) + "\n")
    return path


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _text(value: str | bytes | None) -> str:
    return (value.decode(errors="replace") if isinstance(value, bytes) else value) or ""


def run_local(
    executable: Path,
    run_dir: Path,
    timeout_seconds: int,
    *,
    launcher: Sequence[str] = (),
    env: dict[str, str] | None = None,
) -> RunOutcome:
    if not (run_dir / INPUT_NAME).is_file():
        raise FileNotFoundError(f"{run_dir / INPUT_NAME} not staged; call write_inputs first")
    argv = command(executable, launcher)
    started = _now()
    t0 = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(  # noqa: S603 - argument list, no shell, explicit cwd and timeout
            list(argv),
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
        stdout, stderr, code = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout, stderr, code = _text(exc.stdout), _text(exc.stderr), None
    except OSError as exc:
        stdout, stderr, code = "", f"cannot execute {executable}: {exc}", None
    (run_dir / STDOUT_NAME).write_text(stdout)
    (run_dir / STDERR_NAME).write_text(stderr)
    outcome = RunOutcome(
        mode="local",
        argv=argv,
        cwd=str(run_dir),
        returncode=code,
        timed_out=timed_out,
        end_marker_seen=_markers_seen(run_dir, stdout),
        wall_seconds=round(time.monotonic() - t0, 3),
        started_at=started,
        stdout_tail=stdout[-LOG_TAIL:],
        stderr_tail=stderr[-LOG_TAIL:],
    )
    write_run_record(run_dir, outcome)
    return outcome


def submit_slurm(
    executable: Path,
    run_dir: Path,
    *,
    name: str,
    manifest: Path,
    account: str,
    time_limit: str = "00:30:00",
    qos: str = "debug",
    partition: str = "batch",
    launcher: Sequence[str] = ("srun",),
    runner=None,
) -> RunOutcome:
    """Submit one job through ``execution.slurm_jobs``. Idempotent by spec hash; no polling."""
    spec = JobSpec(
        name=name[:64],
        command=command(executable, launcher),
        time_limit=time_limit,
        gpus=0,
        account=account,
        qos=qos,
        partition=partition,
        cwd=str(run_dir),
    )
    kwargs = {} if runner is None else {"runner": runner}
    base = dict(
        mode="slurm",
        argv=spec.command,
        cwd=str(run_dir),
        returncode=None,
        timed_out=False,
        end_marker_seen=False,
        wall_seconds=0.0,
        started_at=_now(),
    )
    try:
        entry, submitted = submit_job(spec, manifest, **kwargs)
    except SlurmJobError as exc:
        return RunOutcome(**base, stderr_tail=str(exc)[:LOG_TAIL])
    outcome = RunOutcome(**base, job_id=entry.job_id, submitted=submitted)
    write_run_record(run_dir, outcome)
    return outcome


__all__ = [
    "END_MARKER",
    "RUN_RECORD",
    "STDERR_NAME",
    "STDOUT_NAME",
    "WALL_MARKER",
    "RunOutcome",
    "command",
    "run_local",
    "submit_slurm",
    "write_run_record",
]
