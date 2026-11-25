"""Execute an externally installed SPECTRA-PKA.

The command is ``[executable, input_name]`` run with ``cwd=run_dir``; there is no shell and no
invented flag. Process success (exit code 0 plus the ``END as expected`` marker) is not
scientific success: parsing and validation are separate steps.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from matsim_agents.backends.external.spectra_pka.availability import SUCCESS_MARKER
from matsim_agents.backends.external.spectra_pka.config import ExecutionConfig
from matsim_agents.execution.slurm_jobs import JobSpec, SlurmJobError, submit_job

STDOUT_NAME = "stdout.txt"
STDERR_NAME = "stderr.txt"
RUN_RECORD = "run_record.json"
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


def _marker_in(run_dir: Path, stdout: str, results_stub: str) -> bool:
    if SUCCESS_MARKER in stdout:
        return True
    log = run_dir / f"{results_stub}.log"
    if log.is_file():
        try:
            return SUCCESS_MARKER in log.read_text(errors="replace")
        except OSError:
            return False
    return False


def write_run_record(run_dir: Path, outcome: RunOutcome) -> Path:
    path = run_dir / RUN_RECORD
    path.write_text(json.dumps(outcome.as_dict(), indent=2, sort_keys=True) + "\n")
    return path


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_local(
    executable: Path, input_name: str, run_dir: Path, results_stub: str, timeout_seconds: int
) -> RunOutcome:
    argv = (str(executable), input_name)
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
        )
        stdout, stderr, code = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (
            exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        ) or ""
        stderr = (
            exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        ) or ""
        code = None
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
        end_marker_seen=_marker_in(run_dir, stdout, results_stub),
        wall_seconds=round(time.monotonic() - t0, 3),
        started_at=started,
        stdout_tail=stdout[-LOG_TAIL:],
        stderr_tail=stderr[-LOG_TAIL:],
    )
    write_run_record(run_dir, outcome)
    return outcome


def submit_slurm(
    executable: Path,
    input_name: str,
    run_dir: Path,
    cfg: ExecutionConfig,
    *,
    name: str,
    manifest: Path,
    runner=None,
) -> RunOutcome:
    """Submit one job through ``execution.slurm_jobs``. Idempotent by spec hash; no polling."""
    spec = JobSpec(
        name=name[:64],
        command=(str(executable), input_name),
        time_limit=cfg.time_limit or "00:30:00",
        gpus=0,
        account=cfg.account or "",
        qos=cfg.qos or "debug",
        partition=cfg.partition or "batch",
        cwd=str(run_dir),
    )
    kwargs = {} if runner is None else {"runner": runner}
    try:
        entry, submitted = submit_job(spec, manifest, **kwargs)
    except SlurmJobError as exc:
        return RunOutcome(
            mode="slurm",
            argv=spec.command,
            cwd=str(run_dir),
            returncode=None,
            timed_out=False,
            end_marker_seen=False,
            wall_seconds=0.0,
            started_at=_now(),
            stderr_tail=str(exc)[:LOG_TAIL],
        )
    outcome = RunOutcome(
        mode="slurm",
        argv=spec.command,
        cwd=str(run_dir),
        returncode=None,
        timed_out=False,
        end_marker_seen=False,
        wall_seconds=0.0,
        started_at=_now(),
        job_id=entry.job_id,
        submitted=submitted,
    )
    write_run_record(run_dir, outcome)
    return outcome


__all__ = [
    "RUN_RECORD",
    "STDERR_NAME",
    "STDOUT_NAME",
    "RunOutcome",
    "run_local",
    "submit_slurm",
    "write_run_record",
]
