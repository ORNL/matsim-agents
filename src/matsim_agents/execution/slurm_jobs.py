"""Non-blocking Slurm submission with a persistent manifest.

Safety rules enforced here, not left to callers:

* one node, at most ``MAX_GPUS_PER_JOB`` GPUs, and a finite time limit per job;
* a job spec is hashed, and a spec already in the manifest is never submitted twice;
* at most ``MAX_OUTSTANDING_GPU_JOBS`` GPU jobs may be unfinished at once;
* a failed job may be retried once, only with a stated diagnosis, and a retry cannot be retried;
* nothing here waits: ``submit`` calls ``sbatch`` once and ``refresh`` calls ``sacct`` once.

The manifest is created at the first successful submission, never before. Commands are argument
lists; the job body is quoted with ``shlex`` and no shell is used locally. Commands run through
an injectable ``Runner`` so tests never reach a scheduler.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.core.fileio import atomic_write_text
from matsim_agents.execution.locking import exclusive_lock

MANIFEST_SCHEMA = "matsim-validation-jobs/1"
DEFAULT_MANIFEST = Path("outputs/radiation_damage/validation_jobs.json")
MAX_OUTSTANDING_GPU_JOBS = 2
MAX_GPUS_PER_JOB = 8  # the GCDs of one Frontier node
MAX_TIME_SECONDS = 2 * 3600  # the debug QoS wall-time limit on this system
TERMINAL_OK = frozenset({"COMPLETED"})
TERMINAL_FAILED = frozenset(
    {
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "BOOT_FAIL",
        "DEADLINE",
        "PREEMPTED",
    }
)
_DIAGNOSES = {
    "TIMEOUT": "the job reached its time limit",
    "OUT_OF_MEMORY": "the job ran out of memory",
    "NODE_FAIL": "the node failed",
    "BOOT_FAIL": "the node failed to boot",
    "PREEMPTED": "the job was preempted",
    "DEADLINE": "the job missed its deadline",
    "CANCELLED": "the job was cancelled",
    "FAILED": "the job exited with a non-zero status",
}
_TIME = re.compile(r"^(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})$")


class SlurmJobError(RuntimeError):
    """A submission rule was violated or the scheduler call failed."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[Sequence[str]], CommandResult]


def subprocess_runner(argv: Sequence[str]) -> CommandResult:
    """Run a scheduler command without a shell and with a timeout."""
    done = subprocess.run(  # noqa: S603 - argument list, no shell
        list(argv), capture_output=True, text=True, timeout=60, check=False
    )
    return CommandResult(done.returncode, done.stdout, done.stderr)


def parse_time_limit(text: str) -> int:
    """Seconds in ``[D-]HH:MM:SS``; anything else (including ``UNLIMITED``) is rejected."""
    match = _TIME.match(text)
    if match is None:
        raise ValueError(f"time limit {text!r} is not [D-]HH:MM:SS")
    days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
    if minutes > 59 or seconds > 59:
        raise ValueError(f"time limit {text!r} has an out-of-range field")
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


class JobSpec(BaseModel):
    """One validation job. Frozen so its hash is stable."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    command: tuple[str, ...] = Field(min_length=1)
    time_limit: str = "00:15:00"
    gpus: int = Field(default=0, ge=0)
    nodes: int = 1
    account: str = Field(min_length=1)
    qos: str = "debug"
    partition: str = "batch"
    cwd: str | None = None

    @model_validator(mode="after")
    def _limits(self) -> JobSpec:
        seconds = parse_time_limit(self.time_limit)
        if not 0 < seconds <= MAX_TIME_SECONDS:
            raise ValueError(f"time limit must be within (0, {MAX_TIME_SECONDS}] seconds")
        if self.nodes != 1:
            raise ValueError("validation jobs use exactly one node")
        if self.gpus > MAX_GPUS_PER_JOB:
            raise ValueError(f"validation jobs use at most {MAX_GPUS_PER_JOB} GPUs")
        if any(not part for part in self.command):
            raise ValueError("command parts must be non-empty")
        return self

    @property
    def spec_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class JobEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    spec_hash: str
    spec: JobSpec
    job_id: str
    state: str = "SUBMITTED"
    submitted_at: str
    retry_of: str | None = None
    retried: bool = False
    diagnosis: str | None = None
    exit_code: str | None = None
    elapsed: str | None = None
    log_dir: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_OK or self.state in TERMINAL_FAILED

    @property
    def outstanding_gpu(self) -> bool:
        return self.spec.gpus > 0 and not self.terminal


def build_sbatch_argv(spec: JobSpec, log_dir: str | None = None) -> list[str]:
    """The ``sbatch`` argument list for ``spec``. The body is one quoted ``--wrap`` string."""
    argv = [
        "sbatch",
        "--parsable",
        "-A",
        spec.account,
        "-q",
        spec.qos,
        "-p",
        spec.partition,
        "-N",
        str(spec.nodes),
        "-t",
        spec.time_limit,
        "-J",
        spec.name,
    ]
    if spec.gpus:
        argv += [f"--gpus-per-node={spec.gpus}", "--gpu-bind=closest"]
    if spec.cwd:
        argv += ["--chdir", spec.cwd]
    if log_dir:
        argv += ["-o", f"{log_dir}/%x-%j.out", "-e", f"{log_dir}/%x-%j.err"]
    argv += ["--wrap", shlex.join(spec.command)]
    return argv


def load_manifest(path: str | os.PathLike) -> list[JobEntry]:
    """Entries in the manifest; a missing manifest is an empty list."""
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if data.get("schema") != MANIFEST_SCHEMA:
        raise SlurmJobError(f"{path} is not a {MANIFEST_SCHEMA} manifest")
    return [JobEntry.model_validate(item) for item in data.get("jobs", [])]


def _write_manifest(path: Path, entries: Sequence[JobEntry]) -> None:
    payload = {
        "schema": MANIFEST_SCHEMA,
        "jobs": [e.model_dump(mode="json") for e in entries],
    }
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def _parse_job_id(stdout: str) -> str:
    token = stdout.strip().splitlines()[-1].split(";")[0].strip() if stdout.strip() else ""
    if not token.isdigit():
        raise SlurmJobError(f"sbatch did not return a job id: {stdout.strip()[:120]!r}")
    return token


def _submit_locked(
    spec: JobSpec,
    path: Path,
    runner: Runner,
    entries: list[JobEntry],
    *,
    retry_of: str | None,
    diagnosis: str | None,
) -> JobEntry:
    log_dir = str(path.parent / "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    result = runner(build_sbatch_argv(spec, log_dir))
    if result.returncode != 0:
        raise SlurmJobError(f"sbatch failed ({result.returncode}): {result.stderr.strip()[:200]}")
    entry = JobEntry(
        spec_hash=spec.spec_hash,
        spec=spec,
        job_id=_parse_job_id(result.stdout),
        submitted_at=datetime.now(UTC).isoformat(timespec="seconds"),
        retry_of=retry_of,
        diagnosis=diagnosis,
        log_dir=log_dir,
    )
    entries.append(entry)
    return entry


def submit_job(
    spec: JobSpec,
    manifest: str | os.PathLike = DEFAULT_MANIFEST,
    *,
    runner: Runner = subprocess_runner,
    max_outstanding_gpu_jobs: int = MAX_OUTSTANDING_GPU_JOBS,
) -> tuple[JobEntry, bool]:
    """Submit ``spec`` once. Returns ``(entry, submitted)``; ``submitted`` is False when the
    same spec is already in the manifest, in which case nothing is sent to the scheduler.
    A caller that queues a batch of independent jobs passes its own explicit cap."""
    path = Path(manifest)
    with exclusive_lock(_lock_path(path), timeout=30):
        entries = load_manifest(path)
        for existing in entries:
            if existing.spec_hash == spec.spec_hash:
                return existing, False
        if spec.gpus:
            outstanding = [e.job_id for e in entries if e.outstanding_gpu]
            if len(outstanding) >= max_outstanding_gpu_jobs:
                raise SlurmJobError(
                    f"{len(outstanding)} GPU jobs are outstanding "
                    f"(limit {max_outstanding_gpu_jobs}): {', '.join(outstanding)}; "
                    "refresh the manifest first if some have finished"
                )
        entry = _submit_locked(spec, path, runner, entries, retry_of=None, diagnosis=None)
        _write_manifest(path, entries)
        return entry, True


def diagnose(entry: JobEntry) -> str | None:
    """A short diagnosis for a failed job, or None when it has not failed."""
    return _DIAGNOSES.get(entry.state)


def retry_job(
    spec_hash: str,
    manifest: str | os.PathLike = DEFAULT_MANIFEST,
    *,
    diagnosis: str,
    spec: JobSpec | None = None,
    runner: Runner = subprocess_runner,
    max_outstanding_gpu_jobs: int = MAX_OUTSTANDING_GPU_JOBS,
) -> JobEntry:
    """Resubmit a failed job once. ``spec`` may change what the diagnosis blames (for example
    a longer time limit); an identical spec is allowed because the failed entry is marked."""
    if not diagnosis.strip():
        raise SlurmJobError("a retry needs a diagnosis of the failure")
    path = Path(manifest)
    with exclusive_lock(_lock_path(path), timeout=30):
        entries = load_manifest(path)
        index = next((i for i, e in enumerate(entries) if e.spec_hash == spec_hash), None)
        if index is None:
            raise SlurmJobError(f"no job with spec hash {spec_hash} in {path}")
        old = entries[index]
        if old.state not in TERMINAL_FAILED:
            raise SlurmJobError(f"job {old.job_id} is {old.state}; only failed jobs are retried")
        if old.retried or old.retry_of is not None:
            raise SlurmJobError(f"job {old.job_id} was already retried or is itself a retry")
        new_spec = spec or old.spec
        if new_spec.gpus and len([e for e in entries if e.outstanding_gpu]) >= (
            max_outstanding_gpu_jobs
        ):
            raise SlurmJobError("the outstanding GPU job limit is reached")
        entries[index] = old.model_copy(update={"retried": True, "diagnosis": diagnosis.strip()})
        if new_spec.spec_hash == old.spec_hash:
            entries.pop(index)  # keep the hash unique; history lives in the retry's retry_of
        entry = _submit_locked(
            new_spec,
            path,
            runner,
            entries,
            retry_of=old.spec_hash,
            diagnosis=diagnosis.strip(),
        )
        _write_manifest(path, entries)
        return entry


def refresh_manifest(
    manifest: str | os.PathLike = DEFAULT_MANIFEST, *, runner: Runner = subprocess_runner
) -> list[JobEntry]:
    """Update unfinished entries from one ``sacct`` call. Never loops or waits."""
    path = Path(manifest)
    if not path.exists():
        return []
    with exclusive_lock(_lock_path(path), timeout=30):
        entries = load_manifest(path)
        pending = [e for e in entries if not e.terminal]
        if not pending:
            return entries
        result = runner(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                ",".join(e.job_id for e in pending),
                "-o",
                "JobID,State,ExitCode,Elapsed",
            ]
        )
        if result.returncode != 0:
            raise SlurmJobError(
                f"sacct failed ({result.returncode}): {result.stderr.strip()[:200]}"
            )
        rows: dict[str, list[str]] = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 4:
                rows[parts[0]] = parts
        updated: list[JobEntry] = []
        for entry in entries:
            row = rows.get(entry.job_id)
            if row is None or entry.terminal:
                updated.append(entry)
                continue
            state = row[1].split()[0] if row[1] else entry.state
            new = entry.model_copy(update={"state": state, "exit_code": row[2], "elapsed": row[3]})
            if new.state in TERMINAL_FAILED and new.diagnosis is None:
                new = new.model_copy(update={"diagnosis": diagnose(new)})
            updated.append(new)
        _write_manifest(path, updated)
        return updated


def summarize(entries: Sequence[JobEntry]) -> dict[str, Any]:
    """Counts for reporting: submitted, completed, failed, outstanding."""
    return {
        "submitted": len(entries),
        "completed": sum(e.state in TERMINAL_OK for e in entries),
        "failed": sum(e.state in TERMINAL_FAILED for e in entries),
        "outstanding": sum(not e.terminal for e in entries),
        "outstanding_gpu": sum(e.outstanding_gpu for e in entries),
    }


__all__ = [
    "DEFAULT_MANIFEST",
    "MANIFEST_SCHEMA",
    "MAX_OUTSTANDING_GPU_JOBS",
    "CommandResult",
    "JobEntry",
    "JobSpec",
    "Runner",
    "SlurmJobError",
    "build_sbatch_argv",
    "diagnose",
    "load_manifest",
    "parse_time_limit",
    "refresh_manifest",
    "retry_job",
    "submit_job",
    "subprocess_runner",
    "summarize",
]
