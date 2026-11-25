"""Run ``neb.x`` and record exactly what was executed.

The command is always ``<launcher> neb.x [-nimage N] [-npool N] [extra] -inp <input>``; ``neb.x``
does not read standard input. This module only launches and records. A zero exit code is *not*
convergence: the record's ``status`` says whether the process ran and how it ended, and
``convergence_assessed`` is always false here. Deciding convergence is the parser's job.

``run_launcher`` merges stderr into stdout, so the single log file holds both streams; the
record states this instead of pretending to hold a separate stderr capture. An existing log is
never overwritten: a rerun must choose a new ``stdout_name`` (the staged workflow does this).

Nothing site-specific lives here: scheduler and MPI wrappers come in through
``QENebConfig.launcher``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from matsim_agents.backends.dft.process import TIMEOUT_RETURN_CODE, run_launcher
from matsim_agents.backends.dft.qe_neb import QENebConfig

RunStatus = Literal["not_started", "exited_zero", "exited_nonzero", "timed_out"]
RECORD_SUFFIX = ".record.json"

Launcher = Callable[..., int]


class NebRunRecord(BaseModel):
    """What was run and how the process ended; says nothing about NEB convergence."""

    model_config = ConfigDict(frozen=True)

    argv: tuple[str, ...]
    executable: str
    cwd: str
    input_path: str
    input_sha256: str
    config_hash: str
    stdout_path: str
    stdout_sha256: str | None
    streams: str = "stderr merged into stdout"
    started_at: str
    finished_at: str
    exit_code: int | None
    status: RunStatus
    convergence_assessed: bool = False
    error: str | None = None


def build_neb_argv(config: QENebConfig, input_path: str | os.PathLike, cwd: str) -> list[str]:
    """The full argument vector; the input is passed with ``-inp``, never on stdin."""
    target = Path(input_path).resolve()
    try:
        shown = str(target.relative_to(Path(cwd).resolve()))
    except ValueError:
        shown = str(target)
    argv = [*config.launcher, config.neb_executable]
    if config.nimage > 1:
        argv += ["-nimage", str(config.nimage)]
    if config.npool > 1:
        argv += ["-npool", str(config.npool)]
    argv += list(config.extra_args)
    argv += ["-inp", shown]
    return argv


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def write_run_record(record: NebRunRecord, path: str | os.PathLike) -> None:
    """Atomically write ``record`` as JSON."""
    target = Path(path)
    handle, staging = tempfile.mkstemp(prefix=".neb-record-", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(record.model_dump(mode="json"), stream, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(staging, target)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise


def run_neb(
    config: QENebConfig,
    input_path: str | os.PathLike,
    *,
    timeout_sec: float | None = None,
    env: Mapping[str, str] | None = None,
    stdout_name: str = "neb.out",
    launcher: Launcher = run_launcher,
) -> NebRunRecord:
    """Execute ``neb.x`` in ``config.workdir`` and return (and save) the run record.

    Raises ``FileNotFoundError`` for a missing input and ``FileExistsError`` if the log already
    exists. A process that cannot start yields a ``not_started`` record, not an exception.
    """
    input_file = Path(input_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"neb input {input_file} does not exist")
    cwd = Path(config.workdir).resolve()
    cwd.mkdir(parents=True, exist_ok=True)
    stdout_path = cwd / stdout_name
    record_path = cwd / (stdout_name + RECORD_SUFFIX)
    if stdout_path.exists() or record_path.exists():
        raise FileExistsError(f"{stdout_path} already exists; earlier output is never overwritten")
    if timeout_sec is not None and not timeout_sec > 0:
        raise ValueError("timeout_sec must be positive")

    argv = build_neb_argv(config, input_file, str(cwd))
    input_hash = _sha256(input_file)
    started = _now()
    exit_code: int | None = None
    error: str | None = None
    status: RunStatus
    try:
        with open(stdout_path, "x", encoding="utf-8") as out:
            exit_code = launcher(
                argv,
                cwd=str(cwd),
                stdout=out,
                timeout_sec=timeout_sec,
                env=None if env is None else dict(env),
            )
    except OSError as exc:
        error = f"{type(exc).__name__}: {exc}"
    if exit_code is None:
        status = "not_started"
    elif exit_code == TIMEOUT_RETURN_CODE and timeout_sec is not None:
        status = "timed_out"
    elif exit_code == 0:
        status = "exited_zero"
    else:
        status = "exited_nonzero"
    record = NebRunRecord(
        argv=tuple(argv),
        executable=config.neb_executable,
        cwd=str(cwd),
        input_path=str(input_file.resolve()),
        input_sha256=input_hash,
        config_hash=config.config_hash(),
        stdout_path=str(stdout_path),
        stdout_sha256=_sha256(stdout_path) if stdout_path.exists() else None,
        started_at=started,
        finished_at=_now(),
        exit_code=exit_code,
        status=status,
        error=error,
    )
    write_run_record(record, record_path)
    return record


__all__ = [
    "NebRunRecord",
    "RECORD_SUFFIX",
    "RunStatus",
    "build_neb_argv",
    "run_neb",
    "write_run_record",
]
