"""Run a DFT launcher so that a timeout stops everything it started."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Mapping
from typing import IO

TIMEOUT_RETURN_CODE = 124
# srun forwards SIGTERM to its step and waits up to Slurm's KillWait (30 s by default).
TERMINATE_GRACE_SEC = 60.0


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, sig)


def _wait_for_group(proc: subprocess.Popen, deadline: float) -> bool:
    while time.monotonic() < deadline:
        proc.poll()
        if proc.returncode is not None and not _group_alive(proc.pid):
            return True
        time.sleep(0.05)
    return False


def stop_process_group(proc: subprocess.Popen, grace_sec: float = TERMINATE_GRACE_SEC) -> None:
    """SIGTERM the launcher's process group, then SIGKILL what remains, and reap the leader."""
    _signal_group(proc.pid, signal.SIGTERM)
    if not _wait_for_group(proc, time.monotonic() + grace_sec):
        _signal_group(proc.pid, signal.SIGKILL)
        _wait_for_group(proc, time.monotonic() + grace_sec)
    proc.wait()


def run_launcher(
    argv: list[str],
    *,
    cwd: str,
    stdout: IO,
    timeout_sec: float | None,
    env: Mapping[str, str] | None,
    grace_sec: float = TERMINATE_GRACE_SEC,
) -> int:
    """The launcher's return code, or 124 once a timed-out launch has fully stopped."""
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        env=None if env is None else dict(env),
        start_new_session=True,
    )
    try:
        return proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        stop_process_group(proc, grace_sec)
        return TIMEOUT_RETURN_CODE
    except BaseException:
        stop_process_group(proc, grace_sec)
        raise


__all__ = ["TERMINATE_GRACE_SEC", "TIMEOUT_RETURN_CODE", "run_launcher", "stop_process_group"]
