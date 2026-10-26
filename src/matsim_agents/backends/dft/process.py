"""Run a DFT launcher so that a timeout or termination stops everything it started."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
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
    """The launcher's return code, or 124 once a timed-out launch has fully stopped.

    The launcher runs in its own session, so signals to the workflow do not reach it;
    SIGTERM to the workflow (in the main thread) stops the launch group before exiting.
    """
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        env=None if env is None else dict(env),
        start_new_session=True,
    )
    stopping = False
    deferred: list[int] = []

    def on_sigterm(signum: int, _frame: object) -> None:
        if stopping:
            deferred.append(signum)
        else:
            raise SystemExit(128 + signum)

    restore = _handle_sigterm(on_sigterm)
    try:
        try:
            return proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            stopping = True
            stop_process_group(proc, grace_sec)
            if deferred:
                raise SystemExit(128 + deferred[0]) from None
            return TIMEOUT_RETURN_CODE
        except BaseException:
            stopping = True
            stop_process_group(proc, grace_sec)
            raise
    finally:
        restore()


def _handle_sigterm(handler: Callable[[int, object], None]) -> Callable[[], None]:
    """Install ``handler`` for SIGTERM; returns the call that restores the previous one."""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    previous = signal.signal(signal.SIGTERM, handler)
    return lambda: signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)


__all__ = ["TERMINATE_GRACE_SEC", "TIMEOUT_RETURN_CODE", "run_launcher", "stop_process_group"]
