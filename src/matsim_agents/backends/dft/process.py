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

    The launcher runs in its own session, so signals to the workflow do not reach it.
    SIGTERM or Ctrl-C to the workflow (in the main thread) stops and reaps the launch
    group, then exits or raises ``KeyboardInterrupt``; a signal during that cleanup
    waits for it to finish.
    """
    received: list[int] = []
    # Handlers first: a signal while the launcher starts must not escape before it is tracked.
    restore = _handle_signals(lambda signum, _frame: received.append(signum))
    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=None if env is None else dict(env),
            start_new_session=True,
        )
        try:
            returncode = _wait_unless_signalled(proc, timeout_sec, received)
        finally:
            # A launcher that exits as the signal arrives may leave children behind.
            if proc.returncode is None or received:
                stop_process_group(proc, grace_sec)
    finally:
        restore()
    if received:
        if received[0] == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + received[0])
    return TIMEOUT_RETURN_CODE if returncode is None else returncode


def _wait_unless_signalled(
    proc: subprocess.Popen, timeout_sec: float | None, received: list[int]
) -> int | None:
    """The launcher's return code, or ``None`` once it times out or the workflow is signalled."""
    deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
    while proc.poll() is None and not received:
        if deadline is not None and time.monotonic() >= deadline:
            return None
        time.sleep(0.05)
    return proc.returncode


def _handle_signals(handler: Callable[[int, object], None]) -> Callable[[], None]:
    """Install ``handler`` for SIGTERM and a default SIGINT; returns the restoring call."""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    signums = [signal.SIGTERM]
    # An ignored or application-owned SIGINT keeps its handler.
    if signal.getsignal(signal.SIGINT) is signal.default_int_handler:
        signums.append(signal.SIGINT)
    previous = {signum: signal.signal(signum, handler) for signum in signums}

    def restore() -> None:
        for signum, old in previous.items():
            signal.signal(signum, signal.SIG_DFL if old is None else old)

    return restore


__all__ = ["TERMINATE_GRACE_SEC", "TIMEOUT_RETURN_CODE", "run_launcher", "stop_process_group"]
