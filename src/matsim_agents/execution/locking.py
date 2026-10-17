"""Exclusive inter-process locks that work on shared HPC filesystems.

``fcntl.flock`` fails with ``OSError [Errno 524]`` on NERSC CFS (DVS) compute
nodes, so the lock is an atomically created lock file (``O_CREAT | O_EXCL``),
which is supported on CFS, Lustre, GPFS, and local disks. The file records the
holder's host and PID. A lock left by a crashed process on the same host is
reclaimed; a lock held from another host is never broken automatically.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class LockTimeoutError(TimeoutError):
    """Raised when an exclusive lock could not be acquired in time."""


def _holder(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _is_stale(holder: dict | None) -> bool:
    if not holder or holder.get("host") != socket.gethostname():
        return False
    try:
        os.kill(int(holder["pid"]), 0)
    except ProcessLookupError:
        return True
    except (PermissionError, KeyError, TypeError, ValueError):
        return False
    return False


@contextmanager
def exclusive_lock(
    path: str | Path, *, timeout: float | None = None, poll_interval: float = 1.0
) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` for the duration of the ``with`` block."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "time": time.time()})
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if _is_stale(_holder(path)):
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
                continue
            if deadline is not None and time.monotonic() >= deadline:
                raise LockTimeoutError(
                    f"timed out waiting for {path} (held by {_holder(path)})"
                ) from None
            time.sleep(poll_interval)
            continue
        except OSError as exc:  # pragma: no cover - filesystem without O_EXCL support
            if exc.errno in {errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise RuntimeError(f"cannot create exclusive lock file {path}") from exc
            raise
        try:
            os.write(fd, payload.encode())
        finally:
            os.close(fd)
        break
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
