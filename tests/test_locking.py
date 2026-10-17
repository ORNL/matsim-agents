"""Exclusive lock files used for shared reference and calculation caches."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

from matsim_agents.execution.locking import LockTimeoutError, exclusive_lock


def test_lock_is_exclusive_and_released(tmp_path):
    path = tmp_path / "cache" / "entry.lock"
    with exclusive_lock(path):
        holder = json.loads(path.read_text())
        assert holder["pid"] == os.getpid()
        assert holder["host"] == socket.gethostname()
        with pytest.raises(LockTimeoutError), exclusive_lock(path, timeout=0.2, poll_interval=0.05):
            pass
    assert not path.exists()
    with exclusive_lock(path, timeout=0.2):
        pass


def test_lock_is_released_when_the_body_raises(tmp_path):
    path = tmp_path / "entry.lock"
    with pytest.raises(RuntimeError), exclusive_lock(path):
        raise RuntimeError("boom")
    assert not path.exists()


def test_waiter_acquires_after_holder_releases(tmp_path):
    path = tmp_path / "entry.lock"
    order = []

    def waiter():
        with exclusive_lock(path, timeout=5, poll_interval=0.02):
            order.append("waiter")

    with exclusive_lock(path):
        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.1)
        order.append("holder")
    thread.join(timeout=5)
    assert order == ["holder", "waiter"]


def test_lock_left_by_dead_process_on_this_host_is_reclaimed(tmp_path):
    path = tmp_path / "entry.lock"
    dead = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    path.write_text(json.dumps({"host": socket.gethostname(), "pid": int(dead.stdout)}))
    with exclusive_lock(path, timeout=1):
        assert json.loads(path.read_text())["pid"] == os.getpid()


def test_lock_held_from_another_host_is_not_broken(tmp_path):
    path = tmp_path / "entry.lock"
    path.write_text(json.dumps({"host": "other-node", "pid": 1}))
    with pytest.raises(LockTimeoutError), exclusive_lock(path, timeout=0.2, poll_interval=0.05):
        pass
    assert json.loads(path.read_text())["host"] == "other-node"


def test_lock_held_by_live_process_on_this_host_is_not_broken(tmp_path):
    path = tmp_path / "entry.lock"
    path.write_text(json.dumps({"host": socket.gethostname(), "pid": os.getppid()}))
    with pytest.raises(LockTimeoutError), exclusive_lock(path, timeout=0.2, poll_interval=0.05):
        pass
