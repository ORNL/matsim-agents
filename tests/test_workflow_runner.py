import io
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from matsim_agents.execution.workflow_runner import run_native_command


def test_native_output_is_streamed_and_retained_without_interpreting_exit_status():
    output = io.StringIO()
    emitted = []
    status = run_native_command(
        [sys.executable, "-c", "print('prompt', end='', flush=True); raise SystemExit(7)"],
        output,
        emitted.append,
        cwd=Path.cwd(),
    )
    assert status == 7
    assert output.getvalue() == "".join(emitted) == "prompt"


def test_cancellation_signals_only_owned_child_group_and_waits(monkeypatch):
    class Stream(io.BytesIO):
        def fileno(self):
            return 1

    class Process:
        pid = 12345
        stdout = Stream(b"")

        def __init__(self):
            self.waits = []

        def wait(self, timeout=None):
            self.waits.append(timeout)
            if len(self.waits) <= 2:
                raise subprocess.TimeoutExpired("test", timeout)
            return -signal.SIGKILL

    process = Process()
    created = []
    signals = []
    monkeypatch.setattr(
        "matsim_agents.execution.workflow_runner.subprocess.Popen",
        lambda command, **kwargs: created.append(kwargs) or process,
    )
    # Avoid reading a fake descriptor: this stands in for a user interruption.
    monkeypatch.setattr(
        "matsim_agents.execution.workflow_runner.os.read",
        lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        "matsim_agents.execution.workflow_runner.os.killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )
    monkeypatch.setattr("matsim_agents.execution.workflow_runner._group_alive", lambda pid: False)
    with pytest.raises(KeyboardInterrupt):
        run_native_command(["test-only"], io.StringIO(), lambda text: None, cwd=Path.cwd())
    assert created[0]["start_new_session"] is True
    assert signals == [
        (12345, signal.SIGINT),
        (12345, signal.SIGTERM),
        (12345, signal.SIGKILL),
    ]
    assert process.waits == [10, 10, None]
    assert process.stdout.closed


def test_descendants_are_stopped_even_after_leader_exits(monkeypatch):
    class Stream(io.BytesIO):
        def fileno(self):
            return 1

    class Process:
        pid = 12345
        stdout = Stream()

        def wait(self, timeout=None):
            return -signal.SIGINT

    signals = []
    monkeypatch.setattr(
        "matsim_agents.execution.workflow_runner.subprocess.Popen",
        lambda *args, **kwargs: Process(),
    )
    monkeypatch.setattr(
        "matsim_agents.execution.workflow_runner.os.read",
        lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        "matsim_agents.execution.workflow_runner.os.killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )
    monkeypatch.setattr("matsim_agents.execution.workflow_runner._group_alive", lambda pid: True)
    clock = iter([0, 11])
    monkeypatch.setattr(
        "matsim_agents.execution.workflow_runner.time.monotonic", lambda: next(clock)
    )
    with pytest.raises(KeyboardInterrupt):
        run_native_command(["test"], io.StringIO(), lambda text: None, cwd=Path.cwd())
    assert signals == [
        (12345, signal.SIGINT),
        (12345, signal.SIGTERM),
        (12345, signal.SIGKILL),
    ]
