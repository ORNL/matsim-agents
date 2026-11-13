import os
import time

import pytest

from matsim_agents.execution.deadline_worker import DeadlineWorker, TaskDeadlineExceeded


def _initialize():
    return {"calls": 0}


def _execute(task, state):
    state["calls"] += 1
    if task == "stall":
        time.sleep(60)
    return os.getpid(), state["calls"]


def test_deadline_kills_stalled_step_and_restarts_worker_with_fresh_state():
    with DeadlineWorker(_initialize, _execute) as worker:
        first_pid, calls = worker.run("quick", seconds=5)
        assert calls == 1
        assert worker.run("quick", seconds=5) == (first_pid, 2)
        started = time.monotonic()
        with pytest.raises(TaskDeadlineExceeded):
            worker.run("stall", seconds=0.1)
        assert time.monotonic() - started < 5
        assert worker.process is None
        second_pid, calls = worker.run("quick", seconds=5)
        assert second_pid != first_pid and calls == 1
    assert worker.process is None
    with pytest.raises(ProcessLookupError):
        os.kill(second_pid, 0)


def test_invalid_deadline_fails_before_spawning():
    with DeadlineWorker(_initialize, _execute) as worker:
        with pytest.raises(ValueError):
            worker.run("quick", seconds=0)
        assert worker.process is None
