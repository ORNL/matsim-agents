from __future__ import annotations

import os
import time

import pytest

from matsim_agents.backends.dft.process import TIMEOUT_RETURN_CODE, run_launcher


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as stat:
        return stat.read().split(")")[-1].split()[0] != "Z"


@pytest.mark.parametrize("ignore_term", [False, True])
def test_a_timed_out_launch_stops_its_whole_process_tree(tmp_path, ignore_term):
    trap = "trap '' TERM; " if ignore_term else ""
    # A wrapper that runs the code without exec, as the reviewed launcher did.
    script = f"{trap}sleep 60 & echo $! > child.pid; wait"
    with open(tmp_path / "out.log", "w") as out:
        start = time.monotonic()
        rc = run_launcher(
            ["sh", "-c", script],
            cwd=str(tmp_path),
            stdout=out,
            timeout_sec=1,
            env=None,
            grace_sec=1,
        )
    assert rc == TIMEOUT_RETURN_CODE
    assert time.monotonic() - start < 10
    child = int((tmp_path / "child.pid").read_text())
    assert not _alive(child)


def test_a_finished_launch_returns_its_exit_code(tmp_path):
    with open(tmp_path / "out.log", "w") as out:
        rc = run_launcher(
            ["sh", "-c", "echo done; exit 3"], cwd=str(tmp_path), stdout=out, timeout_sec=10, env={}
        )
    assert rc == 3
    assert (tmp_path / "out.log").read_text() == "done\n"
