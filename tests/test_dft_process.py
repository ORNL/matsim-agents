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


def test_sigterm_to_the_workflow_stops_the_launch(tmp_path):
    import signal
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    from matsim_agents.backends.dft import process

    code = textwrap.dedent(
        f"""
        from matsim_agents.backends.dft.process import run_launcher
        with open({str(tmp_path / "out.log")!r}, "w") as out:
            run_launcher(
                ["sh", "-c", "sleep 60 & echo $! > child.pid; wait"],
                cwd={str(tmp_path)!r}, stdout=out, timeout_sec=None, env=None, grace_sec=1,
            )
        """
    )
    source = str(Path(process.__file__).parents[3])
    env = {**os.environ, "PYTHONPATH": source}
    workflow = subprocess.Popen([sys.executable, "-c", code], env=env)
    pid_file = tmp_path / "child.pid"
    deadline = time.monotonic() + 30
    while not (pid_file.is_file() and pid_file.read_text().strip()):
        assert time.monotonic() < deadline and workflow.poll() is None
        time.sleep(0.05)
    workflow.send_signal(signal.SIGTERM)
    assert workflow.wait(timeout=30) == 128 + signal.SIGTERM
    assert not _alive(int(pid_file.read_text()))
