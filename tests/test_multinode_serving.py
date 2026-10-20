"""Multi-node managed vLLM servers (resource dispatch phase 5)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from matsim_agents.backends.llm.servers import _ManagedServer
from matsim_agents.execution.allocation import Allocation
from matsim_agents.execution.placement import ResourcePlan

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deployments/perlmutter/launchers/vllm-server-perlmutter.sh"


def _placement(servers, nodes=("n0", "n1", "n2"), llm_nodes=3):
    plan = ResourcePlan.model_validate(
        {
            "preset": "perlmutter-gpu",
            "pools": {"llm": {"nodes": llm_nodes, "servers": servers}},
        }
    )
    return plan.resolve(Allocation("slurm", tuple(nodes), 4))


def _server(**extra):
    return {
        "name": "big",
        "kind": "managed-vllm",
        "model": "org/Big",
        "nodes": 2,
        "launcher": ["bash", "vllm-server-perlmutter.sh"],
        **extra,
    }


def test_multi_node_server_runs_its_launcher_on_the_head_node(tmp_path):
    placement = _placement([_server(tensor_parallel_size=8)])
    server = placement.pool("llm").server("big")
    managed = _ManagedServer(server, placement, log_dir=tmp_path, local_host="n0")
    # Other nodes of the allocation can reach it by host name.
    assert managed.local and managed.base_url == "http://n0:8000/v1"
    assert managed.command[managed.command.index("--host") + 1] == "0.0.0.0"
    assert managed.env["MATSIM_SERVER_SLICES"] == "n0=0,1,2,3;n1=0,1,2,3"
    assert managed.env["MATSIM_SERVER_RAY_PORT"] == "9000"
    argv = managed.command
    assert argv[argv.index("--tensor-parallel-size") + 1] == "8"
    assert "--pipeline-parallel-size" not in argv


def test_remote_multi_node_head_gets_the_layout_inside_its_step(tmp_path):
    placement = _placement([_server()])
    server = placement.pool("llm").server("big")
    managed = _ManagedServer(server, placement, log_dir=tmp_path, local_host="driver")
    assert managed.base_url == "http://n0:8000/v1"
    assert "--nodelist=n0" in managed.command
    assert "MATSIM_SERVER_SLICES=n0=0,1,2,3;n1=0,1,2,3" in managed.command
    assert "MATSIM_SERVER_IN_STEP=1" in managed.command
    # Default split: tensor parallel within a node, pipeline stages across nodes.
    command = managed.command
    assert command[command.index("--pipeline-parallel-size") + 1] == "2"


def test_multi_node_servers_need_a_facility_launcher(tmp_path):
    placement = _placement([{**_server(), "launcher": None}])
    with pytest.raises(ValueError, match="need a facility launcher that starts the Ray cluster"):
        _ManagedServer(
            placement.pool("llm").server("big"), placement, log_dir=tmp_path, local_host="n0"
        )


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


@pytest.fixture
def stub_tools(tmp_path):
    record = tmp_path / "calls"
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = (
        'printf "%s %s | CUDA=%s\\n" "$(basename "$0")" "$*" "${CUDA_VISIBLE_DEVICES:-}"'
        f" >> {record}\n"
    )
    # `ray start --block` would run until stopped; the stubs return at once.
    _stub(venv / "ray", log)
    _stub(venv / "vllm", log)
    # The GPU probe reports the whole cluster as joined.
    _stub(venv / "python", 'echo "${STUB_GPUS:-8}"\n')
    _stub(bin_dir / "srun", log + 'while [[ "$1" == --* ]]; do shift; done\nexec "$@"\n')
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "MATSIM_VLLM_VENV": str(venv.parent),
        "PROJECT_ROOT": str(ROOT),
        "MATSIM_SERVER_RAY_PORT": "9000",
        "MATSIM_NODE_DEVICES": "4",
        "MATSIM_NODE_CORES": "64",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "SLURM_JOB_ID": "1",
    }
    return env, record


def test_launcher_builds_a_ray_cluster_before_serving(stub_tools):
    env, record = stub_tools
    env["MATSIM_SERVER_SLICES"] = "n0=0,1,2,3;n1=2,3"
    env["STUB_GPUS"] = "6"
    result = subprocess.run(
        ["bash", str(LAUNCHER), "/models/Big", "--port", "8000"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    calls = record.read_text().splitlines()
    head = next(call for call in calls if call.startswith("ray start --head"))
    assert "--port=9000" in head and "--num-gpus=4" in head
    srun = next(call for call in calls if call.startswith("srun "))
    assert "--nodelist=n1" in srun and "--overlap" in srun and "--gpu-bind=none" in srun
    worker = next(call for call in calls if call.startswith("ray start --address="))
    assert "--num-gpus=2" in worker and worker.endswith("CUDA=2,3")
    serve = next(call for call in calls if call.startswith("vllm serve"))
    assert "/models/Big --port 8000 --distributed-executor-backend ray" in serve
    assert any(call.startswith("ray stop --force") for call in calls)
    assert "[ray] 6/6 GPUs available" in result.stdout


def test_launcher_serves_one_node_directly(stub_tools):
    env, record = stub_tools
    env["MATSIM_SERVER_SLICES"] = "n0=0,1"
    result = subprocess.run(
        ["bash", str(LAUNCHER), "/models/Small", "--port", "8000"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    calls = record.read_text().splitlines()
    assert calls == ["vllm serve /models/Small --port 8000 | CUDA=0,1,2,3"]


def test_launcher_fails_when_the_cluster_does_not_form(stub_tools, monkeypatch):
    env, record = stub_tools
    env["MATSIM_SERVER_SLICES"] = "n0=0,1,2,3;n1=0,1,2,3"
    env["STUB_GPUS"] = "4"
    script = LAUNCHER.read_text().replace(
        'wait_for_gpus "$EXPECTED" 600', 'wait_for_gpus "$EXPECTED" 3'
    )
    short = record.parent / "launcher-short-timeout.sh"
    short.write_text(script)
    result = subprocess.run(
        ["bash", str(short), "/models/Big"], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0
    assert "only 4/8 GPUs" in result.stderr
    calls = record.read_text().splitlines()
    assert not any(call.startswith("vllm serve") for call in calls)
    assert any(call.startswith("ray stop --force") for call in calls)


def test_single_node_allocations_keep_servers_on_the_loopback(tmp_path):
    plan = ResourcePlan.model_validate(
        {
            "preset": "perlmutter-gpu",
            "pools": {"llm": {"servers": [{**_server(), "nodes": 1}]}},
        }
    )
    placement = plan.resolve(Allocation("slurm", ("n0",), 4))
    managed = _ManagedServer(
        placement.pool("llm").server("big"), placement, log_dir=tmp_path, local_host="n0"
    )
    assert managed.base_url == "http://127.0.0.1:8000/v1"
    assert managed.command[managed.command.index("--host") + 1] == "127.0.0.1"
