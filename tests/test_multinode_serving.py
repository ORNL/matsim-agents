"""Multi-node managed vLLM servers (resource dispatch phase 5)."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest

from matsim_agents.backends.llm.servers import _ManagedServer
from matsim_agents.execution.allocation import Allocation
from matsim_agents.execution.placement import ResourcePlan

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deployments/perlmutter/launchers/vllm-server-perlmutter.sh"
RAY_NODE = ROOT / "deployments/common/ray-node.sh"


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
    assert managed.env["MATSIM_SERVER_LOG_DIR"] == str(tmp_path / "big-ray")
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
    # `ray start --block` runs until it is terminated and records when it stops.
    # The Ray stub writes a session log naming its node, as Ray does under
    # --temp-dir, before it records the call (which the GPU probe below waits for).
    _stub(
        venv / "ray",
        'for arg in "$@"; do case "$arg" in --temp-dir=*) d=${arg#*=}\n'
        '  mkdir -p "$d/session_1/logs" && ln -sfn session_1 "$d/session_latest"\n'
        '  hostname > "$d/session_1/logs/raylet.out";; esac; done\n'
        'if [[ "$*" == *--block* ]]; then\n'
        "  sleep 30 & sleeper=$!\n"
        f'  trap "kill $sleeper; echo stopped $(hostname) >> {record}; exit 0" TERM\n'
        f"  {log}"
        '  wait "$sleeper"\n'
        f"else\n  {log}"
        "fi\n",
    )
    # vLLM serves until it is stopped when STUB_VLLM_SERVES is set.
    _stub(
        venv / "vllm",
        log + 'if [[ -n "${STUB_VLLM_SERVES:-}" ]]; then sleep 30 & wait; fi\n',
    )
    # The GPU probe reports the whole cluster as joined once a worker has started
    # (or after a few seconds), like Ray counting GPUs as workers register.
    _stub(
        venv / "python",
        f'for _ in $(seq 50); do grep -q "^ray start --address" {record} 2>/dev/null && break; '
        "sleep 0.1; done\n"
        'echo "${STUB_GPUS:-8}"\n',
    )
    # srun runs the step "on" its node: `hostname` reports that node, and the
    # node-local Ray directory is a separate path per node. Signalling it kills
    # the step, as Slurm does, so workers must be stopped another way.
    _stub(
        bin_dir / "srun",
        log + 'while [[ "$1" == --* ]]; do\n'
        '  [[ "$1" == --nodelist=* ]] && host=${1#*=}; shift\n'
        "done\n"
        "export STUB_HOST=$host\n"
        'args=(); for a in "$@"; do\n'
        '  [[ "$a" == --temp-dir=* ]] && a="$a.$host"; args+=("$a")\n'
        "done\n"
        # Like Slurm: a signalled srun kills its step outright.
        '"${args[@]}" & child=$!\n'
        'trap "kill -KILL $child" TERM INT\n'
        'wait "$child"; wait "$child"\n',
    )
    _stub(
        bin_dir / "hostname",
        'if [[ "${1:-}" == -I ]]; then echo 127.0.0.1; else echo "${STUB_HOST:-n0}"; fi\n',
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "MATSIM_VLLM_VENV": str(venv.parent),
        "PROJECT_ROOT": str(ROOT),
        "MATSIM_SERVER_RAY_PORT": "9000",
        "MATSIM_SERVER_LOG_DIR": str(tmp_path / "logs" / "big-ray"),
        "MATSIM_NODE_DEVICES": "4",
        "MATSIM_NODE_CORES": "64",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        # Unique, so the launcher's node-local Ray directory does not collide.
        "SLURM_JOB_ID": f"test{os.getpid()}",
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
    # Each node's Ray logs end up on shared storage; the node-local copies are gone.
    logs = Path(env["MATSIM_SERVER_LOG_DIR"])
    for host in ("n0", "n1"):
        assert (logs / host / "raylet.out").read_text().strip() == host
    ray_tmp = Path(f"/tmp/r{env['SLURM_JOB_ID']}.9000")
    assert not ray_tmp.exists() and not Path(f"{ray_tmp}.n1").exists()
    # Workers leave while the head still runs.
    assert calls.index("stopped n1") < calls.index("stopped n0")


def test_stopping_the_server_stops_workers_before_the_head(stub_tools):
    """matsim-agents stops a server by signalling its process group."""
    env, record = stub_tools
    env["MATSIM_SERVER_SLICES"] = "n0=0,1,2,3;n1=0,1,2,3"
    env["STUB_VLLM_SERVES"] = "1"
    process = subprocess.Popen(
        ["bash", str(LAUNCHER), "/models/Big"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 30
    while "vllm serve" not in (record.read_text() if record.exists() else ""):
        assert time.monotonic() < deadline and process.poll() is None
        time.sleep(0.2)
    os.killpg(process.pid, signal.SIGTERM)
    output, _ = process.communicate(timeout=60)
    calls = record.read_text().splitlines()
    assert calls.index("stopped n1") < calls.index("stopped n0"), output
    logs = Path(env["MATSIM_SERVER_LOG_DIR"])
    for host in ("n0", "n1"):
        assert (logs / host / "raylet.out").read_text().strip() == host


def test_ray_node_saves_logs_when_asked_to_stop_even_if_killed_after(tmp_path):
    ray = tmp_path / "ray"
    pid_file = tmp_path / "ray.pid"
    # A Ray node that writes its session log and then ignores SIGTERM.
    _stub(
        ray,
        'd=${1#--temp-dir=}; mkdir -p "$d/session_1/logs" && ln -sfn session_1 '
        '"$d/session_latest"; echo log > "$d/session_1/logs/raylet.out"\n'
        f'echo $$ > {pid_file}; trap "" TERM; sleep 30\n',
    )
    logs = tmp_path / "logs"
    process = subprocess.Popen(
        ["bash", str(RAY_NODE), str(logs), str(ray), f"--temp-dir={tmp_path / 'session'}"]
    )
    deadline = time.monotonic() + 10
    while not pid_file.exists() or not pid_file.read_text().strip():
        assert time.monotonic() < deadline
        time.sleep(0.1)
    try:
        process.send_signal(signal.SIGTERM)
        host_logs = logs / socket.gethostname()
        while not (host_logs / "raylet.out").exists():
            assert time.monotonic() < deadline + 10
            time.sleep(0.1)
        # The scheduler may kill the step before Ray has shut down.
        process.kill()
        process.wait(timeout=10)
    finally:
        os.kill(int(pid_file.read_text()), signal.SIGKILL)


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
    # The logs that explain the failure are kept.
    assert (Path(env["MATSIM_SERVER_LOG_DIR"]) / "n0" / "raylet.out").exists()


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
