"""Lifecycle of plan-managed LLM servers and their use by the discovery chat (phase 4)."""

from __future__ import annotations

import json
import socket
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

from matsim_agents.backends.llm import provider
from matsim_agents.backends.llm.servers import LLMServices, _ManagedServer, served_models
from matsim_agents.execution.allocation import Allocation
from matsim_agents.execution.placement import ResourcePlan

STUB_SERVER = textwrap.dedent(
    """
    import argparse, json, os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--served-model-name")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args, extra = parser.parse_known_args()
    record = os.environ["STUB_RECORD"]
    with open(record, "a") as stream:
        stream.write(json.dumps({
            "argv": sys.argv[1:],
            "mask": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }) + "\\n")
    if os.environ.get("STUB_FAIL"):
        print("stub failure: out of memory", flush=True)
        sys.exit(3)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": args.served_model_name}]}).encode()
            self.send_response(200 if self.path == "/v1/models" else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    HTTPServer((args.host, args.port), Handler).serve_forever()
    """
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def stub(tmp_path, monkeypatch):
    script = tmp_path / "stub_vllm.py"
    script.write_text(STUB_SERVER)
    record = tmp_path / "record.jsonl"
    monkeypatch.setenv("STUB_RECORD", str(record))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return script, record


def _plan(servers, *, llm=None, extra_pools=None) -> ResourcePlan:
    pools = {"llm": {"servers": servers, **(llm or {})}, **(extra_pools or {})}
    return ResourcePlan.model_validate({"preset": "perlmutter-gpu", "pools": pools})


def _local(plan: ResourcePlan, host: str = "localhost"):
    return plan.resolve(Allocation("local", (host,), 4))


def _managed(script, name="proposer", **extra):
    return {
        "name": name,
        "kind": "managed-vllm",
        "model": "org/Qwen2.5-14B-Instruct",
        "launcher": [sys.executable, str(script)],
        "port": _free_port(),
        "startup_timeout_s": 30,
        **extra,
    }


def test_managed_server_starts_waits_and_stops(stub, tmp_path):
    script, record = stub
    plan = _plan(
        [_managed(script, max_model_len=8192, extra_args=["--enforce-eager"])],
        llm={"devices": [0]},
        extra_pools={"mlip": {"devices": [1, 2, 3]}},
    )
    services = LLMServices.start(_local(plan), log_dir=tmp_path / "logs", poll_interval_s=0.2)
    try:
        endpoint = services.endpoint()
        assert endpoint.provider == "vllm"
        assert endpoint.model == "Qwen2.5-14B-Instruct"
        assert served_models(endpoint.base_url) == ["Qwen2.5-14B-Instruct"]
        launched = json.loads(record.read_text().splitlines()[0])
        assert launched["mask"] == "0"
        argv = launched["argv"]
        assert argv[0] == "org/Qwen2.5-14B-Instruct"
        assert argv[argv.index("--tensor-parallel-size") + 1] == "1"
        assert argv[argv.index("--max-model-len") + 1] == "8192"
        assert "--gpu-memory-utilization" not in argv
        assert argv[-1] == "--enforce-eager"
        process = services._managed[0].process
    finally:
        services.close()
    assert process.poll() is not None


def test_shared_device_passes_memory_budget(stub, tmp_path):
    script, record = stub
    plan = _plan(
        [_managed(script)],
        llm={"devices": [0], "share_with": ["mlip"], "memory_fraction": 0.6},
        extra_pools={"mlip": {"devices": [0, 1], "share_with": ["llm"], "memory_fraction": 0.3}},
    )
    with LLMServices.start(_local(plan), log_dir=tmp_path, poll_interval_s=0.2):
        argv = json.loads(record.read_text().splitlines()[0])["argv"]
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.6"


def test_server_that_exits_reports_its_log(stub, tmp_path, monkeypatch):
    script, _ = stub
    monkeypatch.setenv("STUB_FAIL", "1")
    plan = _plan([_managed(script)])
    with pytest.raises(RuntimeError, match="(?s)exited with code 3.*out of memory"):
        LLMServices.start(_local(plan), log_dir=tmp_path, poll_interval_s=0.2)


def test_server_that_never_becomes_ready_times_out_and_is_stopped(tmp_path):
    sleeper = tmp_path / "sleep.py"
    sleeper.write_text("import time\ntime.sleep(600)\n")
    server = {
        **_managed(sleeper),
        "launcher": [sys.executable, str(sleeper)],
        "startup_timeout_s": 1,
    }
    plan = _plan([server])
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="did not list 'Qwen2.5-14B-Instruct'"):
        LLMServices.start(_local(plan), log_dir=tmp_path, poll_interval_s=0.2)
    assert time.monotonic() - started < 60


def test_two_servers_on_one_node_and_external_endpoint(stub, tmp_path):
    script, record = stub
    servers = [
        _managed(script, name="proposer", devices=[0]),
        {**_managed(script, name="critic", devices=[1]), "model": "org/gemma-4-31B-it"},
        {"name": "remote", "kind": "external", "base_url": "https://llm.example.org/v1"},
    ]
    for server in servers[:2]:
        server["nodes"] = 1
    plan = _plan(servers, llm={"devices": [0, 1]})
    with LLMServices.start(_local(plan), log_dir=tmp_path, poll_interval_s=0.2) as services:
        assert services.endpoint("critic").model == "gemma-4-31B-it"
        assert services.endpoint("remote").base_url == "https://llm.example.org/v1"
        with pytest.raises(ValueError, match="choose one by name"):
            services.endpoint()
        with pytest.raises(KeyError, match="no server 'missing'"):
            services.endpoint("missing")
        masks = sorted(json.loads(line)["mask"] for line in record.read_text().splitlines())
    assert masks == ["0", "1"]


def test_remote_server_commands_use_the_facility_launcher(tmp_path):
    plan = _plan(
        [
            {
                "name": "p",
                "kind": "managed-vllm",
                "model": "org/A",
                "nodes": ["n1"],
                "devices": [2],
            },
        ],
        llm={"nodes": ["n1"]},
    )
    placement = plan.resolve(Allocation("slurm", ("n0", "n1"), 4))
    server = placement.pool("llm").server("p")
    managed = _ManagedServer(server, placement, log_dir=tmp_path, local_host="n0")
    assert managed.base_url == "http://n1:8000/v1"
    assert managed.command[:8] == [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        "--nodelist=n1",
        "--overlap",
        "--gpus-per-node=4",
        "--cpus-per-task=16",
        "env",
    ]
    assert managed.command[8:11] == ["CUDA_VISIBLE_DEVICES=2", "vllm", "serve"]
    assert managed.command[managed.command.index("--host") + 1] == "0.0.0.0"

    aurora = ResourcePlan.model_validate(
        {
            "preset": "aurora",
            "pools": {
                "llm": {
                    "nodes": ["x1"],
                    "servers": [
                        {"name": "p", "kind": "managed-vllm", "model": "org/A", "devices": [0, 1]}
                    ],
                }
            },
        }
    ).resolve(Allocation("pbs", ("x0", "x1")))
    remote = _ManagedServer(
        aurora.pool("llm").server("p"), aurora, log_dir=tmp_path, local_host="x0"
    )
    assert remote.command[:7] == ["mpiexec", "-n", "1", "--ppn", "1", "--hosts", "x1"]
    assert "ZE_AFFINITY_MASK=0,1" in remote.command


def test_missing_model_directory_fails_before_launch(tmp_path):
    plan = _plan([{"name": "p", "kind": "managed-vllm", "model": str(tmp_path / "absent")}])
    with pytest.raises(ValueError, match="model directory not found"):
        LLMServices.start(_local(plan), log_dir=tmp_path)


def test_multi_node_managed_servers_are_phase_5(tmp_path):
    plan = _plan(
        [{"name": "big", "kind": "managed-vllm", "model": "org/A", "nodes": 2}], llm={"nodes": 2}
    )
    placement = plan.resolve(Allocation("slurm", ("n0", "n1"), 4))
    with pytest.raises(ValueError, match="multi-node serving is phase 5"):
        LLMServices.start(placement, log_dir=tmp_path)


def test_in_process_server_limits_its_devices(tmp_path, monkeypatch):
    monkeypatch.setattr(provider, "_IN_PROCESS_DEVICES", {})
    plan = _plan(
        [{"name": "hf", "kind": "in-process", "model": "/models/Qwen2.5-72B-Instruct"}],
        llm={"devices": "all", "share_with": ["mlip"], "memory_fraction": 0.75},
        extra_pools={"mlip": {"devices": [0], "share_with": ["llm"], "memory_fraction": 0.2}},
    )
    with LLMServices.start(_local(plan), log_dir=tmp_path) as services:
        endpoint = services.endpoint("hf")
    assert endpoint.provider == "huggingface"
    assert provider._IN_PROCESS_DEVICES["/models/Qwen2.5-72B-Instruct"] == {
        0: 0.75,
        1: None,
        2: None,
        3: None,
    }


def test_in_process_server_must_run_on_this_node(tmp_path):
    plan = _plan([{"name": "hf", "kind": "in-process", "model": "org/A"}], llm={"nodes": ["n1"]})
    placement = plan.resolve(Allocation("slurm", ("n0", "n1"), 4))
    with pytest.raises(ValueError, match="must be placed on this node"):
        LLMServices.start(placement, log_dir=tmp_path, local_host="n0")


def test_in_process_loader_passes_device_budget(monkeypatch):
    calls = {}

    class Pipeline:
        @classmethod
        def from_model_id(cls, **kwargs):
            calls.update(kwargs)
            return cls()

    monkeypatch.setattr(provider, "_HUGGINGFACE_PIPELINES", {})
    monkeypatch.setattr(provider, "_IN_PROCESS_DEVICES", {"/m/A": {0: 0.5, 2: None}})
    monkeypatch.setattr(provider, "_max_memory", lambda fractions: {0: 10, 2: 20})
    monkeypatch.setitem(sys.modules, "langchain_huggingface", type(sys)("langchain_huggingface"))
    sys.modules["langchain_huggingface"].HuggingFacePipeline = Pipeline
    provider._load_huggingface_pipeline("/m/A", {})
    assert calls["model_kwargs"] == {"max_memory": {0: 10, 2: 20}}
    assert calls["device_map"] == "auto"


def test_chat_session_uses_plan_servers(stub, tmp_path, monkeypatch):
    from matsim_agents.orchestration.interactive_chat import (
        DiscoveryChatConfig,
        DiscoveryChatSession,
    )

    script, _ = stub
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(
            {
                "resources": {
                    "preset": "perlmutter-gpu",
                    "geometry": {"devices_per_node": 4},
                    "pools": {
                        "llm": {
                            "devices": [0, 1],
                            "servers": [
                                {**_managed(script, name="proposer"), "nodes": 1, "devices": [0]},
                                {**_managed(script, name="critic"), "nodes": 1, "devices": [1]},
                            ],
                        }
                    },
                }
            }
        )
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NUM_NODES", raising=False)
    monkeypatch.delenv("PBS_NODEFILE", raising=False)
    config = DiscoveryChatConfig(
        llm_only=True,
        output_dir=str(tmp_path / "out"),
        resources=str(plan_path),
        llm_server="proposer",
        critic_llm_servers=["critic"],
    )
    session = DiscoveryChatSession(config=config)
    try:
        session.open()
        assert session.config.llm_provider == "vllm"
        assert session.config.llm_model == "Qwen2.5-14B-Instruct"
        assert served_models(session.config.llm_base_url) == ["Qwen2.5-14B-Instruct"]
        assert session.config.critic_panel_models == ["Qwen2.5-14B-Instruct"]
        assert session.config.critic_panel_base_urls[0].endswith("/v1")
        assert (tmp_path / "out" / "placement.json").is_file()
    finally:
        session.close()
    assert session.llm_services is None


def test_chat_server_names_need_an_llm_pool(tmp_path):
    from matsim_agents.orchestration.interactive_chat import (
        DiscoveryChatConfig,
        DiscoveryChatSession,
    )

    session = DiscoveryChatSession(config=DiscoveryChatConfig(llm_only=True, llm_server="x"))
    with pytest.raises(ValueError, match="need a plan with an llm pool"):
        session.open()


def test_shipped_discovery_layouts_name_a_proposer(monkeypatch):
    monkeypatch.setenv("MODEL_ROOT", "/models")
    monkeypatch.setenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
    root = Path(__file__).resolve().parents[1] / "deployments" / "perlmutter" / "resources"
    for name in ("discovery-vllm.yaml", "discovery-hf-80gb.yaml"):
        plan = ResourcePlan.from_yaml(root / name)
        placement = plan.resolve(Allocation("local", ("localhost",), 4))
        assert [s.name for s in placement.pool("llm").servers] == ["proposer"]
    vllm = ResourcePlan.from_yaml(root / "discovery-vllm.yaml")
    launcher = vllm.pools["llm"].servers[0].launcher
    assert launcher is not None and Path(launcher[-1]).is_file()


# -- resources serve, llm-check, and debate -----------------------------------------


def _serve_plan(tmp_path, script, servers=None) -> Path:
    plan = tmp_path / "serve.yaml"
    plan.write_text(
        yaml.safe_dump(
            {
                "resources": {
                    "preset": "perlmutter-gpu",
                    "geometry": {"devices_per_node": 4},
                    "pools": {"llm": {"devices": [0], "servers": servers or [_managed(script)]}},
                }
            }
        )
    )
    return plan


def _no_scheduler(monkeypatch):
    for name in ("SLURM_JOB_ID", "SLURM_JOB_NUM_NODES", "PBS_NODEFILE"):
        monkeypatch.delenv(name, raising=False)


def test_resources_serve_runs_command_with_endpoints(stub, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from matsim_agents.cli import app

    script, _ = stub
    _no_scheduler(monkeypatch)
    plan = _serve_plan(tmp_path, script)
    seen = tmp_path / "seen.json"
    child = (
        "import json, os, sys; json.dump({k: os.environ.get(k) for k in "
        "('MATSIM_LLM_ENDPOINTS', 'MATSIM_LLM_PROVIDER', 'MATSIM_LLM_MODEL', "
        f"'MATSIM_VLLM_BASE_URL')}}, open({str(seen)!r}, 'w')); sys.exit(7)"
    )
    output = tmp_path / "servers"
    result = CliRunner().invoke(
        app,
        [
            "resources",
            "serve",
            str(plan),
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            child,
        ],
    )
    assert result.exit_code == 7, result.output
    env = json.loads(seen.read_text())
    assert env["MATSIM_LLM_PROVIDER"] == "vllm"
    assert env["MATSIM_LLM_MODEL"] == "Qwen2.5-14B-Instruct"
    assert env["MATSIM_VLLM_BASE_URL"].startswith("http://127.0.0.1:")
    published = json.loads(Path(env["MATSIM_LLM_ENDPOINTS"]).read_text())
    assert published["proposer"]["tensor_parallel_size"] == 1
    assert (output / "placement.json").is_file()
    with pytest.raises(OSError):
        served_models(env["MATSIM_VLLM_BASE_URL"], timeout=2)


def test_resources_serve_rejects_in_process_servers(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from matsim_agents.cli import app

    _no_scheduler(monkeypatch)
    plan = _serve_plan(tmp_path, None, [{"name": "hf", "kind": "in-process", "model": "org/A"}])
    result = CliRunner().invoke(app, ["resources", "serve", str(plan), "--", "true"])
    assert result.exit_code == 1
    assert "cannot be served" in result.output


def _published(tmp_path, monkeypatch, **extra):
    from matsim_agents.backends.llm.servers import ENDPOINTS_VARIABLE, LLMEndpoint

    endpoints = {
        "qwen": LLMEndpoint(
            "qwen", "vllm", "Qwen2.5-14B-Instruct", "http://n0:8000/v1", 4, 4, 8192
        ),
        "gemma": LLMEndpoint("gemma", "vllm", "gemma-4-31B-it", "http://n0:8001/v1", 1, 1, 4096),
        **extra,
    }
    path = LLMServices(endpoints).write(tmp_path / "llm-endpoints.json")
    monkeypatch.setenv(ENDPOINTS_VARIABLE, str(path))
    return path


def test_llm_check_uses_plan_server_from_published_endpoints(tmp_path, monkeypatch):
    from matsim_agents.workflows.llm_check import LLMCheckConfig, run_llm_check

    _published(tmp_path, monkeypatch)
    seen = {}

    def model_factory(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after load")

    def readiness(cfg):
        seen["cfg"] = cfg
        return {"endpoint": cfg.base_url}

    cfg = LLMCheckConfig(server="qwen", output_root=str(tmp_path / "check"))
    result = run_llm_check(cfg, model_factory=model_factory, readiness_probe=readiness)
    assert result.status == "failed"
    assert seen["provider"] == "vllm" and seen["model"] == "Qwen2.5-14B-Instruct"
    assert seen["base_url"] == "http://n0:8000/v1"
    resolved = seen["cfg"]
    assert (resolved.tensor_parallel_size, resolved.expected_accelerators) == (4, 4)
    assert resolved.context_length == 8192
    assert resolved.api_key == "EMPTY"


def test_llm_check_starts_the_plan_server(stub, tmp_path, monkeypatch):
    from matsim_agents.workflows.llm_check import LLMCheckConfig, run_llm_check

    script, _ = stub
    _no_scheduler(monkeypatch)
    plan = ResourcePlan.model_validate(
        {
            "preset": "perlmutter-gpu",
            "geometry": {"devices_per_node": 4},
            "pools": {"llm": {"devices": [0], "servers": [_managed(script)]}},
        }
    )
    seen = {}

    def readiness(cfg):
        seen["models"] = served_models(cfg.base_url)
        raise RuntimeError("stop after readiness")

    cfg = LLMCheckConfig(resources=plan, output_root=str(tmp_path / "check"))
    result = run_llm_check(cfg, readiness_probe=readiness)
    assert result.status == "failed"
    assert seen["models"] == ["Qwen2.5-14B-Instruct"]
    assert (tmp_path / "check" / "placement.json").is_file()


def test_llm_check_plan_server_excludes_explicit_model():
    from pydantic import ValidationError

    from matsim_agents.workflows.llm_check import LLMCheckConfig

    with pytest.raises(ValidationError, match="remove model, provider"):
        LLMCheckConfig(server="qwen", provider="vllm", model="x")


def test_named_servers_need_a_plan_or_published_endpoints(tmp_path, monkeypatch):
    from matsim_agents.backends.llm.servers import ENDPOINTS_VARIABLE
    from matsim_agents.workflows.llm_check import LLMCheckConfig, run_llm_check

    monkeypatch.delenv(ENDPOINTS_VARIABLE, raising=False)
    with pytest.raises(ValueError, match="resources serve"):
        run_llm_check(LLMCheckConfig(server="qwen", output_root=str(tmp_path)))


def test_debate_participants_resolve_servers(tmp_path, monkeypatch):
    from matsim_agents.workflows.debate import ScientificDebateConfig, run_scientific_debate

    _published(tmp_path, monkeypatch)
    calls = []

    class Model:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def invoke(self, _messages):
            return type("Response", (), {"content": "claim: yes\nconfidence: 0.5"})()

    cfg = ScientificDebateConfig(
        hypothesis="Is Si stable?",
        rounds=1,
        output_root=str(tmp_path / "runs"),
        participants=[
            {"name": "a", "server": "qwen"},
            {"name": "b", "provider": "ollama", "model": "llama3.1:8b"},
        ],
    )
    result = run_scientific_debate(cfg, model_factory=lambda **kw: Model(**kw))
    assert result.status is not None
    assert calls[0] == {
        "provider": "vllm",
        "model": "Qwen2.5-14B-Instruct",
        "base_url": "http://n0:8000/v1",
    }
    assert calls[1]["provider"] == "ollama"


def test_debate_participant_validation():
    from pydantic import ValidationError

    from matsim_agents.workflows.debate import DebateParticipant, ScientificDebateConfig

    with pytest.raises(ValidationError, match="remove provider"):
        DebateParticipant(name="a", server="qwen", provider="vllm")
    with pytest.raises(ValidationError, match="needs provider and model, or server"):
        DebateParticipant(name="a", provider="vllm")
    with pytest.raises(ValidationError, match="no participant names a server"):
        ScientificDebateConfig(
            hypothesis="h",
            participants=[
                {"name": "a", "provider": "ollama", "model": "m"},
                {"name": "b", "provider": "ollama", "model": "m"},
            ],
            resources={"preset": "perlmutter-gpu", "pools": {"mlip": {}}},
        )


def test_debate_example_with_plan_servers_loads(monkeypatch):
    from matsim_agents.cli.workflow_commands import _load_workflow_config
    from matsim_agents.workflows.debate import ScientificDebateConfig

    monkeypatch.setenv("MODEL_ROOT", "/models")
    path = (
        Path(__file__).resolve().parents[1] / "examples/debate/equal-independent-plan-servers.yaml"
    )
    cfg = ScientificDebateConfig.model_validate(_load_workflow_config(path))
    placement = cfg.resources.resolve(Allocation("local", ("localhost",), 4))
    servers = {server.name: server.model for server in placement.pool("llm").servers}
    assert servers == {
        "qwen": "/models/Qwen2.5-14B-Instruct",
        "gemma": "/models/gemma-4-31B-it",
    }


def test_shipped_llm_layouts_resolve(monkeypatch):
    monkeypatch.setenv("MODEL_ROOT", "/models")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    for path in sorted(root.glob("deployments/*/resources/llm-check.yaml")) + sorted(
        root.glob("deployments/perlmutter/resources/hypothesis-debate*.yaml")
    ):
        plan = ResourcePlan.from_yaml(path)
        placement = plan.resolve(Allocation("local", ("localhost",), None))
        for server in placement.pool("llm").servers:
            assert server.kind == "managed-vllm"
            assert Path(server.launcher[-1]).is_file(), path
