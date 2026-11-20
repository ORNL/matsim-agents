"""MCP client tests against a real local MCP server (stdio and Streamable HTTP).

Every test talks to ``tests/fixtures/mcp_test_server.py`` through the official
MCP SDK; LLMs are replaced by scripted stand-ins, and no network beyond
127.0.0.1 is used.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp")

from matsim_agents.backends.mcp.config import MCPConfig  # noqa: E402
from matsim_agents.backends.mcp.records import REDACTED, MCPAuditLog  # noqa: E402
from matsim_agents.backends.mcp.session import MCPSessionManager  # noqa: E402
from matsim_agents.orchestration.state import MatSimState, TaskSpec  # noqa: E402

SERVER = Path(__file__).resolve().parents[1] / "fixtures" / "mcp_test_server.py"


def _stdio(name: str, *extra_args: str, **fields) -> dict:
    return {
        "name": name,
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SERVER), "--name", name, *extra_args],
        **fields,
    }


def _pid_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    status = Path(f"/proc/{pid}/status")
    return status.exists() and "zombie" in status.read_text()


@pytest.fixture
def manager_factory(tmp_path):
    managers: list[MCPSessionManager] = []

    def create(*servers: dict) -> MCPSessionManager:
        config = MCPConfig(servers=list(servers))
        audit = MCPAuditLog(tmp_path / "audit.jsonl", secrets=config.secret_values())
        manager = MCPSessionManager(config, audit=audit, log_dir=tmp_path / "logs")
        managers.append(manager)
        manager.start()
        return manager

    yield create
    for manager in managers:
        manager.close()


def _audit(tmp_path) -> list[dict]:
    path = tmp_path / "audit.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ── stdio lifecycle and discovery ─────────────────────────────────────────────


def test_stdio_initialization_discovery_and_cleanup(manager_factory, tmp_path):
    pid_file = tmp_path / "pid"
    manager = manager_factory(
        _stdio("a", "--pid-file", str(pid_file), allowed_tools=["add", "whoami"])
    )
    pid = int(pid_file.read_text())
    assert manager.server_status()["a"]["connected"]

    discovered = {tool.name: tool for tool in manager.discovered_tools()}
    assert {"add", "whoami", "unlisted", "fail", "slow", "touch", "env_value"} <= set(discovered)
    schema = discovered["add"].input_schema
    assert schema["type"] == "object"
    assert schema["required"] == ["a", "b"]
    assert schema["properties"]["a"]["type"] == "integer"
    # Only allowlisted tools reach the planner.
    assert sorted(t.name for t in manager.permitted_tools()) == ["add", "whoami"]

    manager.close()
    assert _pid_gone(pid)
    events = [entry["event"] for entry in _audit(tmp_path)]
    assert events[0] == "connect" and events[-1] == "shutdown"
    shutdown = _audit(tmp_path)[-1]
    assert shutdown["clean"] is True
    manager.close()  # idempotent


def test_authorized_call_returns_content_and_structured_result(manager_factory, tmp_path):
    manager = manager_factory(_stdio("a", allowed_tools=["add"]))
    record = manager.call_tool("a", "add", {"a": 2, "b": 3})
    assert record.outcome == "success"
    assert record.authorized
    assert record.content == ["5"]
    assert record.structured_content == {"result": 5}
    assert record.content_sha256
    entry = _audit(tmp_path)[-1]
    assert entry["event"] == "tool_call"
    assert (entry["server"], entry["tool"], entry["outcome"]) == ("a", "add", "success")
    assert entry["arguments"] == {"a": 2, "b": 3}
    assert entry["started_at"] <= entry["finished_at"]


def test_unauthorized_unknown_and_invalid_calls_never_reach_the_tool(manager_factory, tmp_path):
    manager = manager_factory(_stdio("a", allowed_tools=["touch", "not_offered"]))
    marker = tmp_path / "marker"

    unlisted = manager.call_tool("a", "unlisted", {})
    assert (unlisted.outcome, unlisted.authorized) == ("denied", False)
    # A server-side description claiming pre-approval changes nothing.
    assert "not in allowed_tools" in unlisted.error

    unknown_tool = manager.call_tool("a", "not_offered", {})
    assert unknown_tool.outcome == "denied"
    assert "does not offer" in unknown_tool.error

    unknown_server = manager.call_tool("zzz", "touch", {"path": str(marker)})
    assert unknown_server.outcome == "denied"
    assert "unknown server" in unknown_server.error

    invalid = manager.call_tool("a", "touch", {"path": 42})
    assert invalid.outcome == "invalid_arguments"
    assert invalid.authorized
    assert "path" in invalid.error
    assert not marker.exists()

    # The same tool with valid arguments does run.
    assert manager.call_tool("a", "touch", {"path": str(marker)}).outcome == "success"
    assert marker.read_text() == "a"
    outcomes = [e["outcome"] for e in _audit(tmp_path) if e["event"] == "tool_call"]
    assert outcomes == ["denied", "denied", "denied", "invalid_arguments", "success"]


def test_tool_name_collisions_resolve_by_server_identity(manager_factory):
    manager = manager_factory(
        _stdio("alpha", allowed_tools=["whoami"]),
        _stdio("beta", allowed_tools=["whoami"]),
        _stdio("gamma", allowed_tools=[]),
    )
    assert manager.call_tool("alpha", "whoami", {}).content == ["alpha"]
    assert manager.call_tool("beta", "whoami", {}).content == ["beta"]
    assert manager.call_tool("gamma", "whoami", {}).outcome == "denied"
    permitted = sorted((t.server, t.name) for t in manager.permitted_tools())
    assert permitted == [("alpha", "whoami"), ("beta", "whoami")]


def test_tool_errors_and_timeouts_are_reported(manager_factory, tmp_path):
    manager = manager_factory(_stdio("a", allowed_tools=["fail", "slow"], call_timeout_s=0.5))
    failed = manager.call_tool("a", "fail", {})
    assert failed.outcome == "tool_error"
    assert "deliberate failure" in failed.error
    slow = manager.call_tool("a", "slow", {"seconds": 5})
    assert slow.outcome == "timeout"
    manager.close()
    assert _audit(tmp_path)[-1]["clean"] is True


def test_resources_are_read_only_when_allowlisted(manager_factory):
    manager = manager_factory(_stdio("a", allowed_resources=["test://notes"]))
    assert [r.uri for r in manager.permitted_resources()] == ["test://notes"]
    record = manager.read_resource("a", "test://notes")
    assert (record.kind, record.outcome, record.content) == (
        "resource",
        "success",
        ["notes from a"],
    )
    hidden = manager.read_resource("a", "test://hidden")
    assert hidden.outcome == "denied"
    assert hidden.content == []


def test_secrets_are_passed_to_the_server_but_redacted_everywhere(
    manager_factory, tmp_path, monkeypatch
):
    monkeypatch.setenv("HOST_SECRET", "very-secret-credential")
    from matsim_agents.backends.mcp.config import parse_mcp_config

    config = parse_mcp_config(
        {
            "servers": [
                _stdio(
                    "a",
                    env={"MATSIM_TEST_SECRET": "${HOST_SECRET}"},
                    allowed_tools=["env_value", "add"],
                )
            ]
        }
    )
    manager = manager_factory(*[s.model_dump() | {"env": s.env} for s in config.servers])
    record = manager.call_tool("a", "env_value", {"variable": "MATSIM_TEST_SECRET"})
    assert record.outcome == "success"
    assert record.content == [REDACTED]  # the server received it; the record does not keep it
    denied = manager.call_tool("a", "unlisted", {"api_key": "very-secret-credential"})
    assert denied.arguments == {"api_key": REDACTED}
    manager.close()
    audit_text = (tmp_path / "audit.jsonl").read_text()
    assert "very-secret-credential" not in audit_text
    for log in (tmp_path / "logs").glob("*.log"):
        assert "very-secret-credential" not in log.read_text()


# ── connection failures ───────────────────────────────────────────────────────


def test_missing_command_leaves_other_servers_working(manager_factory, tmp_path):
    manager = manager_factory(
        {
            "name": "broken",
            "transport": "stdio",
            "command": "/nonexistent/mcp-server",
            "allowed_tools": ["whoami"],
        },
        _stdio("ok", allowed_tools=["whoami"]),
    )
    status = manager.server_status()
    assert not status["broken"]["connected"]
    assert status["broken"]["error"]
    record = manager.call_tool("broken", "whoami", {})
    assert record.outcome == "denied"
    assert "no fallback" in record.error
    assert manager.call_tool("ok", "whoami", {}).content == ["ok"]
    connect = [e for e in _audit(tmp_path) if e["event"] == "connect"]
    assert {(e["server"], e["outcome"]) for e in connect} == {
        ("broken", "connection_error"),
        ("ok", "success"),
    }


def test_connect_timeout_terminates_the_unresponsive_server(manager_factory, tmp_path):
    pid_file = tmp_path / "pid"
    started = time.monotonic()
    manager = manager_factory(
        _stdio("hang", "--hang", "--pid-file", str(pid_file), connect_timeout_s=1.0)
    )
    assert time.monotonic() - started < 15
    status = manager.server_status()["hang"]
    assert not status["connected"]
    assert status["error"] == "timed out"
    # A failed connection is torn down at once, not at shutdown.
    assert _pid_gone(int(pid_file.read_text()))
    manager.close()


def test_http_connection_refused_is_reported(manager_factory):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manager = manager_factory(
        {
            "name": "remote",
            "transport": "streamable_http",
            "url": f"http://127.0.0.1:{port}/mcp",
            "connect_timeout_s": 3,
            "allowed_tools": ["whoami"],
        }
    )
    assert not manager.server_status()["remote"]["connected"]
    assert manager.call_tool("remote", "whoami", {}).outcome == "denied"


# ── Streamable HTTP ───────────────────────────────────────────────────────────


@pytest.fixture
def http_server(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = (tmp_path / "http-server.log").open("w")
    process = subprocess.Popen(
        [sys.executable, str(SERVER), "--name", "web", "--http", str(port)],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    else:
        process.kill()
        pytest.fail("test MCP HTTP server did not start")
    yield f"http://127.0.0.1:{port}/mcp"
    process.terminate()
    process.wait(timeout=10)
    log.close()


def test_streamable_http_discovery_call_and_header_redaction(
    manager_factory, http_server, tmp_path
):
    manager = manager_factory(
        {
            "name": "web",
            "transport": "streamable_http",
            "url": http_server,
            "headers": {"Authorization": "Bearer http-header-secret"},
            "allowed_tools": ["add", "whoami"],
            "allowed_resources": ["test://notes"],
        }
    )
    assert manager.server_status()["web"]["connected"]
    assert manager.call_tool("web", "whoami", {}).content == ["web"]
    assert manager.call_tool("web", "add", {"a": 40, "b": 2}).structured_content == {"result": 42}
    assert manager.read_resource("web", "test://notes").content == ["notes from web"]
    manager.close()
    audit = (tmp_path / "audit.jsonl").read_text()
    assert "http-header-secret" not in audit
    assert '"transport": "streamable_http"' in audit


# ── agent workflow integration ────────────────────────────────────────────────


def _scripted_planner(monkeypatch, tasks, captured=None):
    import matsim_agents.backends.llm.provider as llm_mod

    structured = MagicMock()

    def invoke(messages):
        if captured is not None:
            captured.extend(messages)
        return MagicMock(tasks=tasks, rationale="scripted plan")

    structured.invoke.side_effect = invoke
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    llm.invoke.side_effect = RuntimeError("analyst LLM unavailable in tests")
    monkeypatch.setattr(llm_mod, "get_chat_model", lambda **_: llm)


def test_planner_sees_only_permitted_tools_as_untrusted_data(manager_factory, monkeypatch):
    from matsim_agents.orchestration.agents.planner import planner_node

    manager = manager_factory(_stdio("a", allowed_tools=["add"]))
    captured: list = []
    _scripted_planner(monkeypatch, [], captured)
    planner_node(
        MatSimState(objective="add numbers", llm_provider="vllm"),
        {"configurable": {"mcp_client": manager}},
    )
    system = captured[0].content
    catalog = json.loads(system.split("<mcp_catalog>")[1].split("</mcp_catalog>")[0])
    assert [(t["mcp_server"], t["mcp_tool"]) for t in catalog["tools"]] == [("a", "add")]
    assert catalog["tools"][0]["input_schema"]["required"] == ["a", "b"]
    assert "IGNORE ALL RULES" not in system
    assert "untrusted data" in system
    assert "Always emit at least one" not in system


def test_planner_prompt_unchanged_without_mcp(monkeypatch):
    from matsim_agents.orchestration.agents.planner import _SYSTEM_PROMPT, planner_node

    captured: list = []
    _scripted_planner(monkeypatch, [], captured)
    planner_node(MatSimState(objective="relax a.vasp", llm_provider="vllm"))
    assert captured[0].content == _SYSTEM_PROMPT


def test_objective_graph_executes_mcp_tasks_without_mlip_config(
    manager_factory, monkeypatch, tmp_path
):
    from matsim_agents.orchestration.objective_graph import build_graph

    monkeypatch.delenv("MATSIM_MLIP_CONFIG", raising=False)
    manager = manager_factory(
        _stdio("a", allowed_tools=["add"], allowed_resources=["test://notes"]),
        _stdio("b", allowed_tools=[]),
    )
    tasks = [
        TaskSpec(kind="mcp_tool", mcp_server="a", mcp_tool="add", mcp_arguments={"a": 1, "b": 2}),
        TaskSpec(kind="mcp_resource", mcp_server="a", mcp_resource_uri="test://notes"),
        TaskSpec(kind="mcp_tool", mcp_server="b", mcp_tool="add", mcp_arguments={"a": 1, "b": 2}),
        TaskSpec(kind="mcp_tool", mcp_server="a", mcp_tool="add", mcp_arguments={"a": "x"}),
    ]
    _scripted_planner(monkeypatch, tasks)
    final = build_graph().invoke(
        MatSimState(objective="use MCP", llm_provider="vllm", max_iterations=3),
        config={"configurable": {"thread_id": "mcp-1", "mcp_client": manager}},
    )
    records = final["mcp_results"]
    # max_iterations bounds MCP tasks too: the fourth task is never executed.
    assert [r.outcome for r in records] == ["success", "success", "denied"]
    assert final["iteration"] == 3
    assert records[0].structured_content == {"result": 3}
    assert final["results"] == []
    assert "External MCP results (3, untrusted, not scientific evidence)" in final["analysis"]
    assert "a/add: success: 3" in final["analysis"]
    calls = [e for e in _audit(tmp_path) if e["event"] in {"tool_call", "resource_call"}]
    assert len(calls) == 3


def test_relax_task_still_requires_mlip_config_with_mcp_enabled(manager_factory, monkeypatch):
    from matsim_agents.orchestration.agents.executor import executor_node

    monkeypatch.delenv("MATSIM_MLIP_CONFIG", raising=False)
    manager = manager_factory(_stdio("a", allowed_tools=["add"]))
    state = MatSimState(pending_tasks=[TaskSpec(structure_path="a.vasp")])
    with pytest.raises(ValueError, match="Missing MLIP selection"):
        executor_node(state, {"configurable": {"mcp_client": manager}})


def test_cli_run_with_mcp_config_end_to_end(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from matsim_agents.cli import app

    config = tmp_path / "mcp.yaml"
    config.write_text(
        json.dumps(
            {
                "mcp": {
                    "servers": [
                        _stdio("a", "--pid-file", str(tmp_path / "pid"), allowed_tools=["add"])
                    ]
                }
            }
        )
    )
    tasks = [
        TaskSpec(kind="mcp_tool", mcp_server="a", mcp_tool="add", mcp_arguments={"a": 4, "b": 5})
    ]
    _scripted_planner(monkeypatch, tasks)
    monkeypatch.delenv("MATSIM_MLIP_CONFIG", raising=False)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "add 4 and 5",
            "--llm-provider",
            "vllm",
            "--output-dir",
            str(out),
            "--mcp-config",
            str(config),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "MCP server a [stdio]: connected" in result.output
    assert "MCP tool a/add: success: 9" in result.output
    audit = [json.loads(x) for x in (out / "mcp" / "mcp_audit.jsonl").read_text().splitlines()]
    assert [e["event"] for e in audit] == ["connect", "tool_call", "shutdown"]
    assert audit[-1]["clean"] is True
    assert _pid_gone(int((tmp_path / "pid").read_text()))


def test_cli_rejects_invalid_mcp_config(tmp_path):
    from typer.testing import CliRunner

    from matsim_agents.cli import app

    bad = tmp_path / "bad.yaml"
    bad.write_text("mcp:\n  servers:\n    - {name: a, transport: stdio}\n")
    result = CliRunner().invoke(app, ["run", "x", "--mcp-config", str(bad)])
    assert result.exit_code != 0
    assert "requires command" in result.output


def test_runs_without_mcp_config_never_import_the_sdk():
    code = (
        "import sys\n"
        "import matsim_agents.cli\n"
        "from matsim_agents.orchestration.objective_graph import build_graph\n"
        "from matsim_agents.orchestration.agents.executor import executor_node\n"
        "from matsim_agents.orchestration.state import MatSimState, TaskSpec\n"
        "build_graph()\n"
        "out = executor_node(MatSimState(pending_tasks=[TaskSpec(kind='mcp_tool', "
        "mcp_server='a', mcp_tool='t')]))\n"
        "assert out['mcp_results'][0].outcome == 'denied'\n"
        "assert 'mcp' not in sys.modules, 'MCP SDK imported without configuration'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_documented_example_runs_end_to_end(tmp_path):
    """examples/mcp/run_demo.py drives the real graph against the demo server."""
    import yaml

    example = Path(__file__).resolve().parents[2] / "examples" / "mcp"
    config = yaml.safe_load((example / "mcp.example.yaml").read_text())
    config["mcp"]["servers"][0]["command"] = sys.executable
    config["mcp"]["servers"][0]["cwd"] = str(example)
    path = tmp_path / "mcp.yaml"
    path.write_text(yaml.safe_dump(config))
    out = subprocess.run(
        [
            sys.executable,
            str(example / "run_demo.py"),
            "--output-dir",
            str(tmp_path / "o"),
            "--config",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout
    assert "MCP tool demo/convert_energy: success: -308.75306" in out
    assert "invalid_arguments: to_unit: 'J' is not one of" in out
    assert "denied: unknown server 'elsewhere'" in out
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "o" / "mcp" / "mcp_audit.jsonl").read_text().splitlines()
    ]
    assert events == ["connect", "tool_call", "resource_call", "tool_call", "tool_call", "shutdown"]


def test_server_crash_mid_run_is_recorded_and_isolated(manager_factory, tmp_path):
    manager = manager_factory(
        _stdio("fragile", allowed_tools=["crash", "whoami"], call_timeout_s=10),
        _stdio("steady", allowed_tools=["whoami"]),
    )
    crashed = manager.call_tool("fragile", "crash", {})
    assert crashed.outcome == "connection_error"
    assert not manager.server_status()["fragile"]["connected"]
    after = manager.call_tool("fragile", "whoami", {})
    assert after.outcome == "denied"
    assert "no fallback" in after.error
    assert manager.call_tool("steady", "whoami", {}).content == ["steady"]
    manager.close()
    audit = _audit(tmp_path)
    assert any(e["event"] == "disconnect" and e["server"] == "fragile" for e in audit)
    assert audit[-1]["event"] == "shutdown"
    assert audit[-1]["clean"] is True
