"""MCP client tests against a real local MCP server (stdio and Streamable HTTP).

Every test talks to ``tests/fixtures/mcp_test_server.py`` through the official
MCP SDK; LLMs are replaced by scripted stand-ins, and no network beyond
127.0.0.1 is used.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("mcp")

from matsim_agents.agents.state import MatSimState, TaskSpec  # noqa: E402
from matsim_agents.backends.mcp.config import MCPConfig  # noqa: E402
from matsim_agents.backends.mcp.records import (  # noqa: E402
    MAX_CONTENT_CHARS,
    REDACTED,
    MCPAuditLog,
    sha256_json,
)
from matsim_agents.backends.mcp.session import MCPSessionManager  # noqa: E402

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

    def create(*servers: dict, config: MCPConfig | None = None) -> MCPSessionManager:
        config = config or MCPConfig(servers=list(servers))
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


def test_record_metadata_is_scrubbed_after_authorization(manager_factory, monkeypatch):
    from mcp import types

    secret = "metadata-secret"
    uri = f"test://notes?token={secret}"
    manager = manager_factory(
        _stdio("a", env={"TOKEN": secret}, allowed_resources=[uri], allowed_tools=["add"])
    )
    session = MagicMock()
    session.read_resource = AsyncMock(
        return_value=types.ReadResourceResult(
            contents=[types.TextResourceContents(uri=uri, text="ok")]
        )
    )
    monkeypatch.setattr(manager, "_submit", lambda server, op, timeout: asyncio.run(op(session)))
    record = manager.read_resource("a", uri)
    assert record.outcome == "success"
    session.read_resource.assert_awaited_once_with(uri)
    assert secret not in record.model_dump_json()
    assert secret not in record.summary_line()
    denied = manager.read_resource("a", f"test://unlisted?token={secret}")
    assert denied.outcome == "denied"
    assert secret not in denied.model_dump_json()
    denied_tool = manager.call_tool("a", f"missing-{secret}", {})
    assert secret not in denied_tool.model_dump_json()


def test_resource_discovery_follows_pagination(manager_factory):
    manager = manager_factory(_stdio("a", allowed_resources=["test://second"]))
    session = MagicMock()
    session.initialize = AsyncMock(
        return_value=SimpleNamespace(
            capabilities=SimpleNamespace(tools=None, resources=object()),
            server_info=SimpleNamespace(name="a", version="1"),
        )
    )
    resource = SimpleNamespace(
        uri="test://second", name="second", description="on page two", mime_type="text/plain"
    )
    session.list_resources = AsyncMock(
        side_effect=[
            SimpleNamespace(resources=[], next_cursor="next"),
            SimpleNamespace(resources=[resource], next_cursor=None),
        ]
    )
    asyncio.run(manager._discover(manager.config.server("a"), session))
    assert session.list_resources.await_count == 2
    assert session.list_resources.await_args_list[1].kwargs["params"].cursor == "next"
    assert manager.permitted_resources()[0].description == "on page two"


def test_shutdown_cancels_active_and_queued_calls(manager_factory, monkeypatch, tmp_path):
    import matsim_agents.backends.mcp.session as session_module

    monkeypatch.setattr(session_module, "_SHUTDOWN_GRACE_S", 3.0)
    marker = tmp_path / "started"
    pid_file = tmp_path / "pid"
    manager = manager_factory(
        _stdio("a", "--pid-file", str(pid_file), allowed_tools=["slow"], call_timeout_s=20)
    )
    records = []
    active = threading.Thread(
        target=lambda: records.append(
            manager.call_tool("a", "slow", {"seconds": 20, "marker": str(marker)})
        )
    )
    active.start()
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert marker.exists()
    queued = threading.Thread(
        target=lambda: records.append(manager.call_tool("a", "slow", {"seconds": 20}))
    )
    queued.start()
    deadline = time.monotonic() + 5
    while manager._queues["a"].qsize() == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert manager._queues["a"].qsize() == 1
    manager.close()
    active.join(3)
    queued.join(3)
    assert not active.is_alive() and not queued.is_alive()
    assert sorted(record.outcome for record in records) == ["connection_error", "connection_error"]
    assert _pid_gone(int(pid_file.read_text()))
    shutdown = next(event for event in _audit(tmp_path) if event["event"] == "shutdown")
    assert shutdown["clean"] is True


def test_shutdown_cancels_connection_in_progress(monkeypatch):
    import anyio

    import matsim_agents.backends.mcp.session as session_module

    opened = threading.Event()
    cleaned = threading.Event()

    async def stalled_open(self, server, stack):
        stack.callback(cleaned.set)
        opened.set()
        await anyio.sleep(60)

    monkeypatch.setattr(session_module, "_SHUTDOWN_GRACE_S", 3.0)
    monkeypatch.setattr(MCPSessionManager, "_open", stalled_open)
    manager = MCPSessionManager(MCPConfig(servers=[_stdio("a", connect_timeout_s=60)]))
    starter = threading.Thread(target=manager.start)
    starter.start()
    assert opened.wait(5)
    manager.close()
    starter.join(3)
    assert not starter.is_alive()
    assert cleaned.is_set()


def test_result_content_has_shared_limits_and_error_structured_content(
    manager_factory, monkeypatch
):
    from mcp import types

    manager = manager_factory(
        _stdio("a", allowed_tools=["add"], allowed_resources=["test://notes"])
    )
    session = MagicMock()
    blocks = [types.TextContent(text="x" * 1000) for _ in range(20)]
    session.call_tool = AsyncMock(
        return_value=types.CallToolResult(
            content=blocks,
            structuredContent={"token": "issued-token", "detail": "bad"},
            isError=True,
        )
    )
    session.read_resource = AsyncMock(
        return_value=types.ReadResourceResult(
            contents=[types.TextResourceContents(uri="test://notes", text="x") for _ in range(1000)]
        )
    )
    monkeypatch.setattr(manager, "_submit", lambda server, op, timeout: asyncio.run(op(session)))
    tool = manager.call_tool("a", "add", {"a": 1, "b": 2})
    resource = manager.read_resource("a", "test://notes")
    for record in (tool, resource):
        assert sum(map(len, record.content)) <= MAX_CONTENT_CHARS
        assert len(record.content) <= 64
        assert "omitted" in "".join(record.content)
    assert tool.structured_content == {"token": REDACTED, "detail": "bad"}
    assert tool.content_sha256 == sha256_json(
        {"content": tool.content, "structured_content": tool.structured_content}
    )


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
                    env={"MATSIM_TEST_SECRET": "Bearer ${HOST_SECRET}"},
                    allowed_tools=["env_value", "stderr_env", "touch"],
                )
            ]
        }
    )
    manager = manager_factory(config=config)
    whole = manager.call_tool("a", "env_value", {"variable": "MATSIM_TEST_SECRET"})
    assert whole.outcome == "success"
    assert whole.content == [REDACTED]  # the server received it; the record does not keep it
    token_only = manager.call_tool(
        "a", "env_value", {"variable": "MATSIM_TEST_SECRET", "last_word": True}
    )
    assert token_only.content == [REDACTED]
    assert manager.call_tool("a", "stderr_env", {"variable": "MATSIM_TEST_SECRET"}).content == [
        "written"
    ]
    denied = manager.call_tool("a", "unlisted", {"api_key": "very-secret-credential"})
    assert denied.arguments == {"api_key": REDACTED}
    # Schema errors name the rule, never the rejected value.
    invalid = manager.call_tool("a", "touch", {"path": 987654321})
    assert invalid.outcome == "invalid_arguments"
    assert "987654321" not in invalid.error
    assert "path: violates type" in invalid.error
    manager.close()
    stderr_log = (tmp_path / "logs" / "a.stderr.log").read_text()
    assert stderr_log == f"server env MATSIM_TEST_SECRET={REDACTED}\n"
    for path in [tmp_path / "audit.jsonl", *(tmp_path / "logs").glob("*")]:
        assert "very-secret-credential" not in path.read_text()


def test_unusable_tool_schema_is_rejected_without_network_or_crash(manager_factory, monkeypatch):
    import urllib.request

    manager = manager_factory(_stdio("a", allowed_tools=["add"]))
    # A server-supplied schema with a remote $ref must not be fetched or crash the run.
    manager._tools["a"]["add"].input_schema = {
        "type": "object",
        "properties": {"a": {"$ref": "https://example.invalid/schema.json"}},
    }

    attempts = []

    def no_network(*args, **kwargs):
        attempts.append(args)
        raise AssertionError("schema validation attempted a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    record = manager.call_tool("a", "add", {"a": 1, "b": 2})
    assert record.outcome == "invalid_arguments"
    assert "input schema cannot be used" in record.error
    assert attempts == []

    manager._tools["a"]["add"].input_schema = {
        "$defs": {"number": {"type": "integer"}},
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/number"}},
    }
    assert manager.call_tool("a", "add", {"a": 1, "b": 2}).outcome == "success"


def test_results_are_scrubbed_before_truncation_and_hashing(manager_factory, monkeypatch, tmp_path):
    from mcp import types

    secret = "configured-secret"
    manager = manager_factory(
        _stdio(
            "a",
            env={"SECRET": secret},
            allowed_tools=["add"],
            allowed_resources=["test://notes"],
        )
    )
    text = "x" * (MAX_CONTENT_CHARS - 5) + secret
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=types.CallToolResult(
            content=[types.TextContent(text=text)],
            structuredContent={"access_token": "new-server-issued-token", "echo": secret},
        )
    )
    session.read_resource = AsyncMock(
        return_value=types.ReadResourceResult(
            contents=[types.TextResourceContents(uri="test://notes", text=text)]
        )
    )
    monkeypatch.setattr(manager, "_submit", lambda server, op, timeout: asyncio.run(op(session)))

    result = manager.call_tool("a", "add", {"a": 1, "b": 2})
    assert result.structured_content == {"access_token": REDACTED, "echo": REDACTED}
    assert result.content_sha256 == sha256_json(
        {"content": result.content, "structured_content": result.structured_content}
    )
    assert secret[:5] not in result.content[0]
    resource = manager.read_resource("a", "test://notes")
    assert secret[:5] not in resource.content[0]
    failure = manager._failure(
        {
            "kind": "tool",
            "server": "a",
            "started_at": "t",
            "authorized": True,
            "authorization_reason": "test",
        },
        "protocol_error",
        "x" * 1995 + secret,
    )
    assert secret[:5] not in failure.error
    manager._tools["a"]["add"].input_schema = {
        "type": "object",
        "properties": {"a": {"enum": ["x" * 192 + secret]}},
    }
    invalid = manager.call_tool("a", "add", {"a": "not-allowed"})
    assert invalid.outcome == "invalid_arguments"
    assert secret[:5] not in invalid.error

    session.call_tool.return_value = types.CallToolResult(
        content=[], structuredContent={"result": "x" * (MAX_CONTENT_CHARS + 1)}
    )
    manager._tools["a"]["add"].input_schema = {"type": "object"}
    large = manager.call_tool("a", "add", {})
    assert large.structured_content == {"omitted": "structured content exceeded the size limit"}
    assert large.content_sha256 == sha256_json(
        {"content": [], "structured_content": large.structured_content}
    )
    assert "new-server-issued-token" not in (tmp_path / "audit.jsonl").read_text()


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
    from matsim_agents.agents.roles.planner import planner_node

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
    from matsim_agents.agents.roles.planner import _SYSTEM_PROMPT, planner_node

    captured: list = []
    _scripted_planner(monkeypatch, [], captured)
    planner_node(MatSimState(objective="relax a.vasp", llm_provider="vllm"))
    assert captured[0].content == _SYSTEM_PROMPT


def test_objective_graph_executes_mcp_tasks_without_mlip_config(
    manager_factory, monkeypatch, tmp_path
):
    from matsim_agents.agents.objective_graph import build_graph

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


def test_checkpoints_hold_redacted_arguments_while_the_tool_gets_live_ones(
    manager_factory, monkeypatch
):
    from langgraph.checkpoint.memory import MemorySaver

    from matsim_agents.agents.objective_graph import build_graph

    secret = "live-token-value-123"
    manager = manager_factory(_stdio("a", allowed_tools=["token_length"]))
    tasks = [
        TaskSpec(
            kind="mcp_tool",
            mcp_server="a",
            mcp_tool="token_length",
            mcp_arguments={"token": secret},
        )
    ]
    _scripted_planner(monkeypatch, tasks)
    saver = MemorySaver()
    config = {"configurable": {"thread_id": "redact-1", "mcp_client": manager}}
    final = build_graph(checkpointer=saver).invoke(
        MatSimState(objective="measure", llm_provider="vllm"), config=config
    )
    assert final["mcp_results"][0].structured_content == {"result": len(secret)}
    stored = [saver.serde.dumps_typed(c.checkpoint)[1] for c in saver.list(config)]
    assert stored and all(secret.encode() not in blob for blob in stored)


@pytest.mark.parametrize("configured", [True, False])
def test_denied_resource_uri_credentials_stay_out_of_checkpoints_and_audit(
    manager_factory, monkeypatch, tmp_path, configured
):
    from langgraph.checkpoint.memory import MemorySaver

    from matsim_agents.agents.objective_graph import build_graph

    manager = manager_factory(_stdio("a", allowed_resources=["test://notes"]))
    uri = "https://u:pw-live-9@h/r?%74oken=tk-live-9"
    tasks = [TaskSpec(kind="mcp_resource", mcp_server="a", mcp_resource_uri=uri)]
    _scripted_planner(monkeypatch, tasks)
    saver = MemorySaver()
    configurable = {"thread_id": f"uri-{configured}"}
    if configured:
        configurable["mcp_client"] = manager
    config = {"configurable": configurable}
    final = build_graph(checkpointer=saver).invoke(
        MatSimState(objective="read", llm_provider="vllm"), config=config
    )
    record = final["mcp_results"][0]
    assert record.outcome == "denied" and record.resource_uri == uri
    stored = [saver.serde.dumps_typed(c.checkpoint)[1] for c in saver.list(config)]
    assert stored
    for secret in (b"pw-live-9", b"tk-live-9"):
        assert all(secret not in blob for blob in stored)
    audit = json.dumps(_audit(tmp_path))
    assert "pw-live-9" not in audit and "tk-live-9" not in audit


@pytest.mark.parametrize(
    "uri",
    [
        "https://u:pw-live-9@h/" + "a" * 3000 + "?token=tk-live-9",
        "https://u:pw-live-9@h/r\\s?token=tk-live-9",
        "https://u:pw-live-9@h/r'q\"?token=tk-live-9",
    ],
    ids=["long", "backslash", "both-quotes"],
)
def test_denied_resource_reason_never_quotes_uri_credentials(manager_factory, tmp_path, uri):
    manager = manager_factory(_stdio("a", allowed_resources=["test://notes"]))
    record = manager.read_resource("a", uri)
    assert record.outcome == "denied"
    manager.close()
    text = json.dumps([record.error, record.authorization_reason, _audit(tmp_path)])
    assert "pw-live-9" not in text and "tk-live-9" not in text


def test_resource_connection_error_keeps_uri_credentials_out_of_disconnect_audit(
    manager_factory, monkeypatch, tmp_path
):
    from mcp import ClientSession

    uri = "https://u:pw-live-9@h/r'q?token=tk-live-9"

    async def broken(self, requested):
        raise ConnectionError(f"lost reading {requested!r} ({requested})")

    monkeypatch.setattr(ClientSession, "read_resource", broken)
    manager = manager_factory(_stdio("a", allowed_resources=[uri]))
    record = manager.read_resource("a", uri)
    assert record.outcome == "connection_error"
    status = manager.server_status()["a"]
    manager.close()
    audit = _audit(tmp_path)
    assert any(e["event"] == "disconnect" for e in audit)
    text = json.dumps([record.error, status, audit])
    assert "pw-live-9" not in text and "tk-live-9" not in text


def test_restored_placeholder_arguments_are_never_sent(manager_factory, tmp_path):
    from matsim_agents.backends.mcp.tasks import execute_mcp_task

    manager = manager_factory(_stdio("a", allowed_tools=["touch"]))
    marker = tmp_path / "marker"
    task = TaskSpec(
        kind="mcp_tool",
        mcp_server="a",
        mcp_tool="touch",
        mcp_arguments={"path": str(marker), "token": "x"},
    )
    restored = TaskSpec.model_validate(task.model_dump())
    assert restored.mcp_arguments["token"] == REDACTED
    record = execute_mcp_task(manager, restored)
    assert record.outcome == "invalid_arguments"
    assert "re-plan" in record.error
    assert not marker.exists()


def test_cli_plan_json_redacts_mcp_arguments(monkeypatch):
    from typer.testing import CliRunner

    from matsim_agents.cli import app

    tasks = [
        TaskSpec(
            kind="mcp_tool", mcp_server="a", mcp_tool="t", mcp_arguments={"password": "pw-live-1"}
        ),
        TaskSpec(
            kind="mcp_resource", mcp_server="a", mcp_resource_uri="https://h/r?token=tk-live-2"
        ),
    ]
    _scripted_planner(monkeypatch, tasks)
    result = CliRunner().invoke(
        app,
        ["objective", "x", "--plan-only", "--json"],
        env={"MATSIM_LLM_PROVIDER": "vllm"},
    )
    assert result.exit_code == 0, result.output
    assert "pw-live-1" not in result.output and "tk-live-2" not in result.output
    planned = json.loads(result.output)
    assert planned[0]["mcp_arguments"] == {"password": REDACTED}
    assert planned[1]["mcp_resource_uri"] == f"https://h/r?token={REDACTED}"


def test_relax_task_still_requires_mlip_config_with_mcp_enabled(manager_factory, monkeypatch):
    from matsim_agents.agents.roles.executor import executor_node

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
            "objective",
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
    result = CliRunner().invoke(app, ["objective", "x", "--mcp-config", str(bad)])
    assert result.exit_code != 0
    assert "requires command" in result.output


def test_runs_without_mcp_config_never_import_the_sdk():
    code = (
        "import sys\n"
        "import matsim_agents.cli\n"
        "from matsim_agents.agents.objective_graph import build_graph\n"
        "from matsim_agents.agents.roles.executor import executor_node\n"
        "from matsim_agents.agents.state import MatSimState, TaskSpec\n"
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
    assert "invalid_arguments: to_unit: violates enum" in out
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


def _catalog_reading_model():
    """A deterministic chat model that plans only from the MCP catalog it is shown.

    It reads the planner's system prompt, picks the catalog tool whose required
    arguments are all integers and match the numbers in the objective, and answers
    through LangChain's real structured-output (tool-calling) path, so the
    planner's own parsing and TaskSpec validation run unchanged.
    """
    import re

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class CatalogReadingModel(BaseChatModel):
        seen_catalogs: list = []

        @property
        def _llm_type(self) -> str:
            return "catalog-reading-test-model"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            system = messages[0].content
            if "<mcp_catalog>" not in system:  # the analyst
                return ChatResult(generations=[ChatGeneration(message=AIMessage("analysis"))])
            catalog = json.loads(system.split("<mcp_catalog>")[1].split("</mcp_catalog>")[0])
            self.seen_catalogs.append(catalog)
            numbers = [int(n) for n in re.findall(r"-?\d+", messages[-1].content)]
            tasks = []
            for tool in catalog["tools"]:
                schema = tool["input_schema"]
                required = schema.get("required", [])
                types_ = [schema["properties"][name].get("type") for name in required]
                if required and types_ == ["integer"] * len(numbers):
                    tasks.append(
                        {
                            "kind": "mcp_tool",
                            "mcp_server": tool["mcp_server"],
                            "mcp_tool": tool["mcp_tool"],
                            "mcp_arguments": dict(zip(required, numbers, strict=True)),
                        }
                    )
            plan = {"tasks": tasks, "rationale": "from catalog"}
            call = {"name": "_Plan", "args": plan, "id": "1"}
            message = AIMessage(content="", tool_calls=[call])
            return ChatResult(generations=[ChatGeneration(message=message)])

    return CatalogReadingModel()


def test_real_planner_selects_permitted_tool_from_discovered_schema(manager_factory, monkeypatch):
    """Planner (real structured-output parsing) -> executor -> real MCP server."""
    import matsim_agents.backends.llm.provider as llm_mod
    from matsim_agents.agents.objective_graph import build_graph

    monkeypatch.delenv("MATSIM_MLIP_CONFIG", raising=False)
    model = _catalog_reading_model()
    monkeypatch.setattr(llm_mod, "get_chat_model", lambda **_: model)
    manager = manager_factory(
        # "slow" takes a number, "touch" a string: only "add" fits two integers.
        _stdio("calc", allowed_tools=["add", "slow", "touch"]),
        # Also offers "add", but nothing is allowlisted, so the planner never sees it.
        _stdio("other", allowed_tools=[]),
    )
    final = build_graph().invoke(
        MatSimState(objective="Add 20 and 22", llm_provider="vllm"),
        config={"configurable": {"thread_id": "planner-e2e", "mcp_client": manager}},
    )
    catalog = model.seen_catalogs[0]
    assert sorted((t["mcp_server"], t["mcp_tool"]) for t in catalog["tools"]) == [
        ("calc", "add"),
        ("calc", "slow"),
        ("calc", "touch"),
    ]
    assert [
        t.model_dump(include={"kind", "mcp_server", "mcp_tool", "mcp_arguments"})
        for t in final["plan"]
    ] == [
        {
            "kind": "mcp_tool",
            "mcp_server": "calc",
            "mcp_tool": "add",
            "mcp_arguments": {"a": 20, "b": 22},
        }
    ]
    [record] = final["mcp_results"]
    assert (record.server, record.tool, record.outcome) == ("calc", "add", "success")
    assert record.structured_content == {"result": 42}
    assert final["analysis"] == "analysis"


def test_resource_only_server_connects_and_serves_allowlisted_resources(manager_factory, tmp_path):
    """A server that does not advertise tools must not be asked for them."""
    resource_only = Path(__file__).resolve().parents[1] / "fixtures" / "mcp_resource_only_server.py"
    manager = manager_factory(
        {
            "name": "docs",
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(resource_only)],
            "allowed_tools": ["whoami"],
            "allowed_resources": ["test://notes"],
        }
    )
    assert manager.server_status()["docs"] == {
        "transport": "stdio",
        "enabled": True,
        "connected": True,
        "error": None,
    }
    assert manager.discovered_tools() == []
    assert manager.permitted_tools() == []
    assert [r.uri for r in manager.permitted_resources()] == ["test://notes"]
    record = manager.read_resource("docs", "test://notes")
    assert (record.outcome, record.content) == ("success", ["resource-only notes"])
    # Allowlisting a tool the server does not offer still denies it.
    assert manager.call_tool("docs", "whoami", {}).outcome == "denied"
    connect = next(e for e in _audit(tmp_path) if e["event"] == "connect")
    assert (connect["outcome"], connect["tools"], connect["resources"]) == (
        "success",
        [],
        ["test://notes"],
    )


def test_connect_timeout_bounds_a_stalled_transport_opening(manager_factory, monkeypatch, tmp_path):
    """connect_timeout_s covers opening the transport, and a timeout unwinds it."""
    cleaned_up = []

    async def stalled_open(self, server, stack):
        stack.callback(cleaned_up.append, server.name)  # stands in for an opened resource
        await anyio.sleep(3600)  # e.g. a transport that never finishes opening

    import anyio

    monkeypatch.setattr(MCPSessionManager, "_open", stalled_open)
    started = time.monotonic()
    manager = manager_factory(_stdio("stalled", allowed_tools=["whoami"], connect_timeout_s=1.0))
    elapsed = time.monotonic() - started
    # Bounded by connect_timeout_s, not by start()'s 30 s settle grace.
    assert elapsed < 10, elapsed
    status = manager.server_status()["stalled"]
    assert (status["connected"], status["error"]) == (False, "timed out")
    assert cleaned_up == ["stalled"]
    assert manager.call_tool("stalled", "whoami", {}).outcome == "denied"
    connect = next(e for e in _audit(tmp_path) if e["event"] == "connect")
    assert (connect["server"], connect["outcome"]) == ("stalled", "timeout")


# ── Copilot review follow-ups ─────────────────────────────────────────────────


@pytest.mark.parametrize("secret", ['pa"ss-word-1', "back\\slash-sec", "pässwörd-xyz"])
def test_schema_errors_scrub_json_escaped_secrets(secret):
    """A schema value is JSON-encoded before scrubbing; escaping must not hide a secret."""
    import jsonschema

    from matsim_agents.backends.mcp.session import _schema_errors

    schema = {"type": "object", "properties": {"x": {"enum": [secret, "other"]}}}
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors({"x": "nope"}))
    described = _schema_errors(errors, (secret,))
    for form in (secret, json.dumps(secret)[1:-1], json.dumps(secret, ensure_ascii=False)[1:-1]):
        assert form not in described
    assert REDACTED in described
    assert "x: violates enum" in described


def test_a_queued_call_past_its_deadline_is_never_sent(manager_factory, monkeypatch, tmp_path):
    """A caller that gave up (and recorded a timeout) must not have its call run later."""
    import matsim_agents.backends.mcp.session as session_module

    manager = manager_factory(_stdio("a", allowed_tools=["touch", "whoami"], call_timeout_s=0.5))
    gate = threading.Event()

    async def occupy_queue(session):  # holds the server's queue, like a long tool call
        import anyio

        await anyio.to_thread.run_sync(gate.wait)
        return None

    blocker = threading.Thread(target=lambda: manager._submit("a", occupy_queue, 30))
    blocker.start()
    marker = tmp_path / "late-call"
    with monkeypatch.context() as patch:
        patch.setattr(session_module, "_SHUTDOWN_GRACE_S", 0.0)  # caller waits call_timeout_s
        expired = manager.call_tool("a", "touch", {"path": str(marker)})
    assert expired.outcome == "timeout"
    gate.set()
    blocker.join(10)
    # Anything still queued for "a" has run once this later call returns.
    assert manager.call_tool("a", "whoami", {}).content == ["a"]
    assert not marker.exists(), "the expired call was sent to the server after its deadline"


def test_interrupted_startup_closes_connections_and_server_processes(tmp_path):
    """An interrupt (Ctrl-C) while start() waits must not leave servers running."""
    pid_file = tmp_path / "pid"
    # A long connect timeout, so only close() (not the timeout) can stop the server.
    config = MCPConfig(
        servers=[_stdio("a", "--hang", "--pid-file", str(pid_file), connect_timeout_s=60)]
    )
    manager = MCPSessionManager(config, audit=MCPAuditLog(tmp_path / "audit.jsonl"))
    real_ready = manager._ready

    class InterruptedWait:
        def wait(self, timeout=None):
            deadline = time.monotonic() + 10
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            raise KeyboardInterrupt

        def set(self):
            real_ready.set()

    manager._ready = InterruptedWait()
    try:
        with pytest.raises(KeyboardInterrupt):
            manager.start()
        pid = int(pid_file.read_text())
        deadline = time.monotonic() + 8
        while not _pid_gone(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert _pid_gone(pid), "the server process outlived the interrupted start()"
        assert not manager._thread.is_alive()
        assert _audit(tmp_path)[-1]["event"] == "shutdown"
    finally:
        if pid_file.exists() and not _pid_gone(int(pid_file.read_text())):
            os.kill(int(pid_file.read_text()), 9)
