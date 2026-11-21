"""MCP configuration, authorization, task, and redaction tests (no MCP SDK required)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from matsim_agents.backends.mcp.authorization import MCPAuthorizer
from matsim_agents.backends.mcp.config import MCPConfig, load_mcp_config, parse_mcp_config
from matsim_agents.backends.mcp.records import (
    MAX_DESCRIPTION_CHARS,
    REDACTED,
    MCPAuditLog,
    MCPCallRecord,
    MCPResourceInfo,
    MCPToolInfo,
    redact,
)
from matsim_agents.backends.mcp.tasks import execute_mcp_task, planner_catalog
from matsim_agents.orchestration.state import TaskSpec


def _stdio(name="demo", **extra):
    return {"name": name, "transport": "stdio", "command": "python", **extra}


# ── configuration ─────────────────────────────────────────────────────────────


def test_valid_config_from_yaml_resolves_cwd_and_expands_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_TOKEN", "s3cret-token-value")
    path = tmp_path / "mcp.yaml"
    path.write_text(
        """
mcp:
  servers:
    - name: demo
      transport: stdio
      command: python
      args: [server.py]
      env: {DEMO_TOKEN: "${DEMO_TOKEN}"}
      allowed_tools: [convert_energy]
    - name: remote
      transport: streamable_http
      url: http://127.0.0.1:9/mcp
      headers: {Authorization: "Bearer ${DEMO_TOKEN}"}
      enabled: false
"""
    )
    config = load_mcp_config(path)
    demo, remote = config.servers
    assert demo.cwd == str(tmp_path)
    assert demo.env["DEMO_TOKEN"].get_secret_value() == "s3cret-token-value"
    assert remote.headers["Authorization"].get_secret_value() == "Bearer s3cret-token-value"
    assert [s.name for s in config.enabled_servers()] == ["demo"]
    # Secrets never appear in dumps or reprs.
    assert "s3cret" not in repr(config)
    assert "s3cret" not in config.model_dump_json()
    # The bare token is scrubbed too, so a server echoing only the token is caught.
    assert set(config.secret_values()) == {"s3cret-token-value", "Bearer s3cret-token-value"}
    assert config.secret_values()[0] == "Bearer s3cret-token-value"  # longest first


def test_secret_values_cover_references_and_long_literals_only(monkeypatch):
    monkeypatch.setenv("REF", "abcd1234")
    config = parse_mcp_config(
        {
            "servers": [
                _stdio(
                    env={
                        "WRAPPED": "token=${REF}",
                        "LOG_LEVEL": "INFO",
                        "LITERAL_KEY": "literal-key-value",
                    }
                )
            ]
        }
    )
    secrets = set(config.secret_values())
    assert {"abcd1234", "token=abcd1234", "literal-key-value"} <= secrets
    assert "INFO" not in secrets  # short literal settings are not treated as credentials


def test_credentials_too_short_to_redact_and_url_credentials_are_rejected(monkeypatch):
    monkeypatch.setenv("SHORT", "abc")
    with pytest.raises(ValueError, match="SHORT is shorter than 4"):
        parse_mcp_config({"servers": [_stdio(env={"T": "${SHORT}"})]})
    http = {"name": "h", "transport": "streamable_http", "url": "http://127.0.0.1:1/mcp"}
    with pytest.raises(ValueError, match="header Authorization is shorter"):
        parse_mcp_config({"servers": [http | {"headers": {"Authorization": "Bearer ab"}}]})
    with pytest.raises(ValueError, match="credentials in headers"):
        parse_mcp_config({"servers": [http | {"url": "http://user:pw@127.0.0.1:1/mcp"}]})


@pytest.mark.parametrize(
    ("server", "message"),
    [
        ({"name": "bad name", "transport": "stdio", "command": "x"}, "server name"),
        ({"name": "a", "transport": "stdio"}, "requires command"),
        ({"name": "a", "transport": "streamable_http"}, "requires url"),
        ({"name": "a", "transport": "streamable_http", "url": "ftp://x"}, "http"),
        ({"name": "a", "transport": "streamable_http", "url": "http://x", "command": "y"}, "stdio"),
        ({"name": "a", "transport": "stdio", "command": "x", "url": "http://x"}, "streamable"),
        ({"name": "a", "transport": "websocket", "command": "x"}, "transport"),
        (_stdio(allowed_tools=["*"]), "wildcard"),
        (_stdio(allowed_tools=["t", "t"]), "unique"),
        (_stdio(connect_timeout_s=0), "greater than 0"),
        (_stdio(unknown_field=1), "Extra inputs"),
    ],
)
def test_invalid_server_configs_are_rejected(server, message):
    with pytest.raises((ValidationError, ValueError), match=message):
        parse_mcp_config({"servers": [server]})


def test_duplicate_server_names_and_missing_env_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="duplicate MCP server name"):
        parse_mcp_config({"servers": [_stdio(), _stdio()]})
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    with pytest.raises(ValueError, match="NOT_SET_ANYWHERE is not set"):
        parse_mcp_config({"servers": [_stdio(env={"X": "${NOT_SET_ANYWHERE}"})]})


def test_unknown_mcp_block_fields_are_rejected():
    with pytest.raises(ValidationError, match="unknown_field"):
        parse_mcp_config({"servers": [_stdio()], "unknown_field": 1})


def test_yaml_without_mcp_block_is_rejected(tmp_path):
    path = tmp_path / "other.yaml"
    path.write_text("mlip: {backend: mace}\n")
    with pytest.raises(ValueError, match="mcp block"):
        load_mcp_config(path)


# ── authorization ─────────────────────────────────────────────────────────────


def _authorizer():
    config = MCPConfig(
        servers=[
            _stdio("a", allowed_tools=["whoami"], allowed_resources=["test://notes"]),
            _stdio("b", allowed_tools=[]),
            _stdio("off", enabled=False, allowed_tools=["whoami"]),
        ]
    )
    return MCPAuthorizer(config)


def test_authorization_is_deny_by_default_and_per_server():
    auth = _authorizer()
    connected = {"a": True, "b": True, "off": False}
    discovered = {"a": {"whoami", "unlisted"}, "b": {"whoami"}}

    def check(server, tool):
        return auth.check_tool(server, tool, connected=connected, discovered_tools=discovered)

    assert check("a", "whoami").allowed
    # Same tool name on another server is not approved by server a's allowlist.
    assert not check("b", "whoami").allowed
    assert "not in allowed_tools" in check("a", "unlisted").reason
    assert "unknown server" in check("zzz", "whoami").reason
    assert "disabled" in check("off", "whoami").reason


def test_authorization_requires_connection_and_discovery():
    auth = _authorizer()
    allowlisted_but_down = auth.check_tool(
        "a", "whoami", connected={"a": False}, discovered_tools={"a": {"whoami"}}
    )
    assert not allowlisted_but_down.allowed
    assert "no fallback" in allowlisted_but_down.reason
    not_offered = auth.check_tool("a", "whoami", connected={"a": True}, discovered_tools={})
    assert not not_offered.allowed
    assert "does not offer" in not_offered.reason
    assert auth.check_resource("a", "test://notes", connected={"a": True}).allowed
    assert not auth.check_resource("a", "test://hidden", connected={"a": True}).allowed


# ── tasks, records, redaction ─────────────────────────────────────────────────


def test_mcp_tasks_need_no_relaxation_fields_and_relax_still_needs_a_structure():
    task = TaskSpec(kind="mcp_tool", mcp_server="a", mcp_tool="whoami")
    assert task.structure_path is None
    TaskSpec(kind="mcp_resource", mcp_server="a", mcp_resource_uri="test://notes")
    with pytest.raises(ValidationError, match="mcp_server and mcp_tool"):
        TaskSpec(kind="mcp_tool", mcp_server="a")
    with pytest.raises(ValidationError, match="mcp_resource_uri"):
        TaskSpec(kind="mcp_resource", mcp_server="a")
    with pytest.raises(ValidationError, match="require structure_path"):
        TaskSpec(kind="relax")
    assert TaskSpec(structure_path="a.vasp").kind == "relax"


def test_mcp_task_without_configuration_is_denied_and_recorded():
    task = TaskSpec(
        kind="mcp_tool", mcp_server="a", mcp_tool="whoami", mcp_arguments={"api_key": "k"}
    )
    record = execute_mcp_task(None, task)
    assert record.outcome == "denied"
    assert not record.authorized
    assert "--mcp-config" in record.error
    assert record.arguments == {"api_key": REDACTED}
    assert planner_catalog(None) is None


def test_redaction_of_keys_and_configured_secret_values():
    payload = {
        "query": "uses s3cret-value inline",
        "password": "hunter2",
        "nested": [{"Authorization": "Bearer x"}, {"author": "kept"}],
    }
    redacted = redact(payload, ["s3cret-value"])
    assert redacted["query"] == f"uses {REDACTED} inline"
    assert redacted["password"] == REDACTED
    assert redacted["nested"][0]["Authorization"] == REDACTED
    assert redacted["nested"][1]["author"] == "kept"


def test_planner_catalog_scrubs_secrets_without_mutating_execution_schemas():
    from unittest.mock import MagicMock

    secret = "secret-credential"
    schema = {
        "type": "object",
        "properties": {"token": {"type": "string", "default": secret}, secret: {"type": "string"}},
    }
    tool = MCPToolInfo(
        server="a",
        name="tool",
        description="x" * (MAX_DESCRIPTION_CHARS - 3) + secret,
        input_schema=schema,
    )
    resource = MCPResourceInfo(server="a", uri="test://notes", description=secret)
    client = MagicMock()
    client.config.secret_values.return_value = [secret]
    client.permitted_tools.return_value = [tool]
    client.permitted_resources.return_value = [resource]

    prompt = planner_catalog(client)
    assert secret not in prompt
    catalog = json.loads(prompt.split("<mcp_catalog>")[1].split("</mcp_catalog>")[0])
    assert secret[:5] not in catalog["tools"][0]["description"]
    assert catalog["tools"][0]["input_schema"]["properties"]["token"]["default"] == REDACTED
    assert REDACTED in catalog["tools"][0]["input_schema"]["properties"]
    assert catalog["resources"][0]["description"] == REDACTED
    assert tool.input_schema == schema
    assert tool.input_schema["properties"]["token"]["default"] == secret


def test_audit_log_scrubs_secrets_and_omits_content(tmp_path):
    log = MCPAuditLog(tmp_path / "audit.jsonl", secrets=["s3cret-value"])
    record = MCPCallRecord(
        kind="tool",
        server="a",
        tool="whoami",
        started_at="t0",
        finished_at="t1",
        authorized=True,
        authorization_reason="tool is allowlisted and offered by the server",
        outcome="tool_error",
        error="server echoed s3cret-value",
        content=["output mentioning s3cret-value"],
    )
    log.record_call(record)
    line = (tmp_path / "audit.jsonl").read_text()
    entry = json.loads(line)
    assert "s3cret-value" not in line
    assert entry["event"] == "tool_call"
    assert entry["authorized"] is True
    assert entry["authorization_reason"].startswith("tool is allowlisted")
    assert "content" not in entry
    assert entry["content_chars"] == len("output mentioning s3cret-value")
