"""MCP tool descriptions, call records, redaction, and the JSONL audit log.

Call records travel in workflow state so downstream agents can use the
results; the audit log keeps the same attempts as an append-only JSON-lines
file next to the run's other workflow audit records. MCP outputs are external
data: they are not scientific evidence and are not written to the evidence
ledger.

This module does not import the MCP SDK.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote_plus, urlsplit, urlunsplit

from pydantic import BaseModel, Field, field_serializer

REDACTED = "[REDACTED]"
# Brackets in a netloc make urlsplit reject the URI, so parse with a bracket-free stand-in.
_URI_PLACEHOLDER = "__matsim_redacted__"
MAX_CONTENT_CHARS = 8000
MAX_DESCRIPTION_CHARS = 1000
_SENSITIVE_KEY = re.compile(
    r"pass(word|wd|phrase)?|secret|token|api[_-]?key|^auth(orization)?$|credential|cookie"
    r"|private[_-]?key|session[_-]?id",
    re.IGNORECASE,
)

Outcome = Literal[
    "success",
    "denied",
    "invalid_arguments",
    "tool_error",
    "protocol_error",
    "timeout",
    "connection_error",
]


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def sha256_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scrub_text(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    return text


def redact(value: Any, secrets: Iterable[str] = (), *, sensitive_keys: bool = True) -> Any:
    """Copy ``value`` with configured secret values (and, by default, the values of
    credential-like keys such as ``password`` or ``api_key``) replaced."""
    secrets = tuple(secrets)
    if isinstance(value, dict):
        return {
            scrub_text(str(key), secrets): REDACTED
            if sensitive_keys and _SENSITIVE_KEY.search(str(key))
            else redact(item, secrets, sensitive_keys=sensitive_keys)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact(item, secrets, sensitive_keys=sensitive_keys) for item in value]
    if isinstance(value, str):
        return scrub_text(value, secrets)
    return value


def redact_uri(uri: str) -> str:
    """Copy ``uri`` with a userinfo password and credential-like query values replaced."""
    try:
        parts = urlsplit(uri.replace(REDACTED, _URI_PLACEHOLDER))
        password = parts.password
    except ValueError:
        return REDACTED
    netloc = parts.netloc
    if password is not None:
        user, _, host = netloc.rpartition("@")
        netloc = f"{user.partition(':')[0]}:{_URI_PLACEHOLDER}@{host}"
    query = []
    for pair in parts.query.split("&") if parts.query else []:
        key = pair.partition("=")[0]
        sensitive = _SENSITIVE_KEY.search(unquote_plus(key))
        query.append(f"{key}={_URI_PLACEHOLDER}" if sensitive else pair)
    redacted = urlunsplit(parts._replace(netloc=netloc, query="&".join(query)))
    return redacted.replace(_URI_PLACEHOLDER, REDACTED)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} characters]"


class MCPToolInfo(BaseModel):
    """A tool discovered on a server; the input schema is kept as the server sent it."""

    server: str
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPResourceInfo(BaseModel):
    server: str
    uri: str
    name: str = ""
    description: str = ""
    mime_type: str | None = None


class MCPCallRecord(BaseModel):
    """One attempted MCP tool call or resource read and what came of it."""

    kind: Literal["tool", "resource"]
    server: str
    tool: str | None = None
    resource_uri: str | None = None
    started_at: str
    finished_at: str
    authorized: bool
    authorization_reason: str
    outcome: Outcome
    error: str | None = None
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Invocation arguments with sensitive values redacted."
    )
    arguments_sha256: str | None = None
    content: list[str] = Field(
        default_factory=list, description="Text content returned by the server (untrusted)."
    )
    structured_content: Any = None
    content_sha256: str | None = None

    @field_serializer("resource_uri")
    def _redact_resource_uri(self, value: str | None) -> str | None:
        return redact_uri(value) if value else value

    @property
    def target(self) -> str:
        if self.kind == "tool":
            return f"{self.server}/{self.tool}"
        return f"{self.server}/{redact_uri(self.resource_uri or '')}"

    def summary_line(self) -> str:
        if self.outcome == "success":
            preview = " ".join(self.content).strip().replace("\n", " ")
            return f"MCP {self.kind} {self.target}: success: {truncate(preview, 200)}"
        return f"MCP {self.kind} {self.target}: {self.outcome}: {self.error or ''}".rstrip(": ")

    def audit_payload(self) -> dict[str, Any]:
        """The audit view: identity, decision, outcome, and hashes, not the output itself."""
        payload = self.model_dump(exclude={"content", "structured_content"})
        payload["content_chars"] = sum(len(part) for part in self.content)
        return payload


class MCPAuditLog:
    """Append-only JSON-lines audit of MCP connections, calls, and shutdown."""

    def __init__(self, path: str | Path, *, secrets: Iterable[str] = ()) -> None:
        self.path = Path(path)
        self._secrets = tuple(secrets)
        self._lock = threading.Lock()  # connection events and calls come from two threads

    def write(self, event: str, **fields: Any) -> None:
        record = {"timestamp_utc": now_utc(), "event": event, **fields}
        # Call records are key-redacted already; this scrubs configured secret values.
        scrubbed = redact(record, self._secrets, sensitive_keys=False)
        line = json.dumps(scrubbed, sort_keys=True, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

    def record_call(self, record: MCPCallRecord) -> None:
        self.write(f"{record.kind}_call", **record.audit_payload())
