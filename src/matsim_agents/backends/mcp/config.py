"""Validated configuration for external MCP servers.

A configuration file holds an ``mcp`` block naming each server, its transport,
and the tools and resources the run may use::

    mcp:
      servers:
        - name: demo
          transport: stdio
          command: python
          args: [demo_server.py]
          allowed_tools: [convert_energy]
          allowed_resources: ["matsim-demo://units"]

Authorization is local and deny-by-default: a tool or resource runs only when
its exact name or URI is listed for that server. Values of ``env`` and
``headers`` may reference host environment variables as ``${VAR}`` so that
credentials stay out of configuration files; they are held as secrets and are
never written to logs or audit records.

This module does not import the MCP SDK, so configurations can be validated
without the optional ``mcp`` extra installed.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

SERVER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# Shorter credentials cannot be scrubbed without corrupting unrelated text.
MIN_SECRET_LENGTH = 4
# Shorter literal env values (LOG_LEVEL: INFO) are settings, not credentials.
_LITERAL_ENV_SECRET_LENGTH = 8


def expand_secret(value: str, *, field: str, referenced: list[str] | None = None) -> SecretStr:
    """Substitute ``${VAR}`` references from the host environment.

    A missing or too-short variable is an error that names the variable, never
    its value. Substituted values are appended to ``referenced`` so they can be
    scrubbed on their own (for example the token inside ``Bearer ${TOKEN}``).
    """

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"{field}: environment variable {name} is not set")
        secret = os.environ[name]
        if len(secret) < MIN_SECRET_LENGTH:
            raise ValueError(
                f"{field}: environment variable {name} is shorter than "
                f"{MIN_SECRET_LENGTH} characters and cannot be redacted reliably"
            )
        if referenced is not None:
            referenced.append(secret)
        return secret

    return SecretStr(_ENV_REFERENCE.sub(substitute, value))


class MCPServerConfig(BaseModel):
    """One external MCP server and what the run may use from it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    transport: Literal["stdio", "streamable_http"]
    enabled: bool = True

    # stdio transport
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: dict[str, SecretStr] = Field(default_factory=dict)

    # streamable_http transport
    url: str | None = None
    headers: dict[str, SecretStr] = Field(default_factory=dict)

    connect_timeout_s: float = Field(10.0, gt=0, le=600)
    call_timeout_s: float = Field(60.0, gt=0, le=3600)

    allowed_tools: list[str] = Field(
        default_factory=list,
        description="Exact tool names this run may call on this server; empty denies all.",
    )
    allowed_resources: list[str] = Field(
        default_factory=list,
        description="Exact resource URIs this run may read from this server; empty denies all.",
    )

    _referenced_secrets: list[str] = PrivateAttr(default_factory=list)

    @field_validator("env", "headers")
    @classmethod
    def _single_line_values(cls, values: dict[str, SecretStr]) -> dict[str, SecretStr]:
        if any(
            "\n" in value.get_secret_value() or "\r" in value.get_secret_value()
            for value in values.values()
        ):
            raise ValueError("multiline env and header values are not supported")
        return values

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not SERVER_NAME.fullmatch(value):
            raise ValueError(
                "server name must start with a letter and contain only letters, digits, "
                "'_' or '-' (max 64 characters)"
            )
        return value

    @field_validator("allowed_tools", "allowed_resources")
    @classmethod
    def _no_wildcards(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value or not value.strip() or value != value.strip():
                raise ValueError("allowlist entries must be non-empty and unpadded")
            if value in {"*", "all"}:
                raise ValueError("wildcard allowlists are not supported; list each entry")
        if len(set(values)) != len(values):
            raise ValueError("allowlist entries must be unique")
        return values

    @model_validator(mode="after")
    def _transport_fields(self) -> MCPServerConfig:
        if self.transport == "stdio":
            if not self.command:
                raise ValueError(f"server {self.name!r}: stdio transport requires command")
            if self.url or self.headers:
                raise ValueError(f"server {self.name!r}: url/headers apply only to streamable_http")
        else:
            if not self.url:
                raise ValueError(f"server {self.name!r}: streamable_http transport requires url")
            if not re.match(r"^https?://", self.url):
                raise ValueError(f"server {self.name!r}: url must start with http:// or https://")
            # HTTP client errors echo URL parts, so reject malformed URLs here, unechoed.
            malformed = ValueError(f"server {self.name!r}: url is malformed")
            if any(char.isspace() or not char.isprintable() for char in self.url):
                raise malformed
            try:
                parts = urlsplit(self.url)
                parts.port  # noqa: B018 - raises ValueError for a non-numeric port
            except ValueError:
                raise malformed from None
            if parts.username is not None or parts.password is not None:
                raise ValueError(
                    f"server {self.name!r}: put credentials in headers, not in the url"
                )
            if not parts.hostname:
                raise malformed
            for key, value in self.headers.items():
                token = value.get_secret_value().split()[-1:] or [""]
                if len(token[0]) < MIN_SECRET_LENGTH:
                    raise ValueError(
                        f"server {self.name!r}: header {key} is shorter than "
                        f"{MIN_SECRET_LENGTH} characters and cannot be redacted reliably"
                    )
            if self.command or self.args or self.env or self.cwd:
                raise ValueError(f"server {self.name!r}: command/args/env/cwd apply only to stdio")
        return self

    def secret_values(self) -> list[str]:
        """Credential values used to scrub audit records, logs, and outputs.

        Every header value and its final token (the credential in ``Bearer X``),
        every value substituted from ``${VAR}``, and literal env values of at
        least 8 characters. Longest first, so a value is replaced before any
        shorter value it contains.
        """
        values = set(self._referenced_secrets)
        for header in self.headers.values():
            text = header.get_secret_value()
            values.add(text)
            values.update(text.split()[-1:])
        values.update(
            text
            for text in (v.get_secret_value() for v in self.env.values())
            if len(text) >= _LITERAL_ENV_SECRET_LENGTH
        )
        return sorted((v for v in values if len(v) >= MIN_SECRET_LENGTH), key=len, reverse=True)


class MCPConfig(BaseModel):
    """All configured MCP servers for one run."""

    model_config = ConfigDict(extra="forbid")

    servers: list[MCPServerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_names(self) -> MCPConfig:
        names = [server.name for server in self.servers]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate MCP server name(s): {', '.join(duplicates)}")
        return self

    def server(self, name: str) -> MCPServerConfig | None:
        return next((server for server in self.servers if server.name == name), None)

    def enabled_servers(self) -> list[MCPServerConfig]:
        return [server for server in self.servers if server.enabled]

    def secret_values(self) -> list[str]:
        values = {value for server in self.servers for value in server.secret_values()}
        return sorted(values, key=len, reverse=True)


def parse_mcp_config(block: dict, *, base_dir: Path | None = None) -> MCPConfig:
    """Validate an ``mcp`` block, expanding ``${VAR}`` only in env and header values."""
    if not isinstance(block, dict):
        raise ValueError("mcp block must be a mapping")
    servers = block.get("servers")
    if not isinstance(servers, list):
        raise ValueError("mcp block must define a servers list")
    prepared = []
    referenced: list[list[str]] = []
    for index, raw in enumerate(servers):
        if not isinstance(raw, dict):
            raise ValueError(f"mcp.servers[{index}] must be a mapping")
        server = dict(raw)
        label = f"mcp.servers[{index}]"
        server_secrets: list[str] = []
        referenced.append(server_secrets)
        for key in ("env", "headers"):
            mapping = server.get(key) or {}
            if not isinstance(mapping, dict):
                raise ValueError(f"{label}.{key} must be a mapping")
            server[key] = {
                str(name): expand_secret(
                    str(value), field=f"{label}.{key}.{name}", referenced=server_secrets
                )
                for name, value in mapping.items()
            }
        if base_dir is not None and server.get("transport") == "stdio":
            cwd = server.get("cwd")
            resolved = Path(cwd).expanduser() if cwd else base_dir
            if not resolved.is_absolute():
                resolved = base_dir / resolved
            server["cwd"] = str(resolved)
        prepared.append(server)
    config = MCPConfig.model_validate({**block, "servers": prepared})
    for server, secrets in zip(config.servers, referenced, strict=True):
        server._referenced_secrets = secrets
    return config


def load_mcp_config(path: str | Path) -> MCPConfig:
    """Read the ``mcp`` block of a YAML file.

    A relative stdio ``cwd`` resolves against the file's directory, which is
    also the default working directory for stdio servers.
    """
    import yaml

    path = Path(path).expanduser().resolve()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ValueError(f"{path}: invalid YAML{location}") from None
    if not isinstance(data, dict) or "mcp" not in data:
        raise ValueError(f"{path}: expected a mapping with an mcp block")
    try:
        return parse_mcp_config(data["mcp"], base_dir=path.parent)
    except ValidationError as error:
        details = [
            f"{'.'.join(map(str, item['loc'])) or 'mcp'}: {item['msg']}"
            for item in error.errors(include_input=False, include_context=False, include_url=False)
        ]
        raise ValueError(f"{path}: {'; '.join(details)}") from None
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
