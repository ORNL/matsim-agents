"""Local, deny-by-default authorization for external MCP tools and resources.

Decisions use only the run's MCP configuration and what the connection layer
actually discovered. Nothing a server sends (tool descriptions, annotations,
or results) can widen the allowlist, and approval of one server's tool never
approves the same name on another server.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from matsim_agents.backends.mcp.config import MCPConfig


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: str


def _deny(reason: str) -> AuthorizationDecision:
    return AuthorizationDecision(False, reason)


class MCPAuthorizer:
    """Checks a requested server/tool or server/resource against the local policy."""

    def __init__(self, config: MCPConfig) -> None:
        self._config = config

    def _server_decision(self, server: str) -> AuthorizationDecision | None:
        configured = self._config.server(server)
        if configured is None:
            return _deny(f"unknown server {server!r}")
        if not configured.enabled:
            return _deny(f"server {server!r} is disabled")
        return None

    def check_tool(
        self,
        server: str,
        tool: str,
        *,
        connected: Mapping[str, bool],
        discovered_tools: Mapping[str, set[str]],
    ) -> AuthorizationDecision:
        denied = self._server_decision(server)
        if denied:
            return denied
        if tool not in self._config.server(server).allowed_tools:
            return _deny(f"tool {tool!r} is not in allowed_tools for server {server!r}")
        if not connected.get(server, False):
            return _deny(f"server {server!r} is not connected; no fallback is attempted")
        if tool not in discovered_tools.get(server, set()):
            return _deny(f"server {server!r} does not offer tool {tool!r}")
        return AuthorizationDecision(True, "tool is allowlisted and offered by the server")

    def check_resource(
        self, server: str, uri: str, *, connected: Mapping[str, bool]
    ) -> AuthorizationDecision:
        denied = self._server_decision(server)
        if denied:
            return denied
        if uri not in self._config.server(server).allowed_resources:
            return _deny(f"resource {uri!r} is not in allowed_resources for server {server!r}")
        if not connected.get(server, False):
            return _deny(f"server {server!r} is not connected; no fallback is attempted")
        return AuthorizationDecision(True, "resource is allowlisted")
