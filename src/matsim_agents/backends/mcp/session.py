"""Connections to external MCP servers through the official MCP Python SDK.

:class:`MCPSessionManager` owns every MCP connection of one run. The LangGraph
workflow is synchronous, while the SDK is asynchronous and its transports hold
task groups that must be entered and exited by the same task. The manager
therefore runs one event loop on one background thread for the whole run, and
each server is served by its own task that opens the transport, initializes
the session, discovers tools and resources, then executes requests from a
queue until shutdown. A failing server only ends its own task.

Every request is authorized locally before it reaches a server, and every
attempt (allowed or not) is returned as an :class:`MCPCallRecord` and written to
the audit log.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import math
import os
import threading
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import anyio
import httpx2
import jsonschema
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mcp.shared.exceptions import MCPError

from matsim_agents.backends.mcp.authorization import AuthorizationDecision, MCPAuthorizer
from matsim_agents.backends.mcp.config import MCPConfig, MCPServerConfig
from matsim_agents.backends.mcp.records import (
    MAX_CONTENT_CHARS,
    MCPAuditLog,
    MCPCallRecord,
    MCPResourceInfo,
    MCPToolInfo,
    Outcome,
    now_utc,
    redact,
    scrub_text,
    sha256_json,
    truncate,
)

_MAX_PAGES = 100
_SHUTDOWN_GRACE_S = 30.0

_Op = Callable[[ClientSession], Awaitable[MCPCallRecord]]


def _schema_errors(errors: list[Any]) -> str:
    """Describe schema violations by location and rule, never by the offending value,
    so a credential passed in the wrong field is not echoed into records."""
    parts = []
    for error in sorted(errors, key=lambda e: [str(part) for part in e.path]):
        location = "/".join(map(str, error.path)) or "<arguments>"
        rule = json.dumps(error.validator_value, default=str)
        parts.append(f"{location}: violates {error.validator} {truncate(rule, 200)}")
    return "; ".join(parts)


class MCPUnavailableError(RuntimeError):
    """A request was sent to a server whose connection is closed."""


def _describe(exc: BaseException) -> str:
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    if isinstance(exc, MCPError):
        return f"MCPError {exc.code}: {exc.error.message}"
    if isinstance(exc, TimeoutError):
        return "timed out"
    message = str(exc) or exc.__class__.__name__
    return f"{exc.__class__.__name__}: {message}"


class MCPSessionManager:
    """Owns the run's MCP sessions; use as a context manager or call start/close."""

    def __init__(
        self,
        config: MCPConfig,
        *,
        audit: MCPAuditLog | None = None,
        log_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self._authorizer = MCPAuthorizer(config)
        self._secrets = tuple(config.secret_values())
        self._audit = audit
        self._log_dir = Path(log_dir) if log_dir else None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._queues: dict[str, asyncio.Queue | None] = {}
        self._connected: dict[str, bool] = {}
        self._errors: dict[str, str] = {}
        self._tools: dict[str, dict[str, MCPToolInfo]] = {}
        self._resources: dict[str, list[MCPResourceInfo]] = {}
        self._started = False
        self._closed = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def __enter__(self) -> MCPSessionManager:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def start(self) -> None:
        """Connect to every enabled server; unreachable servers are recorded, not fatal."""
        if self._started:
            raise RuntimeError("MCPSessionManager.start() may only be called once")
        self._started = True
        self._thread = threading.Thread(target=self._run_loop, name="matsim-mcp", daemon=True)
        self._thread.start()
        servers = self.config.enabled_servers()
        wait = max((s.connect_timeout_s for s in servers), default=0.0) + _SHUTDOWN_GRACE_S
        if not self._ready.wait(wait):
            self.close()
            raise RuntimeError("MCP connections did not settle within the connect timeout")
        if self._startup_error is not None:
            self.close()
            raise RuntimeError(
                f"MCP session manager failed: {_describe(self._startup_error)}"
            ) from self._startup_error

    def close(self) -> None:
        """Close every session and stop stdio server processes. Safe to call twice."""
        if self._closed or not self._started:
            self._closed = True
            return
        self._closed = True
        loop, stop = self._loop, self._stop
        if loop is not None and stop is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):  # the loop has already finished
                loop.call_soon_threadsafe(stop.set)
        if self._thread is not None:
            self._thread.join(timeout=_SHUTDOWN_GRACE_S)
        clean = self._thread is None or not self._thread.is_alive()
        if self._audit:
            self._audit.write("shutdown", clean=clean, servers=self.server_status())

    def _run_loop(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as exc:  # noqa: BLE001 - reported to the caller of start()
            self._startup_error = exc
        finally:
            self._ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        servers = self.config.enabled_servers()
        settled = {server.name: asyncio.Event() for server in servers}
        for server in self.config.servers:
            self._connected[server.name] = False
            if not server.enabled:
                self._errors[server.name] = "disabled in configuration"
        async with anyio.create_task_group() as group:
            for server in servers:
                queue: asyncio.Queue = asyncio.Queue()
                self._queues[server.name] = queue
                group.start_soon(self._serve, server, queue, settled[server.name])
            for event in settled.values():
                await event.wait()
            self._ready.set()
            await self._stop.wait()
            for queue in self._queues.values():
                if queue is not None:
                    queue.put_nowait(None)

    async def _open(self, server: MCPServerConfig, stack: AsyncExitStack) -> ClientSession:
        if server.transport == "stdio":
            errlog = self._stderr_sink(server.name, stack)
            params = StdioServerParameters(
                command=server.command,
                args=list(server.args),
                env={key: value.get_secret_value() for key, value in server.env.items()},
                cwd=server.cwd,
            )
            read, write = await stack.enter_async_context(stdio_client(params, errlog=errlog))
        else:
            http_client = create_mcp_http_client(
                headers={key: value.get_secret_value() for key, value in server.headers.items()},
                timeout=httpx2.Timeout(server.call_timeout_s, connect=server.connect_timeout_s),
            )
            await stack.enter_async_context(http_client)
            read, write = await stack.enter_async_context(
                streamable_http_client(server.url, http_client=http_client)
            )
        return await stack.enter_async_context(
            # Deadlines are applied per operation (connect_timeout_s for the
            # handshake and discovery, call_timeout_s for each request).
            ClientSession(
                read,
                write,
                client_info=types.Implementation(name="matsim-agents", version="0.1.0"),
            )
        )

    def _stderr_sink(self, name: str, stack: AsyncExitStack):
        """Where a stdio server's stderr goes: a scrubbed log file, or nowhere.

        The server writes into a pipe; a thread copies each line to the log with
        configured credentials replaced, so a server that prints its environment
        cannot leak them into the run directory.
        """
        if self._log_dir is None:
            sink = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 - closed by stack
            stack.callback(sink.close)
            return sink
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"{name}.stderr.log"
        read_fd, write_fd = os.pipe()
        reader = os.fdopen(read_fd, "r", encoding="utf-8", errors="replace")
        writer = os.fdopen(write_fd, "w", encoding="utf-8")
        secrets = self._secrets

        def copy() -> None:
            with reader, log_path.open("a", encoding="utf-8") as log:
                for line in reader:
                    log.write(scrub_text(line, secrets))
                    log.flush()

        thread = threading.Thread(target=copy, name=f"matsim-mcp-{name}-stderr", daemon=True)
        thread.start()
        # Exit order (last registered runs first): the transport stops the server,
        # then our write end closes, then the copier sees EOF and finishes.
        stack.callback(thread.join, 5.0)
        stack.callback(writer.close)
        return writer

    async def _discover(self, server: MCPServerConfig, session: ClientSession) -> dict[str, Any]:
        init = await session.initialize()
        tools: dict[str, MCPToolInfo] = {}
        cursor = None
        # A server that does not advertise tools (e.g. resources only) is valid;
        # asking it for tools would fail the whole connection.
        for _ in range(_MAX_PAGES if init.capabilities.tools is not None else 0):
            params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
            listing = await session.list_tools(params=params)
            for tool in listing.tools:
                tools[tool.name] = MCPToolInfo(
                    server=server.name,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=dict(tool.input_schema or {}),
                )
            cursor = listing.next_cursor
            if not cursor:
                break
        resources: list[MCPResourceInfo] = []
        if init.capabilities.resources is not None:
            listing = await session.list_resources()
            resources = [
                MCPResourceInfo(
                    server=server.name,
                    uri=str(resource.uri),
                    name=resource.name or "",
                    description=resource.description or "",
                    mime_type=resource.mime_type,
                )
                for resource in listing.resources
            ]
        self._tools[server.name] = tools
        self._resources[server.name] = resources
        info = init.server_info
        return {"server_name": info.name, "server_version": info.version}

    async def _serve_connected(
        self,
        server: MCPServerConfig,
        session: ClientSession,
        identity: dict[str, Any],
        queue: asyncio.Queue,
        settled: asyncio.Event,
    ) -> None:
        """Mark the server connected, then execute queued requests until shutdown."""
        name = server.name
        self._connected[name] = True
        self._errors.pop(name, None)
        self._audit_event(
            "connect",
            server=name,
            transport=server.transport,
            outcome="success",
            tools=sorted(self._tools[name]),
            resources=[r.uri for r in self._resources[name]],
            **identity,
        )
        settled.set()
        while (item := await queue.get()) is not None:
            op, future = item
            record = await op(session)
            if not future.done():
                future.set_result(record)

    async def _serve(
        self, server: MCPServerConfig, queue: asyncio.Queue, settled: asyncio.Event
    ) -> None:
        name = server.name
        try:
            # connect_timeout_s bounds opening the transport, the handshake, and
            # discovery. The scope encloses the exit stack, so a timeout unwinds
            # everything opened so far (the SDK shields stdio process shutdown),
            # and its deadline is lifted once the server is connected.
            deadline = anyio.current_time() + server.connect_timeout_s
            with anyio.CancelScope(deadline=deadline) as connect_scope:
                async with AsyncExitStack() as stack:
                    session = await self._open(server, stack)
                    identity = await self._discover(server, session)
                    connect_scope.deadline = math.inf
                    await self._serve_connected(server, session, identity, queue, settled)
            if connect_scope.cancelled_caught:
                raise TimeoutError
        except Exception as exc:  # noqa: BLE001 - recorded and surfaced per request
            error = scrub_text(_describe(exc), self._secrets)
            event = "disconnect" if self._connected.get(name) else "connect"
            self._errors[name] = error
            outcome = "timeout" if isinstance(exc, TimeoutError) else "connection_error"
            self._audit_event(
                event, server=name, transport=server.transport, outcome=outcome, error=error
            )
        finally:
            self._connected[name] = False
            self._queues[name] = None
            settled.set()
            while not queue.empty():
                item = queue.get_nowait()
                if item is not None and not item[1].done():
                    item[1].set_exception(MCPUnavailableError(f"server {name!r} disconnected"))

    # ── discovery views ───────────────────────────────────────────────────────

    def server_status(self) -> dict[str, dict[str, Any]]:
        return {
            server.name: {
                "transport": server.transport,
                "enabled": server.enabled,
                "connected": self._connected.get(server.name, False),
                "error": self._errors.get(server.name),
            }
            for server in self.config.servers
        }

    def discovered_tools(self) -> list[MCPToolInfo]:
        """Every tool the servers offered, permitted or not."""
        return [tool for tools in self._tools.values() for tool in tools.values()]

    def permitted_tools(self) -> list[MCPToolInfo]:
        """Tools that are allowlisted, offered, and on a connected server."""
        return [
            tool
            for tool in self.discovered_tools()
            if self._check_tool(tool.server, tool.name).allowed
        ]

    def permitted_resources(self) -> list[MCPResourceInfo]:
        permitted = []
        for server in self.config.enabled_servers():
            listed = {r.uri: r for r in self._resources.get(server.name, [])}
            for uri in server.allowed_resources:
                if self._authorizer.check_resource(
                    server.name, uri, connected=self._connected
                ).allowed:
                    fallback = MCPResourceInfo(server=server.name, uri=uri)
                    permitted.append(listed.get(uri) or fallback)
        return permitted

    def _check_tool(self, server: str, tool: str) -> AuthorizationDecision:
        return self._authorizer.check_tool(
            server,
            tool,
            connected=self._connected,
            discovered_tools={name: set(tools) for name, tools in self._tools.items()},
        )

    # ── requests ──────────────────────────────────────────────────────────────

    def _audit_event(self, event: str, **fields: Any) -> None:
        if self._audit:
            self._audit.write(event, **fields)

    def _finish(self, record: MCPCallRecord) -> MCPCallRecord:
        if self._audit:
            self._audit.record_call(record)
        return record

    def _submit(self, server: str, op: _Op, timeout: float) -> MCPCallRecord:
        if self._loop is None or self._closed:
            raise MCPUnavailableError("the MCP session manager is not running")
        future: concurrent.futures.Future[MCPCallRecord] = concurrent.futures.Future()

        def enqueue() -> None:
            queue = self._queues.get(server)
            if queue is None:
                future.set_exception(MCPUnavailableError(f"server {server!r} is not connected"))
            else:
                queue.put_nowait((op, future))

        self._loop.call_soon_threadsafe(enqueue)
        return future.result(timeout=timeout)

    def _base(self, kind: str, server: str, started: str, decision: AuthorizationDecision, **kw):
        return {
            "kind": kind,
            "server": server,
            "started_at": started,
            "authorized": decision.allowed,
            "authorization_reason": decision.reason,
            **kw,
        }

    def _request_failure(self, base: dict, exc: Exception, timeout: float) -> MCPCallRecord:
        """Classify an exception raised while a request was in flight."""
        if isinstance(exc, TimeoutError):
            return self._failure(base, "timeout", f"no result within {timeout:g} s")
        if isinstance(exc, MCPError) and exc.code != types.CONNECTION_CLOSED:
            return self._failure(base, "protocol_error", _describe(exc))
        # The connection is gone: later requests to this server are denied, with
        # no fallback, instead of being sent to a dead session.
        name = base["server"]
        if self._connected.get(name):
            self._connected[name] = False
            self._errors[name] = scrub_text(_describe(exc), self._secrets)
            self._audit_event(
                "disconnect", server=name, outcome="connection_error", error=self._errors[name]
            )
        return self._failure(base, "connection_error", _describe(exc))

    def _failure(self, base: dict, outcome: Outcome, error: str) -> MCPCallRecord:
        return MCPCallRecord(
            **base,
            finished_at=now_utc(),
            outcome=outcome,
            error=scrub_text(truncate(error, 2000), self._secrets),
        )

    def _content(self, blocks: list[Any]) -> list[str]:
        parts = []
        for block in blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(scrub_text(truncate(text, MAX_CONTENT_CHARS), self._secrets))
            else:
                parts.append(f"[{getattr(block, 'type', 'non-text')} content omitted]")
        return parts

    def call_tool(self, server: str, tool: str, arguments: dict[str, Any] | None) -> MCPCallRecord:
        """Authorize, validate, and execute one tool call; never raises for MCP failures."""
        started = now_utc()
        arguments = arguments if arguments is not None else {}
        safe_args = redact(arguments, self._secrets) if isinstance(arguments, dict) else {}
        decision = self._check_tool(server, tool)
        base = self._base(
            "tool",
            server,
            started,
            decision,
            tool=tool,
            arguments=safe_args,
            # Hash what is recorded, so the digest cannot confirm a guessed secret.
            arguments_sha256=sha256_json(safe_args),
        )
        if not decision.allowed:
            return self._finish(self._failure(base, "denied", decision.reason))
        if not isinstance(arguments, dict):
            return self._finish(
                self._failure(base, "invalid_arguments", "arguments must be a JSON object")
            )
        schema = self._tools[server][tool].input_schema
        try:
            validator_cls = jsonschema.validators.validator_for(schema)
            validator_cls.check_schema(schema)
            errors = list(validator_cls(schema).iter_errors(arguments))
        except Exception as exc:  # noqa: BLE001 - e.g. an unresolvable remote $ref
            # The tool is not called when its schema cannot be checked; remote
            # $refs are never fetched.
            return self._finish(
                self._failure(
                    base,
                    "invalid_arguments",
                    f"the server's input schema cannot be used ({exc.__class__.__name__})",
                )
            )
        if errors:
            return self._finish(self._failure(base, "invalid_arguments", _schema_errors(errors)))

        timeout = self.config.server(server).call_timeout_s

        async def op(session: ClientSession) -> MCPCallRecord:
            try:
                with anyio.fail_after(timeout):
                    result = await session.call_tool(tool, arguments)
            except Exception as exc:  # noqa: BLE001 - classified into an explicit record
                return self._request_failure(base, exc, timeout)
            if not isinstance(result, types.CallToolResult):
                return self._failure(
                    base, "protocol_error", f"unsupported result type {type(result).__name__}"
                )
            content = self._content(result.content)
            structured = redact(result.structured_content, self._secrets, sensitive_keys=False)
            if structured is not None and len(json.dumps(structured, default=str)) > (
                MAX_CONTENT_CHARS
            ):
                structured = {"omitted": "structured content exceeded the size limit"}
            digest = sha256_json(
                {"content": content, "structured_content": result.structured_content}
            )
            if result.is_error:
                return MCPCallRecord(
                    **base,
                    finished_at=now_utc(),
                    outcome="tool_error",
                    error=truncate(" ".join(content), 2000) or "tool reported an error",
                    content=content,
                    content_sha256=digest,
                )
            return MCPCallRecord(
                **base,
                finished_at=now_utc(),
                outcome="success",
                content=content,
                structured_content=structured,
                content_sha256=digest,
            )

        return self._finish(self._run_op(server, op, timeout, base))

    def read_resource(self, server: str, uri: str) -> MCPCallRecord:
        """Authorize and read one allowlisted resource; never raises for MCP failures."""
        started = now_utc()
        decision = self._authorizer.check_resource(server, uri, connected=self._connected)
        base = self._base("resource", server, started, decision, resource_uri=uri)
        if not decision.allowed:
            return self._finish(self._failure(base, "denied", decision.reason))
        timeout = self.config.server(server).call_timeout_s

        async def op(session: ClientSession) -> MCPCallRecord:
            try:
                with anyio.fail_after(timeout):
                    result = await session.read_resource(uri)
            except Exception as exc:  # noqa: BLE001 - classified into an explicit record
                return self._request_failure(base, exc, timeout)
            if not isinstance(result, types.ReadResourceResult):
                return self._failure(
                    base, "protocol_error", f"unsupported result type {type(result).__name__}"
                )
            content = []
            for item in result.contents:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    content.append(scrub_text(truncate(text, MAX_CONTENT_CHARS), self._secrets))
                else:
                    content.append("[binary resource content omitted]")
            return MCPCallRecord(
                **base,
                finished_at=now_utc(),
                outcome="success",
                content=content,
                content_sha256=sha256_json(content),
            )

        return self._finish(self._run_op(server, op, timeout, base))

    def _run_op(self, server: str, op: _Op, timeout: float, base: dict) -> MCPCallRecord:
        try:
            return self._submit(server, op, timeout + _SHUTDOWN_GRACE_S)
        except MCPUnavailableError as exc:
            return self._failure(base, "connection_error", str(exc))
        except concurrent.futures.TimeoutError:
            return self._failure(base, "timeout", "the MCP connection did not respond")
