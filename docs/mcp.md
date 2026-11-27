# External MCP tools in `matsim-agents run`

`matsim-agents run` can use tools and resources from external
[Model Context Protocol](https://modelcontextprotocol.io) (MCP) servers. The
planner sees the tools you approved, the executor calls them through the
official MCP Python SDK, and the analyst receives the results. Every attempt,
allowed or not, is recorded in an audit log.

MCP is optional. Without `--mcp-config`, `run` behaves exactly as before and
the MCP SDK is never imported.

## Install

```bash
pip install -e ".[mcp]"   # mcp>=2.0,<3 and jsonschema
```

## Supported transports

| `transport` | Server | Notes |
| --- | --- | --- |
| `stdio` | A local process started by matsim-agents | Gets the SDK's minimal default environment (`HOME`, `PATH`, ...) plus `env`; its stderr goes, with credentials scrubbed, to `<output-dir>/mcp/<name>.stderr.log` |
| `streamable_http` | An MCP endpoint URL, for example `http://127.0.0.1:8765/mcp` | Optional `headers` for credentials; credentials in the URL itself are rejected |

The deprecated HTTP+SSE transport, MCP prompts, resource templates, sampling,
elicitation, and OAuth authorization flows are not supported.

## Configuration

The file holds an `mcp` block. Server names must be unique; tools are always
identified by server *and* tool name, so two servers may offer tools with the
same name without colliding.

```yaml
mcp:
  servers:
    - name: demo                       # letters, digits, _ or -
      transport: stdio
      command: python
      args: [demo_server.py]           # relative to cwd
      cwd: .                           # default: this file's directory
      env: {DEMO_TOKEN: "${DEMO_TOKEN}"}
      connect_timeout_s: 20            # handshake and discovery
      call_timeout_s: 30               # each tool call or resource read
      allowed_tools: [convert_energy]
      allowed_resources: ["matsim-demo://units"]
    - name: remote
      transport: streamable_http
      url: https://mcp.example.org/mcp
      headers: {Authorization: "Bearer ${REMOTE_MCP_TOKEN}"}
      enabled: false
      allowed_tools: [lookup]
```

`${VAR}` is expanded from the host environment only in `env` and `headers`
values, so credentials stay out of the file; a missing variable is an error
naming the variable. matsim-agents never prints, logs, or audits these values,
and it scrubs them from anything a server sends back (results, errors, stderr):
every header value and its last word (the token in `Bearer ...`), every value
substituted from `${VAR}`, and literal `env` values of 8 or more characters.
Credentials shorter than 4 characters (a `${VAR}` value or a header token)
are rejected, because they cannot be scrubbed without corrupting other text.
Multiline `env` and `headers` values are rejected because stderr is scrubbed
line by line.
Put credentials in `${VAR}` references, not literal `env` values. Invalid configurations (unknown fields,
missing `command` or `url`, fields of the other transport, duplicate names,
wildcard allowlists, non-positive timeouts) are rejected before any server is
started.

## Server lifecycle

1. `run` loads the configuration and starts one background event loop for the
   whole run.
2. Each enabled server is connected concurrently: the transport is opened, the
   MCP session is initialized, and tools and resources (all pages) are listed
   within `connect_timeout_s`.
3. A server that cannot be started, refuses the connection, or does not answer
   in time is marked unavailable and its process is stopped at once; other
   servers continue.
4. The planner, executor, and analyst run. Requests to one server are executed
   one at a time.
5. When the graph finishes, or fails, every session is closed and every stdio
   server process is stopped, cancelling in-flight requests. A `shutdown` audit
   record states whether this completed cleanly; a cleanup timeout raises an error.

## Tool authorization

Authorization is local and deny-by-default. A tool call runs only when all of
these hold, checked in this order before anything is sent to a server:

1. the server is configured and `enabled`;
2. the tool name is listed in that server's `allowed_tools` (approving
   `whoami` on one server does not approve `whoami` on another);
3. the server is connected (there is no fallback to another server or tool);
4. the server actually offered the tool during discovery;
5. the arguments are a JSON object that validates against the tool's input
   schema (remote `$ref`s are never fetched; a schema that cannot be used
   blocks the call).

Resource reads require the exact URI in `allowed_resources` and a connected
server. Nothing a server sends (tool names, descriptions, schemas, or results)
can change these rules, and requests a server sends to the client (sampling,
elicitation, roots) are refused. Tool results reach only the analyst; they
cannot add tasks, because the plan is fixed before execution starts. The planner is shown only permitted tools and
resources, inside a block that tells it to treat their text as untrusted data,
and tasks it emits for anything else are denied and recorded.

MCP tasks only call external tools. They do not run relaxations, DFT, or
active learning, so the existing scientific approvals, the MLIP selection
(still required by every `relax` task), and active learning are unchanged. MCP tasks count towards `--max-iterations`.

## Run an MCP-enabled workflow

Reproducible example without an LLM, using the harmless demo server in
[examples/mcp](../examples/mcp/demo_server.py) (unit conversion only). The
planner LLM is replaced by a fixed plan; the MCP traffic is real:

```bash
python examples/mcp/run_demo.py --output-dir /tmp/mcp-demo
```

With an LLM planning the tasks (for example a local Ollama model):

```bash
matsim-agents run "Convert -3.2 eV to kJ/mol with the demo unit tool" \
    --mcp-config examples/mcp/mcp.example.yaml --output-dir /tmp/mcp-run
```

`--mcp-audit-path` overrides the audit location. Run `matsim-agents run` from
the environment where the `mcp` extra is installed, because the example starts
the demo server with `python`.

## Expected output and audit records

`run_demo.py` prints:

```text
server demo: connected
server demo_http: disabled in configuration

MCP results in workflow state:
  MCP tool demo/convert_energy: success: -308.753062784
  MCP resource demo/matsim-demo://units: success: {"per_eV": {...}, "source": "CODATA 2018"}
  MCP tool demo/convert_energy: invalid_arguments: to_unit: violates enum ["eV", "kJ/mol", "kcal/mol", "Ha"]
  MCP tool elsewhere/convert_energy: denied: unknown server 'elsewhere'
```

The analyst lists the same results under "External MCP results (untrusted,
not scientific evidence)". The audit log `<output-dir>/mcp/mcp_audit.jsonl`
has one JSON object per line:

| `event` | Fields |
| --- | --- |
| `connect` | `server`, `transport`, `outcome` (`success`, `timeout`, `connection_error`), `error`, discovered `tools` and `resources`, `server_name`, `server_version` |
| `tool_call`, `resource_call` | `server`, `tool` or `resource_uri`, `started_at`, `finished_at`, `authorized`, `authorization_reason`, `outcome`, `error`, redacted `arguments`, `arguments_sha256`, `content_sha256`, `content_chars` |
| `disconnect` | a server's connection failed after it had connected |
| `shutdown` | `clean` and the final status of each server |

Audit records keep a hash and size of each sanitized result, not the result
itself; secret-scrubbed text content is limited to 8,000 characters and 64
blocks per response in workflow state for the
analyst. Configured secrets are also scrubbed from the planner's discovery
catalog without changing the execution schemas. Schema errors name the field
and rule, never the rejected value. Argument and structured-result values under
credential-like keys (`password`, `token`, `api_key`, `Authorization`, ...)
and any configured secret value are replaced by `[REDACTED]`. MCP results are
not written to the
[evidence ledger](evidence-ledger.md): they are external data, not scientific
evidence.

## Failure behavior

| Situation | Outcome |
| --- | --- |
| No `--mcp-config` but the planner emits an MCP task | `denied` (MCP is not configured) |
| Unknown server, disabled server, unlisted tool or resource, tool not offered | `denied`, nothing sent |
| Arguments fail the input schema | `invalid_arguments`, nothing sent |
| Server reports a tool error (`isError`) | `tool_error` with the server's message |
| JSON-RPC error from the server | `protocol_error` |
| No answer within `call_timeout_s` | `timeout` |
| Server process exits or connection drops | `connection_error`; later calls are `denied` |
| Server unavailable at start | listed as unavailable; its tasks are `denied` with no fallback |

None of these stops the run: each becomes a record in workflow state and in
the audit log. An invalid configuration file stops `run` before anything
starts.

## Security limitations

- Allowlisting a tool trusts its server to do what the tool claims. A stdio
  server runs with your user's permissions; the allowlist controls which tools
  the agents may call, not what the server process can do.
- Tool descriptions and results reach the planner and analyst LLMs. They are
  framed as untrusted data, but no prompt framing fully prevents prompt
  injection; allowlist only servers you trust.
- Scrubbing replaces the configured credentials listed above in text and
  metadata. Credential-like keys are redacted in invocation arguments and
  structured results, **not in free-form text**: even JSON-looking text with
  an `access_token` key can expose a server-issued secret that is not configured.
  Scrubbing also cannot recognize unconfigured secrets under innocuous keys or
  configured credentials transformed by a server (for example base64-encoded).
- Query parameters in an HTTP `url` are not treated as secrets; use headers.
- Arguments and results are size-limited when kept in state, but the server
  decides what a tool does with its arguments.

## Scope

MCP support covers `matsim-agents run` (the objective graph). The `plan`
command, the interactive discovery chat, campaigns, and the scientific
workflow runners do not load MCP servers.
