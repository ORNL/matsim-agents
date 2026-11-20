"""A valid MCP server that offers resources but no tools (stdio).

Built on the SDK's low-level server so that, unlike ``MCPServer``, it does not
advertise the tools capability and rejects ``tools/list``.
"""

from __future__ import annotations

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server


async def list_resources(ctx, params) -> types.ListResourcesResult:
    return types.ListResourcesResult(
        resources=[types.Resource(name="notes", uri="test://notes", description="Fixed notes.")]
    )


async def read_resource(ctx, params) -> types.ReadResourceResult:
    return types.ReadResourceResult(
        contents=[types.TextResourceContents(uri=str(params.uri), text="resource-only notes")]
    )


server = Server("resource-only", on_list_resources=list_resources, on_read_resource=read_resource)


async def main() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)
