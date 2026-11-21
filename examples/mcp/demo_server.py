"""A harmless local MCP server for the matsim-agents MCP example.

It offers one tool, ``convert_energy``, and one resource,
``matsim-demo://units``. It touches no files, makes no network requests, and
runs no calculations. Start it through ``mcp.yaml``; it speaks MCP over stdio,
or over Streamable HTTP with ``--http PORT``.
"""

from __future__ import annotations

import argparse
import json
from typing import Literal

from mcp.server.mcpserver import MCPServer

# CODATA 2018 conversion factors, expressed per eV.
PER_EV = {"eV": 1.0, "kJ/mol": 96.48533212, "kcal/mol": 23.0605478, "Ha": 1 / 27.211386245988}
Unit = Literal["eV", "kJ/mol", "kcal/mol", "Ha"]

server = MCPServer("matsim-demo-units")


@server.tool(description="Convert an energy between eV, kJ/mol, kcal/mol, and Hartree.")
def convert_energy(value: float, from_unit: Unit, to_unit: Unit) -> float:
    return value / PER_EV[from_unit] * PER_EV[to_unit]


@server.resource(
    "matsim-demo://units",
    name="units",
    description="Supported units and conversion factors per eV.",
    mime_type="application/json",
)
def units() -> str:
    return json.dumps({"per_eV": PER_EV, "source": "CODATA 2018"}, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", type=int, help="serve Streamable HTTP on 127.0.0.1:PORT")
    args = parser.parse_args()
    if args.http:
        server.run("streamable-http", host="127.0.0.1", port=args.http)
    else:
        server.run("stdio")
