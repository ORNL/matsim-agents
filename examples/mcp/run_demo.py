"""Run the objective graph with the demo MCP server and no LLM.

The planner LLM is replaced by a fixed plan so the example needs no model, API
key, GPU, or MLIP. Everything after planning is the normal workflow: the
executor authorizes each task, calls the demo server over real MCP stdio, and
the analyst summarizes the results; the audit log records every attempt.

    pip install -e ".[mcp]"
    python examples/mcp/run_demo.py --output-dir /tmp/mcp-demo

To let an LLM plan instead, use ``matsim-agents objective ... --mcp-config
examples/mcp/mcp.example.yaml`` (see docs/mcp.md).
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import matsim_agents.backends.llm.provider as llm_provider
from matsim_agents.backends.mcp.config import load_mcp_config
from matsim_agents.backends.mcp.records import MCPAuditLog
from matsim_agents.backends.mcp.session import MCPSessionManager
from matsim_agents.agents.objective_graph import build_graph
from matsim_agents.agents.state import MatSimState, TaskSpec

HERE = Path(__file__).resolve().parent

PLAN = [
    # Allowed: listed in allowed_tools and offered by the server.
    TaskSpec(
        kind="mcp_tool",
        mcp_server="demo",
        mcp_tool="convert_energy",
        mcp_arguments={"value": -3.2, "from_unit": "eV", "to_unit": "kJ/mol"},
    ),
    TaskSpec(kind="mcp_resource", mcp_server="demo", mcp_resource_uri="matsim-demo://units"),
    # Rejected locally: the schema has no unit "J".
    TaskSpec(
        kind="mcp_tool",
        mcp_server="demo",
        mcp_tool="convert_energy",
        mcp_arguments={"value": 1.0, "from_unit": "eV", "to_unit": "J"},
    ),
    # Denied: the server is not configured.
    TaskSpec(kind="mcp_tool", mcp_server="elsewhere", mcp_tool="convert_energy"),
]


def _scripted_planner() -> None:
    """Stand in for the planner LLM with PLAN; the analyst falls back to its
    deterministic summary because this stand-in cannot answer free text."""
    structured = MagicMock()
    structured.invoke.return_value = MagicMock(tasks=PLAN, rationale="fixed demo plan")
    model = MagicMock()
    model.with_structured_output.return_value = structured
    model.invoke.side_effect = RuntimeError("no LLM in this demo")
    llm_provider.get_chat_model = lambda **_: model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mcp-demo"))
    parser.add_argument("--config", type=Path, default=HERE / "mcp.example.yaml")
    args = parser.parse_args()

    _scripted_planner()
    config = load_mcp_config(args.config)
    audit_path = args.output_dir / "mcp" / "mcp_audit.jsonl"
    audit = MCPAuditLog(audit_path, secrets=config.secret_values())
    with MCPSessionManager(config, audit=audit, log_dir=args.output_dir / "mcp") as mcp:
        for name, status in mcp.server_status().items():
            print(f"server {name}: {'connected' if status['connected'] else status['error']}")
        final = build_graph().invoke(
            MatSimState(objective="Convert -3.2 eV to kJ/mol", llm_provider="scripted"),
            config={"configurable": {"thread_id": str(uuid.uuid4()), "mcp_client": mcp}},
        )

    print("\nMCP results in workflow state:")
    for record in final["mcp_results"]:
        print(f"  {record.summary_line()}")
    print(f"\nAnalyst:\n{final['analysis']}")
    print(f"\nAudit log: {audit_path}")
    for line in audit_path.read_text().splitlines():
        entry = json.loads(line)
        target = entry.get("tool") or entry.get("resource_uri") or ""
        print(
            f"  {entry['event']:<14} {entry.get('server', '-'):<10} {target:<22} "
            f"{entry.get('outcome', entry.get('clean', ''))}"
        )


if __name__ == "__main__":
    main()
