"""Objective mode: ``run`` (planner -> executor -> analyst graph) and ``plan``."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.panel import Panel

from matsim_agents.cli._app import app, console
from matsim_agents.cli.options import (
    load_mlip_option,
    opt_active_learning_config,
    opt_active_learning_dry_run,
    opt_al_handoff_audit_path,
    opt_mlip_config,
    opt_trigger_al_handoff,
    opt_uq_min_relaxations_for_handoff,
    opt_uq_min_unreliable_fraction,
    opt_uq_top_weight_threshold,
    path_or_none,
)

if TYPE_CHECKING:
    from matsim_agents.backends.mcp.session import MCPSessionManager
    from matsim_agents.backends.mlip.config import MLIPConfig


def _start_mcp(
    mcp_config: Path | None, output_dir: Path | None, audit_path: Path | None
) -> MCPSessionManager | None:
    """Connect to the configured MCP servers, or return None when none are configured."""
    if mcp_config is None:
        return None
    from matsim_agents.backends.mcp.config import load_mcp_config
    from matsim_agents.backends.mcp.records import MCPAuditLog

    try:
        config = load_mcp_config(mcp_config)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="--mcp-config") from error
    try:
        from matsim_agents.backends.mcp.session import MCPSessionManager
    except ImportError as error:
        raise typer.BadParameter(
            f"MCP support needs the optional extra: pip install 'matsim-agents[mcp]' ({error})",
            param_hint="--mcp-config",
        ) from error
    mcp_dir = (output_dir or Path("./outputs")) / "mcp"
    audit = MCPAuditLog(audit_path or mcp_dir / "mcp_audit.jsonl", secrets=config.secret_values())
    manager = MCPSessionManager(config, audit=audit, log_dir=mcp_dir)
    manager.start()
    for name, status in manager.server_status().items():
        state = "connected" if status["connected"] else f"unavailable ({status['error']})"
        console.print(f"MCP server {name} [{status['transport']}]: {state}", markup=False)
    return manager


def _build_run_config(
    *,
    thread_id: str,
    mlip: MLIPConfig | None,
    output_dir: Path | None,
    trigger_al_handoff: bool,
    active_learning_config: Path | None,
    active_learning_dry_run: bool,
    uq_top_weight_threshold: float,
    uq_min_unreliable_fraction: float,
    uq_min_relaxations_for_handoff: int,
    al_handoff_audit_path: Path | None,
    mcp_client: MCPSessionManager | None = None,
) -> dict:
    config = {
        "configurable": {
            "thread_id": thread_id,
            "mlip": mlip.model_dump(mode="json") if mlip is not None else None,
            "output_dir": path_or_none(output_dir),
            "trigger_al_handoff": trigger_al_handoff,
            "active_learning_config": path_or_none(active_learning_config),
            "active_learning_dry_run": active_learning_dry_run,
            "uq_top_weight_threshold": uq_top_weight_threshold,
            "uq_min_unreliable_fraction": uq_min_unreliable_fraction,
            "uq_min_relaxations_for_handoff": uq_min_relaxations_for_handoff,
            "al_handoff_audit_path": path_or_none(al_handoff_audit_path),
        }
    }
    if mcp_client is not None:
        config["configurable"]["mcp_client"] = mcp_client
    return config


@app.command()
def run(
    objective: str = typer.Argument(..., help="Free-form objective for the agent system."),
    mlip_config: Path | None = opt_mlip_config(),
    output_dir: Path | None = typer.Option(None, help="Where to write artifacts."),
    max_iterations: int = typer.Option(5, help="Maximum executor iterations."),
    llm_provider: str = typer.Option(
        "ollama",
        "--llm-provider",
        help="LLM backend: ollama | vllm | mag | copilot | openai | anthropic | huggingface.",
        case_sensitive=False,
    ),
    llm_model: str | None = typer.Option(
        None, "--llm-model", help="Model identifier (provider-specific)."
    ),
    llm_base_url: str | None = typer.Option(
        None,
        "--llm-base-url",
        help="Override server URL (Ollama or vLLM /v1 endpoint).",
    ),
    trigger_active_learning_on_high_uq: bool = opt_trigger_al_handoff(
        False,
        "After run relaxations, evaluate UQ and optionally hand off to active learning.",
    ),
    active_learning_config: Path | None = opt_active_learning_config(
        "Path to a base AL YAML config used when UQ-triggered handoff fires."
    ),
    active_learning_dry_run: bool = opt_active_learning_dry_run(
        "Plan/report run->AL handoff only, or execute AL loop."
    ),
    uq_top_weight_threshold: float = opt_uq_top_weight_threshold(),
    uq_min_unreliable_fraction: float = opt_uq_min_unreliable_fraction(),
    uq_min_relaxations_for_handoff: int = opt_uq_min_relaxations_for_handoff(
        1,
        "Minimum relaxations required before evaluating run-path handoff.",
    ),
    al_handoff_audit_path: Path | None = opt_al_handoff_audit_path(
        "Optional JSONL path for UQ decision and run->AL handoff audit records."
    ),
    mcp_config: Path | None = typer.Option(
        None,
        "--mcp-config",
        help="YAML file with an mcp block naming external MCP servers and allowed tools.",
    ),
    mcp_audit_path: Path | None = typer.Option(
        None,
        "--mcp-audit-path",
        help="JSONL path for MCP audit records (default: <output-dir>/mcp/mcp_audit.jsonl).",
    ),
):
    """Run the planner -> executor -> analyst graph for a given objective.

    When to use: you have a written objective and want it planned and executed end to end.
    """
    from matsim_agents.orchestration.objective_graph import build_graph
    from matsim_agents.orchestration.state import MatSimState

    mlip = load_mlip_option(mlip_config)
    graph = build_graph()
    state = MatSimState(
        objective=objective,
        max_iterations=max_iterations,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )
    mcp_client = _start_mcp(mcp_config, output_dir, mcp_audit_path)
    try:
        config = _build_run_config(
            thread_id=str(uuid.uuid4()),
            mlip=mlip,
            output_dir=output_dir,
            trigger_al_handoff=trigger_active_learning_on_high_uq,
            active_learning_config=active_learning_config,
            active_learning_dry_run=active_learning_dry_run,
            uq_top_weight_threshold=uq_top_weight_threshold,
            uq_min_unreliable_fraction=uq_min_unreliable_fraction,
            uq_min_relaxations_for_handoff=uq_min_relaxations_for_handoff,
            al_handoff_audit_path=al_handoff_audit_path,
            mcp_client=mcp_client,
        )

        final = graph.invoke(state, config=config)
    finally:
        if mcp_client is not None:
            mcp_client.close()

    console.print(Panel.fit("matsim-agents run summary", style="bold cyan"))
    console.print(f"Objective: {final['objective']}")
    console.print(f"Tasks executed: {len(final['results'])}")
    for r in final["results"]:
        console.print(f"  • {r.optimized_structure_path}  E={r.final_energy_eV:.4f} eV")
    if final.get("mcp_results"):
        console.print(f"MCP calls: {len(final['mcp_results'])}")
        for record in final["mcp_results"]:
            console.print(f"  • {record.summary_line()}", markup=False, soft_wrap=True)
    console.print(Panel(final.get("analysis") or "(no analysis)", title="Analyst"))


@app.command()
def plan(
    objective: str,
    as_json: bool = typer.Option(False, "--json", help="Print the task list as JSON."),
):
    """Run the planner only and print the proposed task list.

    When to use: preview what ``run`` would do for an objective before executing anything.
    """
    from matsim_agents.orchestration.agents.planner import planner_node
    from matsim_agents.orchestration.state import MatSimState

    state = MatSimState(objective=objective)
    out = planner_node(state)
    if as_json:
        console.print_json(json.dumps([t.model_dump() for t in out["plan"]]))
        return
    console.print(f"Planned {len(out['plan'])} task(s) for: {objective}", markup=False)
    for index, task in enumerate(out["plan"], 1):
        if task.kind in {"mcp_tool", "mcp_resource"}:
            target = task.mcp_tool if task.kind == "mcp_tool" else task.mcp_resource_uri
            detail = f"{task.mcp_server}/{target}"
        else:
            detail = f"{task.structure_path} ({task.optimizer}, maxiter={task.maxiter})"
        console.print(
            f"  {index}. {task.kind} {detail}",
            markup=False,
            soft_wrap=True,
        )
    console.print("Nothing was executed. Use `matsim-agents run` to execute the objective.")
