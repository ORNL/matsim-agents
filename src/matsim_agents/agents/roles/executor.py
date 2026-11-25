"""Executor agent.

Pops the next task from ``pending_tasks`` and dispatches it to the
appropriate tool: "relax" runs an MLIP relaxation, and "mcp_tool" /
"mcp_resource" go to the run's MCP session manager, which authorizes them
before any external call. MCP tasks need no MLIP selection.
"""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from matsim_agents.agents.state import MatSimState
from matsim_agents.backends.mcp.tasks import MCP_TASK_KINDS, execute_mcp_task
from matsim_agents.backends.mlip.config import MLIPConfig, load_mlip_config, parse_mlip_config
from matsim_agents.backends.mlip.relaxation import RelaxStructureInput, relax_with_mlip


def _mlip_config(cfg: dict) -> MLIPConfig:
    """The run's MLIP selection, or the mlip block named by MATSIM_MLIP_CONFIG."""
    if cfg.get("mlip") is not None:
        return parse_mlip_config(cfg["mlip"])
    path = os.environ.get("MATSIM_MLIP_CONFIG")
    if path:
        return load_mlip_config(path)
    raise ValueError(
        "Missing MLIP selection. Pass --mlip-config or set MATSIM_MLIP_CONFIG to a YAML "
        "file with an mlip block."
    )


def executor_node(state: MatSimState, config: RunnableConfig = None) -> dict:
    if not state.pending_tasks:
        return {}

    task = state.pending_tasks[0]
    remaining = state.pending_tasks[1:]
    cfg = (config or {}).get("configurable", {}) if config else {}

    if task.kind in MCP_TASK_KINDS:
        record = execute_mcp_task(cfg.get("mcp_client"), task)
        return {
            "pending_tasks": remaining,
            "mcp_results": [record],
            "iteration": state.iteration + 1,
            "messages": [AIMessage(content=f"[executor] {record.summary_line()}")],
        }

    mlip = _mlip_config(cfg)

    if task.kind != "relax":
        return {
            "pending_tasks": remaining,
            "messages": [AIMessage(content=f"[executor] skipping unsupported task: {task.kind}")],
        }

    args = RelaxStructureInput(
        structure_path=task.structure_path,
        mlip=mlip,
        optimizer=task.optimizer,
        maxiter=task.maxiter,
        maxstep=task.maxstep,
        charge=task.charge,
        spin=task.spin,
        random_displacement=task.random_displacement,
        displacement_seed=task.displacement_seed,
        output_dir=cfg.get("output_dir"),
    )

    result = relax_with_mlip(args)
    msg = (
        f"[executor] relaxed {result.structure_path} -> "
        f"E={result.final_energy_eV:.4f} eV, |F|max={result.final_max_force_eV_per_A:.4f} eV/Å, "
        f"{result.num_steps} step(s)."
    )
    return {
        "pending_tasks": remaining,
        "results": [result],
        "iteration": state.iteration + 1,
        "messages": [AIMessage(content=msg)],
    }
