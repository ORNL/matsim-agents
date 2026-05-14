"""Executor agent.

Pops the next task from ``pending_tasks`` and dispatches it to the
appropriate tool. For now only the "relax" task is implemented.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from matsim_agents.state import MatSimState
from matsim_agents.tools.relaxation import RelaxStructureInput, _run


def executor_node(state: MatSimState, config: RunnableConfig | None = None) -> dict:
    if not state.pending_tasks:
        return {}

    task = state.pending_tasks[0]
    remaining = state.pending_tasks[1:]
    cfg = (config or {}).get("configurable", {}) if config else {}

    if task.kind != "relax":
        return {
            "pending_tasks": remaining,
            "messages": [AIMessage(content=f"[executor] skipping unsupported task: {task.kind}")],
        }

    args = RelaxStructureInput(
        structure_path=task.structure_path,
        logdir=cfg["logdir"],
        mlp_checkpoint=cfg["mlp_checkpoint"],
        checkpoint=cfg.get("checkpoint"),
        optimizer=task.optimizer,
        maxiter=task.maxiter,
        maxstep=task.maxstep,
        charge=task.charge,
        spin=task.spin,
        random_displacement=task.random_displacement,
        precision=cfg.get("precision"),
        mlp_precision=cfg.get("mlp_precision"),
        mlp_device=cfg.get("mlp_device", "cuda"),
        output_dir=cfg.get("output_dir"),
    )

    result = _run(args)
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
