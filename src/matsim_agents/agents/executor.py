"""Executor agent.

Pops the next task from ``pending_tasks`` and dispatches it to the
appropriate tool. For now only the "relax" task is implemented.
"""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from matsim_agents.backends.mlip.relaxation import RelaxStructureInput, _run
from matsim_agents.orchestration.state import MatSimState


def _cfg_or_env(
    cfg: dict,
    cfg_key: str,
    env_key: str,
    default: str | None = None,
) -> str | None:
    value = cfg.get(cfg_key)
    if value is not None:
        return str(value)
    return os.environ.get(env_key, default)


def executor_node(state: MatSimState, config: RunnableConfig | None = None) -> dict:
    if not state.pending_tasks:
        return {}

    task = state.pending_tasks[0]
    remaining = state.pending_tasks[1:]
    cfg = (config or {}).get("configurable", {}) if config else {}
    mlip_backend = _cfg_or_env(cfg, "mlip_backend", "MATSIM_MLIP_BACKEND", "hydragnn")
    logdir = _cfg_or_env(cfg, "logdir", "MATSIM_HYDRAGNN_LOGDIR")
    hydragnn_branch_mlp_checkpoint = _cfg_or_env(
        cfg,
        "hydragnn_branch_mlp_checkpoint",
        "HYDRAGNN_BRANCH_MLP_CHECKPOINT",
    )
    checkpoint = _cfg_or_env(cfg, "checkpoint", "MATSIM_HYDRAGNN_CHECKPOINT")

    if mlip_backend == "hydragnn":
        if not logdir:
            raise ValueError(
                "Missing HydraGNN logdir. Provide via run config or set MATSIM_HYDRAGNN_LOGDIR."
            )
        if not hydragnn_branch_mlp_checkpoint:
            raise ValueError(
                "Missing BranchWeightMLP checkpoint. Provide via run config or set "
                "HYDRAGNN_BRANCH_MLP_CHECKPOINT."
            )

    if task.kind != "relax":
        return {
            "pending_tasks": remaining,
            "messages": [AIMessage(content=f"[executor] skipping unsupported task: {task.kind}")],
        }

    args = RelaxStructureInput(
        structure_path=task.structure_path,
        mlip_backend=mlip_backend,
        logdir=logdir,
        hydragnn_branch_mlp_checkpoint=hydragnn_branch_mlp_checkpoint,
        checkpoint=checkpoint,
        uma_model_name=str(
            cfg.get("uma_model_name") or os.environ.get("MATSIM_UMA_MODEL_NAME") or "uma-s-1p1"
        ),
        uma_task=str(cfg.get("uma_task") or os.environ.get("MATSIM_UMA_TASK") or "omat"),
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
