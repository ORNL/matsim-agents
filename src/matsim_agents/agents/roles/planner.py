"""Planner agent.

Turns a free-form objective into a concrete list of :class:`TaskSpec`
items. Uses an LLM with structured output for robustness; falls back to a
deterministic single-task plan when no LLM is configured.
"""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from matsim_agents.agents.state import MatSimState, TaskSpec
from matsim_agents.backends.mcp.tasks import planner_catalog


class _Plan(BaseModel):
    tasks: list[TaskSpec] = Field(default_factory=list)
    rationale: str = ""


_TASKS_PROMPT = """You are the planning agent of a materials-discovery system.
Decompose the user's objective into an ordered list of atomistic tasks.
Available task kinds:
  - "relax": run HydraGNN-driven structure relaxation on a structure file
  - "analyze": summarize results so far
  - "report": produce the final report
"""
_RELAX_RULE = """Always emit at least one "relax" task pointing at a real structure path
mentioned in the objective."""
_SYSTEM_PROMPT = _TASKS_PROMPT + _RELAX_RULE


def _system_prompt(mcp_catalog: str | None) -> str:
    if mcp_catalog is None:
        return _SYSTEM_PROMPT
    # With approved MCP tools an objective may need no relaxation at all, so the
    # relax rule is replaced by the MCP section's own rule for relax tasks.
    return _TASKS_PROMPT.rstrip() + "\n" + mcp_catalog


def planner_node(state: MatSimState, config: RunnableConfig = None) -> dict:
    if not state.objective:
        return {"plan": [], "pending_tasks": []}
    cfg = (config or {}).get("configurable", {}) if config else {}
    mcp_catalog = planner_catalog(cfg.get("mcp_client"))

    structure_paths = [
        tok
        for tok in state.objective.split()
        if os.path.splitext(tok)[1] in {".vasp", ".cif", ".xyz", ".extxyz", ".poscar", ".pdb"}
    ]

    try:
        from matsim_agents.backends.llm.provider import get_chat_model

        llm = get_chat_model(
            provider=state.llm_provider,
            model=state.llm_model,
            base_url=state.llm_base_url,
        ).with_structured_output(_Plan)
        plan: _Plan = llm.invoke(
            [
                SystemMessage(content=_system_prompt(mcp_catalog)),
                AIMessage(content=f"Objective: {state.objective}"),
            ]
        )
        tasks = plan.tasks
        rationale = plan.rationale
    except Exception as exc:  # pragma: no cover - LLM is optional
        tasks = [TaskSpec(kind="relax", structure_path=p) for p in structure_paths]
        rationale = f"LLM unavailable ({exc!s}); falling back to default plan."

    return {
        "plan": tasks,
        "pending_tasks": list(tasks),
        "messages": [
            AIMessage(content=f"[planner] {rationale or f'{len(tasks)} task(s) queued.'}")
        ],
    }
