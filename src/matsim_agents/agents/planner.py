"""Planner agent.

Turns a free-form objective into a concrete list of :class:`TaskSpec`
items. Uses an LLM with structured output for robustness; falls back to a
deterministic single-task plan when no LLM is configured.
"""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from matsim_agents.state import MatSimState, TaskSpec


class _Plan(BaseModel):
    tasks: list[TaskSpec] = Field(default_factory=list)
    rationale: str = ""


_SYSTEM_PROMPT = """You are the planning agent of a materials-discovery system.
Decompose the user's objective into an ordered list of atomistic tasks.
Available task kinds:
  - "relax": run HydraGNN-driven structure relaxation on a structure file
  - "analyze": summarize results so far
  - "report": produce the final report
Always emit at least one "relax" task pointing at a real structure path
mentioned in the objective."""


def planner_node(state: MatSimState) -> dict:
    if not state.objective:
        return {"plan": [], "pending_tasks": []}

    structure_paths = [
        tok
        for tok in state.objective.split()
        if os.path.splitext(tok)[1] in {".vasp", ".cif", ".xyz", ".extxyz", ".poscar", ".pdb"}
    ]

    try:
        from matsim_agents.llm import get_chat_model

        llm = get_chat_model(
            provider=state.llm_provider,
            model=state.llm_model,
            base_url=state.llm_base_url,
        ).with_structured_output(_Plan)
        plan: _Plan = llm.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
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
