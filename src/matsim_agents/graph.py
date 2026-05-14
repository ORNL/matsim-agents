"""Assemble the LangGraph workflow.

The graph implements a simple Plan -> Execute (loop) -> Analyze pipeline:

    planner ─► executor ──┐
                  ▲       │
                  └───────┤  while pending_tasks
                          ▼
                       analyst ─► END
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from matsim_agents.agents import analyst_node, executor_node, planner_node
from matsim_agents.state import MatSimState


def _route_after_executor(state: MatSimState) -> Literal["executor", "analyst"]:
    if state.pending_tasks and state.iteration < state.max_iterations:
        return "executor"
    return "analyst"


def build_graph(checkpointer=None):
    """Compile and return the matsim-agents LangGraph workflow."""
    graph = StateGraph(MatSimState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("analyst", analyst_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", _route_after_executor,
                                {"executor": "executor", "analyst": "analyst"})
    graph.add_edge("analyst", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
