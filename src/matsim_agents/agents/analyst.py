"""Analyst agent.

Summarizes accumulated relaxation results into a human-readable report.
Uses the configured LLM when available, otherwise a deterministic summary.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

from matsim_agents.state import MatSimState

_SYSTEM_PROMPT = """You are the analyst agent of a materials-discovery system.
Given a JSON list of structure relaxation results, write a concise scientific
summary highlighting: lowest-energy structure, convergence behaviour, and any
branch-weighting trends from the auxiliary MLP."""


def _deterministic_summary(state: MatSimState) -> str:
    if not state.results:
        return "No relaxation results to analyze."
    best = min(state.results, key=lambda r: r.final_energy_eV)
    return (
        f"Analyzed {len(state.results)} relaxation(s). "
        f"Lowest energy: {best.final_energy_eV:.4f} eV "
        f"({best.optimized_structure_path}) in {best.num_steps} step(s); "
        f"final |F|max = {best.final_max_force_eV_per_A:.4f} eV/Å."
    )


def analyst_node(state: MatSimState) -> dict:
    summary = _deterministic_summary(state)

    try:
        from matsim_agents.llm import get_chat_model

        llm = get_chat_model(
            provider=state.llm_provider,
            model=state.llm_model,
            base_url=state.llm_base_url,
        )
        results_json = [r.model_dump() for r in state.results]
        rsp = llm.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                AIMessage(content=f"Results: {results_json}\nDeterministic summary: {summary}"),
            ]
        )
        summary = rsp.content if isinstance(rsp.content, str) else str(rsp.content)
    except Exception:  # pragma: no cover - LLM is optional
        pass

    return {
        "analysis": summary,
        "messages": [AIMessage(content=f"[analyst] {summary}")],
    }
