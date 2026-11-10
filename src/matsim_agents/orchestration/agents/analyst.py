"""Analyst agent.

Summarizes accumulated relaxation results into a human-readable report.
Uses the configured LLM when available, otherwise a deterministic summary.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

from matsim_agents.orchestration.state import MatSimState

_SYSTEM_PROMPT = """You are the analyst agent of a materials-discovery system.
Given a JSON list of structure relaxation results, write a concise scientific
summary highlighting: lowest-energy structure, convergence behaviour, and any
branch-weighting trends from the auxiliary MLP. Compare energies only per atom
and only between structures of the same composition; raw total energies of
different sizes or compositions are not comparable."""


def _deterministic_summary(state: MatSimState) -> str:
    if not state.results:
        return "No relaxation results to analyze."
    from ase.io import read

    # Raw totals are only comparable per atom and within one composition.
    by_formula: dict[str, list[tuple[float, object]]] = {}
    unreadable = []
    for result in state.results:
        try:
            atoms = read(result.optimized_structure_path)
        except Exception:  # noqa: BLE001
            unreadable.append(result.optimized_structure_path)
            continue
        by_formula.setdefault(atoms.get_chemical_formula(mode="hill", empirical=True), []).append(
            (result.final_energy_eV / len(atoms), result)
        )
    lines = [f"Analyzed {len(state.results)} relaxation(s)."]
    for formula, entries in sorted(by_formula.items()):
        energy_per_atom, best = min(entries, key=lambda entry: entry[0])
        lines.append(
            f"{formula}: lowest energy {energy_per_atom:.4f} eV/atom "
            f"(total {best.final_energy_eV:.4f} eV, {best.optimized_structure_path}) "
            f"in {best.num_steps} step(s); final |F|max = "
            f"{best.final_max_force_eV_per_A:.4f} eV/Å."
        )
    if unreadable:
        lines.append(
            f"{len(unreadable)} result(s) not ranked (structure unreadable): "
            + ", ".join(unreadable)
        )
    summary = "\n".join(lines)
    if state.handoff_events:
        summary += "\nAL handoff events:\n- " + "\n- ".join(state.handoff_events)
    return summary


def analyst_node(state: MatSimState) -> dict:
    summary = _deterministic_summary(state)

    try:
        from matsim_agents.backends.llm.provider import get_chat_model
        from matsim_agents.backends.llm.replies import invoke_model

        llm = get_chat_model(
            provider=state.llm_provider,
            model=state.llm_model,
            base_url=state.llm_base_url,
        )
        results_json = [r.model_dump() for r in state.results]
        handoff_events = state.handoff_events
        summary = invoke_model(
            llm,
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                AIMessage(
                    content=(
                        f"Results: {results_json}\n"
                        f"AL handoff events: {handoff_events}\n"
                        f"Deterministic summary: {summary}"
                    )
                ),
            ],
        ).text
    except Exception:  # pragma: no cover - LLM is optional
        pass

    return {
        "analysis": summary,
        "messages": [AIMessage(content=f"[analyst] {summary}")],
    }
