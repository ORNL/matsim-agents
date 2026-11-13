"""Read-only exploration summaries and cumulative relative-energy reports."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from matsim_agents.discovery.stability import score_stability
from matsim_agents.execution.run_directory import ScientificRunDirectory
from matsim_agents.workflows.exploration_checkpoint import RoundState, _state


class ExplorationSummary(BaseModel):
    run_directory: str
    status: Literal["completed", "failed"]
    outcome: str
    counts: dict[str, int]
    termination_reasons: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    calculator_loads: int | None = None
    rounds_completed: int = 0
    rounds_requested: int = 1


def summarize_rounds(run: ScientificRunDirectory) -> ExplorationSummary:
    state = _state(run)
    attempts = [attempt for round_ in state.rounds for attempt in round_.attempts]
    results = [attempt.result for attempt in attempts if attempt.result is not None]
    errors = [attempt.error for attempt in attempts if attempt.error is not None]
    if state.error:
        errors.append(state.error)
    counts = {
        "generated": sum(len(round_.candidates) for round_ in state.rounds),
        "attempted": len(attempts),
        "completed": len(results),
        "converged": sum(result.converged for result in results),
        "nonconverged": sum(not result.converged for result in results),
        "backend_errors": sum(attempt.error is not None for attempt in attempts),
    }
    counts["failed"] = counts["nonconverged"] + counts["backend_errors"]
    if state.config.selection == "diverse":
        counts.update(
            {
                disposition: sum(
                    decision.disposition == disposition
                    for round_ in state.rounds
                    if round_.selection is not None
                    for decision in round_.selection.decisions
                )
                for disposition in ("selected", "duplicate", "deferred")
            }
        )
    reasons: dict[str, int] = {}
    for result in results:
        reason = result.termination_reason or "not_recorded"
        reasons[reason] = reasons.get(reason, 0) + 1
    errors.extend(
        f"No candidates generated in round {index + 1}."
        for index, round_ in enumerate(state.rounds)
        if not round_.candidates
    )
    return ExplorationSummary(
        run_directory=str(run.path),
        status="failed" if errors or not state.complete else "completed",
        outcome=(
            "incomplete"
            if not state.complete
            else "usable_minimum"
            if counts["converged"]
            else "relaxation_non_convergence"
        ),
        counts=counts,
        errors=errors,
        termination_reasons=reasons,
        calculator_loads=state.calculator_loads,
        rounds_completed=sum(round_.complete for round_ in state.rounds),
        rounds_requested=state.config.iterations,
    )


def summarize_exploration(run_directory: str | Path) -> ExplorationSummary:
    """Read the canonical persisted evidence, without importing a model."""
    run = ScientificRunDirectory.open(run_directory)
    return summarize_rounds(run)


def _write_ranking(run: ScientificRunDirectory, state: RoundState) -> None:
    results = [
        attempt.result
        for round_ in state.rounds
        for attempt in round_.attempts
        if attempt.result is not None
        and attempt.result.converged
        and attempt.result.final_max_force_eV_per_A <= state.config.relaxation.fmax
    ]
    report = (
        score_stability(
            state.config.formula,
            results,
            force_tol_eV_per_A=state.config.relaxation.fmax,
            candidates=[candidate for round_ in state.rounds for candidate in round_.candidates],
        )
        if results
        else None
    )
    if report is None:
        run.write_json("cumulative-ranking.json", None)
        return
    payload = report.model_dump(
        mode="json",
        include={
            "formula",
            "ground_state",
            "ranking",
            "evidence_level",
            "ranking_mode",
            "limitations",
        },
    )
    payload["summary"] = (
        f"Relative energy ranking of {len(report.ranking)} eligible relaxation attempt(s). "
        "The ground_state entry is the lowest-energy observed candidate, "
        "not a proven ground state. "
        "Equivalent relaxed structures may appear more than once; no polymorph-degeneracy "
        "or chemical-stability proxy is assessed."
    )
    run.write_json("cumulative-ranking.json", payload)
