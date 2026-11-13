"""One checkpointed execution path for single- and multi-round exploration.

An uncommitted attempt can run again after interruption. Committed attempts,
including backend failures and nonconvergence, are never retried implicitly.
"""

from __future__ import annotations

import fcntl
import logging
from collections.abc import Callable
from pathlib import Path

from matsim_agents.backends.mlip.relaxation import CalculatorCache, relax_with_mlip
from matsim_agents.discovery.seeds import generate_seed_action
from matsim_agents.discovery.selection import select_diverse_candidates
from matsim_agents.execution.run_directory import ScientificRunDirectory
from matsim_agents.workflows.exploration_checkpoint import (
    Attempt,
    Round,
    RoundState,
    _remember,
    _selection_references,
    _state,
    create_run,
    validate_resume,
)
from matsim_agents.workflows.exploration_config import ExplorationConfig
from matsim_agents.workflows.exploration_reporting import (
    ExplorationSummary,
    _write_ranking,
    summarize_rounds,
)

log = logging.getLogger(__name__)


def run_rounds(
    cfg: ExplorationConfig | None,
    *,
    run_directory: str | Path | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> ExplorationSummary:
    """Run/resume bounded rounds. One lock covers generation, attempts, and ranking."""
    if run_directory is None:
        if cfg is None:
            raise ValueError("A configuration is required for a new run")
        run = create_run(cfg)
    else:
        run = ScientificRunDirectory.open(run_directory)

    with (run.path / ".exploration.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This exploration is already running in another process") from exc
        state = _state(run)
        cfg = state.config
        if on_progress is not None:
            on_progress(f"Run directory: {run.path}")
        validate_resume(run, state)
        if state.complete:
            return summarize_rounds(run)
        state.error = None
        run.write_json("rounds-state.json", state)
        cache = CalculatorCache(capacity=1)
        previous_loads = state.calculator_loads
        try:
            for index in range(cfg.iterations):
                folder = run.path / f"round-{index + 1:04d}"
                if index == len(state.rounds):
                    state.rounds.append(_generate_round(state, index, folder))
                    run.write_json("rounds-state.json", state)
                round_ = state.rounds[index]
                if round_.complete:
                    continue
                _execute_round(run, state, index, cache, previous_loads, on_progress)
                round_.complete = True
                run.write_json("rounds-state.json", state)
                run.append_event("round_finished", {"round": index + 1})
                _write_ranking(run, state)
            _write_ranking(run, state)
            _remember(state, [str(run.path / "cumulative-ranking.json")])
            state.complete = True
        except BaseException as exc:
            # Only publish the last atomically committed work, not partial
            # hashes/results left in memory by an interrupted checkpoint.
            state = _state(run)
            state.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            state.calculator_loads = previous_loads + cache.loads
            run.write_json("rounds-state.json", state)
            run.write_json("exploration-summary.json", summarize_rounds(run))
        return summarize_rounds(run)


def _generate_round(state: RoundState, index: int, folder: Path) -> Round:
    cfg = state.config
    action = cfg.generation_action().model_copy(
        update={
            "action_id": f"round-{index + 1:04d}",
            "random_seed": cfg.generation.random_seed + index,
        }
    )
    candidates = [
        candidate.model_copy(
            update={"structure_path": str(Path(candidate.structure_path).resolve())}
        )
        for candidate in generate_seed_action(action, str(folder / "seeds"))
    ]
    _remember(state, [candidate.structure_path for candidate in candidates])
    selection = (
        select_diverse_candidates(
            [candidate.structure_path for candidate in candidates],
            _selection_references(state.rounds, cfg.relaxation.fmax),
            formula=cfg.formula,
            budget=cfg.max_relaxations,
        )
        if cfg.selection == "diverse"
        else None
    )
    return Round(candidates=candidates, selection=selection)


def _execute_round(
    run: ScientificRunDirectory,
    state: RoundState,
    index: int,
    cache: CalculatorCache,
    previous_loads: int,
    on_progress: Callable[[str], None] | None,
) -> None:
    cfg = state.config
    round_ = state.rounds[index]
    folder = run.path / f"round-{index + 1:04d}"
    selected = (
        round_.selection.selected
        if round_.selection is not None
        else list(range(min(cfg.max_relaxations, len(round_.candidates))))
    )
    if on_progress is not None:
        on_progress(
            f"Round {index + 1}/{cfg.iterations}: "
            f"{len(round_.attempts)} attempts already checkpointed"
        )
        if round_.selection is not None:
            on_progress(
                f"Round {index + 1}: {len(selected)} selected; "
                f"{sum(d.disposition == 'duplicate' for d in round_.selection.decisions)} "
                "duplicates skipped"
            )
    for candidate_index in selected[len(round_.attempts) :]:
        candidate = round_.candidates[candidate_index]
        request = cfg.relaxation_request(
            candidate.structure_path, str(folder / f"attempt-{candidate_index:04d}")
        )
        try:
            result = relax_with_mlip(request, calculators=cache)
        except Exception as exc:
            log.exception("Round %s candidate %s failed", index + 1, candidate_index)
            attempt = Attempt(candidate=candidate_index, error=f"{type(exc).__name__}: {exc}")
        else:
            _remember(
                state,
                [result.optimized_structure_path, result.trajectory_path, result.log_csv_path],
            )
            attempt = Attempt(candidate=candidate_index, result=result)
        round_.attempts.append(attempt)
        state.calculator_loads = previous_loads + cache.loads
        run.write_json("rounds-state.json", state)
        if on_progress is not None:
            reason = attempt.result.termination_reason if attempt.result else attempt.error
            on_progress(f"Round {index + 1}, attempt {candidate_index + 1}: {reason}")
