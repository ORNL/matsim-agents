"""Candidate-checkpointed, single-writer multi-round exploration.

An uncommitted attempt can run again after interruption. Committed attempts,
including backend failures and nonconvergence, are never retried implicitly.
"""

from __future__ import annotations

import fcntl
import logging
import platform
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.backends.mlip.relaxation import (
    CalculatorCache,
    RelaxStructureInput,
    relax_with_mlip,
)
from matsim_agents.backends.mlip.results import RelaxationResult
from matsim_agents.discovery.seeds import PhaseCandidate, generate_seed_action
from matsim_agents.discovery.stability import score_stability
from matsim_agents.execution.contracts import EvidenceLevel, ProvenanceRecord
from matsim_agents.execution.ledger import file_sha256
from matsim_agents.execution.run_directory import ScientificRunDirectory
from matsim_agents.workflows.exploration import (
    ExplorationConfig,
    ExplorationSummary,
    check_exploration,
)

log = logging.getLogger(__name__)


class Attempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: int = Field(ge=0)
    result: RelaxationResult | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _one_outcome(self) -> Attempt:
        if (self.result is None) == (self.error is None):
            raise ValueError("An attempt must contain exactly one result or error")
        return self


class Round(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[PhaseCandidate]
    attempts: list[Attempt] = Field(default_factory=list)
    complete: bool = False


class RoundState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    config: ExplorationConfig
    checkpoint_sha256: str
    compatibility: dict[str, str]
    rounds: list[Round] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    calculator_loads: int = Field(0, ge=0)
    complete: bool = False
    error: str | None = None

    @model_validator(mode="after")
    def _validate_progress(self) -> RoundState:
        if len(self.rounds) > self.config.iterations:
            raise ValueError("Checkpoint contains too many rounds")
        for index, round_ in enumerate(self.rounds):
            allowance = min(self.config.max_relaxations, len(round_.candidates))
            if [attempt.candidate for attempt in round_.attempts] != list(
                range(len(round_.attempts))
            ):
                raise ValueError("Checkpoint attempts must be ordered without gaps or duplicates")
            if len(round_.attempts) > allowance:
                raise ValueError("Checkpoint exceeds the per-round attempt budget")
            if round_.complete and len(round_.attempts) != allowance:
                raise ValueError("Completed round has missing attempts")
            if not round_.complete and index != len(self.rounds) - 1:
                raise ValueError("Checkpoint contains rounds after an incomplete round")
        if self.complete and (
            len(self.rounds) != self.config.iterations
            or not all(round_.complete for round_ in self.rounds)
        ):
            raise ValueError("Completed checkpoint has missing or incomplete rounds")
        return self


def _compatibility() -> dict[str, str]:
    # Refuse to combine evidence produced by changed workflow/optimizer code
    # or scientific dependencies. Users can inspect the original run instead.
    from matsim_agents.backends.mlip import calculator, mace_inference, relaxation
    from matsim_agents.discovery import composition, seeds, stability
    from matsim_agents.workflows import exploration

    result = {"python": platform.python_version()}
    for module in (
        calculator,
        mace_inference,
        relaxation,
        composition,
        seeds,
        stability,
        exploration,
    ):
        result[module.__name__] = file_sha256(module.__file__)
    result["rounds_source"] = file_sha256(__file__)
    for package in (
        "mace-torch",
        "torch",
        "e3nn",
        "ase",
        "numpy",
        "scipy",
        "pyxtal",
        "pymatgen",
        "pydantic",
        "torchmetrics",
        "lightning-utilities",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _state(run: ScientificRunDirectory) -> RoundState:
    return RoundState.model_validate_json((run.path / "rounds-state.json").read_text())


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


def _verify_artifacts(state: RoundState) -> None:
    required = {
        candidate.structure_path for round_ in state.rounds for candidate in round_.candidates
    }
    for round_ in state.rounds:
        for attempt in round_.attempts:
            if attempt.result is not None:
                required.update(
                    (
                        attempt.result.optimized_structure_path,
                        attempt.result.trajectory_path,
                        attempt.result.log_csv_path,
                    )
                )
    if any(str(Path(filename).resolve()) not in state.artifacts for filename in required):
        raise ValueError("Resume refused: checkpoint is missing required artifact hashes")
    for filename, digest in state.artifacts.items():
        path = Path(filename)
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Resume refused: missing or changed artifact: {path}")


def _remember(state: RoundState, filenames: list[str]) -> None:
    for filename in filenames:
        path = Path(filename).resolve()
        state.artifacts[str(path)] = file_sha256(path)


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
        # Normalize programmatic paths before storing portable resume identity.
        cfg = cfg.model_copy(
            update={
                "checkpoint": cfg.checkpoint.expanduser().resolve(),
                "output_dir": cfg.output_dir.expanduser().resolve(),
            }
        )
        check = check_exploration(cfg)
        if not check.ready:
            raise ValueError("Exploration preflight failed:\n" + "\n".join(check.errors))
        if check.checkpoint_sha256 is None:
            raise ValueError("Preflight did not identify checkpoint")
        run = ScientificRunDirectory.create(
            cfg.output_dir,
            workflow="exploration_rounds",
            request=cfg.model_dump(mode="json"),
            resolved_config=cfg.model_dump(mode="json"),
            provenance=ProvenanceRecord(
                workflow="exploration_rounds",
                evidence_level=EvidenceLevel.MLIP_RELAXATION,
                model_checkpoint_hash=check.checkpoint_sha256,
                numerical_settings=cfg.relaxation.model_dump(),
            ),
        )
        state = RoundState(
            config=cfg,
            checkpoint_sha256=check.checkpoint_sha256,
            compatibility=_compatibility(),
        )
        run.write_json("rounds-state.json", state)
    else:
        run = ScientificRunDirectory.open(run_directory)
        if not (run.path / "rounds-state.json").is_file():
            raise ValueError(
                "Only checkpointed multi-round exploration can resume; "
                "start a new run with iterations > 1."
            )

    with (run.path / ".exploration.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This exploration is already running in another process") from exc
        state = _state(run)
        cfg = state.config
        original_config = ExplorationConfig.model_validate_json(
            (run.path / "resolved_config.json").read_text()
        )
        if cfg != original_config:
            raise ValueError("Resume refused: checkpoint configuration differs from original run")
        if on_progress is not None:
            on_progress(f"Run directory: {run.path}")
        check = check_exploration(cfg)
        if not check.ready or check.checkpoint_sha256 != state.checkpoint_sha256:
            raise ValueError(
                "Resume preflight failed: "
                + "; ".join([*check.errors, "checkpoint must match the original run"])
            )
        if state.compatibility != _compatibility():
            raise ValueError(
                "Resume refused: workflow code or scientific dependency versions changed"
            )
        _verify_artifacts(state)
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
                    state.rounds.append(Round(candidates=candidates))
                    run.write_json("rounds-state.json", state)
                round_ = state.rounds[index]
                if round_.complete:
                    continue
                if on_progress is not None:
                    on_progress(
                        f"Round {index + 1}/{cfg.iterations}: "
                        f"{len(round_.attempts)} attempts already checkpointed"
                    )
                allowance = min(cfg.max_relaxations, len(round_.candidates))
                for candidate_index in range(len(round_.attempts), allowance):
                    candidate = round_.candidates[candidate_index]
                    request = RelaxStructureInput(
                        structure_path=candidate.structure_path,
                        output_dir=str(folder / f"attempt-{candidate_index:04d}"),
                        mlip_backend="mace",
                        mace_family="checkpoint",
                        mace_model=str(cfg.checkpoint),
                        mlp_device=cfg.device,
                        precision=cfg.precision,
                        **cfg.relaxation.model_dump(),
                    )
                    try:
                        result = relax_with_mlip(request, calculators=cache)
                    except Exception as exc:
                        # Match the existing discovery workflow: persist the
                        # failed attempt, report it, and consume its budget.
                        log.exception("Round %s candidate %s failed", index + 1, candidate_index)
                        attempt = Attempt(
                            candidate=candidate_index, error=f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        _remember(
                            state,
                            [
                                result.optimized_structure_path,
                                result.trajectory_path,
                                result.log_csv_path,
                            ],
                        )
                        attempt = Attempt(candidate=candidate_index, result=result)
                    round_.attempts.append(attempt)
                    state.calculator_loads = previous_loads + cache.loads
                    run.write_json("rounds-state.json", state)
                    if on_progress is not None:
                        reason = (
                            attempt.result.termination_reason if attempt.result else attempt.error
                        )
                        on_progress(f"Round {index + 1}, attempt {candidate_index + 1}: {reason}")
                round_.complete = True
                run.write_json("rounds-state.json", state)
                run.append_event("round_finished", {"round": index + 1})
                _write_ranking(run, state)
            _write_ranking(run, state)
            state.complete = True
        except BaseException as exc:
            state.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            state.calculator_loads = previous_loads + cache.loads
            run.write_json("rounds-state.json", state)
            run.write_json("exploration-summary.json", summarize_rounds(run))
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
    run.write_json("cumulative-ranking.json", report)
