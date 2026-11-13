"""Exploration journal contracts, compatibility checks, and artifact integrity."""

from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.backends.mlip.results import RelaxationResult
from matsim_agents.discovery.seeds import PhaseCandidate
from matsim_agents.discovery.selection import CandidateSelection
from matsim_agents.execution.contracts import EvidenceLevel, ProvenanceRecord
from matsim_agents.execution.ledger import file_sha256
from matsim_agents.execution.run_directory import ScientificRunDirectory
from matsim_agents.workflows.exploration_config import ExplorationConfig, check_exploration


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
    selection: CandidateSelection | None = None
    attempts: list[Attempt] = Field(default_factory=list)
    complete: bool = False


class RoundState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2]
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
            if self.config.selection == "diverse":
                if round_.selection is None:
                    raise ValueError("Diverse round is missing checkpointed selection")
                if len(round_.selection.decisions) != len(round_.candidates):
                    raise ValueError("Selection must account for every generated candidate")
                selected = round_.selection.selected
                if len(selected) > self.config.max_relaxations:
                    raise ValueError("Selection exceeds the per-round attempt budget")
                available = sum(
                    decision.disposition != "duplicate" for decision in round_.selection.decisions
                )
                if len(selected) != min(available, self.config.max_relaxations):
                    raise ValueError(
                        "Selection must fill the budget from available unique candidates"
                    )
                if round_.selection.references != _selection_references(
                    self.rounds[:index], self.config.relaxation.fmax
                ):
                    raise ValueError("Selection references do not match prior checkpointed work")
                representatives = set(round_.selection.references)
                for decision, candidate in zip(
                    round_.selection.decisions, round_.candidates, strict=True
                ):
                    if decision.disposition == "duplicate":
                        if decision.duplicate_of not in representatives:
                            raise ValueError("Duplicate decision has no prior representative")
                    else:
                        representatives.add(candidate.structure_path)
            else:
                if round_.selection is not None:
                    raise ValueError("Sequential round cannot contain a diverse selection")
                selected = list(range(min(self.config.max_relaxations, len(round_.candidates))))
            allowance = len(selected)
            if [attempt.candidate for attempt in round_.attempts] != selected[
                : len(round_.attempts)
            ]:
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
    from matsim_agents.discovery import composition, seeds, selection, stability
    from matsim_agents.workflows import (
        exploration,
        exploration_checkpoint,
        exploration_config,
        exploration_reporting,
        exploration_rounds,
    )

    result = {"python": platform.python_version()}
    for module in (
        calculator,
        mace_inference,
        relaxation,
        composition,
        seeds,
        selection,
        stability,
        exploration,
        exploration_checkpoint,
        exploration_config,
        exploration_reporting,
        exploration_rounds,
    ):
        result[module.__name__] = file_sha256(module.__file__)
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
    path = run.path / "rounds-state.json"
    if not path.is_file():
        raise ValueError(
            "Unsupported exploration run format: no candidate checkpoint. "
            "Start a new run with explore run CONFIG."
        )
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise ValueError(
            "Unsupported exploration checkpoint version; this workflow requires version 2. "
            "Start a new run with explore run CONFIG."
        )
    return RoundState.model_validate(raw)


def create_run(cfg: ExplorationConfig) -> ScientificRunDirectory:
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
            random_seed=cfg.generation.random_seed,
        ),
    )
    state = RoundState(
        schema_version=2,
        config=cfg,
        checkpoint_sha256=check.checkpoint_sha256,
        compatibility=_compatibility(),
    )
    run.write_json("rounds-state.json", state)
    return run


def validate_resume(run: ScientificRunDirectory, state: RoundState) -> None:
    original_config = ExplorationConfig.model_validate_json(
        (run.path / "resolved_config.json").read_text()
    )
    if state.config != original_config:
        raise ValueError("Resume refused: checkpoint configuration differs from original run")
    check = check_exploration(state.config)
    if not check.ready or check.checkpoint_sha256 != state.checkpoint_sha256:
        raise ValueError(
            "Resume preflight failed: "
            + "; ".join([*check.errors, "checkpoint must match the original run"])
        )
    if state.compatibility != _compatibility():
        raise ValueError("Resume refused: workflow code or scientific dependency versions changed")
    _verify_artifacts(run, state)


def _verify_artifacts(run: ScientificRunDirectory, state: RoundState) -> None:
    required = {
        candidate.structure_path for round_ in state.rounds for candidate in round_.candidates
    }
    if state.complete:
        required.add(str(run.path / "cumulative-ranking.json"))
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


def _selection_references(rounds: list[Round], fmax: float) -> list[str]:
    references = []
    for round_ in rounds:
        for attempt in round_.attempts:
            references.append(round_.candidates[attempt.candidate].structure_path)
            result = attempt.result
            if result is not None and result.converged and result.final_max_force_eV_per_A <= fmax:
                references.append(result.optimized_structure_path)
    return list(dict.fromkeys(references))
