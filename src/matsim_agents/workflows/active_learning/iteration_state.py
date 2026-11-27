"""State of one active-learning iteration and its on-disk record."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from matsim_agents.workflows.active_learning.config import (
    ALConfig,
)


@dataclass
class IterationState:
    iteration: int
    status: str = "running"  # running | complete | failed
    n_candidates: int = 0
    n_selected: int = 0
    n_dft_converged: int = 0
    n_dft_failed: int = 0
    n_dft_reused: int = 0
    n_dft_neb_runs: int = 0
    neb_pathways: list[dict[str, Any]] = field(default_factory=list)
    dft_backend: str | None = None
    score_min: float | None = None
    score_max: float | None = None
    score_mean: float | None = None
    selected_candidate_ids: list[str] = field(default_factory=list)
    candidate_uncertainty: dict[str, float] = field(default_factory=dict)
    dft_force_errors_eV_per_A: dict[str, float] = field(default_factory=dict)
    accuracy_triggered: bool | None = None
    dataset_path: str | None = None
    validation_dataset_path: str | None = None
    n_training_frames: int = 0
    n_validation_frames: int = 0
    n_training_frames_total: int = 0
    n_validation_frames_total: int = 0
    training_status: str = "disabled"
    training_deferred_reasons: list[str] = field(default_factory=list)
    candidate_model_path: str | None = None
    model_comparison: dict[str, Any] | None = None
    model_promoted: bool = False
    promotion_validation: dict[str, Any] | None = None
    lammps_export: dict[str, Any] | None = None
    new_logdir: str | None = None
    timings_sec: dict[str, float] = field(default_factory=dict)
    notes: str | None = None
    # Scores of the model in use on loop.benchmark_sets, after this iteration's decisions.
    benchmark: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


IterationAssessor = Callable[[ALConfig, int, Path, IterationState], None]


def _iter_dir(root: Path, i: int) -> Path:
    return root / f"iteration_{i:04d}"


def _write_state(it_dir: Path, state: IterationState) -> None:
    (it_dir / "state.json").write_text(json.dumps(state.to_dict(), indent=2))
