"""Public config-driven exploration API, shared by CLI and Python callers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from matsim_agents.workflows.exploration_config import (
    ExplorationCheck,
    ExplorationConfig,
    ExplorationGenerationConfig,
    ExplorationRelaxationConfig,
    check_exploration,
    load_exploration_config,
)
from matsim_agents.workflows.exploration_reporting import (
    ExplorationSummary,
    summarize_exploration,
)

__all__ = [
    "ExplorationCheck",
    "ExplorationConfig",
    "ExplorationGenerationConfig",
    "ExplorationRelaxationConfig",
    "ExplorationSummary",
    "check_exploration",
    "load_exploration_config",
    "resume_exploration",
    "run_exploration",
    "summarize_exploration",
]


def run_exploration(
    cfg: ExplorationConfig, *, on_progress: Callable[[str], None] | None = None
) -> ExplorationSummary:
    """Run one or more bounded, checkpointed rounds with one calculator cache."""
    from matsim_agents.workflows.exploration_rounds import run_rounds

    return run_rounds(cfg, on_progress=on_progress)


def resume_exploration(
    run_directory: str | Path, *, on_progress: Callable[[str], None] | None = None
) -> ExplorationSummary:
    """Resume checkpointed work without repeating completed attempts."""
    from matsim_agents.workflows.exploration_rounds import run_rounds

    return run_rounds(None, run_directory=run_directory, on_progress=on_progress)
