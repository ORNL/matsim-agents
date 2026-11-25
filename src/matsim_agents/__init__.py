"""matsim-agents: a multi-agent AI framework for atomistic materials simulation.

Subpackages (see ``docs/restructuring/target-architecture.md`` for the layers):
    - core: contracts, seeding, hashing, statistics, energy conventions
    - execution: scheduler-neutral resources, launch contracts, and provenance
    - data: dataset governance, splits, and energy alignment
    - backends: LLM, DFT, MLIP (with training), MCP, and external-code adapters
    - science: structures, defects, NEB, magnetism, kinetics, irradiation, properties
    - workflows: relaxation, exploration, active learning, NEB, and debate workflows
    - agents: LangGraph state, graphs, and planner / executor / analyst roles
    - cli: ``matsim-agents`` command-line interface
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["MatSimState", "__version__"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    # Deferred: the state module imports LangGraph, which is slow to load.
    if name == "MatSimState":
        from matsim_agents.agents.state import MatSimState

        return MatSimState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from matsim_agents.agents.state import MatSimState
