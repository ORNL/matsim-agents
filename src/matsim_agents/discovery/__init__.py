"""Materials-discovery building blocks.

Submodules:
    - composition:   parse chemical compositions from free text.
    - phase_explorer: enumerate candidate crystal phases / atomistic
      arrangements for a given composition and write them as ASE-readable
      structure files.
    - stability:     score chemical / dynamical stability from a batch of
      relaxed structures.
    - wrapper:       high-level helper that ties phase enumeration,
      relaxation, and stability scoring together.
"""

from matsim_agents.discovery.composition import (
    Composition,
    extract_compositions,
    parse_composition,
)
from matsim_agents.discovery.phase_explorer import (
    PhaseCandidate,
    enumerate_phases,
)
from matsim_agents.discovery.stability import StabilityReport, score_stability
from matsim_agents.discovery.wrapper import (
    CompositionExplorationResult,
    explore_composition,
)

__all__ = [
    "Composition",
    "CompositionExplorationResult",
    "PhaseCandidate",
    "StabilityReport",
    "enumerate_phases",
    "explore_composition",
    "extract_compositions",
    "parse_composition",
    "score_stability",
]
