"""Materials-discovery building blocks.

Submodules:
    - composition:   parse chemical compositions from free text.
    - seeds:         unified seed generator (AFLOW prototype decoration
      + optional pyXtal random search) for a given composition.
    - phase_explorer: backward-compatible shim around ``seeds.generate_seeds``.
    - stability:     score chemical / dynamical stability from a batch of
      relaxed structures.
    - wrapper:       high-level helper that ties seed generation,
      relaxation, and stability scoring together.
"""

from matsim_agents.discovery.composition import (
    Composition,
    extract_compositions,
    parse_composition,
)
from matsim_agents.discovery.phase_explorer import enumerate_phases
from matsim_agents.discovery.seeds import (
    PhaseCandidate,
    compatible_prototypes,
    generate_seeds,
    load_prototypes,
    random_search,
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
    "compatible_prototypes",
    "enumerate_phases",
    "explore_composition",
    "extract_compositions",
    "generate_seeds",
    "load_prototypes",
    "parse_composition",
    "random_search",
    "score_stability",
]
