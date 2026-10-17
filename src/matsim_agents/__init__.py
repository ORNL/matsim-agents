"""matsim-agents: a multi-agent AI framework for atomistic materials simulation.

Subpackages (see ``docs/architecture.md`` for the full layer description):
    - orchestration: typed workflow state and LangGraph objective/composition graphs
    - agents: planner / executor / analyst graph nodes
    - workflows: composable relaxation, phase-exploration, and debate workflows
    - discovery: composition parsing, seed generation, and stability analysis
    - active_learning: uncertainty-driven acquisition, labeling, and retraining
    - campaign: multi-formula campaign planning, execution, and review
    - backends: LLM providers, MLIP calculators, and DFT (QE/VASP) interfaces
    - execution: scheduler-neutral resources, launch contracts, and provenance
    - cli: ``matsim-agents`` command-line interface
"""

from matsim_agents.orchestration.state import MatSimState

__all__ = ["MatSimState", "__version__"]
__version__ = "0.1.0"
