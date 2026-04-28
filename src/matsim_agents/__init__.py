"""matsim-agents: a multi-agent AI framework for atomistic materials simulation.

Top-level package. Submodules:
    - state: typed shared state for the agent graph
    - tools: callable tools that wrap atomistic backends (HydraGNN, ASE, ...)
    - agents: planner / executor / analyst nodes
    - graph: assembled LangGraph workflow
    - llm: LLM provider factory
    - cli: command-line interface
"""

from matsim_agents.state import MatSimState

__all__ = ["MatSimState", "__version__"]
__version__ = "0.1.0"
