"""Tools exposed to LangGraph agents.

Each tool is a thin, typed wrapper around a scientific backend. Heavy
dependencies (torch, HydraGNN, ASE optimizers, ...) are imported lazily
inside the tool body so the package remains importable in lightweight
environments.
"""

from matsim_agents.tools.relaxation import relax_structure

__all__ = ["relax_structure"]
