"""Compatibility alias for :mod:`matsim_agents.orchestration.objective_graph`."""

import sys

from matsim_agents.orchestration import objective_graph as _implementation

sys.modules[__name__] = _implementation
