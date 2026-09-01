"""Compatibility alias for :mod:`matsim_agents.orchestration.composition_graph`."""

import sys

from matsim_agents.orchestration import composition_graph as _implementation

sys.modules[__name__] = _implementation
