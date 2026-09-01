"""Compatibility alias for :mod:`matsim_agents.orchestration.state`."""

import sys

from matsim_agents.orchestration import state as _implementation

sys.modules[__name__] = _implementation
