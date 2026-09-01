"""Compatibility alias for :mod:`matsim_agents.backends.llm.provider`."""

import sys

from matsim_agents.backends.llm import provider as _implementation

sys.modules[__name__] = _implementation
