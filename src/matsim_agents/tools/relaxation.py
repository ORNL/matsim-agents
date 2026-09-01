"""Compatibility alias for :mod:`matsim_agents.backends.mlip.relaxation`."""

import sys

from matsim_agents.backends.mlip import relaxation as _implementation

sys.modules[__name__] = _implementation
