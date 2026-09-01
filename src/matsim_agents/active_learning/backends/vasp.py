"""Compatibility alias for :mod:`matsim_agents.backends.dft.vasp`."""

import sys

from matsim_agents.backends.dft import vasp as _implementation

sys.modules[__name__] = _implementation
