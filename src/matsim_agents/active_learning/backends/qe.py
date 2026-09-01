"""Compatibility alias for :mod:`matsim_agents.backends.dft.qe`."""

import sys

from matsim_agents.backends.dft import qe as _implementation

sys.modules[__name__] = _implementation
