"""Compatibility alias for :mod:`matsim_agents.backends.dft.qe_relax`."""

import sys

from matsim_agents.backends.dft import qe_relax as _implementation

sys.modules[__name__] = _implementation
