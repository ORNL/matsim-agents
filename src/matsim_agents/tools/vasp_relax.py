"""Compatibility alias for :mod:`matsim_agents.backends.dft.vasp_relax`."""

import sys

from matsim_agents.backends.dft import vasp_relax as _implementation

sys.modules[__name__] = _implementation
