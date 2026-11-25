"""Deprecated alias of :mod:`matsim_agents.active_learning.md_sampler`."""

import sys
import warnings

from matsim_agents.active_learning import md_sampler

warnings.warn(
    "matsim_agents.active_learning.candidates is deprecated; "
    "use matsim_agents.active_learning.md_sampler",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = md_sampler
