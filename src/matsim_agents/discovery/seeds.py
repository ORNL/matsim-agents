"""Deprecated alias of :mod:`matsim_agents.discovery.seed_structures`."""

import sys
import warnings

from matsim_agents.discovery import seed_structures

warnings.warn(
    "matsim_agents.discovery.seeds is deprecated; use matsim_agents.discovery.seed_structures",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = seed_structures
