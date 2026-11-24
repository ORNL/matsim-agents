"""Deprecated alias of :mod:`matsim_agents.discovery.structural_screening`."""

import sys
import warnings

from matsim_agents.discovery import structural_screening

warnings.warn(
    "matsim_agents.discovery.selection is deprecated; "
    "use matsim_agents.discovery.structural_screening",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = structural_screening
