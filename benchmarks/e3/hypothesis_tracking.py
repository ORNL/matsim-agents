"""E3 protocol version using the shared scientific hypothesis contract."""

from matsim_agents.backends.llm.hypotheses import (
    FIELDS,
    INSTRUCTIONS,
    STATUSES,
    validate_hypotheses,
)

VERSION = "e3-mlip-hypotheses-v1"

__all__ = ["FIELDS", "INSTRUCTIONS", "STATUSES", "VERSION", "validate_hypotheses"]
