"""Command-line interface for matsim-agents (entry point: ``matsim-agents``)."""

import warnings

# Upstream LangGraph notice, printed on import; not actionable by CLI users.
warnings.filterwarnings("ignore", message=r"The default value of `allowed_objects` will change")

# Import order defines the order commands are listed in ``--help``.
from matsim_agents.cli import (  # noqa: E402, F401, I001  (register commands)
    workflow_commands,
    exploration,
    workflows,
    objective,
    interactive,
    composition,
    active_learning,
    resources,
    ledger,
    neb,
    magnetic,
    radiation_damage,
    benchmark,
    capabilities,
    coordination,
    inventory,
    e1,
    system,
    config_tools,
    tools,
)
from matsim_agents.cli._app import app  # noqa: E402

__all__ = ["app"]
