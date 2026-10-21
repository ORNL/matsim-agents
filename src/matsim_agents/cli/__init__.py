"""Command-line interface for matsim-agents (entry point: ``matsim-agents``)."""

# Import order defines the order commands are listed in ``--help``.
from matsim_agents.cli import (  # noqa: F401, I001  (register commands)
    workflow_commands,
    objective,
    interactive,
    composition,
    active_learning,
    resources,
    ledger,
)
from matsim_agents.cli._app import app

__all__ = ["app"]
