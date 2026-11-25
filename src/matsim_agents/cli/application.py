"""Application commands: ``matsim-agents application check FILE``."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from matsim_agents.cli._app import app, console

application_app = typer.Typer(
    add_completion=False,
    help="Validate applications (examples/<app>/application.yaml) against their workflow.",
)
app.add_typer(application_app, name="application")


@application_app.command("check")
def check(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Dry run: load the application and its workflow config; nothing is computed."""
    from matsim_agents.core.application import load_application
    from matsim_agents.workflows.applications import check_application

    try:
        summary = check_application(load_application(path))
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print_json(json.dumps(summary, sort_keys=True))
