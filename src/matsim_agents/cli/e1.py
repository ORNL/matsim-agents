"""E1 development screening command."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from matsim_agents.cli._app import app, console


@app.command("e1-development")
def e1_development(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", "-o", file_okay=False),
    retry_errors: bool = typer.Option(
        False,
        "--retry-errors",
        help="Retry recorded errors only within the manifest's approved attempt limit.",
    ),
) -> None:
    """Run development questions for one candidate; held-out questions are not accepted."""
    from matsim_agents.workflows.e1 import E1DevelopmentManifest, run_e1_development

    document = json.loads(manifest.read_text(encoding="utf-8"))
    config = E1DevelopmentManifest.model_validate(document)
    result = run_e1_development(config, output, retry_errors=retry_errors)
    console.print_json(json.dumps(result, sort_keys=True))