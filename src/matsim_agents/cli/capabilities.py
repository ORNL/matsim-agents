"""Backend-capability subcommand group: ``matsim-agents capabilities ...``."""

from __future__ import annotations

import json
import typing
from pathlib import Path

import typer

from matsim_agents.cli._app import app, console

capabilities_app = typer.Typer(
    add_completion=False,
    help="List DFT, MLIP and LLM backends and check which can run here.",
)
app.add_typer(capabilities_app, name="capabilities")


@capabilities_app.command("check")
def capabilities_check(
    kind: list[str] = typer.Option(None, "--kind", help="Only dft, mlip or llm (repeatable)."),
    require: list[str] = typer.Option(
        None, "--require", help="Exit non-zero unless this backend is available (repeatable)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the checks as JSON."),
):
    """Report which backends have their modules, launchers and env vars present."""

    from matsim_agents.execution.capabilities import (
        BackendKind,
        check,
        check_all,
        get_capability,
    )

    unknown = sorted(set(kind or ()) - set(typing.get_args(BackendKind)))
    if unknown:
        typer.echo(f"unknown kind(s): {', '.join(unknown)} (expected dft, mlip or llm)", err=True)
        raise typer.Exit(code=2)
    try:
        required = [check(get_capability(name)) for name in require or ()]
    except KeyError as exc:
        typer.echo(exc.args[0], err=True)
        raise typer.Exit(code=2) from None
    results = check_all(kind)
    if as_json:
        console.print_json(json.dumps([r.model_dump(mode="json") for r in results]))
    else:
        for r in results:
            status = "[green]available[/green]" if r.available else "[yellow]missing[/yellow]"
            detail = f" ({'; '.join(r.missing)})" if r.missing else ""
            if r.python:
                detail = f" (via {r.python})"
            console.print(f"{r.kind:4} {r.name:12} {status}{detail}")
    unavailable = [r.name for r in required if not r.available]
    if unavailable:
        typer.echo(f"required backend(s) unavailable: {', '.join(unavailable)}", err=True)
        raise typer.Exit(code=1)


@capabilities_app.command("matrix")
def capabilities_matrix(
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the markdown here instead of stdout."
    ),
):
    """Print the static capability matrix as markdown (source of docs/capabilities.md)."""

    from matsim_agents.execution.capabilities import render_markdown

    text = render_markdown()
    if output is None:
        typer.echo(text, nl=False)
    else:
        output.write_text(text, encoding="utf-8")
        console.print(f"wrote {output}")
