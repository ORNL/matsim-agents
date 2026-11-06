"""Read-only inventory commands for retained scientific runs."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from matsim_agents.cli._app import app, console

inventory_app = typer.Typer(add_completion=False, help="Inventory retained run artifacts.")
app.add_typer(inventory_app, name="inventory")


@inventory_app.command("dft")
def inventory_dft(
    root: Path = typer.Argument(..., exists=True, file_okay=False, help="Run tree to scan."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write JSON report here."),
    accounting_csv: Path | None = typer.Option(
        None,
        "--accounting-csv",
        exists=True,
        dir_okay=False,
        help="Optional CSV: run_directory,elapsed_seconds,nodes,gpus.",
    ),
):
    """Report native DFT application timers and explicitly supplied allocations."""

    from matsim_agents.execution.dft_inventory import inventory_dft_runs

    try:
        report = inventory_dft_runs(root, accounting_csv)
    except (OSError, ValueError) as exc:
        console.print(f"[red]inventory error[/red]: {exc}")
        raise typer.Exit(code=2) from None
    rendered = json.dumps(report, indent=2)
    if output is None:
        console.print(rendered, markup=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        console.print(f"Wrote {report['run_count']} DFT run record(s) to {output}")
