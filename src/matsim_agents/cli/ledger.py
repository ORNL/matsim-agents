"""Evidence-ledger subcommand group: ``matsim-agents ledger ...``."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from matsim_agents.cli._app import app, console

ledger_app = typer.Typer(
    add_completion=False,
    help="Inspect and verify an append-only evidence ledger (evidence-ledger.jsonl).",
)
app.add_typer(ledger_app, name="ledger")

LEDGER_ARGUMENT = typer.Argument(..., exists=True, dir_okay=False, help="Ledger JSONL file.")


@ledger_app.command("verify")
def ledger_verify(
    ledger: Path = LEDGER_ARGUMENT,
    artifacts: bool = typer.Option(
        True, "--artifacts/--no-artifacts", help="Also re-hash recorded input and output files."
    ),
):
    """Exit non-zero when the hash chain is broken or recorded artifacts changed."""

    from matsim_agents.execution.ledger import EvidenceLedger, LedgerIntegrityError

    try:
        opened = EvidenceLedger(ledger)
    except LedgerIntegrityError as exc:
        console.print(f"[red]tampered[/red]: {exc}")
        raise typer.Exit(code=1) from None
    problems = opened.verify_artifacts() if artifacts else []
    for problem in problems:
        console.print(f"[red]artifact[/red]: {problem}")
    if problems:
        raise typer.Exit(code=1)
    console.print(f"[green]ok[/green]: {len(opened)} record(s), hash chain intact")


@ledger_app.command("show")
def ledger_show(
    ledger: Path = LEDGER_ARGUMENT,
    hypothesis: str | None = typer.Option(
        None, "--hypothesis", help="Print every revision of this hypothesis instead."
    ),
):
    """Print a summary: record counts and the current revision of every hypothesis."""

    from matsim_agents.execution.ledger import EvidenceLedger

    opened = EvidenceLedger(ledger)
    if hypothesis is not None:
        history = [r.model_dump(mode="json") for r in opened.history(hypothesis)]
        if not history:
            console.print(f"[red]unknown hypothesis[/red]: {hypothesis}")
            raise typer.Exit(code=1)
        console.print_json(json.dumps(history))
        return
    console.print_json(json.dumps(opened.summary()))
