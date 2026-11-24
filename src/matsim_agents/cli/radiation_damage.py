"""Radiation-damage subcommand group: ``matsim-agents radiation-damage ...``.

Dry run unless ``--dispatch`` is given. Nothing here claims a qualified result.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from matsim_agents.cli._app import app, console

rd_app = typer.Typer(
    add_completion=False,
    help="Vacancy-hop recovery workflow for seeded BCC alloys (exploratory, unqualified).",
)
app.add_typer(rd_app, name="radiation-damage")

CONFIG_ARGUMENT = typer.Argument(..., exists=True, dir_okay=False, help="Run YAML config.")
DIRECTORY_ARGUMENT = typer.Argument(..., exists=True, file_okay=False, help="Run directory.")


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error[/red]: {message}")
    return typer.Exit(code=1)


def _load_config(path: Path):
    import yaml
    from pydantic import ValidationError

    from matsim_agents.workflows.radiation_damage import RadiationDamageConfig

    try:
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("the config must be a mapping")
        return RadiationDamageConfig(**data)
    except (ValidationError, TypeError, ValueError, OSError) as exc:
        raise _fail(f"{path}: {exc}") from None


@rd_app.command("run")
def run(
    config: Path = CONFIG_ARGUMENT,
    out: Path = typer.Option(..., "--out", help="Run directory (new, or existing with --resume)."),
    dispatch: bool = typer.Option(
        False, "--dispatch", help="Actually run neb.x. Default is a dry run."
    ),
    resume: bool = typer.Option(False, "--resume", help="Continue the run in --out."),
    allow_unrelaxed_endpoints: bool = typer.Option(
        False, "--allow-unrelaxed-endpoints", help="Permit dispatch with generated endpoints."
    ),
    ledger: Path | None = typer.Option(None, "--ledger", help="Evidence ledger to append to."),
):
    """Run or resume the staged workflow and write its report."""

    from matsim_agents.execution.ledger import EvidenceLedger
    from matsim_agents.workflows.radiation_damage import (
        RadiationDamageError,
        run_radiation_damage,
    )
    from matsim_agents.workflows.radiation_report import write_report

    cfg = _load_config(config)
    try:
        result = run_radiation_damage(
            cfg,
            out,
            dispatch=dispatch,
            resume=resume,
            allow_unrelaxed_endpoints=allow_unrelaxed_endpoints,
            ledger=EvidenceLedger(ledger) if ledger else None,
        )
    except RadiationDamageError as exc:
        raise _fail(str(exc)) from None
    path = write_report(result)
    mode = "dry run" if result.state.dry_run else "dispatched"
    console.print(f"[green]{mode}[/green]: {result.state.transition_id}")
    for stage in result.state.stages:
        console.print(f"  {stage.name}: {stage.status} {stage.detail}".rstrip())
    for blocker in result.blockers:
        console.print(f"[yellow]blocker[/yellow]: {blocker}")
    console.print(f"report: {path}")


@rd_app.command("status")
def status(directory: Path = DIRECTORY_ARGUMENT):
    """Print the stage status of a run directory."""

    from matsim_agents.workflows.radiation_damage import RadiationDamageError, read_state

    try:
        state = read_state(directory)
    except RadiationDamageError as exc:
        raise _fail(str(exc)) from None
    console.print(f"{'dry run' if state.dry_run else 'dispatched'}: {state.transition_id}")
    for stage in state.stages:
        console.print(f"  {stage.name}: {stage.status} {stage.detail}".rstrip())


@rd_app.command("report")
def report(
    directory: Path = DIRECTORY_ARGUMENT,
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
):
    """Print the saved report of a run directory."""

    from matsim_agents.workflows.radiation_report import read_report

    try:
        data = read_report(directory)
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from None
    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return
    console.print(f"{data['mode']}: {data['transition_id']} ({data['qualification']})")
    kinetics = data["kinetics"]
    if kinetics is None:
        console.print("kinetics: not run")
    else:
        console.print(
            f"kinetics (exploratory): {kinetics['n_trajectories']} trajectories at "
            f"{kinetics['temperature_K']:g} K, censored {kinetics['censored_fraction']:.3f}"
        )
        for kind, outcome in kinetics["outcomes"].items():
            console.print(
                f"  {kind}: {outcome['probability']:.3f} "
                f"[{outcome['wilson_low']:.3f}, {outcome['wilson_high']:.3f}]"
            )
    for blocker in data["blockers"]:
        console.print(f"[yellow]blocker[/yellow]: {blocker}")
    for caveat in data["caveats"]:
        console.print(f"caveat: {caveat}")


@rd_app.command("capabilities")
def capabilities(
    as_json: bool = typer.Option(False, "--json", help="Print as JSON."),
):
    """Software and runtime readiness of each capability; none is scientifically qualified."""

    from matsim_agents.workflows.radiation_capabilities import evaluate_all

    items = evaluate_all()
    if as_json:
        typer.echo(json.dumps([i.model_dump(mode="json") for i in items], indent=2))
        return
    for item in items:
        console.print(f"{item.name}: {item.software} / {item.runtime} / {item.qualification}")
