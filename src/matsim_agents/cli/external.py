"""External-application subcommands: ``matsim-agents external spectra-pka ...``.

SPECTRA-PKA is optional, separately installed GPL-3.0 software. These commands only drive an
executable the user configured; ``run`` is a dry run unless ``--execute`` is given.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from matsim_agents.cli._app import app, console

external_app = typer.Typer(
    add_completion=False,
    help="Optional interfaces to externally installed scientific software.",
)
spka_app = typer.Typer(
    add_completion=False,
    help=(
        "Optional interface to externally installed SPECTRA-PKA. MATSIM-AGENTS does not "
        "implement its nuclear physics, bundle it, or download it."
    ),
)
external_app.add_typer(spka_app, name="spectra-pka")
app.add_typer(external_app, name="external")

CONFIG_OPTION = typer.Option(
    ..., "--config", "-c", exists=True, dir_okay=False, help="YAML config."
)


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error[/red]: {message}")
    return typer.Exit(code=1)


def _load(config: Path):
    from matsim_agents.backends.external.spectra_pka.config import (
        ConfigError,
        load_config_file,
    )

    try:
        return load_config_file(config)
    except ConfigError as exc:
        raise _fail(str(exc)) from None


@spka_app.command("check")
def check(
    config: Path = CONFIG_OPTION,
    no_probe: bool = typer.Option(False, "--no-probe", help="Do not start the executable."),
) -> None:
    """Report installation, data and readiness. READY is tool readiness, not qualification."""
    from matsim_agents.backends.external.spectra_pka.availability import check_availability

    parsed = _load(config)
    result = check_availability(
        parsed.external_tools.spectra_pka, parsed.campaign, invoke=not no_probe
    )
    console.print_json(json.dumps(result.as_dict()))


@spka_app.command("prepare")
def prepare(config: Path = CONFIG_OPTION) -> None:
    """Stage the native input file and links to the user's data; execute nothing."""
    from matsim_agents.backends.external.spectra_pka import adapter

    parsed = _load(config)
    if parsed.campaign is None:
        raise _fail("the config has no 'campaign' section")
    try:
        result = adapter.run(parsed.external_tools.spectra_pka, parsed.campaign, execute=False)
    except adapter.AdapterError as exc:
        raise _fail(str(exc)) from None
    console.print(f"staged in {result.run_dir}\ncommand: {' '.join(result.command)}")


@spka_app.command("run")
def run(
    config: Path = CONFIG_OPTION,
    execute: bool = typer.Option(
        False, "--execute", help="Actually run the executable. Without it this is a dry run."
    ),
) -> None:
    """Dry run by default. With --execute: run, parse and validate the output."""
    from matsim_agents.backends.external.spectra_pka import adapter

    parsed = _load(config)
    if parsed.campaign is None:
        raise _fail("the config has no 'campaign' section")
    try:
        result = adapter.run(parsed.external_tools.spectra_pka, parsed.campaign, execute=execute)
    except adapter.AdapterError as exc:
        raise _fail(str(exc)) from None
    console.print(
        json.dumps(
            {
                "status": result.status,
                "run_dir": str(result.run_dir),
                "command": list(result.command),
                "message": result.message,
                "cached": result.cached,
                "numerical_validation": "NOT_VALIDATED",
            },
            indent=2,
        )
    )
    if result.status in {"EXECUTION_FAILED", "PARSE_FAILED", "STRUCTURALLY_INVALID"}:
        raise typer.Exit(code=1)


@spka_app.command(
    "fetch-data",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Extract needed PKA matrices from pkas.zip. Arguments: see data_fetch --help.",
)
def fetch_data(ctx: typer.Context) -> None:
    from matsim_agents.backends.external.spectra_pka import data_fetch

    code = data_fetch.main(list(ctx.args))
    if code:
        raise typer.Exit(code=code)


@spka_app.command("status")
def status(config: Path = CONFIG_OPTION) -> None:
    """Show the state of a staged or finished run."""
    from matsim_agents.backends.external.spectra_pka import adapter

    parsed = _load(config)
    if parsed.campaign is None:
        raise _fail("the config has no 'campaign' section")
    console.print_json(
        json.dumps(adapter.status(parsed.external_tools.spectra_pka, parsed.campaign))
    )


@spka_app.command("import")
def import_results(config: Path = CONFIG_OPTION) -> None:
    """Parse and validate existing raw output in the run directory (no execution)."""
    from matsim_agents.backends.external.spectra_pka import adapter

    parsed = _load(config)
    if parsed.campaign is None:
        raise _fail("the config has no 'campaign' section")
    cfg = parsed.external_tools.spectra_pka
    result = adapter.collect(cfg, parsed.campaign, adapter.run_directory(cfg, parsed.campaign))
    report = result.validation.as_dict() if result.validation else None
    console.print(
        json.dumps(
            {
                "status": result.status,
                "message": result.message,
                "validation": report,
                "species_probabilities": (
                    result.spectrum.species_probabilities() if result.spectrum else None
                ),
                "numerical_validation": "NOT_VALIDATED",
            },
            indent=2,
        )
    )
    if result.spectrum is None:
        raise typer.Exit(code=1)
