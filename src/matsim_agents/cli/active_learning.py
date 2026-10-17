"""Active-learning subcommand group: ``matsim-agents al ...``."""

from __future__ import annotations

from pathlib import Path

import typer

from matsim_agents.cli._app import app, console

al_app = typer.Typer(
    add_completion=False,
    help="HydraGNN <-> DFT (VASP or Quantum ESPRESSO) active-learning loop.",
)
app.add_typer(al_app, name="al")


@al_app.command("run")
def al_run(
    config: Path = typer.Argument(..., help="Path to AL YAML config (see ALConfig schema)."),
    log_level: str = typer.Option("INFO", help="Python logging level."),
):
    """Run the active-learning loop end-to-end."""
    import logging

    from matsim_agents.active_learning import ALConfig
    from matsim_agents.active_learning.loop import run_active_learning

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    cfg = ALConfig.from_yaml(config)
    run_active_learning(cfg)


@al_app.command("validate-config")
def al_validate_config(config: Path = typer.Argument(...)):
    """Validate an AL YAML config without running anything."""
    from matsim_agents.active_learning import ALConfig

    cfg = ALConfig.from_yaml(config)
    console.print_json(cfg.model_dump_json(indent=2))
