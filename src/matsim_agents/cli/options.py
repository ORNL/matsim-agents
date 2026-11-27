"""Reusable Typer option factories shared by several commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from matsim_agents.backends.mlip.config import MLIPConfig


def opt_mlip_config():
    return typer.Option(
        None,
        "--mlip-config",
        help=(
            "YAML file whose `mlip:` block selects the model (backend: hydragnn | uma | "
            "mace plus that backend's settings), e.g. examples/mlip/*.yaml or an "
            "active-learning config."
        ),
    )


def load_mlip_option(path: Path | None) -> MLIPConfig | None:
    """Load --mlip-config, reporting an invalid file as a usage error."""
    if path is None:
        return None
    from matsim_agents.backends.mlip.config import load_mlip_config

    try:
        return load_mlip_config(path)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="--mlip-config") from error


def opt_active_learning_config(help_text: str):
    return typer.Option(None, "--al-config", help=help_text)


def opt_active_learning_dry_run(help_text: str):
    return typer.Option(True, "--al-dry-run/--al-run", help=help_text)


def path_or_none(value: Path | None) -> str | None:
    return str(value) if value else None
