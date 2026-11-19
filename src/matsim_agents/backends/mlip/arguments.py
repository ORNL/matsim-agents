"""Shared argparse option that selects the MLIP through an ``mlip`` YAML block."""

from __future__ import annotations

import argparse
from pathlib import Path

from matsim_agents.backends.mlip.config import MLIPConfig, load_mlip_config


def add_mlip_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--mlip-config`` (an mlip block: backend plus its own settings)."""
    parser.add_argument(
        "--mlip-config",
        type=Path,
        help="YAML file with an mlip block (backend: hydragnn | uma | mace and its settings).",
    )


def mlip_config_from_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> MLIPConfig | None:
    """Load ``--mlip-config``, or exit with a usage error if it is invalid."""
    if args.mlip_config is None:
        return None
    try:
        return load_mlip_config(args.mlip_config)
    except (OSError, ValueError) as error:
        parser.error(f"--mlip-config: {error}")
