"""Read YAML configuration files as mappings with one error contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def parse_yaml_mapping(text: str, source: str | Path) -> dict[str, Any]:
    """Parse YAML text that must be a top-level mapping; failures are ``ValueError``."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(f"{source}: invalid YAML: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{source}: expected a mapping at the top level")
    return data


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Read a YAML or JSON file that must hold a top-level mapping."""
    path = Path(path)
    return parse_yaml_mapping(path.read_text(encoding="utf-8"), path)
