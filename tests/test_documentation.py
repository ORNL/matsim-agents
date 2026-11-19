"""Regression checks for claims made by repository documentation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

VALIDATOR = (
    Path(__file__).resolve().parents[1] / "scripts" / "diagnostics" / "validate_documentation.py"
)


def _collect_violations() -> list[str]:
    # Load by path: the top-level name ``scripts`` is also claimed by installed
    # packages (mace-torch ships one), which would shadow the repository folder.
    spec = importlib.util.spec_from_file_location("validate_documentation", VALIDATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collect_violations()


def test_documentation_matches_repository() -> None:
    violations = _collect_violations()
    assert not violations, "Documentation contract violations:\n" + "\n".join(violations)
