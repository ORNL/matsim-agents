"""Regression checks for claims made by repository documentation."""

from __future__ import annotations

from scripts.diagnostics.validate_documentation import collect_violations


def test_documentation_matches_repository() -> None:
    violations = collect_violations()
    assert not violations, "Documentation contract violations:\n" + "\n".join(violations)
