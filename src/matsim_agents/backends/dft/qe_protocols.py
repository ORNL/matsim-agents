"""Numerical protocol records and the rule that decides when one is qualified."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matsim_agents.backends.dft.qe_convergence import AcceptanceCriteria
from matsim_agents.core.yamlio import load_yaml_mapping

SOFTWARE_ONLY = "software verification only"


@dataclass(frozen=True)
class Protocol:
    name: str
    domain: str
    case_kinds: tuple[str, ...]
    status: str
    settings: Mapping[str, Any]


def load_protocols(path: str | Path) -> tuple[dict[str, Protocol], AcceptanceCriteria]:
    data = load_yaml_mapping(path)
    protocols = {
        name: Protocol(
            name=name,
            domain=entry["domain"],
            case_kinds=tuple(entry["case_kinds"]),
            status=entry["status"],
            settings=dict(entry["settings"]),
        )
        for name, entry in data["protocols"].items()
    }
    return protocols, AcceptanceCriteria.from_mapping(data.get("criteria_default"))


def qualified_by(
    protocol: Protocol, report: Mapping[str, Any], case_kinds_covered: set[str]
) -> tuple[bool, list[str]]:
    """Whether ``report`` qualifies ``protocol``; the reasons when it does not."""
    reasons: list[str] = []
    if report.get("status") != "qualified":
        reasons.append(f"report status is {report.get('status')!r}")
    if report.get("label"):
        reasons.append(f"report is labelled {report['label']!r}, not production data")
    missing = sorted(set(protocol.case_kinds) - case_kinds_covered)
    if missing:
        reasons.append(f"cell kinds not covered: {missing}")
    return not reasons, reasons


__all__ = ["Protocol", "SOFTWARE_ONLY", "load_protocols", "qualified_by"]
