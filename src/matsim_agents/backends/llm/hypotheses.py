"""Shared scientific hypothesis schema and evidence-scoped revision validation."""

from __future__ import annotations

from typing import Any

VERSION = "scientific-hypotheses-v1"
FIELDS = {"id", "claim", "prediction", "status", "reason", "evidence_ids"}
STATUSES = {
    "proposed",
    "unresolved",
    "refined",
    "supported_by_mlip",
    "challenged_by_mlip",
    "supported_by_dft",
    "challenged_by_dft",
    "retired",
}

INSTRUCTIONS = (
    "Include scientific_hypotheses: a non-empty list of objects with exactly id, claim, "
    "prediction, status, reason, evidence_ids. Use stable IDs across rounds and include "
    "every previous hypothesis, marking retired ones retired. State falsifiable material "
    "claims and predictions separately from computational convergence or cost. Status "
    "must be proposed, unresolved, refined, supported_by_mlip, challenged_by_mlip, "
    "supported_by_dft, challenged_by_dft, or retired. Explain whether and why each claim "
    "changed, cite only supplied evidence IDs, and explicitly retain uncertainty. Initial "
    "claims are proposals, not established facts. MLIP predictions and optimization "
    "convergence do not establish thermodynamic stability or DFT accuracy. DFT support "
    "is limited to the measured observable and declared method and reference scope. "
    "Peer opinions are not numerical evidence. Do not invent observations or comparisons."
)


def validate_hypotheses(
    value: Any,
    valid_evidence: set[str],
    previous: list[dict[str, Any]],
    *,
    mlip_evidence: set[str] | None = None,
    dft_evidence: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("scientific_hypotheses must be a non-empty list")
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("scientific hypothesis must contain exactly the required fields")
        for field in ("id", "claim", "prediction", "status", "reason"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"scientific hypothesis {field} must be a non-empty string")
        if row["id"] in seen:
            raise ValueError("scientific hypothesis IDs must be unique")
        seen.add(row["id"])
        if row["status"] not in STATUSES:
            raise ValueError("invalid scientific hypothesis status")
        citations = row["evidence_ids"]
        if (
            not isinstance(citations, list)
            or not citations
            or not all(isinstance(item, str) for item in citations)
            or not set(citations) <= valid_evidence
        ):
            raise ValueError("scientific hypothesis must cite supplied evidence IDs")
        if row["status"] in {"supported_by_mlip", "challenged_by_mlip"} and not (
            set(citations) & (mlip_evidence or set())
        ):
            raise ValueError("MLIP support or challenge requires outcome evidence")
        if row["status"] in {"supported_by_dft", "challenged_by_dft"} and not (
            set(citations) & (dft_evidence or set())
        ):
            raise ValueError("DFT support or challenge requires measured DFT evidence")
    missing = {row["id"] for row in previous} - seen
    if missing:
        raise ValueError(f"scientific hypotheses omit previous IDs: {sorted(missing)}")
    return value
