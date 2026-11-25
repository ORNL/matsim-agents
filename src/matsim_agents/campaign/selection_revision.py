"""Connect deterministic selection scores to the hypothesis-revision ledger.

The status of a revision is decided by rules from the :class:`SelectionResult`; a language
model may draft rationale text only. Its draft is accepted only if every number in it is
present in the evidence packet and every evidence identifier it cites was supplied, so it
cannot invent evidence or change a score. A rejected draft falls back to the deterministic
text, and the rejection is recorded.

Status rules for the claim "candidate X meets the stated requirements":

``SUPPORTED``
    X is ``QUALIFIED`` and nondominated, on measured or calculated evidence;
``CONTRADICTED``
    X is ``INFEASIBLE`` because a *measured or calculated* value violates a constraint;
``UNRESOLVED``
    anything else (missing, surrogate-only, unqualified, dominated, or marginal evidence).

``SUPPORTED`` is computational support for the declared requirements. It is not scientific
qualification, and the ledger still enforces that the cited evidence is measured.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from matsim_agents.campaign.selection import (
    CandidateEvidence,
    CandidateScore,
    SelectionResult,
    SelectionStatus,
)
from matsim_agents.core.revision import (
    MAX_RATIONALE_CHARS,
    RevisionBudget,
    RevisionBudgetError,
)
from matsim_agents.execution.ledger import (
    EvidenceLedger,
    EvidenceRecord,
    HypothesisRecord,
    HypothesisStatus,
)
from matsim_agents.execution.properties import (
    PropertyEvidence,
    PropertyStatus,
    to_ledger_record,
)

PACKET_VERSION = "selection-packet-v1"
RationaleDrafter = Callable[[str], str]
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?(?![\w])")
_EVIDENCE_ID = re.compile(r"\b(?:ev|pe|sel|hr|dc)-[0-9a-f]{12,}\b")
_SMALL_INTEGER_LIMIT = 10
_TOLERANCE = 5e-3
_RELIABLE = {PropertyStatus.MEASURED, PropertyStatus.CALCULATED}


class PacketValue(BaseModel):
    property: str
    value: float
    uncertainty: float
    unit: str
    status: str
    evidence_level: str
    qualification: str
    property_id: str
    ledger_evidence_id: str | None


class PacketCandidate(BaseModel):
    candidate_id: str
    status: str
    reasons: list[str]
    values: list[PacketValue]
    missing: dict[str, str]
    violated: list[str]
    pareto: str
    dominated_by: list[str]
    best_in: list[str]
    worst_in: list[str]
    constraint_probability: float | None
    pareto_robustness: float | None
    desirability: float | None


class EvidencePacket(BaseModel):
    """Everything a drafter may mention: the scores, the evidence, and the numbers."""

    packet_version: str = PACKET_VERSION
    scoring_version: str
    score_digest: str
    constraints: list[dict]
    candidates: list[PacketCandidate]
    pareto_front: list[str]
    limitations: list[str]
    allowed_evidence_ids: list[str]
    numbers: list[float]
    digest: str = ""

    def canonical_text(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def append_property_evidence(
    ledger: EvidenceLedger, evidence: PropertyEvidence, *, campaign_id: str | None = None
) -> str:
    """Append one property record and return its ledger evidence id.

    Parent ids that are not ledger records (for example an external source label) are kept
    as a stated assumption instead of a parent link.
    """
    record = to_ledger_record(evidence, campaign_id=campaign_id)
    unknown = [i for i in record.parent_record_ids if i not in ledger]
    if unknown:
        payload = record.model_dump(exclude={"evidence_id"})
        payload["parent_record_ids"] = [i for i in record.parent_record_ids if i in ledger]
        payload["physical_assumptions"] = [
            *record.physical_assumptions,
            f"source identifiers outside this ledger: {sorted(unknown)}",
        ]
        record = EvidenceRecord.model_validate(payload)
    return ledger.append(record).evidence_id  # type: ignore[union-attr]


def record_candidate_evidence(
    ledger: EvidenceLedger,
    candidates: Sequence[CandidateEvidence],
    result: SelectionResult,
    *,
    campaign_id: str | None = None,
) -> dict[str, str]:
    """Append the evidence the scores used; returns ``property_id -> ledger evidence id``."""
    used = {v.property_id for c in result.candidates for v in c.values.values()}
    mapping: dict[str, str] = {}
    for cand in candidates:
        for ev in cand.evidence:
            pid = ev.property_id()
            if pid not in used or pid in mapping:
                continue
            mapping[pid] = append_property_evidence(ledger, ev, campaign_id=campaign_id)
    return mapping


def _numbers(candidate: CandidateScore, packet_values: list[PacketValue]) -> list[float]:
    out: list[float] = []
    for v in packet_values:
        out.extend([v.value, v.uncertainty])
    for x in (
        candidate.constraint_probability,
        candidate.pareto_robustness,
        candidate.desirability,
    ):
        if x is not None:
            out.append(x)
    return out


def build_evidence_packet(
    result: SelectionResult, ledger_ids: dict[str, str] | None = None
) -> EvidencePacket:
    ledger_ids = ledger_ids or {}
    packet_candidates: list[PacketCandidate] = []
    numbers: list[float] = []
    allowed: set[str] = {result.score_digest}
    for c in result.candidates:
        values = [
            PacketValue(
                property=v.property,
                value=v.value,
                uncertainty=v.uncertainty,
                unit=_unit(v.property),
                status=v.status.value,
                evidence_level=v.evidence_level.value,
                qualification=v.qualification.value,
                property_id=v.property_id,
                ledger_evidence_id=ledger_ids.get(v.property_id),
            )
            for v in c.values.values()
        ]
        allowed.update(v.property_id for v in values)
        allowed.update(v.ledger_evidence_id for v in values if v.ledger_evidence_id)
        numbers.extend(_numbers(c, values))
        packet_candidates.append(
            PacketCandidate(
                candidate_id=c.candidate_id,
                status=c.status.value,
                reasons=c.reasons,
                values=values,
                missing=c.missing,
                violated=c.violated,
                pareto=c.pareto,
                dominated_by=c.dominated_by,
                best_in=c.best_in,
                worst_in=c.worst_in,
                constraint_probability=c.constraint_probability,
                pareto_robustness=c.pareto_robustness,
                desirability=c.desirability,
            )
        )
    constraints = [x.model_dump(mode="json") for x in result.spec.constraints]
    for x in constraints:
        numbers.extend(v for k, v in x.items() if k != "property" and isinstance(v, int | float))
    numbers.extend(
        x
        for x in (result.spec.min_constraint_confidence, float(result.spec.n_samples))
        if x is not None
    )
    packet = EvidencePacket(
        scoring_version=result.scoring_version,
        score_digest=result.score_digest,
        constraints=constraints,
        candidates=packet_candidates,
        pareto_front=result.pareto_front,
        limitations=result.limitations,
        allowed_evidence_ids=sorted(allowed),
        numbers=sorted(set(numbers)),
    )
    packet.digest = "pk-" + hashlib.sha256(packet.canonical_text().encode()).hexdigest()[:24]
    return packet


def _unit(prop: str) -> str:
    from matsim_agents.core.contracts import PROPERTY_UNITS

    return PROPERTY_UNITS[prop]


def validate_draft(draft: str, packet: EvidencePacket) -> list[str]:
    """Problems that make an LLM draft unacceptable; empty means it only restates the packet."""
    problems: list[str] = []
    cited = set(_EVIDENCE_ID.findall(draft))
    unknown_ids = sorted(cited - set(packet.allowed_evidence_ids))
    if unknown_ids:
        problems.append(f"cites evidence not in the packet: {unknown_ids}")
    text = _EVIDENCE_ID.sub(" ", draft)
    for token in _NUMBER.findall(text):
        x = float(token)
        if x == int(x) and abs(x) <= _SMALL_INTEGER_LIMIT:
            continue
        # a value may be quoted rounded to fewer significant figures than the packet holds
        if not any(
            abs(x - n) <= _TOLERANCE * max(abs(n), 1e-12) or _rounds_to(n, token)
            for n in packet.numbers
        ):
            problems.append(f"number {token} is not in the evidence packet")
    lowered = draft.lower()
    values = [v for c in packet.candidates for v in c.values]
    if "scientifically qualified" in lowered and not any(
        v.qualification == "scientifically_qualified" for v in values
    ):
        problems.append("claims scientific qualification that no cited value has")
    if "experimentally" in lowered and not any(v.status == "measured" for v in values):
        problems.append("claims experimental support but no measured value is in the packet")
    return problems


def _rounds_to(n: float, token: str) -> bool:
    digits = len(token.lstrip("+-").replace(".", "").lstrip("0").split("e")[0]) or 1
    return float(f"{n:.{digits}g}") == float(token)


@dataclass(frozen=True)
class SelectionRevisionOutcome:
    record: HypothesisRecord
    packet: EvidencePacket
    used_llm: bool
    notes: tuple[str, ...]


def _status_for(c: CandidateScore) -> tuple[HypothesisStatus, str]:
    reliable = all(v.reliable for v in c.values.values())
    if c.status == SelectionStatus.QUALIFIED and c.pareto == "nondominated" and reliable:
        return HypothesisStatus.SUPPORTED, "qualified, nondominated, measured or calculated"
    if c.status == SelectionStatus.INFEASIBLE and c.violated:
        violating = [v for p, v in c.values.items() if any(p in text for text in c.violated)]
        if violating and all(v.reliable for v in violating):
            return HypothesisStatus.CONTRADICTED, "a measured or calculated value violates a limit"
        return HypothesisStatus.UNRESOLVED, "the violating value is not measured or calculated"
    if c.status == SelectionStatus.QUALIFIED and c.pareto == "dominated":
        return HypothesisStatus.UNRESOLVED, f"dominated by {', '.join(c.dominated_by)}"
    return HypothesisStatus.UNRESOLVED, f"selection status {c.status.value}"


def _next_tests(c: CandidateScore) -> list[str]:
    tests = [f"calculate {p}: {why}" for p, why in sorted(c.missing.items())]
    tests += [
        f"replace the {v.status.value} value of {p} with a measured or converged-DFT value"
        for p, v in sorted(c.values.items())
        if not v.reliable
    ]
    tests += [
        f"qualify {p} (currently {v.qualification.value})"
        for p, v in sorted(c.values.items())
        if v.reliable and v.qualification.value == "unvalidated"
    ]
    return tests or ["experimental confirmation of the computed properties"]


def _deterministic_rationale(c: CandidateScore, reason: str) -> str:
    parts = [f"status={c.status.value}; {reason}."]
    for p, v in sorted(c.values.items()):
        parts.append(f"{p}={v.value:.6g}+/-{v.uncertainty:.3g} ({v.status.value})")
    if c.violated:
        parts.append("violated: " + "; ".join(c.violated))
    if c.missing:
        parts.append("missing: " + "; ".join(f"{p} ({w})" for p, w in sorted(c.missing.items())))
    if c.constraint_probability is not None:
        parts.append(f"constraint probability {c.constraint_probability:.3f}")
    if c.pareto_robustness is not None:
        parts.append(f"Pareto robustness {c.pareto_robustness:.3f}")
    return "Deterministic summary: " + " | ".join(parts)


def revise_selection_hypothesis(
    ledger: EvidenceLedger,
    hypothesis_id: str,
    candidate_id: str,
    result: SelectionResult,
    candidates: Sequence[CandidateEvidence],
    *,
    budget: RevisionBudget | None = None,
    drafter: RationaleDrafter | None = None,
    agent_model_id: str | None = None,
    campaign_id: str | None = None,
) -> SelectionRevisionOutcome:
    """Append one revision of the claim that ``candidate_id`` meets the requirements."""
    budget = budget or RevisionBudget()
    history = ledger.history(hypothesis_id)
    if len(history) >= budget.max_revisions:
        raise RevisionBudgetError(
            f"{hypothesis_id} already has {len(history)} of {budget.max_revisions} revisions"
        )
    c = result.by_id(candidate_id)
    ledger_ids = record_candidate_evidence(ledger, candidates, result, campaign_id=campaign_id)
    packet = build_evidence_packet(result, ledger_ids)
    status, reason = _status_for(c)
    cited = [
        ledger_ids[v.property_id]
        for v in c.values.values()
        if v.property_id in ledger_ids and v.status in _RELIABLE
    ]
    if status in {HypothesisStatus.SUPPORTED, HypothesisStatus.CONTRADICTED} and not cited:
        status = HypothesisStatus.UNRESOLVED

    notes: list[str] = []
    rationale = _deterministic_rationale(c, reason)
    llm = {"requested": drafter is not None, "calls": 0, "accepted": False, "rejected": []}
    if drafter is not None:
        if budget.llm_calls_left == 0:
            notes.append("LLM budget exhausted; deterministic rationale used")
        else:
            budget.llm_calls_used += 1
            llm["calls"] = 1
            try:
                draft = drafter(packet.canonical_text()).strip()
            except Exception as exc:  # a drafter failure must not block the record
                notes.append(f"drafter failed ({type(exc).__name__}); deterministic")
            else:
                problems = validate_draft(draft, packet) if draft else ["empty draft"]
                if problems:
                    llm["rejected"] = problems
                    notes.append("LLM draft rejected: " + "; ".join(problems))
                else:
                    llm["accepted"] = True
                    rationale += f"\n[LLM-drafted, numbers verified] {draft[:MAX_RATIONALE_CHARS]}"

    traceability = {
        "scoring_version": result.scoring_version,
        "score_digest": result.score_digest,
        "packet_version": packet.packet_version,
        "packet_digest": packet.digest,
        "candidate_id": candidate_id,
        "selection_status": c.status.value,
        "status_rule": reason,
        "property_ids": {v.property: v.property_id for v in c.values.values()},
        "llm": llm,
    }
    record = HypothesisRecord(
        hypothesis_id=hypothesis_id,
        prior_revision_id=history[-1].revision_id if history else None,
        campaign_id=campaign_id,
        claim_type="property",
        scientific_claim=(
            f"Candidate {candidate_id} meets the stated thermal, electrical, mechanical, and "
            "radiation requirements."
        ),
        falsifiable_prediction=(
            "A higher-fidelity or measured value of any scored property moves the candidate "
            "across a stated limit or off the Pareto front."
        ),
        status=status,
        evidence_ids=cited,
        support_level=ledger.support_level(cited),
        scientific_rationale=rationale,
        proposed_tests=_next_tests(c),
        agent_model_id=agent_model_id if llm["accepted"] else None,
        traceability=traceability,
    )
    stored = ledger.append(record)
    assert isinstance(stored, HypothesisRecord)
    return SelectionRevisionOutcome(
        record=stored, packet=packet, used_llm=bool(llm["accepted"]), notes=tuple(notes)
    )


__all__ = [
    "PACKET_VERSION",
    "EvidencePacket",
    "RationaleDrafter",
    "SelectionRevisionOutcome",
    "append_property_evidence",
    "build_evidence_packet",
    "record_candidate_evidence",
    "revise_selection_hypothesis",
    "validate_draft",
]
