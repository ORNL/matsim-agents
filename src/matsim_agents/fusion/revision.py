"""LLM-assisted hypothesis revision for fusion PFC candidates (F7).

The deterministic score decides the revision status and the evidence lists. A
language model may only draft a structured rationale (:class:`HypothesisDraft`).
A draft is rejected if it carries extra fields, cites evidence outside the
packet, quotes a number the packet does not contain, or asks to change scores,
weights or constraints. A rejected draft falls back to deterministic text and
the rejection is recorded in the revision's traceability.

Status rules for the claim "candidate X is the preferred PFC material under
policy P in scenario S":

``SUPPORTED``
    X is rank 1, ``complete_qualified``, on an authoritative Pareto front, and
    every dominance it holds over the others is ``established``;
``CONTRADICTED``
    X is ``complete_qualified`` and another candidate dominates it with
    ``established`` dominance;
``UNRESOLVED``
    anything else.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from matsim_agents.campaign.selection_revision import (
    _NUMBER,
    _SMALL_INTEGER_LIMIT,
    _TOLERANCE,
    _rounds_to,
    append_property_evidence,
)
from matsim_agents.execution.ledger import EvidenceLedger, HypothesisRecord, HypothesisStatus
from matsim_agents.fusion.objectives import objective
from matsim_agents.fusion.policy import FusionScoringPolicy
from matsim_agents.fusion.scenario import FusionEvidence
from matsim_agents.fusion.scoring import (
    CandidateResult,
    Dominance,
    RankingStatus,
    ScoringResult,
)
from matsim_agents.workflows.radiation_hypothesis import (
    MAX_RATIONALE_CHARS,
    RevisionBudget,
    RevisionBudgetError,
)

PACKET_VERSION = "fusion-pfc-packet-v1"
Drafter = Callable[[str], str]
_EVIDENCE_ID = re.compile(r"\b(?:ev|pe|fsc|fpk|nuc|hr)-[0-9a-f]{12,}\b")
_ALTERATION = re.compile(
    r"\b(?:re-?weight\w*|(?:increase|decrease|raise|lower|change|adjust|alter|set)\s+the\s+"
    r"(?:\w+\s+)?(?:weight|score|constraint|limit|benchmark)s?|relax\w*\s+the\s+"
    r"(?:\w+\s+)?(?:constraint|limit)s?|override\w*)\b",
    re.IGNORECASE,
)


class PacketObservable(BaseModel):
    observable: str
    value: float | None
    uncertainty: float | None
    desirability: float | None
    status: str | None
    qualification: str | None
    property_id: str | None
    ledger_evidence_id: str | None
    reason: str = ""


class PacketObjective(BaseModel):
    objective: str
    weight: float
    desirability: float | None
    d_low: float | None
    d_high: float | None
    observables: list[PacketObservable]
    reason: str = ""


class PacketCandidate(BaseModel):
    candidate_id: str
    ranking_status: str
    feasibility: str
    score: float | None
    score_low: float | None
    score_high: float | None
    partial_weighted_sum: float
    covered_weight: float
    rank: int | None
    pareto: bool | None
    missing_objectives: list[str]
    objectives: list[PacketObjective]


class FusionEvidencePacket(BaseModel):
    """Everything the drafter may see and quote; nothing else is admissible."""

    packet_version: str = PACKET_VERSION
    policy_version: str
    policy_digest: str
    weights: dict[str, float]
    benchmark_version: str
    benchmark_digest: str
    synthetic_benchmarks: bool
    scenario_id: str
    score_digest: str
    candidates: list[PacketCandidate]
    ranking: list[str]
    pareto_front: list[str]
    pareto_authoritative: bool
    dominance: list[dict[str, str]]
    warnings: list[str]
    notes: list[str]
    allowed_evidence_ids: list[str]
    numbers: list[float]
    digest: str = ""

    def canonical_text(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


class HypothesisDraft(BaseModel):
    """The only structure a drafter may return; extra fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    rationale: str = Field(min_length=1, max_length=MAX_RATIONALE_CHARS)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    recommended_calculation: str = ""


def build_packet(
    result: ScoringResult,
    policy: FusionScoringPolicy,
    ledger_ids: Mapping[str, str] | None = None,
) -> FusionEvidencePacket:
    """Evidence packet for a scoring result; ``ledger_ids`` maps property id to ledger id."""
    if policy.policy_version != result.policy_version or policy.digest() != result.policy_digest:
        raise ValueError("policy does not match the one used for scoring")
    ledger_ids = ledger_ids or {}
    score_digest = result.digest()
    allowed: set[str] = {score_digest}
    numbers: list[float] = list(policy.weights.values())
    cands = []
    for c in result.candidates:
        objs = []
        for o in c.objectives:
            obs = [
                PacketObservable(
                    observable=x.observable,
                    value=x.value,
                    uncertainty=x.uncertainty,
                    desirability=x.desirability,
                    status=x.status.value if x.status else None,
                    qualification=x.qualification.value if x.qualification else None,
                    property_id=x.property_id,
                    ledger_evidence_id=ledger_ids.get(x.property_id or ""),
                    reason=x.reason,
                )
                for x in o.observables
            ]
            for x in obs:
                allowed.update(i for i in (x.property_id, x.ledger_evidence_id) if i)
                numbers += [v for v in (x.value, x.uncertainty, x.desirability) if v is not None]
            numbers += [v for v in (o.desirability, o.d_low, o.d_high) if v is not None]
            objs.append(
                PacketObjective(
                    objective=o.objective,
                    weight=o.weight,
                    desirability=o.desirability,
                    d_low=o.d_low,
                    d_high=o.d_high,
                    observables=obs,
                    reason=o.reason,
                )
            )
        numbers += [
            v
            for v in (c.score, c.score_low, c.score_high, c.partial_weighted_sum, c.covered_weight)
            if v is not None
        ]
        numbers += [cn.value for cn in c.constraints if cn.value is not None]
        cands.append(
            PacketCandidate(
                candidate_id=c.candidate_id,
                ranking_status=c.ranking_status.value,
                feasibility=c.feasibility.value,
                score=c.score,
                score_low=c.score_low,
                score_high=c.score_high,
                partial_weighted_sum=c.partial_weighted_sum,
                covered_weight=c.covered_weight,
                rank=c.rank,
                pareto=c.pareto,
                missing_objectives=list(c.missing_objectives),
                objectives=objs,
            )
        )
    packet = FusionEvidencePacket(
        policy_version=policy.policy_version,
        policy_digest=policy.digest(),
        weights=dict(policy.weights),
        benchmark_version=result.benchmark_version,
        benchmark_digest=result.benchmark_digest,
        synthetic_benchmarks=result.synthetic_benchmarks,
        scenario_id=result.scenario_id,
        score_digest=score_digest,
        candidates=cands,
        ranking=list(result.ranking),
        pareto_front=list(result.pareto_front),
        pareto_authoritative=result.pareto_authoritative,
        dominance=[p.model_dump(mode="json") for p in result.dominance],
        warnings=list(result.correlation_warnings),
        notes=list(result.notes),
        allowed_evidence_ids=sorted(allowed),
        numbers=sorted(set(numbers)),
    )
    packet.digest = "fpk-" + hashlib.sha256(packet.canonical_text().encode()).hexdigest()[:24]
    return packet


def parse_draft(
    text: str, packet: FusionEvidencePacket
) -> tuple[HypothesisDraft | None, list[str]]:
    """Parse and check a drafter reply; returns the draft and the problems found."""
    try:
        draft = HypothesisDraft.model_validate_json(text)
    except ValidationError as exc:
        fields = sorted({".".join(map(str, e["loc"])) for e in exc.errors()})
        return None, [f"draft does not match the HypothesisDraft schema: {fields}"]
    problems: list[str] = []
    allowed = set(packet.allowed_evidence_ids)
    listed = draft.supporting_evidence_ids + draft.contradicting_evidence_ids
    body = " ".join([draft.rationale, draft.recommended_calculation, *draft.missing_evidence])
    unknown = sorted((set(listed) | set(_EVIDENCE_ID.findall(body))) - allowed)
    if unknown:
        problems.append(f"cites evidence not in the packet: {unknown}")
    for token in _NUMBER.findall(_EVIDENCE_ID.sub(" ", body)):
        x = float(token)
        if x == int(x) and abs(x) <= _SMALL_INTEGER_LIMIT:
            continue
        if not any(
            abs(x - n) <= _TOLERANCE * max(abs(n), 1e-12) or _rounds_to(n, token)
            for n in packet.numbers
        ):
            problems.append(f"number {token} is not in the evidence packet")
    if m := _ALTERATION.search(body):
        problems.append(f"asks to alter scoring inputs ({m.group(0)!r}); policy is fixed")
    quals = {
        x.qualification for c in packet.candidates for o in c.objectives for x in o.observables
    }
    if "scientifically qualified" in body.lower() and "scientifically_qualified" not in quals:
        problems.append("claims scientific qualification that no cited value has")
    return draft, problems


def _status(c: CandidateResult, result: ScoringResult) -> tuple[HypothesisStatus, str]:
    held = [p for p in result.dominance if p.winner == c.candidate_id]
    beaten = [p for p in result.dominance if p.loser == c.candidate_id]
    qualified = c.ranking_status == RankingStatus.COMPLETE_QUALIFIED
    if qualified and any(p.dominance == Dominance.ESTABLISHED for p in beaten):
        by = sorted(p.winner for p in beaten if p.dominance == Dominance.ESTABLISHED)
        return HypothesisStatus.CONTRADICTED, f"established dominance by {', '.join(by)}"
    if (
        qualified
        and c.rank == 1
        and result.pareto_authoritative
        and c.pareto
        and all(p.dominance == Dominance.ESTABLISHED for p in held)
        and len(held) == len(result.ranking) - 1
    ):
        return HypothesisStatus.SUPPORTED, "rank 1, qualified, established dominance over all"
    if c.missing_objectives:
        return HypothesisStatus.UNRESOLVED, f"missing {', '.join(c.missing_objectives)}"
    return HypothesisStatus.UNRESOLVED, f"ranking status {c.ranking_status.value}"


def _evidence_split(
    c: CandidateResult, result: ScoringResult, ledger_ids: Mapping[str, str]
) -> tuple[list[str], list[str]]:
    """Ledger ids of objectives where ``c`` is best (supporting) or worst (contradicting)."""
    others = [x for x in result.candidates if x.candidate_id != c.candidate_id]
    support, contra = [], []
    for o in c.objectives:
        if o.desirability is None:
            continue
        rivals = [r.objective(o.objective).desirability for r in others]
        rivals = [r for r in rivals if r is not None]
        ids = [ledger_ids[x.property_id] for x in o.observables if x.property_id in ledger_ids]
        if rivals and o.desirability >= max(rivals):
            support += ids
        elif rivals and o.desirability <= min(rivals):
            contra += ids
    return sorted(set(support)), sorted(set(contra))


def recommended_calculations(c: CandidateResult) -> list[str]:
    """Deterministic next calculations: missing objectives first, then widest intervals."""
    out = []
    for name in c.missing_objectives:
        d = objective(name)
        out.append(f"{name}: {d.evidence_requirement} (methods: {', '.join(d.methods)})")
    spread = sorted(
        (o for o in c.objectives if o.d_low is not None and o.d_high is not None),
        key=lambda o: -o.weight * (o.d_high - o.d_low),  # type: ignore[operator]
    )
    for o in spread[:2]:
        if o.d_high != o.d_low:
            out.append(f"{o.objective}: reduce uncertainty ({objective(o.objective).methods[0]})")
    return out or ["experimental confirmation of the scored observables"]


def _deterministic_rationale(c: CandidateResult, reason: str) -> str:
    parts = [f"status={c.ranking_status.value}; {reason}"]
    if c.score is not None:
        parts.append(f"S={c.score:.4g} [{c.score_low:.4g}, {c.score_high:.4g}] rank {c.rank}")
    else:
        parts.append(f"no total score; covered weight {c.covered_weight:.3g}")
    for o in c.objectives:
        if o.desirability is not None:
            parts.append(f"{o.objective} d={o.desirability:.3g}")
    return "Deterministic summary: " + " | ".join(parts)


@dataclass(frozen=True)
class FusionRevisionOutcome:
    record: HypothesisRecord
    packet: FusionEvidencePacket
    used_llm: bool
    notes: tuple[str, ...]


def revise_fusion_hypothesis(
    ledger: EvidenceLedger,
    hypothesis_id: str,
    candidate_id: str,
    result: ScoringResult,
    policy: FusionScoringPolicy,
    evidence: Mapping[str, Sequence[FusionEvidence]],
    *,
    budget: RevisionBudget | None = None,
    drafter: Drafter | None = None,
    agent_model_id: str | None = None,
    campaign_id: str | None = None,
) -> FusionRevisionOutcome:
    """Append one revision of the claim that ``candidate_id`` is the preferred PFC material."""
    budget = budget or RevisionBudget()
    history = ledger.history(hypothesis_id)
    if len(history) >= budget.max_revisions:
        raise RevisionBudgetError(
            f"{hypothesis_id} already has {len(history)} of {budget.max_revisions} revisions"
        )
    c = result.by_id(candidate_id)
    used = {
        x.property_id
        for cand in result.candidates
        for o in cand.objectives
        for x in o.observables
        if x.property_id
    }
    ledger_ids: dict[str, str] = {}
    for items in evidence.values():
        for item in items:
            pid = item.evidence.property_id()
            if pid in used and pid not in ledger_ids:
                ledger_ids[pid] = append_property_evidence(
                    ledger, item.evidence, campaign_id=campaign_id
                )
    packet = build_packet(result, policy, ledger_ids)
    status, reason = _status(c, result)
    supporting, contradicting = _evidence_split(c, result, ledger_ids)
    cited = sorted(set(supporting + contradicting))
    if status != HypothesisStatus.UNRESOLVED and not cited:
        status, reason = HypothesisStatus.UNRESOLVED, f"{reason}; no ledger evidence to cite"
    missing = [f"{n}: {c.objective(n).reason or 'no evidence'}" for n in c.missing_objectives]
    calcs = recommended_calculations(c)

    notes: list[str] = []
    rationale = _deterministic_rationale(c, reason)
    llm: dict[str, Any] = {"requested": drafter is not None, "calls": 0, "accepted": False}
    if drafter is not None:
        if budget.llm_calls_left == 0:
            notes.append("LLM budget exhausted; deterministic rationale used")
        else:
            budget.llm_calls_used += 1
            llm["calls"] = 1
            try:
                reply = drafter(packet.canonical_text()).strip()
            except Exception as exc:  # a drafter failure must not block the record
                notes.append(f"drafter failed ({type(exc).__name__}); deterministic")
            else:
                draft, problems = parse_draft(reply, packet)
                if problems or draft is None:
                    llm["rejected"] = problems
                    notes.append("LLM draft rejected: " + "; ".join(problems))
                else:
                    llm["accepted"] = True
                    llm["draft"] = draft.model_dump()
                    rationale += f"\n[LLM-drafted, verified against packet] {draft.rationale}"

    record = HypothesisRecord(
        hypothesis_id=hypothesis_id,
        prior_revision_id=history[-1].revision_id if history else None,
        campaign_id=campaign_id,
        claim_type="property",
        scientific_claim=(
            f"Candidate {candidate_id} is the preferred plasma-facing material under policy "
            f"{policy.policy_version} in scenario {result.scenario_id}."
        ),
        falsifiable_prediction=(
            "Higher-fidelity evidence for any objective moves another candidate above it or "
            "makes its dominance conditional."
        ),
        status=status,
        evidence_ids=cited,
        scientific_rationale=rationale,
        proposed_tests=calcs,
        agent_model_id=agent_model_id if llm["accepted"] else None,
        traceability={
            "parent_revision_id": history[-1].revision_id if history else None,
            "policy_version": policy.policy_version,
            "policy_digest": policy.digest(),
            "benchmark_version": result.benchmark_version,
            "scenario_id": result.scenario_id,
            "score_digest": packet.score_digest,
            "packet_version": packet.packet_version,
            "packet_digest": packet.digest,
            "candidate_id": candidate_id,
            "status_rule": reason,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "missing_evidence": missing,
            "recommended_calculation": calcs[0],
            "synthetic_benchmarks": result.synthetic_benchmarks,
            "llm": llm,
        },
    )
    stored = ledger.append(record)
    assert isinstance(stored, HypothesisRecord)
    return FusionRevisionOutcome(
        record=stored, packet=packet, used_llm=bool(llm["accepted"]), notes=tuple(notes)
    )


__all__ = [
    "PACKET_VERSION",
    "FusionEvidencePacket",
    "FusionRevisionOutcome",
    "HypothesisDraft",
    "build_packet",
    "parse_draft",
    "recommended_calculations",
    "revise_fusion_hypothesis",
]
