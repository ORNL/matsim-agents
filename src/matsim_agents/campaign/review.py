"""Structured multi-LLM evidence review for campaign steering."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from matsim_agents.backends.llm.provider import get_chat_model
from matsim_agents.campaign.orchestrator import CampaignReviewDecision, ReviewRunner
from matsim_agents.campaign.state import CampaignState, FormulaRunRecord
from matsim_agents.discovery.composition import parse_composition
from matsim_agents.discovery.formula_merge import LLMFormulaProposal
from matsim_agents.workflows.debate import (
    DebateParticipant,
    ScientificDebateConfig,
    ScientificDebateResult,
    run_scientific_debate,
)


class CampaignDebateReviewConfig(BaseModel):
    participants: list[DebateParticipant] = Field(min_length=2)
    output_root: str
    rounds: int = Field(2, ge=1)
    minimum_agreement_fraction: float = Field(1.0, gt=0.5, le=1.0)


class FormulaReview(BaseModel):
    formula: str
    action: Literal["keep", "deactivate", "reactivate"]
    rationale: str


class HypothesisRevision(BaseModel):
    claim: str
    status: Literal["supported", "contradicted", "unresolved", "superseded"]
    evidence: list[str] = Field(default_factory=list)
    proposed_test: str | None = None


class ReviewFormulaProposal(BaseModel):
    formula: str
    rationale: str
    falsification_tests: list[str] = Field(default_factory=list)


class ReviewVerdict(BaseModel):
    decisions: list[FormulaReview] = Field(default_factory=list)
    revisions: list[HypothesisRevision] = Field(default_factory=list)
    new_formula_proposals: list[ReviewFormulaProposal] = Field(default_factory=list)


_REVIEW_INSTRUCTION = """Return JSON only with this exact structure:
{"decisions": [{"formula": "...", "action": "keep|deactivate|reactivate", "rationale": "..."}],
 "revisions": [{"claim": "...", "status": "supported|contradicted|unresolved|superseded",
 "evidence": ["..."], "proposed_test": "..."}],
 "new_formula_proposals": [{"formula": "...", "rationale": "...",
 "falsification_tests": ["..."]}]}
Only review formulas present in the evidence. Deactivate only when numerical evidence makes
continued computation scientifically unjustified; uncertainty alone is not evidence of failure.
New formulas are proposals only and will undergo deterministic chemical and budget validation.
Do not wrap the JSON in Markdown."""


def _parse_verdict(text: str) -> ReviewVerdict:
    decoder = json.JSONDecoder()
    parsed: list[ReviewVerdict] = []
    validation_errors: list[ValueError] = []
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text, offset)
            if not isinstance(candidate, dict) or "decisions" not in candidate:
                continue
            parsed.append(ReviewVerdict.model_validate(candidate))
        except (json.JSONDecodeError, ValueError) as exc:
            validation_errors.append(exc)
    if parsed:
        return parsed[-1]
    if validation_errors:
        raise ValueError("response contains no valid review verdict") from validation_errors[-1]
    raise ValueError("response contains no JSON object")


def _review_evidence(campaign: CampaignState, records: list[FormulaRunRecord]) -> str:
    formulas: list[dict[str, Any]] = []
    for record in records:
        report = campaign.stability_reports.get(record.formula)
        formulas.append(
            {
                "formula": record.formula,
                "run": record.model_dump(mode="json"),
                "stability": report.model_dump(mode="json") if report is not None else None,
            }
        )
    return json.dumps(
        {
            "campaign_id": campaign.campaign_id,
            "iteration": campaign.iteration,
            "formulas": formulas,
        },
        sort_keys=True,
    )


def run_campaign_debate_review(
    campaign: CampaignState,
    records: list[FormulaRunRecord],
    *,
    config: CampaignDebateReviewConfig,
    model_factory: Callable[..., Any] = get_chat_model,
    debate_runner: Callable[..., ScientificDebateResult] = run_scientific_debate,
) -> CampaignReviewDecision:
    """Review one campaign iteration and apply only cross-model-agreed actions."""

    debate = debate_runner(
        ScientificDebateConfig(
            hypothesis=(
                "Review this materials-discovery campaign evidence and decide which formulas "
                f"should remain active. Evidence:\n{_review_evidence(campaign, records)}"
            ),
            participants=config.participants,
            rounds=config.rounds,
            output_root=config.output_root,
            debate_mode="equal",
            synthesis_method="independent_verdicts",
            final_response_instruction=_REVIEW_INSTRUCTION,
        ),
        model_factory=model_factory,
    )
    threshold = math.ceil(config.minimum_agreement_fraction * len(debate.verdicts))
    votes: Counter[tuple[str, str]] = Counter()
    notes: list[str] = []
    proposals: list[LLMFormulaProposal] = []
    revisions: list[dict[str, str | list[str]]] = []
    allowed = set(campaign.formulas)
    for verdict in debate.verdicts:
        try:
            payload = _parse_verdict(verdict.response)
        except ValueError as exc:
            notes.append(f"[{verdict.participant}] invalid review verdict: {exc}")
            continue
        seen: set[str] = set()
        for decision in payload.decisions:
            if decision.formula not in allowed or decision.formula in seen:
                continue
            seen.add(decision.formula)
            votes[(decision.formula, decision.action)] += 1
            notes.append(
                f"[{verdict.participant}] {decision.formula}: "
                f"{decision.action} - {decision.rationale}"
            )
        for revision in payload.revisions:
            revisions.append(
                {
                    "participant": verdict.participant,
                    "claim": revision.claim,
                    "status": revision.status,
                    "evidence": revision.evidence,
                    "proposed_test": revision.proposed_test or "",
                }
            )
        for proposal in payload.new_formula_proposals:
            composition = parse_composition(proposal.formula)
            if composition is None or not set(composition.elements).issubset(
                campaign.formula_policy.elements
            ):
                notes.append(
                    f"[{verdict.participant}] rejected out-of-scope formula proposal: "
                    f"{proposal.formula}"
                )
                continue
            proposals.append(
                LLMFormulaProposal(
                    participant=verdict.participant,
                    formula=composition.formula,
                    elements=composition.elements,
                )
            )

    deactivate = sorted(
        formula
        for (formula, action), count in votes.items()
        if action == "deactivate" and count >= threshold
    )
    reactivate = sorted(
        formula
        for (formula, action), count in votes.items()
        if action == "reactivate" and count >= threshold
    )
    return CampaignReviewDecision(
        debate_run_id=debate.run_id,
        deactivate_formulas=deactivate,
        reactivate_formulas=reactivate,
        formula_proposals=proposals,
        hypothesis_revisions=revisions,
        notes=notes,
    )


def make_debate_review_runner(
    config: CampaignDebateReviewConfig,
    *,
    model_factory: Callable[..., Any] = get_chat_model,
) -> ReviewRunner:
    def review(campaign: CampaignState, records: list[FormulaRunRecord]) -> CampaignReviewDecision:
        return run_campaign_debate_review(
            campaign,
            records,
            config=config,
            model_factory=model_factory,
        )

    return review
