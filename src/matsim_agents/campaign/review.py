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
Treat authoritative_facts and evidence_semantics as constraints, not hypotheses. Do not infer
composition from prototype_id, reinterpret charge balance, equate missing or zero summary counters
with zero attempted calculations, or treat MLIP convergence/relative energies as thermodynamic
stability. Use outcome_class as the authoritative stage classification and do not infer a software,
generation, or pre-relaxation root cause beyond that classification unless a corresponding recorded
exception identifies it. Surrogate hull values are MLIP proxies, not DFT validation. Cite the
numerical field and evidence level that support every decision.
New formulas are proposals only and will undergo deterministic chemical and budget validation.
Do not wrap the JSON in Markdown."""


_EVIDENCE_SEMANTICS = {
    "composition": (
        "canonical_formula, element_counts, and charge_balanced_by_policy are authoritative. "
        "Prototype A/B labels are abstract sites whose species mapping may be reordered; never "
        "infer composition from prototype_id."
    ),
    "execution_counts": (
        "candidate_counts separately reports generated, attempted, completed, converged, and "
        "failed candidates. outcome_class is the authoritative conservative classification. "
        "Do not infer a more specific root cause without a recorded exception from that stage."
    ),
    "failed_formula": (
        "A formula-level failure or absence of converged candidates is not evidence that the "
        "composition is chemically impossible and is not a campaign/infrastructure failure."
    ),
    "energy_scope": (
        "MLIP total or per-atom energies are comparable only within their declared ranking scope. "
        "Relative phase ranking does not establish formation energy, convex-hull stability, or "
        "experimental synthesizability."
    ),
    "dft_scope": (
        "DFT evidence supports thermodynamic claims only when compatible elemental and competing-"
        "phase references produce an explicit formation energy or energy above hull."
    ),
    "cross_model_scope": (
        "Cross-model ranking disagreement is an uncertainty signal. Agreement is corroboration "
        "among related MLIPs, not independent thermodynamic validation."
    ),
    "surrogate_hull_scope": (
        "Surrogate formation and hull energies are model-specific MLIP proxies. They must be "
        "reported by model with reference completeness and cannot be described as DFT evidence."
    ),
}


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
        candidate = campaign.formulas[record.formula]
        labels = record.evidence.get("mlip_labels", [])
        if not isinstance(labels, list):
            labels = []
        converged_labels = sum(
            isinstance(label, dict) and bool(label.get("converged")) for label in labels
        )
        ranking_mode = report.ranking_mode if report is not None else None
        has_formation_energy = bool(
            report is not None
            and any(entry.formation_energy_eV_per_atom is not None for entry in report.ranking)
        )
        has_hull_energy = bool(
            report is not None
            and any(entry.energy_above_hull_eV_per_atom is not None for entry in report.ranking)
        )
        cross_model = record.evidence.get("cross_model_validation", {})
        robustness = record.evidence.get("perturbation_robustness", {})
        formulas.append(
            {
                "formula": record.formula,
                "authoritative_facts": {
                    "canonical_formula": candidate.reduced_formula,
                    "element_counts": candidate.elements,
                    "charge_balanced_by_policy": candidate.charge_balanced,
                    "run_status": record.status,
                    "failure_reason": record.failure_reason,
                    "outcome_class": record.outcome_class,
                    "candidate_counts": record.candidate_counts,
                    "observed_mlip_labels": len(labels),
                    "converged_mlip_labels": converged_labels,
                    "dft_calculations": record.n_dft_calculations,
                    "ranking_mode": ranking_mode,
                    "has_formation_energy": has_formation_energy,
                    "has_energy_above_hull": has_hull_energy,
                    "cross_model_ranking_disagreement": cross_model.get("ranking_disagreement"),
                    "perturbation_robust_fraction": robustness.get("robust_fraction"),
                    "surrogate_hull_evidence_level": (
                        "mlip_proxy"
                        if any(
                            str(key).endswith(":surrogate_hull")
                            for key in cross_model.get("models", {})
                        )
                        else None
                    ),
                },
                "run": record.model_dump(mode="json"),
                "stability": report.model_dump(mode="json") if report is not None else None,
            }
        )
    return json.dumps(
        {
            "campaign_id": campaign.campaign_id,
            "iteration": campaign.iteration,
            "evidence_semantics": _EVIDENCE_SEMANTICS,
            "formulas": formulas,
        },
        sort_keys=True,
    )


def _deactivation_block_reason(
    campaign: CampaignState,
    record: FormulaRunRecord,
) -> str | None:
    report = campaign.stability_reports.get(record.formula)
    if report is None:
        return "authoritative numerical evidence has no stability report"
    ground_state = report.ground_state
    if not ground_state.converged or not ground_state.eligible_for_ranking:
        return "authoritative numerical evidence has no converged, ranking-eligible ground state"
    if any(entry.energy_above_hull_eV_per_atom is not None for entry in report.ranking):
        return None
    return (
        "authoritative numerical evidence contains a converged, ranking-eligible candidate "
        "but no energy-above-hull result; relative MLIP ranking cannot justify deactivation"
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
    records_by_formula = {record.formula: record for record in records}
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
            if decision.action == "deactivate" and decision.formula in records_by_formula:
                block_reason = _deactivation_block_reason(
                    campaign,
                    records_by_formula[decision.formula],
                )
                if block_reason is not None:
                    notes.append(
                        f"[{verdict.participant}] rejected {decision.formula} deactivation: "
                        f"{block_reason}"
                    )
                    continue
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
