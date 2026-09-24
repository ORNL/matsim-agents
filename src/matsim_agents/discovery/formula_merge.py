"""Merge deterministic formula enumeration with LLM-proposed formulas.

Per the multi-LLM debate design: popularity does not establish chemical
validity. A formula proposed by every panel member may still fail the
deterministic charge-balance screen; a formula proposed by only one model
may still be scientifically valuable. The merge step therefore never
discards a proposal because other models disagreed with it -- it only
applies the same deterministic constraints used for the enumerated grid.
"""

from __future__ import annotations

from pydantic import BaseModel

from matsim_agents.discovery.composition import extract_compositions
from matsim_agents.discovery.formula import (
    FormulaCandidate,
    FormulaGenerationPolicy,
    _charge_balance_possible,
)
from matsim_agents.workflows.debate import DebateVerdict


class LLMFormulaProposal(BaseModel):
    """One formula mention attributed to one debate participant."""

    participant: str
    formula: str
    elements: dict[str, int]


def extract_llm_formula_proposals(
    verdicts: list[DebateVerdict],
    policy: FormulaGenerationPolicy,
) -> list[LLMFormulaProposal]:
    """Pull formula mentions out of free-text debate verdicts.

    Only formulas whose elements are a subset of ``policy.elements`` are kept:
    the campaign is scoped to a fixed element set, so a model proposing an
    out-of-scope element is not a formula proposal for this campaign.
    """
    allowed = set(policy.elements)
    proposals: list[LLMFormulaProposal] = []
    for verdict in verdicts:
        for composition in extract_compositions(verdict.response):
            if set(composition.elements).issubset(allowed):
                proposals.append(
                    LLMFormulaProposal(
                        participant=verdict.participant,
                        formula=composition.formula,
                        elements=composition.elements,
                    )
                )
    return proposals


def merge_formulas(
    deterministic: list[FormulaCandidate],
    llm_proposals: list[LLMFormulaProposal],
    policy: FormulaGenerationPolicy,
    *,
    iteration: int = 0,
) -> list[FormulaCandidate]:
    """Merge the deterministic grid with LLM proposals, preserving attribution."""
    by_formula = {candidate.reduced_formula: candidate for candidate in deterministic}
    for proposal in llm_proposals:
        existing = by_formula.get(proposal.formula)
        if existing is not None:
            if proposal.participant not in existing.llm_contributors:
                existing.llm_contributors.append(proposal.participant)
            if existing.generation_source == "deterministic":
                existing.generation_source = "deterministic+llm"
            continue
        # LLM proposed a formula outside the deterministic grid (e.g. a
        # coefficient beyond the configured range) -- validate it the same way.
        charge_ok = True
        if policy.require_charge_balance:
            charge_ok = _charge_balance_possible(proposal.elements, policy.oxidation_states)
        by_formula[proposal.formula] = FormulaCandidate(
            formula_id=f"llm-{proposal.formula}",
            reduced_formula=proposal.formula,
            elements=proposal.elements,
            allowed_oxidation_states={
                symbol: policy.oxidation_states.get(symbol, []) for symbol in proposal.elements
            },
            charge_balanced=charge_ok,
            generation_source="llm",
            llm_contributors=[proposal.participant],
            active=charge_ok,
            iteration_created=iteration,
            acceptance_reason="proposed by LLM panel" if charge_ok else None,
            rejection_reason=(
                None if charge_ok else "fails charge balance for given oxidation states"
            ),
        )
    return sorted(
        by_formula.values(),
        key=lambda candidate: (len(candidate.elements), candidate.reduced_formula),
    )
