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


def _policy_rejection_reasons(
    elements: dict[str, int], policy: FormulaGenerationPolicy
) -> list[str]:
    species_count = len(elements)
    reasons = []
    undeclared = sorted(set(elements) - set(policy.elements))
    if undeclared:
        reasons.append(f"undeclared elements outside formula policy: {', '.join(undeclared)}")
    if not policy.minimum_species <= species_count <= policy.maximum_species:
        reasons.append("species count is outside formula policy bounds")
    if species_count == 2 and not policy.include_binary_endmembers:
        reasons.append("binary endmembers are disabled")
    if species_count >= 3 and not policy.include_mixed_oxides:
        reasons.append("mixed compositions are disabled")
    if any(
        coefficient < policy.minimum_coefficient or coefficient > policy.maximum_coefficient
        for coefficient in elements.values()
    ):
        reasons.append("coefficient is outside formula policy bounds")
    if sum(elements.values()) > policy.maximum_atoms_in_reduced_formula:
        reasons.append("reduced formula exceeds atom-count limit")
    return reasons


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
        rejection_reasons = _policy_rejection_reasons(proposal.elements, policy)
        charge_ok = True
        if policy.require_charge_balance:
            charge_ok = _charge_balance_possible(proposal.elements, policy.oxidation_states)
            if not charge_ok:
                rejection_reasons.append("fails charge balance for given oxidation states")
        active = not rejection_reasons
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
            active=active,
            iteration_created=iteration,
            acceptance_reason="proposed by LLM panel" if active else None,
            rejection_reason="; ".join(rejection_reasons) or None,
        )
    return sorted(
        by_formula.values(),
        key=lambda candidate: (len(candidate.elements), candidate.reduced_formula),
    )
