from __future__ import annotations

from matsim_agents.discovery.formula import FormulaGenerationPolicy, enumerate_formulas
from matsim_agents.discovery.formula_merge import extract_llm_formula_proposals, merge_formulas
from matsim_agents.workflows.debate import DebateVerdict


def _policy(**overrides) -> FormulaGenerationPolicy:
    defaults = dict(
        elements=["Nb", "Ta", "O"],
        maximum_coefficient=6,
        maximum_atoms_in_reduced_formula=12,
        require_charge_balance=True,
        oxidation_states={"Nb": [3, 4, 5], "Ta": [3, 4, 5], "O": [-2]},
    )
    defaults.update(overrides)
    return FormulaGenerationPolicy(**defaults)


def _verdict(participant: str, response: str) -> DebateVerdict:
    return DebateVerdict(
        contribution_id=f"verdict-{participant}",
        participant=participant,
        provider="vllm",
        model=f"{participant}-model",
        response=response,
    )


def test_extract_llm_formula_proposals_reads_mentions_in_prose():
    verdicts = [
        _verdict("qwen", "I propose NbTaO4 as a mixed +4 oxidation-state compound."),
        _verdict("gemma", "Competing binary oxides such as Nb2O5 and Ta2O5 may dominate instead."),
    ]
    proposals = extract_llm_formula_proposals(verdicts, _policy())
    formulas = {p.formula for p in proposals}
    # Canonical formulas are alphabetical Hill order (e.g. Ta2O5 -> "O5Ta2").
    assert formulas == {"NbO4Ta", "Nb2O5", "O5Ta2"}
    assert {p.participant for p in proposals if p.formula == "NbO4Ta"} == {"qwen"}


def test_extract_llm_formula_proposals_drops_out_of_scope_elements():
    verdicts = [_verdict("deepseek", "Consider Li2MnO3 as an alternative cathode chemistry.")]
    proposals = extract_llm_formula_proposals(verdicts, _policy())
    assert proposals == []


def test_merge_formulas_attributes_llm_agreement_on_deterministic_formula():
    deterministic = enumerate_formulas(_policy())
    verdicts = [
        _verdict("qwen", "NbTaO4 is a strong candidate."),
        _verdict("deepseek", "I also favor NbTaO4 based on rutile-derived ordering."),
    ]
    proposals = extract_llm_formula_proposals(verdicts, _policy())
    merged = {c.reduced_formula: c for c in merge_formulas(deterministic, proposals, _policy())}
    candidate = merged["NbO4Ta"]
    assert candidate.generation_source == "deterministic+llm"
    assert set(candidate.llm_contributors) == {"qwen", "deepseek"}


def test_merge_formulas_adds_llm_only_formula_outside_deterministic_grid():
    deterministic = enumerate_formulas(_policy(maximum_coefficient=2))
    verdicts = [_verdict("gemma", "Nb2Ta2O7 deserves consideration despite its larger cell.")]
    proposals = extract_llm_formula_proposals(verdicts, _policy(maximum_coefficient=2))
    merged = {
        c.reduced_formula: c
        for c in merge_formulas(deterministic, proposals, _policy(maximum_coefficient=2))
    }
    assert "Nb2O7Ta2" not in {c.reduced_formula for c in deterministic}
    candidate = merged["Nb2O7Ta2"]
    assert candidate.generation_source == "llm"
    assert candidate.active
    assert candidate.llm_contributors == ["gemma"]


def test_merge_formulas_marks_charge_imbalanced_llm_proposal_inactive():
    deterministic: list = []
    verdicts = [_verdict("qwen", "NbO might form under reducing conditions.")]
    proposals = extract_llm_formula_proposals(verdicts, _policy())
    merged = {c.reduced_formula: c for c in merge_formulas(deterministic, proposals, _policy())}
    assert merged["NbO"].active is False
    assert merged["NbO"].rejection_reason is not None
