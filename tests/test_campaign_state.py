from __future__ import annotations

from matsim_agents.campaign.orchestrator import run_formula_discovery_stage
from matsim_agents.campaign.state import CampaignState
from matsim_agents.discovery.formula import FormulaGenerationPolicy
from matsim_agents.discovery.stability import (
    PhaseStability,
    RankingMode,
    ReferenceEnergySet,
    StabilityReport,
)
from matsim_agents.execution.contracts import WorkflowStatus
from matsim_agents.workflows.debate import DebateVerdict, ScientificDebateResult


def _policy() -> FormulaGenerationPolicy:
    return FormulaGenerationPolicy(
        elements=["Nb", "Ta", "O"],
        maximum_coefficient=6,
        maximum_atoms_in_reduced_formula=12,
        require_charge_balance=True,
        oxidation_states={"Nb": [3, 4, 5], "Ta": [3, 4, 5], "O": [-2]},
    )


def _campaign() -> CampaignState:
    return CampaignState(
        campaign_id="nb-ta-o-001",
        element_set=["Nb", "Ta", "O"],
        formula_policy=_policy(),
        reference_energies=ReferenceEnergySet(
            identifier="nb-ta-o-pbe-v1",
            method_signature="pbe-v1",
            elemental_energies_eV_per_atom={"Nb": -10.1, "Ta": -11.9, "O": -4.9},
        ),
    )


def _debate_result() -> ScientificDebateResult:
    return ScientificDebateResult(
        run_id="run-001",
        run_directory="/tmp/run-001",
        status=WorkflowStatus.COMPLETE,
        hypothesis="Which Nb-Ta-O formula is most promising?",
        rounds_completed=1,
        turns=[],
        verdicts=[
            DebateVerdict(
                contribution_id="verdict-qwen",
                participant="qwen",
                provider="vllm",
                model="qwen-model",
                response="NbTaO4 looks promising given mixed +4 oxidation states.",
            )
        ],
        synthesis="",
        transcript_path="/tmp/run-001/debate_transcript.json",
        dialogue_path="/tmp/run-001/dialogue.json",
    )


def test_upsert_and_active_formulas_filter_inactive():
    campaign = _campaign()
    campaign = run_formula_discovery_stage(campaign, _debate_result())
    assert campaign.iteration == 1
    assert campaign.debate_run_ids == ["run-001"]
    active = {c.reduced_formula for c in campaign.active_formulas()}
    inactive = {c.reduced_formula for c in campaign.formulas.values()} - active
    # Canonical formulas are alphabetical Hill order (e.g. NbTaO4 -> "NbO4Ta").
    assert "NbO4Ta" in active
    assert all(not campaign.formulas[f].active for f in inactive)
    # LLM agreement on an already-enumerated formula is recorded, not duplicated.
    assert campaign.formulas["NbO4Ta"].generation_source == "deterministic+llm"


def test_record_stability_feeds_hull_reference_set():
    campaign = _campaign()
    ground_state = PhaseStability(
        structure_path="candidates/NbTaO4-P003.vasp",
        optimized_structure_path="candidates/NbTaO4-P003-relaxed.vasp",
        final_energy_eV=-62.56,
        energy_per_atom_eV=-7.82,
        delta_e_above_min_eV_per_atom=0.0,
        final_max_force_eV_per_A=0.01,
        converged=True,
        dynamically_stable_proxy=True,
        formation_energy_eV_per_atom=-0.45,
        energy_above_hull_eV_per_atom=0.0,
    )
    report = StabilityReport(
        formula="NbTaO4",
        ground_state=ground_state,
        ranking=[ground_state],
        chemically_stable_proxy=True,
        summary="NbTaO4 relaxed to a new hull vertex.",
        ranking_mode=RankingMode.CONVEX_HULL,
        reference_set_id="nb-ta-o-pbe-v1",
    )
    campaign.record_stability(report)
    assert campaign.stability_reports["NbTaO4"] is report
    assert campaign.reference_energies.competing_phases["NbTaO4"] == -0.45
