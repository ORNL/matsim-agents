from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from matsim_agents.campaign.orchestrator import run_formula_discovery_stage
from matsim_agents.campaign.state import CampaignState
from matsim_agents.discovery.formula import FormulaGenerationPolicy
from matsim_agents.discovery.stability import (
    PhaseStability,
    RankingMode,
    ReferenceEnergySet,
    StabilityReport,
)
from matsim_agents.execution.contracts import EvidenceLevel, WorkflowStatus
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
        transcript_path="/tmp/run-001/transcript.md",
        contributions_path="/tmp/run-001/contributions.jsonl",
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
        composition={"Nb": 1, "Ta": 1, "O": 4},
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
        evidence_level=EvidenceLevel.CONVERGED_DFT,
        reference_set_id="nb-ta-o-pbe-v1",
    )
    campaign.record_stability(report)
    assert campaign.stability_reports["NbTaO4"] is report
    assert campaign.reference_energies.competing_phases["NbTaO4"] == -0.45
    assert campaign.current_hull is not None
    assert campaign.current_hull.version == 1
    assert campaign.current_hull.new_hull_vertices == ["NbTaO4"]
    assert campaign.current_hull.hull_vertices["NbTaO4"].endswith("P003-relaxed.vasp")
    assert campaign.current_hull.provisional is True
    assert campaign.current_hull.reference_completeness is not None
    assert campaign.current_hull.reference_completeness.missing_binary_subsystems


def test_record_stability_admits_near_degenerate_hull_vertex():
    campaign = _campaign()
    ground_state = PhaseStability(
        structure_path="candidates/NbTaO4-P003.vasp",
        optimized_structure_path="candidates/NbTaO4-P003-relaxed.vasp",
        composition={"Nb": 1, "Ta": 1, "O": 4},
        final_energy_eV=-62.56,
        energy_per_atom_eV=-7.82,
        delta_e_above_min_eV_per_atom=0.0,
        final_max_force_eV_per_A=0.01,
        converged=True,
        eligible_for_ranking=True,
        dynamically_stable_proxy=True,
        formation_energy_eV_per_atom=-0.45,
        energy_above_hull_eV_per_atom=0.0,
    )
    report = StabilityReport(
        formula="NbTaO4",
        ground_state=ground_state,
        ranking=[ground_state],
        chemically_stable_proxy=False,
        summary="NbTaO4 has a near-degenerate polymorph.",
        ranking_mode=RankingMode.CONVEX_HULL,
        evidence_level=EvidenceLevel.CONVERGED_DFT,
        reference_set_id="nb-ta-o-pbe-v1",
        near_degenerate_structure_paths=["candidates/NbTaO4-P004-relaxed.vasp"],
    )

    campaign.record_stability(report)

    assert campaign.reference_energies.competing_phases["NbTaO4"] == -0.45
    assert campaign.stability_reports["NbTaO4"].chemically_stable_proxy is False


def _hull_report(formula: str, energy: float, formation_energy: float) -> StabilityReport:
    from pymatgen.core import Composition

    phase = PhaseStability(
        structure_path=f"candidates/{formula}.vasp",
        optimized_structure_path=f"candidates/{formula}-relaxed.vasp",
        composition=Composition(formula).as_dict(),
        final_energy_eV=energy,
        energy_per_atom_eV=energy,
        delta_e_above_min_eV_per_atom=0.0,
        final_max_force_eV_per_A=0.01,
        converged=True,
        dynamically_stable_proxy=True,
        formation_energy_eV_per_atom=formation_energy,
        energy_above_hull_eV_per_atom=0.0,
    )
    return StabilityReport(
        formula=formula,
        ground_state=phase,
        ranking=[phase],
        chemically_stable_proxy=True,
        summary=f"{formula} hull result",
        ranking_mode=RankingMode.CONVEX_HULL,
        evidence_level=EvidenceLevel.CONVERGED_DFT,
        reference_set_id="nb-o-pbe-v1",
    )


def test_record_stability_recalibrates_prior_reports_when_hull_changes():
    campaign = CampaignState(
        campaign_id="nb-o-001",
        element_set=["Nb", "O"],
        formula_policy=FormulaGenerationPolicy(
            elements=["Nb", "O"],
            require_charge_balance=False,
        ),
        reference_energies=ReferenceEnergySet(
            identifier="nb-o-pbe-v1",
            method_signature="pbe-v1",
            elemental_energies_eV_per_atom={"Nb": 0.0, "O": 0.0},
        ),
    )
    campaign.record_stability(_hull_report("NbO", -2.0, -1.0))
    campaign.record_stability(_hull_report("Nb2O", -4.8, -1.6))

    assert campaign.stability_reports[
        "NbO"
    ].ground_state.energy_above_hull_eV_per_atom == pytest.approx(0.2)
    assert campaign.current_hull is not None
    assert campaign.current_hull.version == 2
    assert campaign.current_hull.new_hull_vertices == ["Nb2O"]
    assert campaign.current_hull.removed_hull_vertices == ["NbO"]
    assert campaign.current_hull.uncompetitive_formulas == ["NbO"]


@pytest.mark.parametrize(
    "reference_set_id", ["different-dft-method", "different-calibration", None]
)
@pytest.mark.parametrize("energy_above_hull", [0.0, 0.02, 0.2])
def test_hull_snapshot_excludes_incompatible_reports(reference_set_id, energy_above_hull):
    campaign = _campaign()
    campaign.reference_energies.elemental_energies_eV_per_atom = {
        "Nb": 0.0,
        "Ta": 0.0,
        "O": 0.0,
    }
    incompatible = _hull_report("NbO", -2.0, -1.0)
    incompatible.reference_set_id = reference_set_id
    incompatible.ground_state.energy_above_hull_eV_per_atom = energy_above_hull
    incompatible.ground_state.decomposition = {"Nb": 0.5, "O": 0.5}
    campaign.record_stability(incompatible)
    assert campaign.current_hull is None
    assert "NbO" not in campaign.reference_energies.competing_phases

    compatible = _hull_report("Nb2O", -4.8, -1.6)
    compatible.reference_set_id = campaign.reference_energies.identifier
    campaign.record_stability(compatible)

    assert campaign.stability_reports["NbO"] is incompatible
    assert incompatible.ground_state.energy_above_hull_eV_per_atom == energy_above_hull
    assert campaign.current_hull is not None
    assert campaign.current_hull.reference_set_id == campaign.reference_energies.identifier
    assert set(campaign.current_hull.energy_above_hull_eV_per_atom) == {"Nb2O"}
    assert set(campaign.current_hull.decomposition_products) == {"Nb2O"}
    assert set(campaign.current_hull.hull_vertices) == {"Nb2O"}
    assert campaign.current_hull.new_hull_vertices == ["Nb2O"]
    assert campaign.current_hull.removed_hull_vertices == []
    assert campaign.current_hull.near_hull_phases == {}
    assert campaign.current_hull.uncompetitive_formulas == []
    assert campaign.hull_history == [campaign.current_hull]


def test_campaign_requires_matching_unique_element_set():
    with pytest.raises(ValidationError, match="element_set must match"):
        CampaignState(
            campaign_id="mismatch",
            element_set=["Nb", "O"],
            formula_policy=_policy(),
        )
    with pytest.raises(ValidationError, match="element_set must not contain duplicates"):
        CampaignState(
            campaign_id="duplicates",
            element_set=["Nb", "Nb", "Ta", "O"],
            formula_policy=_policy(),
        )


def test_record_stability_does_not_admit_above_hull_phase():
    campaign = _campaign()
    report = _hull_report("NbO", -2.0, -1.0)
    report.reference_set_id = campaign.reference_energies.identifier
    report.ground_state.energy_above_hull_eV_per_atom = 0.1

    campaign.record_stability(report)

    assert "NbO" not in campaign.reference_energies.competing_phases


def _vertex_report(level: EvidenceLevel) -> StabilityReport:
    ground_state = PhaseStability(
        structure_path="candidates/NbTaO4-P003.vasp",
        optimized_structure_path="candidates/NbTaO4-P003-relaxed.vasp",
        composition={"Nb": 1, "Ta": 1, "O": 4},
        final_energy_eV=-62.56,
        energy_per_atom_eV=-7.82,
        delta_e_above_min_eV_per_atom=0.0,
        final_max_force_eV_per_A=0.01,
        converged=True,
        dynamically_stable_proxy=True,
        formation_energy_eV_per_atom=-0.45,
        energy_above_hull_eV_per_atom=0.0,
    )
    return StabilityReport(
        formula="NbTaO4",
        ground_state=ground_state,
        ranking=[ground_state],
        chemically_stable_proxy=True,
        summary="NbTaO4 on the hull.",
        ranking_mode=RankingMode.CONVEX_HULL,
        evidence_level=level,
        reference_set_id="nb-ta-o-pbe-v1",
    )


def test_mlip_hull_reports_never_enter_the_campaign_hull():
    campaign = _campaign()
    campaign.record_stability(_vertex_report(EvidenceLevel.MLIP_RELAXATION))
    assert "NbTaO4" in campaign.stability_reports
    assert "NbTaO4" not in campaign.reference_energies.competing_phases
    assert campaign.current_hull is None


def test_version_1_campaign_state_migrates_explicitly():
    campaign = _campaign()
    campaign.record_stability(_vertex_report(EvidenceLevel.CONVERGED_DFT))
    saved = campaign.model_dump(mode="json")
    saved.pop("schema_version")
    del saved["stability_reports"]["NbTaO4"]["evidence_level"]
    del saved["current_hull"]["evidence_level"]
    for hull in saved["hull_history"]:
        del hull["evidence_level"]
    original = json.loads(json.dumps(saved))
    restored = CampaignState.model_validate(saved)
    assert saved == original
    assert restored.schema_version == 3
    assert restored.stability_reports["NbTaO4"].evidence_level == EvidenceLevel.CONVERGED_DFT
    assert restored.current_hull.evidence_level == EvidenceLevel.CONVERGED_DFT
    with pytest.raises(ValidationError, match="newer than this code"):
        CampaignState.model_validate({**saved, "schema_version": 4})


def _version_2_state() -> dict:
    from matsim_agents.campaign.state import CampaignReviewRecord, FormulaRunRecord

    campaign = _campaign()
    campaign.record_stability(_vertex_report(EvidenceLevel.CONVERGED_DFT))
    campaign.formula_runs["NbTaO4"] = FormulaRunRecord(
        formula="NbTaO4",
        status=WorkflowStatus.COMPLETE,
        output_dir="formulas/NbTaO4",
        evidence={"dft_refinement": {"backend": "qe", "method_signature": "pbe-v1"}},
    )
    campaign.formula_runs["Nb2O5"] = FormulaRunRecord(
        formula="Nb2O5", status=WorkflowStatus.FAILED, failure_reason="RuntimeError('pw.x')"
    )
    campaign.review_history.append(
        CampaignReviewRecord(
            iteration=0,
            debate_run_id="review-001",
            deactivate_formulas=["Nb2O5"],
            notes=["[a] Nb2O5: deactivate - no converged candidates"],
            metadata={
                "hypothesis_revisions": [
                    {
                        "participant": "model-a",
                        "claim": "NbTaO4 is a new hull vertex",
                        "status": "supported",
                        "evidence": ["DFT hull distance 0.0"],
                        "proposed_test": "phonons",
                    }
                ]
            },
        )
    )
    saved = campaign.model_dump(mode="json")
    saved["schema_version"] = 2
    del saved["evidence_ledger_file"]
    return saved


def test_version_2_campaign_is_backfilled_into_its_ledger(tmp_path):
    from matsim_agents.campaign.evidence import open_campaign_ledger
    from matsim_agents.execution.ledger import EvidenceOutcome

    campaign = CampaignState.model_validate(_version_2_state())
    assert campaign.schema_version == 3 and campaign.evidence_ledger_file is None
    ledger = open_campaign_ledger(campaign, tmp_path / "one")
    assert campaign.evidence_ledger_file == "evidence-ledger.jsonl"
    [hull] = ledger.evidence(subject="NbTaO4")
    assert hull.kind == "stability" and hull.evidence_level == EvidenceLevel.CONVERGED_DFT
    assert hull.method.method_signature == "pbe-v1"
    [failure] = ledger.evidence(subject="Nb2O5")
    assert failure.outcome == EvidenceOutcome.FAILED_TO_COMPUTE
    [decision] = ledger.decisions()
    assert decision.subject == "Nb2O5" and decision.imported_from
    [claim] = ledger.current_hypotheses().values()
    assert claim.imported_from and claim.evidence_ids == []
    assert "DFT hull distance 0.0" in claim.scientific_rationale
    # Opening again does not backfill twice; replaying the saved state gives the same IDs.
    assert len(open_campaign_ledger(campaign, tmp_path / "one")) == len(ledger)
    replay = open_campaign_ledger(
        CampaignState.model_validate(_version_2_state()), tmp_path / "two"
    )
    assert [r.model_dump(exclude={"recorded_at"}) for r in replay.records()] == [
        r.model_dump(exclude={"recorded_at"}) for r in ledger.records()
    ]


def test_recalibrated_reports_become_new_evidence_with_their_parent(tmp_path):
    from matsim_agents.campaign.evidence import open_campaign_ledger, sync_stability_evidence

    campaign = _campaign()
    campaign.record_stability(_vertex_report(EvidenceLevel.CONVERGED_DFT))
    ledger = open_campaign_ledger(campaign, tmp_path)
    [first] = sync_stability_evidence(ledger, campaign)
    assert sync_stability_evidence(ledger, campaign) == []
    report = campaign.stability_reports["NbTaO4"]
    shifted = report.ground_state.model_copy(update={"energy_above_hull_eV_per_atom": 0.02})
    campaign.stability_reports["NbTaO4"] = report.model_copy(
        update={"ground_state": shifted, "ranking": [shifted]}
    )
    [second] = sync_stability_evidence(ledger, campaign)
    assert ledger.get(second).parent_record_ids == [first]
