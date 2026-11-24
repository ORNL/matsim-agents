"""MP9: radiation properties as evidence. All numbers are synthetic; no radiation physics runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from matsim_agents.campaign import radiation_properties as rp
from matsim_agents.campaign.budget import BudgetLedger
from matsim_agents.campaign.property_acquisition import (
    ProposedCalculation,
    plan_property_acquisition,
)
from matsim_agents.campaign.selection import (
    CandidateEvidence,
    Constraint,
    Objective,
    SelectionSpec,
    SelectionStatus,
    select,
)
from matsim_agents.execution.contracts import PROPERTY_UNITS, EvidenceLevel
from matsim_agents.execution.properties import (
    PropertyConditions,
    PropertyEvidence,
    PropertyQualification,
    PropertyStatus,
)
from matsim_agents.integrations.external.spectra_pka.config import SpectraPKAConfig
from matsim_agents.workflows.full_supercell_campaign import FullSupercellCampaign

CONFIG = Path(__file__).parents[1] / "deployments/frontier/campaigns/qe-1024/campaign.yaml"
T = 900.0
STATE = rp.RadiationState(irradiation="synthetic-1dpa", temperature_K=T, defect_state="V1He1")
KAPPA = "thermal_conductivity_total"


def kappa(value, unc, *, irradiated: bool, sid="A-s"):
    cond = PropertyConditions(
        temperature_K=T,
        irradiation="synthetic-1dpa" if irradiated else None,
        defect_state="V1He1" if irradiated else None,
    )
    return PropertyEvidence(
        structure_id=sid,
        composition="A",
        property=KAPPA,
        status=PropertyStatus.CALCULATED,
        value=value,
        unit=PROPERTY_UNITS[KAPPA],
        uncertainty=unc,
        conditions=cond,
        evidence_ids=[f"{sid}-{'irr' if irradiated else 'pri'}"],
        evidence_level=EvidenceLevel.CONVERGED_DFT,
        converged=True,
        qualification=PropertyQualification.NUMERICALLY_VALIDATED,
    )


def swelling(value, *, irradiation="synthetic-1dpa", sid="A-s"):
    return PropertyEvidence(
        structure_id=sid,
        composition="A",
        property="irradiation_swelling",
        status=PropertyStatus.MEASURED,
        value=value,
        unit=PROPERTY_UNITS["irradiation_swelling"],
        uncertainty=0.002,
        conditions=PropertyConditions(
            temperature_K=T, irradiation=irradiation, defect_state="V1He1"
        ),
        evidence_ids=["synthetic-meas"],
        evidence_level=EvidenceLevel.EXPERIMENTAL,
    )


def test_reference_alloy_is_the_1024_site_parent():
    cfg = FullSupercellCampaign.from_yaml(CONFIG)
    assert rp.check_reference_alloy(cfg) == rp.REFERENCE_ALLOY_COUNTS
    assert sum(rp.REFERENCE_ALLOY_COUNTS.values()) == rp.REFERENCE_SITES == 1024


def test_a_different_parent_is_refused():
    cfg = FullSupercellCampaign.from_yaml(CONFIG)
    smaller = FullSupercellCampaign.model_validate(
        {
            **cfg.model_dump(mode="json"),
            "parent_supercell": {
                **cfg.parent_supercell.model_dump(mode="json"),
                "conventional_repeats": [2, 2, 2],
                "expected_parent_atoms": 16,
                "composition": {"W": 8, "Ta": 2, "Cr": 2, "V": 2, "Hf": 1, "Mo": 1},
            },
        }
    )
    with pytest.raises(rp.RadiationError):
        rp.check_reference_alloy(smaller)


def test_no_supplied_data_gives_missing_for_every_radiation_property_with_a_reason():
    out = rp.radiation_evidence("A-s", "A", STATE)
    assert [e.property for e in out] == list(rp.RADIATION_PROPERTIES)
    for e in out:
        assert e.status is PropertyStatus.MISSING
        assert e.value is None and e.uncertainty is None
        assert e.reason and not e.is_scoreable()
        assert e.conditions.irradiation == "synthetic-1dpa"


def test_matching_measured_swelling_is_used_unchanged():
    out = rp.radiation_evidence("A-s", "A", STATE, [swelling(0.013)])
    got = {e.property: e for e in out}
    assert got["irradiation_swelling"].value == 0.013
    assert got["irradiation_swelling"].status is PropertyStatus.MEASURED
    assert got["embrittlement_dbtt_shift"].status is PropertyStatus.MISSING


def test_swelling_from_another_dose_or_structure_is_not_borrowed():
    other_dose = swelling(0.5, irradiation="synthetic-10dpa")
    other_structure = swelling(0.5, sid="B-s")
    out = rp.radiation_evidence("A-s", "A", STATE, [other_dose, other_structure])
    assert {e.property: e.status for e in out}["irradiation_swelling"] is PropertyStatus.MISSING


def test_degradation_is_derived_only_from_matching_pristine_and_irradiated_records():
    pri, irr = kappa(40.0, 2.0, irradiated=False), kappa(30.0, 1.5, irradiated=True)
    out = {e.property: e for e in rp.radiation_evidence("A-s", "A", STATE, [pri, irr])}
    d = out["thermal_conductivity_degradation"]
    assert d.status is PropertyStatus.APPROXIMATED
    assert d.value == pytest.approx(0.25)
    assert d.uncertainty > 0
    assert d.qualification is PropertyQualification.UNVALIDATED
    assert set(d.evidence_ids) == {"A-s-pri", "A-s-irr"}
    assert d.approximation


def test_pristine_record_alone_never_stands_in_for_the_irradiated_one():
    out = rp.radiation_evidence("A-s", "A", STATE, [kappa(40.0, 2.0, irradiated=False)])
    d = {e.property: e for e in out}["thermal_conductivity_degradation"]
    assert d.status is PropertyStatus.MISSING and d.value is None


def test_derivation_rejects_a_pristine_record_that_carries_a_defect_state():
    irr = kappa(30.0, 1.5, irradiated=True)
    with pytest.raises(rp.RadiationError):
        rp.derive_conductivity_degradation(irr, irr, STATE)


def test_readiness_reports_missing_spectra_and_pseudopotentials_as_blockers(tmp_path):
    ready = rp.assess_radiation_readiness(SpectraPKAConfig(), tmp_path, {})
    assert ready.spectra_pka == "NOT_INSTALLED"
    assert set(ready.pseudopotentials) == set(rp.RADIATION_SPECIES)
    assert all(v == "MISSING" for v in ready.pseudopotentials.values())
    assert set(ready.blockers) == {"spectra_pka", "qe_gas_defects"}
    assert "no degradation is claimed" in ready.statement


def test_blockers_are_carried_into_the_missing_reason(tmp_path):
    ready = rp.assess_radiation_readiness(SpectraPKAConfig(), tmp_path, {})
    out = rp.radiation_evidence("A-s", "A", STATE, readiness=ready)
    assert all("SPECTRA-PKA is NOT_INSTALLED" in e.reason for e in out)


def spec():
    return SelectionSpec(
        objectives=[
            Objective(property=KAPPA, direction="maximize", temperature_K=T),
            Objective(property="irradiation_swelling", direction="minimize", temperature_K=T),
        ],
        constraints=[Constraint(property="irradiation_swelling", maximum=0.05, temperature_K=T)],
        seed=3,
    )


def test_a_missing_radiation_objective_makes_the_candidate_incomplete_not_zero():
    base = [kappa(30.0, 1.0, irradiated=True)]
    out = select(
        spec(),
        [
            CandidateEvidence(
                candidate_id="A", evidence=base + rp.radiation_evidence("A-s", "A", STATE)
            )
        ],
    )
    score = out.candidates[0]
    assert score.status is not SelectionStatus.QUALIFIED
    assert score.status in {SelectionStatus.INCOMPLETE, SelectionStatus.INSUFFICIENT_EVIDENCE}


def test_measured_swelling_enters_selection_with_its_own_value():
    base = [kappa(30.0, 1.0, irradiated=True)]
    rad = rp.radiation_evidence("A-s", "A", STATE, [swelling(0.013)])
    out = select(spec(), [CandidateEvidence(candidate_id="A", evidence=base + rad)])
    rows = {o.property: o for o in out.candidates[0].objectives}
    assert rows["irradiation_swelling"].value == 0.013


def test_swelling_acquisition_is_blocked_by_the_missing_backend_and_says_why(tmp_path):
    ready = rp.assess_radiation_readiness(SpectraPKAConfig(), tmp_path, {})
    base = [kappa(30.0, 1.0, irradiated=True)]
    cand = CandidateEvidence(
        candidate_id="A", evidence=base + rp.radiation_evidence("A-s", "A", STATE)
    )
    proposal = ProposedCalculation(
        calc_id="A-swelling",
        candidate_id="A",
        property="irradiation_swelling",
        temperature_K=T,
        backend="spectra_pka",
        cost_node_hours=10.0,
        expected_uncertainty=0.01,
    )
    ledger = BudgetLedger(total_node_hours=1000.0, allocation={"property": 200.0})
    plan = plan_property_acquisition(
        spec(), [cand], [proposal], ledger, blockers={"spectra_pka": ready.blockers["spectra_pka"]}
    )
    row = plan.items[0]
    assert row.decision == "blocked" and "NOT_INSTALLED" in row.reason
    assert plan.selected == []
