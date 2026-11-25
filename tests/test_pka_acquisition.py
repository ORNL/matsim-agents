"""PKA-informed ranking on a synthetic spectrum (not real nuclear data)."""

from __future__ import annotations

import pytest

from matsim_agents.integrations.external.spectra_pka.parser import (
    parse_output_text,
    to_pka_spectrum,
)
from matsim_agents.science.acquisition.pka_acquisition import PKACandidate, select_pka_informed
from matsim_agents.science.irradiation.contracts import DefectStatus, PKASourceProvenance
from tests.spectra_pka_fixtures import SYNTHETIC_OUT, make_campaign


@pytest.fixture
def spectrum(tmp_path):
    from matsim_agents.integrations.external.spectra_pka.adapter import scenario_from_campaign

    campaign = make_campaign(tmp_path)
    prov = PKASourceProvenance(source="synthetic", nuclear_data_identity="synthetic")
    return to_pka_spectrum(parse_output_text(SYNTHETIC_OUT), scenario_from_campaign(campaign), prov)


def _cands():
    return [
        PKACandidate(candidate_id="w-a", species="W182", node_hours=1.0),
        PKACandidate(candidate_id="w-b", species="W182", node_hours=1.0),
        PKACandidate(candidate_id="ta-a", species="Ta181", node_hours=1.0),
        PKACandidate(candidate_id="re-a", species="Re187", node_hours=1.0),
    ]


def test_higher_production_first_then_diminishing_returns(spectrum):
    result = select_pka_informed(spectrum, _cands(), 3.0)
    # P(W182)=0.75, P(Ta181)=0.25: scores 0.75, then 0.75/2=0.375 > 0.25, then 0.25.
    assert [s.candidate_id for s in result.selected] == ["w-a", "w-b", "ta-a"]
    assert result.selected[1].representativeness == 0.5
    assert result.selected[1].score == pytest.approx(0.375)


def test_unmatched_species_reported_not_hidden(spectrum):
    result = select_pka_informed(spectrum, _cands(), 10.0)
    assert result.unmatched_candidates == ("re-a",)
    assert result.used_node_hours <= 10.0


def test_budget_is_respected_and_deterministic(spectrum):
    a = select_pka_informed(spectrum, _cands(), 1.5)
    b = select_pka_informed(spectrum, _cands(), 1.5)
    assert a == b
    assert a.used_node_hours <= 1.5 and len(a.selected) == 1


def test_nan_budget_is_rejected(spectrum):
    with pytest.raises(ValueError):
        select_pka_informed(spectrum, _cands(), float("nan"))


def test_element_candidate_covers_its_isotopes(spectrum):
    cands = [PKACandidate(candidate_id="w-el", species="W", node_hours=1.0)]
    result = select_pka_informed(spectrum, cands, 1.0)
    assert [s.candidate_id for s in result.selected] == ["w-el"]
    assert "W182" not in result.uncovered_species
    assert "Ta181" in result.uncovered_species


def test_defect_production_never_claimed(spectrum):
    result = select_pka_informed(spectrum, _cands(), 3.0)
    assert result.surviving_defect_production is DefectStatus.NOT_ESTABLISHED
    with pytest.raises(ValueError):
        select_pka_informed(spectrum, _cands(), 0.0)
