"""Deterministic synthetic chain: alloy -> transmutation -> gas complex -> PKA -> ranking.

Every stage runs on synthetic inputs and a mock SPECTRA-PKA executable. This verifies that the
stages compose, are deterministic, and carry their qualification labels. It establishes no
nuclear, DFT, MLIP or kinetic result.
"""

from __future__ import annotations

import pytest

from matsim_agents.backends.external.spectra_pka import adapter
from matsim_agents.science.acquisition.pka_acquisition import PKACandidate, select_pka_informed
from matsim_agents.science.defects.gas import GasComplexSpec, GasPlacement, generate_gas_complex
from matsim_agents.science.irradiation.contracts import (
    DefectStatus,
    NoCascadeModel,
    ValidationStatus,
)
from matsim_agents.science.irradiation.pka_sampling import sample_pka_events
from matsim_agents.science.irradiation.transmutation import (
    TransmutationScenario,
    apply_transmutation,
)
from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy
from tests.spectra_pka_fixtures import make_campaign, make_config, write_mock_executable

COUNTS = {"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}


def _chain(tmp_path):
    parent = generate_alloy(AlloySpec(counts=COUNTS, repeats=(8, 8, 8), seed=3))
    transmuted, record = apply_transmutation(
        parent,
        "alloy",
        TransmutationScenario(
            scenario_id="s",
            parent_alloy_id="W640Ta128Cr64V96Hf32Mo64",
            exposure_scenario="synthetic-illustration",
            host="W",
            products={"Re": 8, "Os": 2},
            seed=5,
        ),
    )
    spec = GasComplexSpec(
        vacancy_sites=(10, 11),
        gas=(
            GasPlacement(species="He", placement="vacancy", anchor_site=10),
            GasPlacement(species="H", placement="vacancy", anchor_site=11),
        ),
        lattice_parameter=3.165,
    )
    complex_atoms, complex_cfg = generate_gas_complex(transmuted, "transmuted", spec)
    cfg = make_config(tmp_path, write_mock_executable(tmp_path / "spka-mock"))
    run = adapter.run(cfg, make_campaign(tmp_path), execute=True)
    sampled = sample_pka_events(
        run.spectrum, 50, seed=11, lattice_elements=[a.symbol for a in complex_atoms]
    )
    return record, complex_atoms, complex_cfg, run, sampled


def test_chain_composes_with_conserved_counts_and_labels(tmp_path):
    record, atoms, complex_cfg, run, sampled = _chain(tmp_path)
    assert record.final_composition["Re"] == 8 and record.final_composition["Os"] == 2
    assert len(atoms) == 1024 - 2 + 2 == complex_cfg.n_atoms
    assert run.status == "STRUCTURALLY_VALID"
    assert run.spectrum.provenance.numerical_validation is ValidationStatus.NOT_VALIDATED
    assert len(sampled.events) == 50
    assert sampled.excluded_rate_fraction == 0.0


def test_chain_is_deterministic(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _chain(tmp_path / "a")
    b = _chain(tmp_path / "b")
    assert a[0].final_composition == b[0].final_composition
    assert [x.symbol for x in a[1]] == [x.symbol for x in b[1]]
    assert a[4].events == b[4].events


def test_defects_and_ranking_stay_unestablished(tmp_path):
    _, _, _, run, sampled = _chain(tmp_path)
    outcome = NoCascadeModel().surviving_defects(sampled.events[0])
    assert outcome.status is DefectStatus.NOT_ESTABLISHED
    ranking = select_pka_informed(
        run.spectrum,
        [
            PKACandidate(candidate_id="w", species="W182", node_hours=1.0),
            PKACandidate(candidate_id="ta", species="Ta181", node_hours=1.0),
        ],
        2.0,
    )
    assert ranking.surviving_defect_production is DefectStatus.NOT_ESTABLISHED
    assert [s.candidate_id for s in ranking.selected] == ["w", "ta"]
    with pytest.raises(ValueError):
        select_pka_informed(run.spectrum, [], -1.0)
