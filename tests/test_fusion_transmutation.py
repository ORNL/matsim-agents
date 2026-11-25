from collections import Counter

import pytest

from matsim_agents.campaign.radiation_properties import REFERENCE_ALLOY_COUNTS
from matsim_agents.fusion.nuclear_evidence import (
    ExternalNuclearEvidence,
    NuclearEvidenceError,
    NuclearFamily,
    NuclearQuantity,
)
from matsim_agents.fusion.transmutation import reference_site_map, transmuted_configuration
from matsim_agents.irradiation.contracts import RecoilSpecies

H = "a" * 64


def _q(el, a, value, name="atom_fraction", unit="1"):
    return NuclearQuantity(
        name=name, value=value, unit=unit, isotope=RecoilSpecies(element=el, mass_number=a)
    )


def _ev(quantities, family=NuclearFamily.TRANSMUTATION, **kw):
    data = dict(
        family=family,
        external_application="FISPACT-II",
        application_version="5.0",
        nuclear_data_library="TENDL-2019",
        input_sha256={"in": H},
        output_sha256={"out": H},
        material_id="W640Ta128Cr64V96Hf32Mo64",
        irradiation_scenario_id="demo-fw",
        quantities=tuple(quantities),
        exposure_time_s=3e7,
        isotope_inventory={"Re187": 1e-2},
        synthetic=True,
    )
    data.update(kw)
    return ExternalNuclearEvidence(**data)


EV = _ev(
    [
        _q("Re", 185, 4e-3),
        _q("Re", 187, 6e-3),  # Re total 1e-2 -> 10.24 -> 10 sites
        _q("Os", 186, 3e-3),  # 3.072 -> 3
        _q("H", 1, 1e-3),
        _q("H", 2, 2e-4),
        _q("H", 3, 1e-4),
        _q("He", 3, 1e-5),
        _q("He", 4, 5e-4),
    ]
)


def test_site_map_is_reference_alloy_with_stable_ids():
    sites = reference_site_map()
    assert len(sites) == 1024
    assert len({s.site_id for s in sites}) == 1024
    assert len({s.frac for s in sites}) == 1024
    assert Counter(s.element for s in sites) == Counter(REFERENCE_ALLOY_COUNTS)
    assert sites == reference_site_map()


def test_substitution_counts_and_composition():
    cfg = transmuted_configuration(EV, seed=3)
    assert cfg.counts == {"Re": 10, "Os": 3}
    assert cfg.expected_counts["Re"] == pytest.approx(10.24)
    assert cfg.composition["W"] == 640 - 13 and cfg.composition["Re"] == 10
    assert sum(cfg.composition.values()) == 1024
    by_id = {s.site_id: s for s in reference_site_map()}
    assert all(by_id[s.site_id].element == "W" for s in cfg.substitutions)
    assert len({s.site_id for s in cfg.substitutions}) == 13


def test_gas_isotopes_stay_separate_and_unplaced():
    gas = transmuted_configuration(EV, seed=0).gas
    assert set(gas.hydrogen) == {"H", "D", "T"}
    assert gas.hydrogen["T"] == pytest.approx(1e-4 * 1024)
    assert set(gas.helium) == {"He-3", "He-4"}
    assert gas.helium["He-3"] == pytest.approx(1e-5 * 1024)
    assert gas.placed is False


def test_deterministic_and_seed_sensitive():
    a, b = transmuted_configuration(EV, seed=1), transmuted_configuration(EV, seed=1)
    assert a.digest() == b.digest() and a.digest().startswith("ftx-")
    assert transmuted_configuration(EV, seed=2).substitutions != a.substitutions


def test_appm_is_converted():
    ev = _ev([_q("Re", 187, 5000.0, name="concentration_appm", unit="appm")])
    cfg = transmuted_configuration(ev, quantity="concentration_appm", seed=0)
    assert cfg.counts == {"Re": 5, "Os": 0}


def test_refusals():
    with pytest.raises(NuclearEvidenceError, match="per-atom"):
        transmuted_configuration(
            _ev([_q("Re", 187, 1.0, name="production_rate", unit="appm/s")]),
            quantity="production_rate",
            seed=0,
        )
    with pytest.raises(NuclearEvidenceError, match="exceed"):
        transmuted_configuration(_ev([_q("Re", 187, 0.7)]), seed=0)
    with pytest.raises(NuclearEvidenceError, match="duplicate"):
        transmuted_configuration(_ev([_q("Re", 187, 1e-3), _q("Re", 187, 2e-3)]), seed=0)
    dpa = _ev(
        [NuclearQuantity(name="dpa", value=1.0, unit="dpa")],
        family=NuclearFamily.PKA_DPA,
        dpa_convention="NRT",
    )
    with pytest.raises(NuclearEvidenceError, match="needs transmutation evidence"):
        transmuted_configuration(dpa, seed=0)
    assert any("SYNTHETIC" in n for n in transmuted_configuration(EV, seed=0).notes)
