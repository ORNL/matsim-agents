import math

import pytest

from matsim_agents.science.defects.ensemble import EnsembleEstimate, weighted_estimate
from matsim_agents.science.defects.source_terms import (
    YieldBin,
    YieldTable,
    cluster_source_term,
    pka_weighted_source,
)
from matsim_agents.science.irradiation.contracts import (
    RATE_BASIS_PER_ATOM_PER_S,
    ContractError,
    IrradiationScenario,
    PKASourceProvenance,
    PKASpectrum,
    RecoilSpecies,
    SpeciesSpectrum,
)


def _est(mean, se=0.0):
    return EnsembleEstimate(
        mean=mean,
        standard_error=se,
        ci_low=mean - 2 * se,
        ci_high=mean + 2 * se,
        confidence=0.95,
        method="t",
        n=10,
        effective_n=10.0,
        seed=None,
    )


def _spectrum(spectra, basis=RATE_BASIS_PER_ATOM_PER_S):
    return PKASpectrum(
        scenario=IrradiationScenario(
            scenario_id="s",
            parent_alloy_id="W",
            composition={"W": 1.0},
            neutron_spectrum_identity="test",
            flux_normalization="test",
        ),
        spectra=tuple(spectra),
        rate_unit="1/atom/s",
        rate_basis=basis,
        provenance=PKASourceProvenance(source="unit-test", nuclear_data_identity="none"),
    )


def _w(lows, highs, rates):
    return SpeciesSpectrum(
        species=RecoilSpecies(element="W"),
        bin_low_mev=tuple(lows),
        bin_high_mev=tuple(highs),
        rates=tuple(rates),
    )


def _table(bins):
    return YieldTable(
        defect_class="frenkel_pair",
        bins=tuple(bins),
        material="W bcc",
        temperature_K=300.0,
        cascade_provenance="test cascades",
    )


def test_rate_is_sum_of_rate_times_yield():
    spec = _spectrum([_w([0.0, 0.1], [0.1, 0.2], [2.0, 3.0])])
    yields = _table([YieldBin("W", 0.0, 0.1, _est(10.0)), YieldBin("W", 0.1, 0.2, _est(40.0))])
    src = pka_weighted_source(spec, yields)
    assert src.rate_per_target_atom_per_s == pytest.approx(2.0 * 10.0 + 3.0 * 40.0)
    assert src.mean_yield_per_pka == pytest.approx(140.0 / 5.0)
    assert src.excluded_rate_fraction == 0.0 and src.extrapolated_rate_fraction == 0.0


def test_uncertainty_propagation():
    spec = _spectrum([_w([0.0, 0.1], [0.1, 0.2], [2.0, 3.0])])
    yields = _table([YieldBin("W", 0.0, 0.2, _est(10.0, 1.0))])
    src = pka_weighted_source(spec, yields)
    assert src.standard_error == pytest.approx(math.sqrt((2.0 * 1.0) ** 2 + (3.0 * 1.0) ** 2))
    with_spec = pka_weighted_source(spec, yields, spectrum_relative_sigma=0.1)
    expected = math.sqrt(
        (2.0 * 1.0) ** 2 + (20.0 * 0.1) ** 2 + (3.0 * 1.0) ** 2 + (30.0 * 0.1) ** 2
    )
    assert with_spec.standard_error == pytest.approx(expected)
    assert any("spectrum" in item for item in src.not_propagated)
    assert not any("spectrum uncertainty" in item for item in with_spec.not_propagated)


def test_uncovered_bins_error_by_default():
    spec = _spectrum([_w([0.0, 0.1], [0.1, 0.2], [2.0, 3.0])])
    yields = _table([YieldBin("W", 0.0, 0.1, _est(10.0))])
    with pytest.raises(ContractError):
        pka_weighted_source(spec, yields)


def test_exclude_reports_the_dropped_rate():
    spec = _spectrum([_w([0.0, 0.1], [0.1, 0.2], [2.0, 3.0])])
    yields = _table([YieldBin("W", 0.0, 0.1, _est(10.0))])
    src = pka_weighted_source(spec, yields, on_uncovered="exclude")
    assert src.rate_per_target_atom_per_s == pytest.approx(20.0)
    assert src.excluded_rate_fraction == pytest.approx(3.0 / 5.0)
    assert src.used_pka_rate == pytest.approx(2.0)


def test_hold_nearest_is_labelled_extrapolated():
    spec = _spectrum([_w([0.0, 0.1], [0.1, 0.2], [2.0, 3.0])])
    yields = _table([YieldBin("W", 0.0, 0.1, _est(10.0))])
    src = pka_weighted_source(spec, yields, on_uncovered="hold_nearest")
    assert src.rate_per_target_atom_per_s == pytest.approx(50.0)
    assert src.extrapolated_rate_fraction == pytest.approx(3.0 / 5.0)
    assert [c.extrapolated for c in src.contributions] == [False, True]


def test_species_without_yield_is_not_borrowed_from_another():
    ta = SpeciesSpectrum(
        species=RecoilSpecies(element="Ta"),
        bin_low_mev=(0.0,),
        bin_high_mev=(0.1,),
        rates=(1.0,),
    )
    spec = _spectrum([_w([0.0], [0.1], [2.0]), ta])
    yields = _table([YieldBin("W", 0.0, 0.1, _est(10.0))])
    with pytest.raises(ContractError):
        pka_weighted_source(spec, yields)
    src = pka_weighted_source(spec, yields, on_uncovered="hold_nearest")
    assert src.extrapolated_rate_fraction == 0.0
    assert src.excluded_rate_fraction == pytest.approx(1.0 / 3.0)


def test_rate_basis_must_be_per_atom_per_second():
    spec = _spectrum([_w([0.0], [0.1], [2.0])], basis="unspecified")
    with pytest.raises(ContractError):
        pka_weighted_source(spec, _table([YieldBin("W", 0.0, 0.1, _est(1.0))]))


def test_yield_from_real_ensemble_and_unit_scaling():
    est = weighted_estimate([4.0, 6.0, 5.0, 5.0])
    spec = _spectrum([_w([0.0], [0.1], [2.0])])
    src = pka_weighted_source(spec, _table([YieldBin("W", 0.0, 0.1, est)]))
    assert src.rate_per_target_atom_per_s == pytest.approx(10.0)
    vol = src.per_volume(6.3e28)
    assert vol.value == pytest.approx(10.0 * 6.3e28)
    assert vol.unit == "defects per m^3 per second"
    dpa = src.per_dpa(2.0)
    assert dpa.value == pytest.approx(5.0)
    assert dpa.unit == "defects per target atom per dpa"
    with pytest.raises(ValueError):
        src.per_volume(0.0)


def test_cluster_source_term_scales_with_cell_size_and_records_basis():
    spec = _spectrum([_w([0.0], [0.1], [2.0])])
    src = pka_weighted_source(spec, _table([YieldBin("W", 0.0, 0.1, _est(5.0))]))
    term = cluster_source_term(src, state_key="V1", n_target_atoms=100.0, reservoir="irradiation")
    assert term.rate_per_s == pytest.approx(1000.0)
    assert "100 target atoms" in term.basis and "frenkel_pair" in term.basis


def test_yield_table_validation():
    with pytest.raises(ValueError):
        _table([YieldBin("W", 0.0, 0.2, _est(1.0)), YieldBin("W", 0.1, 0.3, _est(1.0))])
    with pytest.raises(ValueError):
        YieldBin("W", 0.2, 0.1, _est(1.0))
    with pytest.raises(ValueError):
        YieldTable("x", (), "m", 300.0, "p")
    with pytest.raises(ValueError):
        YieldTable("x", (YieldBin("W", 0.0, 0.1, _est(1.0)),), "m", 300.0, " ")
