"""Software verification of the exposure history on constructed sources and networks."""

import math

import pytest

from matsim_agents.science.defects.cascade_injection import SourceChannel
from matsim_agents.science.defects.exposure_history import (
    ExposureSegment,
    HistoryError,
    run_exposure_history,
)
from matsim_agents.science.defects.source_terms import DefectSource
from matsim_agents.science.kinetics.cluster_dynamics import (
    ClusterNetwork,
    ClusterState,
    SinkTerm,
)

V1 = "V1He0H0@bulk"
V4 = "V4He0H0@bulk"
REF = 1.0e14
CELL = 1000.0


def source(cls="vacancy_cluster", rate=2e-9, temperature=300.0, spectrum="unit-test", mat="W"):
    return DefectSource(
        defect_class=cls,
        rate_per_target_atom_per_s=rate,
        standard_error=0.0,
        mean_yield_per_pka=None,
        total_pka_rate=1.0,
        used_pka_rate=1.0,
        excluded_rate_fraction=0.0,
        extrapolated_rate_fraction=0.0,
        contributions=(),
        spectrum_source=spectrum,
        spectrum_rate_unit="1/atom/s",
        material=mat,
        temperature_K=temperature,
        cascade_provenance="synthetic",
        spectrum_relative_sigma=0.0,
    )


def build(temperature, sink_rate=None):
    net = ClusterNetwork(temperature)
    for n in (1, 4):
        net.add_state(ClusterState(n_vacancies=n, n_helium=0, n_hydrogen=0, environment="bulk"))
    if sink_rate:
        net.add_sink(SinkTerm(state_key=V4, rate_per_s=sink_rate, sink="gb", basis="b"))
    return net


def channel(**kw):
    return SourceChannel(source(**kw), V4, "cascade", "correlated", "A", 4)


def run(segments, channels=None, build_network=build, **kw):
    return run_exposure_history(
        segments,
        build_network,
        [channel()] if channels is None else channels,
        reference_flux_n_cm2_s=REF,
        n_target_atoms=CELL,
        material="W",
        **kw,
    )


def seg(duration=100.0, temperature=300.0, flux=REF, **kw):
    return ExposureSegment(duration, temperature, flux, **kw)


def test_closed_history_injects_exactly_the_integrated_source():
    result = run([seg(100.0), seg(50.0, flux=2 * REF), seg(25.0, flux=0.0)])
    expected = 2e-9 * CELL * (100.0 + 2 * 50.0)
    assert result.final[V4] == pytest.approx(expected)
    assert result.total_injected == {"V": pytest.approx(4 * expected)}
    assert result.segments[2].injected == {}
    assert "zero flux" in result.segments[2].notes[-1]
    assert result.segments[1].flux_scale == 2.0


def test_flux_scaling_is_linear_and_recorded():
    base = run([seg(10.0)]).final[V4]
    doubled = run([seg(10.0, flux=2 * REF)]).final[V4]
    assert doubled == pytest.approx(2 * base)
    assert any("linearly with flux" in a for a in run([seg()]).assumptions)


def test_initial_microstructure_carries_through_and_decays_with_a_sink():
    def with_sink(t):
        return build(t, sink_rate=0.01)

    out = run([seg(100.0, flux=0.0)], build_network=with_sink, initial={V4: 8.0})
    assert out.final[V4] == pytest.approx(8.0 * math.exp(-1.0))


def test_temperature_changes_rebuild_the_network_and_flag_source_temperature():
    seen = []

    def spy(t):
        seen.append(t)
        return build(t)

    result = run([seg(temperature=300.0), seg(temperature=900.0)], build_network=spy)
    assert seen == [300.0, 900.0]
    assert result.segments[0].notes == ()
    assert "computed at 300 K, used at 900 K" in result.segments[1].notes[0]


def test_dpa_channels_need_an_explicit_request_and_are_labelled():
    dpa = SourceChannel(source("frenkel", 1e-9), V1, "dpa", "independent", "B")
    with pytest.raises(HistoryError, match="allow_dpa_reference"):
        run([seg()], channels=[dpa])
    out = run([seg()], channels=[dpa], allow_dpa_reference=True)
    assert out.dpa_reference_used
    assert any("dpa-based" in a for a in out.assumptions)


def test_double_counting_is_refused_per_segment():
    dpa = SourceChannel(source("frenkel", 1e-9), V1, "dpa", "independent", "A")
    both = [channel(), dpa]
    with pytest.raises(HistoryError, match="counted twice|injected twice"):
        run([seg()], channels=both, allow_dpa_reference=True)


def test_spectrum_and_material_mismatches_are_rejected():
    with pytest.raises(HistoryError, match="spectrum"):
        run([seg(spectrum_id="other")])
    run([seg(spectrum_id="unit-test")])
    with pytest.raises(HistoryError, match="not 'W'"):
        run([seg()], channels=[channel(mat="Mo")])


def test_population_in_a_missing_state_is_rejected():
    with pytest.raises(HistoryError, match="unknown states"):
        run([seg()], initial={"V9He0H0@bulk": 1.0})


def test_bad_segments_and_empty_history_are_rejected():
    with pytest.raises(HistoryError, match="duration"):
        seg(0.0)
    with pytest.raises(HistoryError, match="flux"):
        seg(flux=-1.0)
    with pytest.raises(HistoryError, match="at least one"):
        run([])


def test_builder_with_wrong_temperature_is_caught():
    with pytest.raises(HistoryError, match="wrong temperature"):
        run([seg(temperature=300.0)], build_network=lambda t: build(400.0))
