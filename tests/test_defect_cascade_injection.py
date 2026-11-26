import pytest

from matsim_agents.science.defects.cascade_injection import (
    InjectionError,
    SourceChannel,
    check_channels,
    inject_sources,
)
from matsim_agents.science.defects.source_terms import DefectSource
from matsim_agents.science.kinetics.cluster_dynamics import ClusterNetwork, ClusterState


def _source(defect_class, rate):
    return DefectSource(
        defect_class=defect_class,
        rate_per_target_atom_per_s=rate,
        standard_error=0.0,
        mean_yield_per_pka=None,
        total_pka_rate=1.0,
        used_pka_rate=1.0,
        excluded_rate_fraction=0.0,
        extrapolated_rate_fraction=0.0,
        contributions=(),
        spectrum_source="unit-test",
        spectrum_rate_unit="1/atom/s",
        material="W",
        temperature_K=300.0,
        cascade_provenance="synthetic",
        spectrum_relative_sigma=0.0,
    )


def _net():
    net = ClusterNetwork(temperature_K=300.0)
    for n in (1, 4):
        net.add_state(ClusterState(n_vacancies=n, n_helium=0, n_hydrogen=0, environment="bulk"))
    return net


V1 = "V1He0H0@bulk"
V4 = "V4He0H0@bulk"


def test_cluster_channel_adds_term_scaled_by_cell_size():
    net = _net()
    ch = SourceChannel(_source("vacancy_cluster", 2e-9), V4, "cascade", "correlated", "A", 4)
    rep = inject_sources(net, [ch], n_target_atoms=1000, reservoir="cascades")
    assert net.sources[0].rate_per_s == pytest.approx(2e-6)
    assert rep.species_rate_per_s == {"V": pytest.approx(8e-6)}


def test_independent_channel_rejects_multi_defect_state():
    net = _net()
    ch = SourceChannel(_source("frenkel", 1e-9), V4, "dpa", "independent", "A")
    with pytest.raises(InjectionError, match="one at a time"):
        check_channels(net, [ch])


def test_cluster_size_mismatch_is_rejected():
    net = _net()
    ch = SourceChannel(_source("c", 1e-9), V4, "cascade", "correlated", "A", cluster_size=3)
    with pytest.raises(InjectionError, match="declares cluster size"):
        check_channels(net, [ch])


def test_same_origin_cascade_and_dpa_on_same_species_is_double_count():
    net = _net()
    chans = [
        SourceChannel(_source("cluster", 1e-9), V4, "cascade", "correlated", "A"),
        SourceChannel(_source("frenkel", 1e-9), V1, "dpa", "independent", "A"),
    ]
    with pytest.raises(InjectionError, match="counted twice"):
        check_channels(net, chans)


def test_different_origins_may_use_different_kinds():
    net = _net()
    chans = [
        SourceChannel(_source("cluster", 1e-9), V4, "cascade", "correlated", "A"),
        SourceChannel(_source("frenkel", 1e-9), V1, "dpa", "independent", "B"),
    ]
    check_channels(net, chans)


def test_duplicate_defect_class_from_one_origin_is_rejected():
    net = _net()
    chans = [
        SourceChannel(_source("cluster", 1e-9), V4, "cascade", "correlated", "A"),
        SourceChannel(_source("cluster", 2e-9), V4, "cascade", "correlated", "A"),
    ]
    with pytest.raises(InjectionError, match="twice"):
        check_channels(net, chans)


def test_failed_validation_adds_no_terms():
    net = _net()
    chans = [
        SourceChannel(_source("cluster", 1e-9), V4, "cascade", "correlated", "A"),
        SourceChannel(_source("frenkel", 1e-9), V4, "dpa", "independent", "B"),
    ]
    with pytest.raises(InjectionError):
        inject_sources(net, chans, n_target_atoms=10, reservoir="r")
    assert net.sources == ()


def test_unknown_state_is_rejected():
    net = _net()
    ch = SourceChannel(_source("c", 1e-9), "V9He0H0@bulk", "cascade", "correlated", "A")
    with pytest.raises(ValueError):
        check_channels(net, [ch])


def test_injected_source_drives_populations_with_inventory_balance():
    net = _net()
    ch = SourceChannel(_source("cluster", 1e-3), V4, "cascade", "correlated", "A", 4)
    inject_sources(net, [ch], n_target_atoms=1.0, reservoir="cascades")
    pops = net.evolve({V1: 0.0, V4: 0.0}, 100.0)
    assert pops[V4] == pytest.approx(0.1)
    assert net.inventory(pops)["V"] == pytest.approx(0.4)
