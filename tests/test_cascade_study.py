import numpy as np
import pytest

from matsim_agents.science.cascade.damage_functions import (
    arc_dpa_displacements,
    arc_dpa_efficiency,
    nrt_displacements,
)
from matsim_agents.workflows.radiation.cascade_study import (
    build_cell,
    choose_pka,
    displacement_probability,
    plan_threshold,
    shell_mask,
    threshold_estimates,
)


def test_nrt_regimes():
    assert nrt_displacements(30.0, 40.0) == 0.0
    assert nrt_displacements(60.0, 40.0) == 1.0
    assert nrt_displacements(1000.0, 40.0) == pytest.approx(0.8 * 1000.0 / 80.0)
    with pytest.raises(ValueError):
        nrt_displacements(10.0, 0.0)


def test_arc_dpa_is_continuous_at_the_knee_and_reduces_above_it():
    ed, b, c = 40.0, -0.568, 0.286
    knee = 2 * ed / 0.8
    assert arc_dpa_efficiency(knee, ed, b, c) == pytest.approx(1.0)
    assert arc_dpa_efficiency(1e5, ed, b, c) < 0.5
    assert arc_dpa_displacements(1e5, ed, b, c) < nrt_displacements(1e5, ed)


def test_build_cell_alloy_counts_are_exact_and_seeded():
    a = build_cell("Fe", 2.87, 3, {"Fe": 30, "Cr": 24}, seed=4)
    b = build_cell("Fe", 2.87, 3, {"Fe": 30, "Cr": 24}, seed=4)
    assert a.get_chemical_symbols().count("Cr") == 24
    assert a.get_chemical_symbols() == b.get_chemical_symbols()
    with pytest.raises(ValueError, match="sum"):
        build_cell("Fe", 2.87, 3, {"Fe": 10}, seed=1)


def test_shell_and_pka_selection():
    atoms = build_cell("Fe", 2.87, 6, None, 0)
    mask = shell_mask(atoms)
    assert 0.1 < mask.mean() < 0.5
    pka = choose_pka(atoms, "Fe", np.random.default_rng(0))
    frac = atoms.get_scaled_positions(wrap=True)[pka]
    assert np.all(np.abs(frac - 0.5) < 0.2)
    assert not mask[pka]


def test_plan_threshold_is_complete_and_has_distinct_seeds():
    tasks = plan_threshold("Fe", 2.87, [20.0, 30.0], ["100", "111"], 3, "zbl")
    assert len(tasks) == 12
    assert len({t["seed"] for t in tasks}) == 12
    assert [t["id"] for t in tasks] == list(range(12))


def _rec(e, fp):
    return {"energy_eV": e, "n_frenkel_pairs": fp}


def test_threshold_estimates_flags_the_grid_limits():
    recs = [_rec(10.0, 0)] * 3 + [_rec(20.0, 1), _rec(20.0, 0), _rec(20.0, 1)] + [_rec(30.0, 2)] * 3
    prob = displacement_probability(recs)
    assert prob[20.0] == (pytest.approx(2 / 3), 3)
    est = threshold_estimates(prob)
    assert est["energy_at_p_half_eV"] == 20.0
    assert est["grid_starts_below_all_thresholds"] and est["grid_ends_above_all_thresholds"]
    assert est["monotonic"]
    assert est["mean_threshold_eV"] == pytest.approx(10.0 + 25.0 / 3.0)

    open_grid = threshold_estimates(displacement_probability([_rec(10.0, 1), _rec(20.0, 1)]))
    assert not open_grid["grid_starts_below_all_thresholds"]
