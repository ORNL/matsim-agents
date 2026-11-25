"""DFT/NEB convergence checks. Numbers are synthetic; no calculation is run."""

import math

import pytest

from matsim_agents.workflows.neb_convergence import (
    NebInputs,
    check_barrier_series,
    check_neb_run,
    qualifying_criteria,
)


def _good(**over):
    base = {
        "endpoint_max_force_eV_per_A": (0.01, 0.02),
        "interior_max_error_eV_per_A": 0.03,
        "climbing_image_used": True,
        "n_imaginary_modes_at_saddle": 1,
        "n_images": 7,
    }
    base.update(over)
    return NebInputs(**base)


def test_flat_tail_is_converged():
    out = check_barrier_series("ecut", [30, 40, 50, 60], [0.80, 0.70, 0.705, 0.706])
    assert out.verdict == "converged" and out.last_change_eV < 0.02


def test_one_small_step_is_not_enough():
    out = check_barrier_series("ecut", [30, 40, 50], [0.50, 0.70, 0.705])
    assert out.verdict == "not_converged" and "last changes" in out.reasons[0]


def test_still_moving_is_not_converged():
    out = check_barrier_series("kpts", [2, 3, 4], [0.5, 0.6, 0.7])
    assert out.verdict == "not_converged"


@pytest.mark.parametrize(
    "settings,barriers",
    [
        ([1, 2], [0.1, 0.1]),
        ([1, 2, 3], [0.1, None, 0.1]),
        ([1, 2, 3], [0.1, math.nan, 0.1]),
        ([1, 3, 2], [0.1, 0.1, 0.1]),
        ([1, 2, 2], [0.1, 0.1, 0.1]),
        ([1, 2, 3], [0.1, 0.1]),
    ],
)
def test_bad_series_are_invalid(settings, barriers):
    assert check_barrier_series("p", settings, barriers).verdict == "invalid"


def test_missing_barrier_is_not_treated_as_zero():
    out = check_barrier_series("p", [1, 2, 3], [0.0, 0.0, None])
    assert out.verdict == "invalid" and "not zero" in out.reasons[0]


def test_bad_tolerance_is_rejected():
    with pytest.raises(ValueError):
        check_barrier_series("p", [1, 2, 3], [0.1] * 3, tolerance_eV=0.0)


def test_complete_neb_run_passes():
    assert check_neb_run(_good()).passed


@pytest.mark.parametrize(
    "over,text",
    [
        ({"endpoint_max_force_eV_per_A": (0.2, 0.01)}, "initial endpoint"),
        ({"endpoint_max_force_eV_per_A": (0.01, math.nan)}, "final endpoint"),
        ({"interior_max_error_eV_per_A": None}, "not zero"),
        ({"interior_max_error_eV_per_A": 0.5}, "path not converged"),
        ({"climbing_image_used": False}, "no climbing image"),
        ({"n_imaginary_modes_at_saddle": None}, "not assessed"),
        ({"n_imaginary_modes_at_saddle": 2}, "2 imaginary"),
        ({"n_images": 3}, "3 images"),
    ],
)
def test_each_failure_is_reported(over, text):
    out = check_neb_run(_good(**over))
    assert not out.passed and any(text in f for f in out.failures)


def test_criteria_map_only_to_what_was_checked():
    flat = check_barrier_series("p", [1, 2, 3], [0.1, 0.1, 0.1])
    moving = check_barrier_series("p", [1, 2, 3], [0.1, 0.3, 0.6])
    out = qualifying_criteria(flat, moving, check_neb_run(_good()))
    assert out == {
        "relaxed-endpoints": True,
        "cutoff-convergence": True,
        "kpoint-convergence": False,
        "climbing-image": True,
    }
    failed = qualifying_criteria(flat, flat, check_neb_run(_good(climbing_image_used=False)))
    assert failed["relaxed-endpoints"] is True and failed["climbing-image"] is False
