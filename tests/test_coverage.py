"""Coverage and statistical convergence reports (M18)."""

from __future__ import annotations

import pytest

from matsim_agents.discovery.exploration.coverage import (
    bootstrap_mean_interval,
    convergence_report,
    coverage_report,
    wilson_interval,
)


def test_wilson_matches_known_values_and_edge_cases():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo0, hi0 = wilson_interval(0, 20)
    assert lo0 == 0.0 and hi0 == pytest.approx(0.1611, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson_interval(3, 2)
    narrow = wilson_interval(500, 1000)
    assert narrow[1] - narrow[0] < hi - lo


def test_bootstrap_is_deterministic_and_none_for_tiny_samples():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    a = bootstrap_mean_interval(values, seed=3)
    assert a == bootstrap_mean_interval(values, seed=3)
    assert a[0] < 3.5 < a[1] and a[0] >= 1.0 and a[1] <= 6.0
    assert bootstrap_mean_interval([1.0], seed=3) is None


def test_coverage_lists_unmet_and_unrepresented_strata():
    population = {f"a{i}": "A" for i in range(10)} | {f"b{i}": "B" for i in range(4)}
    population |= {"r0": "rare"}
    sampled = ["a0", "a1", "a2", "b0", "b1", "b1"]  # duplicate id counted once
    report = coverage_report(population, sampled, {"A": 3, "B": 3, "rare": 1, "ghost": 1})
    by = {s.stratum: s for s in report.strata}
    assert by["A"].sampled == 3 and by["A"].met and by["A"].fraction == pytest.approx(0.3)
    assert by["B"].sampled == 2 and not by["B"].met
    assert by["ghost"].population == 0 and not by["ghost"].met
    assert report.unmet == ("B", "ghost", "rare")
    assert report.unrepresented == ("rare",)
    assert not report.complete
    lo, hi = by["A"].fraction_ci
    assert lo < 0.3 < hi


def test_coverage_complete_and_rejects_foreign_ids():
    population = {"x": "A", "y": "A"}
    assert coverage_report(population, ["x"], {"A": 1}).complete
    with pytest.raises(ValueError, match="not in the population"):
        coverage_report(population, ["zzz"], {"A": 1})


def test_convergence_needs_enough_samples_and_separates_uncertainties():
    few = convergence_report("dE", [0.1, 0.2], 0.05)
    assert not few.converged and few.mean is None and "fewer than" in few.note
    tight = convergence_report("dE", [1.0 + 0.001 * (i % 3) for i in range(60)], 0.01)
    assert tight.converged and tight.half_width < 0.01
    loose = convergence_report("dE", [float(i) for i in range(10)], 0.01)
    assert not loose.converged and loose.half_width > 0.01
    for report in (tight, loose):
        assert "not assessed" in report.physical_uncertainty
        assert "not assessed" in report.numerical_uncertainty
