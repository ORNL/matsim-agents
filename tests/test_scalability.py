"""E5: toy-scale scaling, seeded reproducibility and environment record."""

import pytest

from matsim_agents.kinetics.benchmarks import chain_network
from matsim_agents.kinetics.ensemble import run_ensemble
from matsim_agents.workflows.scalability import (
    check_reproducibility,
    digest,
    environment_record,
    measure_scaling,
)


def _ensemble(seed):
    net, start = chain_network(6)
    return run_ensemble(net, start, 300.0, n_trajectories=25, seed=seed, require="exploratory")


def test_chain_event_count_grows_linearly_with_size():
    report = measure_scaling([8, 16, 32, 64], seed=1, n_trajectories=5)
    assert [p.mean_events for p in report.points] == [7.0, 15.0, 31.0, 63.0]
    assert report.events_exponent == pytest.approx(1.0, abs=0.1)
    assert all(p.wall_s > 0 and p.peak_bytes > 0 for p in report.points)


def test_scaling_supports_nothing_and_says_why():
    report = measure_scaling([4, 8, 16], seed=1, n_trajectories=3)
    assert report.supports == ()
    assert any("not atoms" in c for c in report.caveats)


@pytest.mark.parametrize("sizes", [[4, 8], [8, 4, 16], [4, 4, 8]])
def test_bad_sizes_are_rejected(sizes):
    with pytest.raises(ValueError):
        measure_scaling(sizes, seed=1)


def test_seeded_ensemble_is_bitwise_reproducible_and_seed_sensitive():
    report = check_reproducibility(_ensemble, seed=9, repeats=3)
    assert report.identical and report.seed_sensitive and report.verdict == "pass"
    assert len(set(report.digests)) == 1 and report.control_digest != report.digests[0]
    assert report.supports == ("bitwise-seeded",)


def test_unseeded_function_is_caught():
    counter = iter(range(100))
    report = check_reproducibility(lambda seed: next(counter), seed=1)
    assert not report.identical and report.verdict == "fail" and report.supports == ()


def test_seed_insensitive_function_is_caught():
    report = check_reproducibility(lambda seed: {"fixed": 1}, seed=1)
    assert report.identical and not report.seed_sensitive and report.verdict == "fail"


def test_repeats_must_be_at_least_two():
    with pytest.raises(ValueError):
        check_reproducibility(_ensemble, seed=1, repeats=1)


def test_digest_is_order_independent_for_dicts():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def test_environment_without_lock_is_not_pinned(tmp_path):
    rec = environment_record(tmp_path)
    assert rec.pinned is False and rec.lock_file is None and rec.supports == ()
    assert any("not pinned" in c for c in rec.caveats)
    assert rec.packages["numpy"] is not None


def test_environment_with_lock_is_pinned(tmp_path):
    (tmp_path / "uv.lock").write_text("# synthetic fixture\n")
    rec = environment_record(tmp_path)
    assert rec.pinned and rec.lock_file == "uv.lock" and rec.supports == ("environment-pinned",)
    assert rec.as_dict()["pinned"] is True
