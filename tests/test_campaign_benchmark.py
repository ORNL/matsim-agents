from __future__ import annotations

import pytest

from matsim_agents.campaign.benchmark import (
    BenchmarkArm,
    BenchmarkMetric,
    BenchmarkObservation,
    BenchmarkProtocol,
    compare_paired_metric,
    paired_metric_differences,
)
from matsim_agents.execution.contracts import ComputeBudget


def _protocol() -> BenchmarkProtocol:
    return BenchmarkProtocol(
        protocol_id="nb-ta-o-v1",
        element_set=["Nb", "Ta", "O"],
        candidate_pool_regime="fixed",
        candidate_pool_manifest="sha256:candidates-v1",
        initial_model_identifier="uma-s-1p1",
        reference_set_identifier="nb-ta-o-pbe-v1",
        dft_method_signature="qe-pbe-v1",
        budget=ComputeBudget(max_dft_calculations=60, max_node_hours=120.0),
        random_seeds=[101, 202, 303],
        arms=[BenchmarkArm.RANDOM, BenchmarkArm.ADAPTIVE],
        metrics=[
            BenchmarkMetric(
                name="near_hull_per_dft",
                direction="maximize",
                success_threshold=0.1,
            )
        ],
        stopping_rule_identifier="budget-or-three-stale-iterations",
        decoding_parameters={"temperature": 0.2},
    )


def test_benchmark_protocol_builds_paired_frozen_run_matrix():
    protocol = _protocol()
    matrix = protocol.run_matrix()

    assert len(matrix) == 6
    assert {item.random_seed for item in matrix} == {101, 202, 303}
    assert all(item.protocol_digest == protocol.digest() for item in matrix)
    assert all(item.budget == protocol.budget for item in matrix)
    assert len({item.run_id for item in matrix}) == len(matrix)


def test_benchmark_requires_independent_replicates_and_hard_budget():
    values = _protocol().model_dump()
    values["random_seeds"] = [101]
    with pytest.raises(ValueError, match="at least 2 items"):
        BenchmarkProtocol.model_validate(values)

    values = _protocol().model_dump()
    values["budget"] = ComputeBudget()
    with pytest.raises(ValueError, match="hard DFT or node-hour budget"):
        BenchmarkProtocol.model_validate(values)


def test_paired_metric_differences_match_seed_and_protocol():
    digest = _protocol().digest()
    observations = []
    for seed, random_value, adaptive_value in [(101, 0.1, 0.3), (202, 0.2, 0.25)]:
        observations.extend(
            [
                BenchmarkObservation(
                    protocol_digest=digest,
                    arm=BenchmarkArm.RANDOM,
                    random_seed=seed,
                    metrics={"near_hull_per_dft": random_value},
                    dft_calculations_attempted=60,
                    node_hours_consumed=100.0,
                    completed=True,
                ),
                BenchmarkObservation(
                    protocol_digest=digest,
                    arm=BenchmarkArm.ADAPTIVE,
                    random_seed=seed,
                    metrics={"near_hull_per_dft": adaptive_value},
                    dft_calculations_attempted=60,
                    node_hours_consumed=100.0,
                    completed=True,
                ),
            ]
        )

    assert paired_metric_differences(
        observations,
        treatment=BenchmarkArm.ADAPTIVE,
        control=BenchmarkArm.RANDOM,
        metric="near_hull_per_dft",
    ) == pytest.approx([0.2, 0.05])

    comparison = compare_paired_metric(
        observations,
        treatment=BenchmarkArm.ADAPTIVE,
        control=BenchmarkArm.RANDOM,
        metric=BenchmarkMetric(
            name="near_hull_per_dft",
            direction="maximize",
            success_threshold=0.1,
        ),
        bootstrap_samples=1000,
        random_seed=7,
    )
    assert comparison.n_pairs == 2
    assert comparison.mean_difference == pytest.approx(0.125)
    assert comparison.confidence_interval == pytest.approx((0.05, 0.2))
    assert comparison.favorable
