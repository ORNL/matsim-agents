"""MP5: transferable property surrogates on synthetic transfer datasets."""

from __future__ import annotations

import pytest

from matsim_agents.campaign.property_surrogates import (
    SourceRecord,
    TargetSpec,
    evaluate_transfer,
    incompatibilities,
    predict_property_evidence,
    split_by_compatibility,
    train_surrogate,
)
from matsim_agents.execution.properties import PropertyConditions, PropertyStatus
from matsim_agents.seeding import numpy_rng

SIG = "qe-pbe-synthetic"
TARGET = TargetSpec(
    property="thermal_conductivity_total",
    method_signature=SIG,
    temperature_K=1000.0,
    defect_state=None,
)


def make(n, *, seed, offset=0, start=0, order=3, shift=0.0, **kw):
    """Synthetic: value = 3 a + 2 b - c + 10 (+ noise); explicitly synthetic."""
    rng = numpy_rng(seed, "synthetic-property-data", offset)
    rows = []
    for i in range(n):
        f = rng.uniform(0, 1, 3) + shift
        base = dict(
            record_id=f"r{start + i}",
            property="thermal_conductivity_total",
            unit="W/(m K)",
            method_signature=SIG,
            temperature_K=1000.0,
            defect_state=None,
            composition_id=f"c{(start + i) % 7}",
            ordering_id=f"o{i % 3}",
            n_components=order,
            features=list(f),
            value=float(3 * f[0] + 2 * f[1] - f[2] + 10 + rng.normal(0, 0.01)),
            dataset_id="synthetic",
        )
        base.update(kw)
        rows.append(SourceRecord(**base))
    return rows


def test_incompatible_records_are_excluded_with_reasons():
    good = make(1, seed=0)[0]
    bad = [
        good.model_copy(update={"record_id": "b1", "method_signature": "vasp-other"}),
        good.model_copy(update={"record_id": "b2", "temperature_K": 300.0}),
        good.model_copy(update={"record_id": "b3", "defect_state": "vacancy"}),
        good.model_copy(update={"record_id": "b4", "unit": "mW/(cm K)"}),
        good.model_copy(update={"record_id": "b5", "property": "bulk_modulus", "unit": "GPa"}),
        good.model_copy(update={"record_id": "b6", "value": float("nan")}),
    ]
    keep, excluded = split_by_compatibility([good, *bad], TARGET)
    assert [r.record_id for r in keep] == [good.record_id]
    assert set(excluded) == {f"b{i}" for i in range(1, 7)}
    assert "method" in excluded["b1"][0] and "temperature" in excluded["b2"][0]
    assert incompatibilities(good, TARGET) == []


def test_surrogate_recovers_a_linear_property_on_an_independent_test_set():
    train = make(60, seed=1, start=0)
    test = make(30, seed=2, offset=1, start=1000)
    surrogate = train_surrogate(train, TARGET, seed=0)
    report = evaluate_transfer(surrogate, test)
    assert report.mae < 0.05
    assert report.n_test == 30 and report.independent
    assert set(report.mae_by_composition) and set(report.mae_by_temperature) == {"1000.0"}
    assert report.negative_transfer is None


def test_refuses_overlapping_test_rows_and_too_little_data():
    train = make(30, seed=1)
    surrogate = train_surrogate(train, TARGET, seed=0)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_transfer(surrogate, train[:20])
    with pytest.raises(ValueError, match="compatible training"):
        train_surrogate(train[:3], TARGET, seed=0)
    with pytest.raises(ValueError, match="not evaluable"):
        evaluate_transfer(surrogate, make(3, seed=5, start=500))


def test_incompatible_test_rows_are_counted_not_scored():
    train = make(40, seed=1)
    surrogate = train_surrogate(train, TARGET, seed=0)
    test = make(15, seed=3, start=2000) + make(5, seed=4, start=3000, temperature_K=300.0)
    report = evaluate_transfer(surrogate, test)
    assert report.n_test == 15 and report.n_excluded_incompatible == 5


def test_out_of_distribution_query_is_refused_not_extrapolated():
    surrogate = train_surrogate(make(50, seed=1), TARGET, seed=0)
    inside = predict_property_evidence(
        surrogate,
        structure_id="s",
        composition="W",
        features=[0.5, 0.5, 0.5],
        conditions=PropertyConditions(temperature_K=1000.0),
    )
    assert inside.status == PropertyStatus.SURROGATE_PREDICTED
    assert inside.value == pytest.approx(3 * 0.5 + 2 * 0.5 - 0.5 + 10, abs=0.1)
    assert inside.uncertainty is not None and inside.evidence_ids[0].startswith("surrogate-")
    outside = predict_property_evidence(
        surrogate,
        structure_id="s",
        composition="W",
        features=[40.0, -30.0, 25.0],
        conditions=PropertyConditions(temperature_K=1000.0),
    )
    assert outside.status == PropertyStatus.UNSUPPORTED and outside.value is None
    assert "out of distribution" in outside.reason


def test_wrong_temperature_query_is_an_error():
    surrogate = train_surrogate(make(30, seed=1), TARGET, seed=0)
    with pytest.raises(ValueError, match="different temperature"):
        predict_property_evidence(
            surrogate,
            structure_id="s",
            composition="W",
            features=[0.5] * 3,
            conditions=PropertyConditions(temperature_K=300.0),
        )


def test_negative_transfer_is_detected_against_a_target_only_model():
    # Source rows follow a different law than the target rows.
    source = [
        r.model_copy(update={"value": -r.value + 40.0, "n_components": 2})
        for r in make(80, seed=1, start=0, order=2)
    ]
    target_train = make(12, seed=2, offset=1, start=500)
    test = make(30, seed=3, offset=2, start=1000)
    transferred = train_surrogate([*source, *target_train], TARGET, seed=0)
    target_only = train_surrogate(target_train, TARGET, seed=0)
    report = evaluate_transfer(transferred, test, target_only=target_only)
    assert report.negative_transfer is True
    assert report.mae > report.target_only_mae
    assert any("Negative transfer" in x for x in report.limitations)


def test_calibration_inflates_an_overconfident_uncertainty():
    train = make(60, seed=1)
    noisy_cal = [
        r.model_copy(update={"value": r.value + (1.0 if i % 2 else -1.0)})
        for i, r in enumerate(make(40, seed=7, offset=3, start=4000))
    ]
    surrogate = train_surrogate(train, TARGET, seed=0, calibration=noisy_cal)
    assert surrogate.uncertainty_scale > 1.0
    with pytest.raises(ValueError, match="in the training set"):
        train_surrogate(train, TARGET, seed=0, calibration=train[:10])
