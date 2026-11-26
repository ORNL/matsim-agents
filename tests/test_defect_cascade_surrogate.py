"""WP3.10 on synthetic data: outcome = a fixed linear function of the inputs plus noise."""

from __future__ import annotations

import math

import pytest

from matsim_agents.core.seeding import numpy_rng
from matsim_agents.workflows.radiation.cascade_surrogate import (
    CascadeFeatureSpec,
    CascadeQuery,
    CascadeSample,
    CascadeTarget,
    evaluate_cascade_surrogate,
    incompatibilities,
    predict_cascades,
    select_cascades_to_run,
    train_cascade_surrogate,
)

SPEC = CascadeFeatureSpec(elements=("W", "Re"), transmutation_species=("Re",))
TARGET = CascadeTarget(
    name="surviving_frenkel_pairs", unit="count", potential_id="pot-A", non_negative=True
)


def _truth(e_log, temp):
    return 60.0 + 15.0 * e_log - 2.0 * temp


def _sample(i, rng, *, e_log=None, temp=None, potential="pot-A", noise=0.2, prefix="s"):
    e_log = rng.uniform(-2.0, -1.0) if e_log is None else e_log
    temp = rng.uniform(0.3, 0.9) if temp is None else temp
    d = rng.normal(size=3)
    return CascadeSample(
        query_id=f"{prefix}{i}",
        composition={"W": 1.0},
        recoil_element="W",
        recoil_energy_mev=10.0**e_log,
        temperature_K=temp * 1000.0,
        direction=tuple(float(x) for x in d),
        value=_truth(e_log, temp) + float(rng.normal(0.0, noise)),
        target="surviving_frenkel_pairs",
        unit="count",
        potential_id=potential,
        simulation_id=f"sim-{prefix}{i}",
    )


def _data(n, seed, **kw):
    rng = numpy_rng(seed, "cascade-surrogate-test", kw.get("prefix", "s"))
    return [_sample(i, rng, **kw) for i in range(n)]


def test_fit_is_accurate_on_independent_rows_and_reports_honestly():
    sur = train_cascade_surrogate(_data(80, 1), TARGET, SPEC, seed=3)
    rep = evaluate_cascade_surrogate(sur, _data(40, 2, prefix="t"))
    assert rep.n_train == 80 and rep.n_test == 40
    assert rep.mae is not None and rep.mae < 1.0
    assert rep.coverage_95 is not None
    assert any("channelling" in item for item in rep.limitations)


def test_training_is_reproducible_for_a_seed():
    a = train_cascade_surrogate(_data(40, 1), TARGET, SPEC, seed=5)
    b = train_cascade_surrogate(_data(40, 1), TARGET, SPEC, seed=5)
    q = _data(5, 9, prefix="q")
    pa, pb = predict_cascades(a, q), predict_cascades(b, q)
    assert [p.value for p in pa] == [p.value for p in pb]
    assert [p.standard_error for p in pa] == [p.standard_error for p in pb]


def test_out_of_distribution_query_gets_no_value():
    sur = train_cascade_surrogate(_data(60, 1), TARGET, SPEC, seed=3)
    rng = numpy_rng(4, "ood")
    far = _sample(0, rng, e_log=1.5, prefix="far")
    near = _sample(1, rng, prefix="near")
    p_far, p_near = predict_cascades(sur, [far, near])
    assert not p_far.in_distribution and p_far.value is None and p_far.standard_error is None
    assert p_near.in_distribution and p_near.value is not None
    assert p_near.standard_error is not None and p_near.standard_error >= 0.0


def test_incompatible_samples_are_excluded_with_reasons():
    mixed = _data(40, 1) + _data(5, 2, potential="pot-B", prefix="b")
    sur = train_cascade_surrogate(mixed, TARGET, SPEC, seed=3)
    assert len(sur.training_ids) == 40
    assert len(sur.excluded) == 5
    assert all("potential" in v[0] for v in sur.excluded.values())
    assert incompatibilities(mixed[0], TARGET) == []


def test_refuses_to_fit_on_too_few_or_overlapping_rows():
    with pytest.raises(ValueError, match="refusing to fit"):
        train_cascade_surrogate(_data(5, 1), TARGET, SPEC, seed=3)
    train = _data(40, 1)
    with pytest.raises(ValueError, match="calibration rows are in the training set"):
        train_cascade_surrogate(train, TARGET, SPEC, seed=3, calibration=train[:6])
    with pytest.raises(ValueError, match="duplicate"):
        train_cascade_surrogate(train + train[:1], TARGET, SPEC, seed=3)


def test_evaluation_rejects_training_rows_in_the_test_set():
    train = _data(40, 1)
    sur = train_cascade_surrogate(train, TARGET, SPEC, seed=3)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_cascade_surrogate(sur, train[:10])


def test_evaluation_withholds_statistics_with_too_few_in_distribution_rows():
    sur = train_cascade_surrogate(_data(40, 1), TARGET, SPEC, seed=3)
    rep = evaluate_cascade_surrogate(sur, _data(3, 7, prefix="t"))
    assert rep.mae is None and rep.rmse is None and rep.coverage_95 is None


def test_calibration_never_shrinks_the_uncertainty():
    sur = train_cascade_surrogate(
        _data(40, 1), TARGET, SPEC, seed=3, calibration=_data(20, 8, prefix="c", noise=3.0)
    )
    assert sur.uncertainty_scale >= 1.0


def test_negative_predictions_are_clipped_and_flagged_for_count_targets():
    train = [s.model_copy(update={"value": s.value - 60.0}) for s in _data(60, 1)]
    sur = train_cascade_surrogate(train, TARGET, SPEC, seed=3)
    preds = predict_cascades(sur, _data(5, 9, prefix="q"))
    assert all(p.value == 0.0 and p.clipped_at_zero for p in preds if p.in_distribution)


def test_query_validation():
    base = dict(
        query_id="q",
        composition={"W": 1.0},
        recoil_element="W",
        recoil_energy_mev=0.1,
        temperature_K=300.0,
        direction=(0.0, 0.0, 1.0),
    )
    CascadeQuery(**base)
    with pytest.raises(ValueError):
        CascadeQuery(**{**base, "direction": (0.0, 0.0, 0.0)})
    with pytest.raises(ValueError):
        CascadeQuery(**{**base, "composition": {"W": 0.5}})
    with pytest.raises(ValueError):
        CascadeQuery(**{**base, "recoil_element": "Re"})
    with pytest.raises(ValueError):
        SPEC.encode(CascadeQuery(**{**base, "composition": {"W": 0.9, "Ta": 0.1}}))


def test_direction_is_normalised_in_the_features():
    q1 = CascadeQuery(
        query_id="a",
        composition={"W": 1.0},
        recoil_element="W",
        recoil_energy_mev=0.1,
        temperature_K=300.0,
        direction=(0.0, 0.0, 1.0),
    )
    q2 = q1.model_copy(update={"query_id": "b", "direction": (0.0, 0.0, 7.0)})
    assert SPEC.encode(q1) == SPEC.encode(q2)
    assert math.isclose(sum(c * c for c in q2.unit_direction()), 1.0)


def test_acquisition_prefers_uncertain_inputs_and_does_not_repeat_one_input():
    sur = train_cascade_surrogate(_data(60, 1), TARGET, SPEC, seed=3)
    rng = numpy_rng(6, "acq")
    proto = _sample(0, rng, e_log=-1.5, temp=0.6, prefix="p")
    dup = [proto.model_copy(update={"query_id": f"dup{i}"}) for i in range(4)]
    other = _sample(0, rng, e_log=-1.1, temp=0.35, prefix="o")
    picks = select_cascades_to_run(sur, [*dup, other], budget_node_hours=2.0, batch_size=2)
    ids = [p.query_id for p in picks]
    assert len(ids) == 2 and "o0" in ids
    assert sum(i.startswith("dup") for i in ids) == 1
    assert ids == [
        p.query_id
        for p in select_cascades_to_run(sur, [*dup, other], budget_node_hours=2.0, batch_size=2)
    ]


def test_acquisition_respects_budget_and_scores_ood_candidates():
    sur = train_cascade_surrogate(_data(60, 1), TARGET, SPEC, seed=3)
    rng = numpy_rng(6, "acq2")
    far = _sample(0, rng, e_log=1.5, prefix="far").model_copy(update={"node_hours": 3.0})
    cheap = [_sample(i, rng, prefix="c").model_copy(update={"query_id": f"c{i}"}) for i in range(3)]
    picks = select_cascades_to_run(sur, [far, *cheap], budget_node_hours=2.5)
    assert sum(p.node_hours for p in picks) <= 2.5
    assert "far0" not in [p.query_id for p in picks]
    picks = select_cascades_to_run(sur, [far], budget_node_hours=5.0)
    assert picks and not picks[0].in_distribution and picks[0].score > 0
    with pytest.raises(ValueError):
        select_cascades_to_run(sur, [far], budget_node_hours=0.0)
