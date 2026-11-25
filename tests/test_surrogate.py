"""Surrogate interface, leakage prevention and active selection (M17)."""

from __future__ import annotations

import pytest

from matsim_agents.workflows.exploration.magnetic_surrogate import (
    LabeledExample,
    LeakageError,
    MagneticSurrogate,
    StratumMeanSurrogate,
    check_leakage,
    evaluate,
    fit_without_leakage,
    select_for_labeling,
    surrogate_or_fallback,
)


def _ex(i, stratum="a", target=0.0, key=None, source="dft"):
    return LabeledExample(f"c{i}", stratum, target, key or f"k{i}", source)


class Recording:
    name = "recording"

    def __init__(self):
        self.seen = None

    def fit(self, examples):
        self.seen = list(examples)

    def predict(self, candidate_id, stratum):
        raise NotImplementedError


def test_leakage_by_id_and_by_content_is_rejected_and_nothing_is_fitted():
    train = [_ex(1), _ex(2)]
    with pytest.raises(LeakageError, match="shared candidate ids"):
        check_leakage(train, [_ex(1, key="other")])
    with pytest.raises(LeakageError, match="structure or symmetry keys"):
        check_leakage(train, [_ex(9, key="k2")])
    model = Recording()
    with pytest.raises(LeakageError):
        fit_without_leakage(model, train, [_ex(9, key="k2")])
    assert model.seen is None
    fit_without_leakage(model, train, [_ex(7), _ex(8)])
    assert [e.candidate_id for e in model.seen] == ["c1", "c2"]


def test_fallback_means_stds_and_unseen_stratum_uncertainty():
    model = StratumMeanSurrogate()
    with pytest.raises(RuntimeError):
        model.predict("x", "a")
    with pytest.raises(ValueError):
        model.fit([])
    model.fit([_ex(1, "a", 1.0), _ex(2, "a", 3.0), _ex(3, "b", 10.0)])
    a = model.predict("z", "a")
    assert a.mean == pytest.approx(2.0) and a.std == pytest.approx(2**0.5)
    assert model.predict("z", "b").std == 1.0  # single sample: maximally uncertain, not zero
    unseen = model.predict("z", "never")
    assert unseen.std > a.std and unseen.mean == pytest.approx(14 / 3)
    assert isinstance(model, MagneticSurrogate)


def test_accuracy_is_withheld_for_small_or_synthetic_validation():
    model = StratumMeanSurrogate()
    model.fit([_ex(1, "a", 1.0), _ex(2, "a", 3.0)])
    small = evaluate(model, [_ex(i, "a", 2.0) for i in range(5)])
    assert small.rmse is None and small.mae is None and "fewer than" in small.note
    synthetic = evaluate(model, [_ex(i, "a", 2.0, source="synthetic") for i in range(20)])
    assert synthetic.rmse is None and "not DFT evidence" in synthetic.note


def test_evaluation_on_sufficient_dft_validation_matches_hand_computation():
    model = StratumMeanSurrogate()
    model.fit([_ex(1, "a", 1.0), _ex(2, "a", 3.0)])  # predicts 2.0 for stratum a
    validation = [_ex(10 + i, "a", 2.0 + (1.0 if i % 2 else -1.0)) for i in range(10)]
    result = evaluate(model, validation)
    assert result.n == 10 and result.rmse == pytest.approx(1.0) and result.mae == pytest.approx(1.0)


def test_active_selection_prefers_uncertain_strata_and_is_deterministic():
    model = StratumMeanSurrogate()
    model.fit([_ex(i, "known", float(i % 2)) for i in range(20)])
    pool = {f"p{i}": ("novel" if i < 4 else "known") for i in range(40)}
    picks = select_for_labeling(model, pool, 8, seed=1, round_index=0, exploration_fraction=0.0)
    assert {s.candidate_id for s in picks if pool[s.candidate_id] == "novel"} == {
        "p0", "p1", "p2", "p3"
    }  # fmt: skip
    assert picks == select_for_labeling(
        model, pool, 8, seed=1, round_index=0, exploration_fraction=0.0
    )


def test_deterministic_fallback_replaces_a_missing_model():
    chosen = surrogate_or_fallback(None, [_ex(1, "a", 1.0), _ex(2, "a", 2.0)])
    assert chosen.name == "stratum_mean" and chosen.predict("x", "a").mean == pytest.approx(1.5)
    custom = Recording()
    assert surrogate_or_fallback(custom, []) is custom
