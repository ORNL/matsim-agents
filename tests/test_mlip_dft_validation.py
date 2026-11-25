"""MLIP validation protocol. All data are synthetic arrays; nothing here validates a model."""

import json

import numpy as np
import pytest

from matsim_agents.core.seeding import numpy_rng
from matsim_agents.workflows.mlip_dft_validation import (
    ValidationThresholds,
    evaluate_validation,
    write_validation_report,
)

TH = ValidationThresholds(n_bootstrap=200, min_samples=20)


def _pair(n, scale_eV, seed=3):
    rng = numpy_rng(seed, "test")
    ref = rng.normal(size=n)
    return ref + rng.normal(scale=scale_eV, size=n), ref


def _run(**kw):
    base = {
        "model_id": "m",
        "reference_id": "r",
        "held_out": True,
        "synthetic": True,
        "seed": 1,
        "thresholds": TH,
    }
    base.update(kw)
    return evaluate_validation(**base)


def test_small_errors_pass_and_report_ci_and_threshold():
    pred, ref = _pair(200, 0.002)
    out = _run(energy_pred_eV_per_atom=pred, energy_ref_eV_per_atom=ref)
    assert out.verdict == "pass" and out.synthetic
    metric = out.metrics[0]
    assert metric.ci_low <= metric.mean <= metric.ci_high <= metric.threshold
    assert metric.maximum >= metric.mean and metric.n == 200


def test_large_errors_fail_with_a_reason():
    pred, ref = _pair(200, 0.3)
    out = _run(force_pred_eV_per_A=pred, force_ref_eV_per_A=ref)
    assert out.verdict == "fail" and "force_mae" in out.reasons[0]


def test_a_small_sample_is_insufficient_even_if_errors_are_tiny():
    pred, ref = _pair(5, 0.0001)
    out = _run(energy_pred_eV_per_atom=pred, energy_ref_eV_per_atom=ref)
    assert out.verdict == "insufficient" and out.metrics == ()


def test_training_data_cannot_validate_a_model():
    pred, ref = _pair(200, 0.001)
    out = _run(energy_pred_eV_per_atom=pred, energy_ref_eV_per_atom=ref, held_out=False)
    assert out.verdict == "insufficient" and "held out" in out.reasons[0]


def test_non_finite_and_mismatched_inputs_are_invalid():
    pred, ref = _pair(50, 0.001)
    bad = pred.copy()
    bad[3] = np.nan
    assert _run(energy_pred_eV_per_atom=bad, energy_ref_eV_per_atom=ref).verdict == "invalid"
    mismatch = _run(energy_pred_eV_per_atom=pred[:-1], energy_ref_eV_per_atom=ref)
    assert mismatch.verdict == "invalid" and "shape" in mismatch.reasons[0]


def test_no_data_is_insufficient():
    assert _run().verdict == "insufficient"


def test_any_failing_metric_fails_the_whole_report():
    good = _pair(100, 0.001)
    bad = _pair(100, 0.5, seed=4)
    out = _run(
        energy_pred_eV_per_atom=good[0],
        energy_ref_eV_per_atom=good[1],
        barrier_pred_eV=bad[0],
        barrier_ref_eV=bad[1],
    )
    assert out.verdict == "fail" and len(out.metrics) == 2
    assert [m.within_threshold for m in out.metrics] == [True, False]


def test_result_is_deterministic_for_a_seed():
    pred, ref = _pair(100, 0.003)
    a = _run(energy_pred_eV_per_atom=pred, energy_ref_eV_per_atom=ref)
    b = _run(energy_pred_eV_per_atom=pred, energy_ref_eV_per_atom=ref)
    c = _run(energy_pred_eV_per_atom=pred, energy_ref_eV_per_atom=ref, seed=2)
    assert a == b and a.metrics[0].ci_low != c.metrics[0].ci_low


def test_report_is_written_atomically_with_its_schema(tmp_path):
    pred, ref = _pair(100, 0.003)
    out = _run(energy_pred_eV_per_atom=pred, energy_ref_eV_per_atom=ref)
    path = write_validation_report(out, tmp_path / "v" / "report.json")
    data = json.loads(path.read_text())
    assert data["schema"] == "matsim-mlip-validation/1" and data["synthetic"] is True
    assert not list(path.parent.glob(".mlip-val-*"))


@pytest.mark.parametrize("bad", [{"min_samples": 1}, {"confidence": 0.4}, {"n_bootstrap": 10}])
def test_invalid_thresholds_are_rejected(bad):
    with pytest.raises(ValueError):
        ValidationThresholds(**bad)
