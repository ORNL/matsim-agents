from pathlib import Path

from matsim_agents.active_learning.config import TrainerConfig
from matsim_agents.active_learning.evaluate import EvalMetrics, assess_promotion


def _metrics(*, energy_mae: float, force_mae: float, model_path: str) -> EvalMetrics:
    return EvalMetrics(
        backend="uma",
        model_path=model_path,
        iteration=1,
        test_set="held-out.extxyz",
        n_frames_total=10,
        n_frames_evaluated=10,
        n_atoms_total=80,
        energy_mae_eV=0.5,
        energy_rmse_eV=0.6,
        energy_mae_eV_per_atom=energy_mae,
        energy_rmse_eV_per_atom=energy_mae,
        energy_mae_eV_per_atom_shifted=energy_mae,
        energy_rmse_eV_per_atom_shifted=energy_mae,
        energy_mean_offset_eV_per_atom=0.0,
        force_mae_eV_per_A=force_mae,
        force_rmse_eV_per_A=force_mae,
    )


def _trainer(tmp_path: Path) -> TrainerConfig:
    train_script = tmp_path / "train.py"
    train_script.touch()
    validation_set = tmp_path / "held-out.extxyz"
    validation_set.touch()
    return TrainerConfig(
        enabled=True,
        promote_model=True,
        promotion_approved=True,
        train_script=train_script,
        validation_set=validation_set,
        promotion_max_energy_mae_eV_per_atom=0.1,
        promotion_max_force_mae_eV_per_A=0.2,
        promotion_max_relative_regression=0.05,
        promotion_min_evaluated_frames=5,
    )


def test_promotion_gate_accepts_accurate_non_regressing_candidate(tmp_path: Path) -> None:
    decision = assess_promotion(
        _metrics(energy_mae=0.04, force_mae=0.08, model_path="candidate"),
        _metrics(energy_mae=0.05, force_mae=0.1, model_path="incumbent"),
        _trainer(tmp_path),
    )

    assert decision.approved is True
    assert decision.reasons == []
    assert decision.candidate_metrics["model_path"] == "candidate"


def test_promotion_gate_rejects_inaccurate_or_regressing_candidate(tmp_path: Path) -> None:
    decision = assess_promotion(
        _metrics(energy_mae=0.11, force_mae=0.11, model_path="candidate"),
        _metrics(energy_mae=0.05, force_mae=0.1, model_path="incumbent"),
        _trainer(tmp_path),
    )

    assert decision.approved is False
    assert any("exceeds limit" in reason for reason in decision.reasons)
    assert any("incumbent regression limit" in reason for reason in decision.reasons)
