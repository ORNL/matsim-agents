from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

from matsim_agents.active_learning.config import MLIPConfig, TrainerConfig, UMAConfig
from matsim_agents.active_learning.evaluate import (
    EvalMetrics,
    _validate_shift_reference_coverage,
    assess_promotion,
    evaluate_frames,
    evaluate_promotion_candidate,
)


def _metrics(
    *,
    energy_mae: float,
    force_mae: float,
    model_path: str,
    energy_frames: int = 10,
    force_frames: int = 10,
) -> EvalMetrics:
    return EvalMetrics(
        backend="uma",
        model_path=model_path,
        iteration=1,
        test_set="held-out.extxyz",
        n_frames_total=10,
        n_frames_evaluated=10,
        n_energy_frames_evaluated=energy_frames,
        n_force_frames_evaluated=force_frames,
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


def test_shift_reference_accepts_mixed_compositions_that_span_evaluation() -> None:
    evaluation = [[1, 1, 1], [2, 0, 1], [0, 2, 1]]
    mixed_references = [[1, 0, 1], [0, 1, 1], [1, 1, 1]]

    _validate_shift_reference_coverage(
        np.asarray(evaluation, dtype=float),
        np.asarray(mixed_references, dtype=float),
    )


def test_shift_reference_rejects_unspanned_composition_direction() -> None:
    evaluation = [[1, 1, 1], [2, 0, 1]]
    deficient_references = [[1, 0, 1], [2, 0, 2]]

    with pytest.raises(ValueError, match="do not span"):
        _validate_shift_reference_coverage(
            np.asarray(evaluation, dtype=float),
            np.asarray(deficient_references, dtype=float),
        )


def test_cross_composition_shift_requires_independent_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CompositionCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] = float(np.sum(atoms.numbers))
            self.results["forces"] = np.zeros((len(atoms), 3))

    monkeypatch.setattr(
        "matsim_agents.active_learning.calculator.make_mlip_calculator",
        lambda _cfg: CompositionCalculator(),
    )
    frames = [
        Atoms("NbTaO", positions=np.zeros((3, 3)), info={"energy": 0.0}),
        Atoms("Nb2O", positions=np.zeros((3, 3)), info={"energy": 0.0}),
    ]
    mlip = MLIPConfig(backend="uma", uma=UMAConfig())

    with pytest.raises(ValueError, match="requires independent ref_frames"):
        evaluate_frames(mlip, frames)

    evaluate_frames(mlip, [frames[0], frames[0].copy()])


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


@pytest.mark.parametrize("label", ["energy", "force"])
def test_promotion_gate_requires_enough_frames_for_each_metric(tmp_path: Path, label: str) -> None:
    counts = {f"{label}_frames": 1}
    decision = assess_promotion(
        _metrics(
            energy_mae=0.04,
            force_mae=0.08,
            model_path="candidate",
            **counts,
        ),
        _metrics(energy_mae=0.05, force_mae=0.1, model_path="incumbent"),
        _trainer(tmp_path),
    )

    assert decision.approved is False
    assert any(f"1 {label}-labelled frames" in reason for reason in decision.reasons)


@pytest.mark.parametrize("reference_kind", [None, "training", "validation"])
def test_promotion_requires_distinct_held_out_paths(
    tmp_path: Path, reference_kind: str | None
) -> None:
    training_set = tmp_path / "training.extxyz"
    validation_set = training_set if reference_kind is None else tmp_path / "validation.extxyz"
    reference_set = {
        None: None,
        "training": training_set,
        "validation": validation_set,
    }[reference_kind]
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            validation_set=validation_set,
            validation_reference_set=reference_set,
        )
    )

    with pytest.raises(ValueError, match="held out|must differ"):
        evaluate_promotion_candidate(
            cfg,
            "candidate.pt",
            iteration=1,
            training_set=training_set,
        )


def test_promotion_rejects_training_geometry_copied_to_validation(tmp_path: Path) -> None:
    training_set = tmp_path / "training.extxyz"
    validation_set = tmp_path / "held-out.extxyz"
    training_frame = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]], cell=[5, 5, 5], pbc=True)
    validation_frame = training_frame[[1, 0]]
    validation_frame.translate([1.0, 1.0, 1.0])
    write(training_set, training_frame)
    write(validation_set, validation_frame)
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            validation_set=validation_set,
            validation_reference_set=None,
        )
    )

    with pytest.raises(ValueError, match="overlapping geometries"):
        evaluate_promotion_candidate(
            cfg,
            "candidate.pt",
            iteration=1,
            training_set=training_set,
        )
