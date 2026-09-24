import numpy as np
import pytest
from ase import Atoms

from matsim_agents.active_learning.config import TrainerConfig
from matsim_agents.active_learning.dataset_governance import validate_labelled_frames
from matsim_agents.active_learning.trainer import LabelledFrame
from matsim_agents.discovery.stability import RankingMode, score_stability
from matsim_agents.orchestration.state import RelaxationResult
from matsim_agents.workflows.phase_exploration import PhaseExplorationPolicy
from matsim_agents.workflows.relaxation import ScientificRelaxationConfig


def test_active_learning_defaults_to_label_collection_without_retraining():
    trainer = TrainerConfig()
    assert trainer.enabled is False
    assert trainer.promote_model is False


def test_model_promotion_requires_retraining():
    with pytest.raises(ValueError, match="requires trainer.enabled"):
        TrainerConfig(promote_model=True)


def test_model_promotion_requires_explicit_approval(tmp_path):
    script = tmp_path / "train.py"
    script.touch()
    with pytest.raises(ValueError, match="promotion_approved"):
        TrainerConfig(enabled=True, train_script=script, promote_model=True)


def test_phase_reevaluation_requires_retraining():
    with pytest.raises(ValueError, match="requires retrain_mlip"):
        PhaseExplorationPolicy(reevaluate_after_retraining=True)


def test_dft_relaxation_requires_dft_configuration():
    with pytest.raises(ValueError, match="requires dft configuration"):
        ScientificRelaxationConfig(mode="dft", structure_path="Si.vasp")


def test_label_validation_rejects_duplicates_and_nonfinite_values():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    valid = LabelledFrame(atoms, -1.0, np.zeros((1, 3)), None, "job", 0, "qe")
    invalid = LabelledFrame(atoms, float("nan"), np.zeros((1, 3)), None, "job2", 0, "qe")
    accepted, summary = validate_labelled_frames([valid, valid, invalid])
    assert accepted == [valid]
    assert summary.duplicate == 1
    assert summary.rejected == 1


def test_convex_hull_claim_requires_reference_energies(tmp_path):
    structure = tmp_path / "H.xyz"
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.write(structure)
    relaxation = RelaxationResult(
        structure_path=str(structure),
        optimized_structure_path=str(structure),
        trajectory_path="unused",
        log_csv_path="unused",
        final_energy_eV=-1.0,
        final_max_force_eV_per_A=0.0,
        num_steps=1,
        converged=True,
    )
    with pytest.raises(ValueError, match="reference-energy set"):
        score_stability("H", [relaxation], ranking_mode=RankingMode.CONVEX_HULL)
