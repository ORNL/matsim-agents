import numpy as np
import pytest
from ase import Atoms

from matsim_agents.active_learning.config import TrainerConfig
from matsim_agents.active_learning.dataset_governance import validate_labelled_frames
from matsim_agents.active_learning.trainer import LabelledFrame
from matsim_agents.discovery.composition import parse_composition
from matsim_agents.discovery.stability import RankingMode, score_stability
from matsim_agents.discovery.wrapper import CompositionExplorationResult
from matsim_agents.execution.contracts import ComputeBudget
from matsim_agents.orchestration.state import RelaxationResult
from matsim_agents.workflows.phase_exploration import PhaseExplorationPolicy, run_phase_exploration
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


def test_model_promotion_requires_held_out_validation(tmp_path):
    script = tmp_path / "train.py"
    script.touch()
    with pytest.raises(ValueError, match="validation_set"):
        TrainerConfig(
            enabled=True,
            train_script=script,
            promote_model=True,
            promotion_approved=True,
        )


def test_external_validation_paths_must_exist_before_training(tmp_path):
    with pytest.raises(ValueError, match="trainer.validation_set must be an existing file"):
        TrainerConfig(validation_set=tmp_path / "missing.extxyz")
    with pytest.raises(
        ValueError, match="trainer.validation_reference_set must be an existing file"
    ):
        TrainerConfig(validation_reference_set=tmp_path / "missing-references.extxyz")


def test_phase_reevaluation_requires_retraining():
    with pytest.raises(ValueError, match="requires retrain_mlip"):
        PhaseExplorationPolicy(reevaluate_after_retraining=True)


def test_phase_model_promotion_requires_retraining():
    with pytest.raises(ValueError, match="promote_model requires retrain_mlip"):
        PhaseExplorationPolicy(promote_model=True)


def test_phase_reevaluation_requires_requested_promotion():
    with pytest.raises(ValueError, match="reevaluate_after_retraining requires promote_model"):
        PhaseExplorationPolicy(
            active_learning=True,
            retrain_mlip=True,
            reevaluate_after_retraining=True,
        )
    policy = PhaseExplorationPolicy(
        active_learning=True,
        retrain_mlip=True,
        promote_model=True,
        reevaluate_after_retraining=True,
    )
    assert policy.reevaluate_after_retraining


def test_phase_active_learning_requires_dft_approval(tmp_path):
    policy = PhaseExplorationPolicy(active_learning=True)
    with pytest.raises(PermissionError, match="DFT approval"):
        run_phase_exploration(
            "Si",
            policy=policy,
            output_dir=str(tmp_path),
            active_learning_runner=lambda *_: {},
        )


@pytest.mark.parametrize("continue_on_rejection", [False, True])
def test_phase_rejected_promotion_retains_incumbent_only_when_requested(
    tmp_path, monkeypatch, caplog, continue_on_rejection
):
    parsed = parse_composition("Si")
    assert parsed is not None
    initial = CompositionExplorationResult(composition=parsed, phase_candidates=[])
    calls = []

    def explore(*args, **kwargs):
        calls.append(kwargs)
        return initial

    monkeypatch.setattr("matsim_agents.workflows.phase_exploration.explore_composition", explore)
    policy = PhaseExplorationPolicy(
        active_learning=True,
        retrain_mlip=True,
        promote_model=True,
        reevaluate_after_retraining=True,
        continue_on_promotion_rejection=continue_on_rejection,
        dft_approved=True,
        retraining_approved=True,
        promotion_approved=True,
    )

    def run():
        return run_phase_exploration(
            "Si",
            policy=policy,
            output_dir=str(tmp_path),
            active_learning_runner=lambda *_: {"model_promoted": False},
        )

    if continue_on_rejection:
        result = run()
        assert result.initial == initial
        assert result.after_retraining is None
        assert not result.model_promoted
        assert "retaining incumbent" in caplog.text
    else:
        with pytest.raises(RuntimeError, match="did not promote"):
            run()
    assert len(calls) == 1


def test_phase_retraining_requires_approval(tmp_path):
    policy = PhaseExplorationPolicy(
        active_learning=True,
        retrain_mlip=True,
        dft_approved=True,
    )
    with pytest.raises(PermissionError, match="retraining requires explicit approval"):
        run_phase_exploration(
            "Si",
            policy=policy,
            output_dir=str(tmp_path),
            active_learning_runner=lambda *_: {},
        )


def test_phase_model_promotion_requires_approval(tmp_path, monkeypatch):
    parsed = parse_composition("Si")
    assert parsed is not None
    called = False

    monkeypatch.setattr(
        "matsim_agents.workflows.phase_exploration.explore_composition",
        lambda *args, **kwargs: CompositionExplorationResult(
            composition=parsed,
            phase_candidates=[],
        ),
    )
    policy = PhaseExplorationPolicy(
        active_learning=True,
        retrain_mlip=True,
        promote_model=True,
        dft_approved=True,
        retraining_approved=True,
    )

    def active_learning_runner(*_args):
        nonlocal called
        called = True
        return {"model_promoted": True}

    with pytest.raises(PermissionError, match="model promotion requires explicit approval"):
        run_phase_exploration(
            "Si",
            policy=policy,
            output_dir=str(tmp_path),
            active_learning_runner=active_learning_runner,
        )
    assert called is False


@pytest.mark.parametrize("controls", [(False, False), (False, True), (True, True)])
def test_phase_passes_promotion_controls_before_al(tmp_path, monkeypatch, controls):
    parsed = parse_composition("Si")
    assert parsed is not None
    monkeypatch.setattr(
        "matsim_agents.workflows.phase_exploration.explore_composition",
        lambda *args, **kwargs: CompositionExplorationResult(
            composition=parsed, phase_candidates=[]
        ),
    )
    promote_model, promotion_approved = controls
    calls = []

    def active_learning_runner(composition, output_dir, retrain, promote, approved):
        calls.append((composition, output_dir, retrain, promote, approved))
        return {"model_promoted": False}

    result = run_phase_exploration(
        "Si",
        policy=PhaseExplorationPolicy(
            active_learning=True,
            retrain_mlip=True,
            promote_model=promote_model,
            promotion_approved=promotion_approved,
            dft_approved=True,
            retraining_approved=True,
        ),
        output_dir=str(tmp_path),
        active_learning_runner=active_learning_runner,
    )
    assert calls == [("Si", str(tmp_path), True, *controls)]
    assert not result.model_promoted


def test_seed_only_phase_is_not_a_usable_minimum(tmp_path, monkeypatch):
    parsed = parse_composition("Si")
    assert parsed is not None
    monkeypatch.setattr(
        "matsim_agents.discovery.seeds.generate_seeds",
        lambda *args, **kwargs: [
            {"formula": "Si", "source": "prototype", "structure_path": "Si.vasp"}
        ],
    )

    result = run_phase_exploration(
        "Si",
        policy=PhaseExplorationPolicy(relax_structures=False),
        output_dir=str(tmp_path),
    )

    assert result.initial.outcome_class == "seed_only"


def test_formula_budget_does_not_truncate_random_structure_quota(tmp_path, monkeypatch):
    observed = {}

    def fake_explore(composition, *, output_dir, **kwargs):
        observed.update(kwargs)
        parsed = parse_composition(composition)
        assert parsed is not None
        return CompositionExplorationResult(composition=parsed, phase_candidates=[])

    monkeypatch.setattr(
        "matsim_agents.workflows.phase_exploration.explore_composition",
        fake_explore,
    )
    run_phase_exploration(
        "Si",
        policy=PhaseExplorationPolicy(budget=ComputeBudget(max_candidates=3)),
        output_dir=str(tmp_path),
        exploration_kwargs={"n_random": 50},
    )

    assert observed["n_random"] == 50


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


def test_label_validation_checks_existing_dataset_and_element_scope():
    existing_atoms = Atoms("Nb", positions=[[0.0, 0.0, 0.0]])
    existing = LabelledFrame(
        existing_atoms,
        -1.0,
        np.zeros((1, 3)),
        None,
        "existing",
        0,
        "qe",
    )
    duplicate = LabelledFrame(
        existing_atoms.copy(),
        -2.0,
        np.zeros((1, 3)),
        None,
        "duplicate",
        1,
        "qe",
    )
    wrong_atoms = Atoms("Ta", positions=[[0.0, 0.0, 0.0]])
    wrong_element = LabelledFrame(
        wrong_atoms,
        -3.0,
        np.zeros((1, 3)),
        None,
        "wrong",
        1,
        "qe",
    )

    accepted, summary = validate_labelled_frames(
        [duplicate, wrong_element],
        existing_frames=[existing],
        expected_atomic_numbers={41},
    )

    assert accepted == []
    assert summary.duplicate == 1
    assert summary.rejected == 1
    assert "outside the campaign element set" in summary.rejection_reasons[0]


def test_label_validation_rejects_reordered_translated_periodic_duplicate():
    existing_atoms = Atoms(
        "NbO",
        scaled_positions=[[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    duplicate_atoms = existing_atoms[[1, 0]]
    duplicate_atoms.translate([4.5, -3.5, 0.5])
    existing = LabelledFrame(existing_atoms, -3.0, np.zeros((2, 3)), None, "existing", 0, "qe")
    duplicate = LabelledFrame(duplicate_atoms, -3.0, np.zeros((2, 3)), None, "duplicate", 1, "qe")

    accepted, summary = validate_labelled_frames([duplicate], existing_frames=[existing])

    assert accepted == []
    assert summary.duplicate == 1


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
