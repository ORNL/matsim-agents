"""Physical training and reconstructed inference contracts without real models."""

import json
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from matsim_agents.active_learning.config import HydraGNNConfig, TrainerConfig
from matsim_agents.active_learning.formation_training import (
    prepare_hydragnn_training_frames,
    save_hydragnn_energy_convention,
)
from matsim_agents.active_learning.hydragnn_energy import restore_hydragnn_total_energy
from matsim_agents.active_learning.hydragnn_references import resolve_training_references
from matsim_agents.active_learning.trainer import retrain_hydragnn
from matsim_agents.discovery.energy_references import predict_elemental_references


@pytest.fixture
def training_inputs(tmp_path, elemental_manifest, dataset_method_sidecar):
    manifest = elemental_manifest({"Nb": -10, "O": -5})
    dataset = tmp_path / "raw.extxyz"
    frames = []
    for formula, energy in (("NbO2", -23), ("NbO2", -22), ("Nb2O4", -46)):
        atoms = Atoms(formula, positions=np.zeros((len(Atoms(formula)), 3)))
        atoms.calc = SinglePointCalculator(
            atoms, energy=energy, forces=np.full((len(atoms), 3), 0.2)
        )
        frames.append(atoms)
    write(dataset, frames)
    dataset_method_sidecar(dataset, manifest)
    return dataset, manifest


def _checkpoint(tmp_path, training_inputs):
    dataset, manifest = training_inputs
    model = tmp_path / "model"
    prepare_hydragnn_training_frames(dataset, model, manifest)
    checkpoint = model / "ft_model.pk"
    checkpoint.write_text("synthetic-checkpoint")
    (model / "config.json").write_text("{}")
    save_hydragnn_energy_convention(model, checkpoint)
    return model


class FormationCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(offset=0)
        self.last_branch_weights = [1]
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        self.results = {
            "energy": -len(atoms) + self.model.offset,
            "forces": np.full((len(atoms), 3), 0.2),
            "stress": np.arange(6) * 0.01,
        }


@pytest.mark.parametrize("trainer_name", ["finetune_hydragnn", "finetune_hydragnn_newhead"])
def test_trainers_reject_missing_references_before_model_import(tmp_path, trainer_name):
    from importlib import import_module

    module = import_module(f"matsim_agents.active_learning.{trainer_name}")
    kwargs = {"gfm_logdir": tmp_path / "missing-foundation"}
    if trainer_name == "finetune_hydragnn":
        kwargs["branch_mlp_path"] = tmp_path / "missing-mlp"
    with pytest.raises(ValueError, match="elemental_reference_manifest"):
        getattr(module, trainer_name)(tmp_path / "missing-data", tmp_path / "output", **kwargs)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("trainer_name", ["finetune_hydragnn", "finetune_hydragnn_newhead"])
def test_both_trainers_prepare_physical_labels_before_graphs(
    tmp_path, training_inputs, monkeypatch, trainer_name
):
    from importlib import import_module

    module = import_module(f"matsim_agents.active_learning.{trainer_name}")
    dataset, manifest = training_inputs
    model = tmp_path / "output"

    def check_preparation(_root):
        frames = read(model / "training-reference/formation.extxyz", index=":")
        assert [atoms.get_potential_energy() for atoms in frames] == pytest.approx([-3, -2, -6])
        for atoms in frames:
            np.testing.assert_array_equal(atoms.get_forces(), np.full((len(atoms), 3), 0.2))
        raise RuntimeError("verified physical frames before model construction")

    monkeypatch.setattr(module, "_resolve_hydragnn_paths", check_preparation)
    kwargs = {"gfm_logdir": tmp_path, "elemental_reference_manifest": manifest}
    if trainer_name == "finetune_hydragnn":
        kwargs["branch_mlp_path"] = tmp_path / "mlp"
    with pytest.raises(RuntimeError, match="verified physical frames"):
        getattr(module, trainer_name)(dataset, model, **kwargs)


@pytest.mark.parametrize("routed", [False, True])
def test_losses_use_physical_energy_without_second_subtraction(routed):
    import torch

    from matsim_agents.active_learning.finetune_hydragnn import _batch_loss
    from matsim_agents.active_learning.finetune_hydragnn_newhead import _batch_loss_single_head

    batch = SimpleNamespace(
        num_graphs=1,
        pos=torch.zeros((3, 3), dtype=torch.float64),
        batch=torch.zeros(3, dtype=torch.long),
        energy=torch.tensor([-3.0]),
        forces=torch.zeros((3, 3)),
        e_ref=torch.tensor([-2000.0]),
        chemical_composition=torch.zeros((1, 118)),
        dataset_name=torch.tensor([[7]]),
    )

    def model(graph):
        return ((graph.pos**2).sum() - 3).reshape(1, 1)

    def loss():
        if routed:
            return _batch_loss(
                model,
                lambda _: torch.zeros((1, 16)),
                batch,
                torch.tensor([7]),
                1,
                1,
                torch.float64,
            )
        return _batch_loss_single_head(model, batch, 1, 1)

    assert loss().item() == pytest.approx(0)
    batch.energy = torch.tensor([-2.0])
    assert loss().item() == pytest.approx(1 / 9)
    if routed:
        assert batch.dataset_name.item() == 7


def test_single_head_resume_loads_readout_after_surgery(monkeypatch):
    import torch

    import matsim_agents.active_learning.finetune_hydragnn_newhead as trainer

    trained = torch.nn.Linear(1, 1)
    with torch.no_grad():
        trained.weight.fill_(9)
        trained.bias.fill_(-3)
    rebuilt = torch.nn.Linear(1, 1)
    monkeypatch.setattr(trainer, "apply_newhead_surgery", lambda *_args, **kw: rebuilt)
    model = trainer._resume_single_head_model(
        None, trained.state_dict(), {}, ft_repo=None, freeze_mode="none"
    )
    assert model.weight.item() == pytest.approx(9)
    assert model.bias.item() == pytest.approx(-3)
    with pytest.raises(RuntimeError, match="Missing key"):
        trainer._resume_single_head_model(None, {}, {}, ft_repo=None, freeze_mode="none")


@pytest.mark.parametrize("formula,expected", [("NbO2", -23), ("Nb2O4", -46), ("O2", -12)])
def test_checkpoint_reconstructs_totals_and_preserves_forces_and_stress(
    tmp_path, training_inputs, formula, expected
):
    model = _checkpoint(tmp_path, training_inputs)
    underlying = FormationCalculator()
    calculator = restore_hydragnn_total_energy(underlying, model)
    atoms = Atoms(formula, positions=np.zeros((len(Atoms(formula)), 3)))
    atoms.calc = calculator
    assert atoms.get_potential_energy() == pytest.approx(expected)
    np.testing.assert_array_equal(atoms.get_forces(), np.full((len(atoms), 3), 0.2))
    np.testing.assert_array_equal(atoms.get_stress(), np.arange(6) * 0.01)
    assert calculator.model is underlying.model
    assert calculator.last_branch_weights == [1]
    calculator.model.offset = 2
    calculator.reset()
    assert atoms.get_potential_energy() == pytest.approx(expected + 2)
    assert underlying.calls == 2


@pytest.mark.parametrize(
    "defect", ["checkpoint", "selected_checkpoint", "metadata", "geometry", "data"]
)
def test_checkpoint_rejects_mismatches(tmp_path, training_inputs, defect):
    model = _checkpoint(tmp_path, training_inputs)
    selected = None
    if defect == "checkpoint":
        (model / "ft_model.pk").write_text("tampered")
    elif defect == "selected_checkpoint":
        selected = "other.pk"
        (model / selected).write_text("synthetic-checkpoint")
    elif defect == "metadata":
        path = model / "energy-convention.json"
        metadata = json.loads(path.read_text())
        metadata["elemental_energies_eV_per_atom"]["Nb"] = -99
        path.write_text(json.dumps(metadata))
    elif defect == "geometry":
        path = model / "training-reference/references/Nb.extxyz"
        atoms = read(path)
        atoms.positions[0, 0] += 0.1
        write(path, atoms)
    else:
        (model / "training-reference/formation.extxyz").write_text("tampered")
    with pytest.raises(ValueError):
        restore_hydragnn_total_energy(FormationCalculator(), model, selected)


def test_missing_elements_and_double_wrapping_fail(tmp_path, training_inputs):
    model = _checkpoint(tmp_path, training_inputs)
    calculator = restore_hydragnn_total_energy(FormationCalculator(), model)
    atoms = Atoms("Si", calculator=calculator)
    with pytest.raises(ValueError, match="coverage"):
        atoms.get_potential_energy()
    with pytest.raises(ValueError, match="already"):
        restore_hydragnn_total_energy(calculator, model)


def test_evaluation_formation_subtracts_model_references_once(tmp_path, training_inputs):
    model = _checkpoint(tmp_path, training_inputs)
    calculator = restore_hydragnn_total_energy(FormationCalculator(), model)
    atoms = Atoms("NbO2", positions=np.zeros((3, 3)), calculator=calculator)
    reference = predict_elemental_references(
        model / "training-reference/elemental-references.json",
        calculator,
        required_elements={"Nb", "O"},
    )
    assert atoms.get_potential_energy() == pytest.approx(-23)
    assert reference.formation_energy(atoms, -23, model=False) == pytest.approx(-1)
    assert reference.formation_energy(
        atoms, atoms.get_potential_energy(), model=True
    ) == pytest.approx(0)


@pytest.mark.parametrize("name", ["routing.json", "newhead.json"])
def test_legacy_fitted_offsets_are_not_silently_ignored(tmp_path, name):
    (tmp_path / name).write_text(json.dumps({"reference_energies": {"8": -5}}))
    with pytest.raises(ValueError, match="retrain"):
        restore_hydragnn_total_energy(FormationCalculator(), tmp_path)


def test_unmarked_foundation_retains_existing_behavior(tmp_path):
    calculator = FormationCalculator()
    assert restore_hydragnn_total_energy(calculator, tmp_path) is calculator


def test_selected_head_factory_restores_total_energy(tmp_path, training_inputs, monkeypatch):
    import matsim_agents.active_learning.calculator as factory

    model = _checkpoint(tmp_path, training_inputs)
    inference = ModuleType("inference_random_structures")
    selected = []

    def load(logdir, checkpoint, precision):
        selected.append(checkpoint)
        return (
            SimpleNamespace(num_branches=16),
            {"NeuralNetwork": {"Architecture": {"radius": 5, "max_neighbours": 20}}},
            "cpu",
            nullcontext(),
            "float64",
        )

    inference.load_config_and_model = load
    monkeypatch.setitem(sys.modules, "inference_random_structures", inference)
    monkeypatch.setattr(
        factory, "_build_selected_head_calculator", lambda *a, **kw: FormationCalculator()
    )
    calculator = factory.build_hydragnn_calculator(HydraGNNConfig(logdir=model, inference_head=7))
    assert selected == ["ft_model.pk"]
    assert Atoms("NbO2", calculator=calculator).get_potential_energy() == pytest.approx(-23)


@pytest.mark.parametrize("newhead", [False, True])
def test_fused_and_auto_newhead_reload_restore_totals(
    tmp_path, training_inputs, monkeypatch, newhead
):
    import torch

    import matsim_agents.active_learning.calculator as factory
    import matsim_agents.active_learning.finetune_hydragnn_newhead as trainer
    import matsim_agents.backends.mlip.relaxation as relaxation

    model_dir = _checkpoint(tmp_path, training_inputs)
    captured = {}
    model = torch.nn.Linear(1, 1)
    if newhead:
        config = {
            "NeuralNetwork": {
                "Architecture": {"radius": 5, "max_neighbours": 20},
                "Training": {"precision": "fp64"},
            },
        }
        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "newhead.json").write_text(json.dumps({"ft_config": {}}))
        create = ModuleType("hydragnn.models.create")
        create.create_model_config = lambda **kw: model
        precision = ModuleType("hydragnn.train.train_validate_test")
        precision.resolve_precision = lambda _: (None, torch.float64, nullcontext())
        monkeypatch.setitem(sys.modules, "hydragnn.models.create", create)
        monkeypatch.setitem(sys.modules, "hydragnn.train.train_validate_test", precision)
        monkeypatch.setattr(trainer, "apply_newhead_surgery", lambda m, *_args, **kw: m)
        monkeypatch.setattr(torch, "load", lambda *a, **kw: model.state_dict())
        monkeypatch.setattr(
            factory, "_build_single_head_calculator", lambda **kw: FormationCalculator()
        )
        save_hydragnn_energy_convention(model_dir, model_dir / "ft_model.pk")
    else:
        inference = ModuleType("inference_fused")
        inference.load_fused_stack = lambda *a: (
            model,
            torch.nn.Linear(118, 16),
            {"NeuralNetwork": {"Architecture": {"radius": 5, "max_neighbours": 20}}},
            "cpu",
            nullcontext(),
            torch.float64,
            16,
            "cpu",
            nullcontext(),
            True,
            "fp64",
            "fp64",
        )
        monkeypatch.setitem(sys.modules, "inference_fused", inference)

        def build(**kw):
            captured.update(kw)
            return FormationCalculator()

        monkeypatch.setattr(relaxation, "_build_calculator", build)
        branch_mlp = model_dir / "branch-mlp.pt"
        branch_mlp.write_text("routing-checkpoint")
        from matsim_agents.active_learning.dataset_governance import sha256_file

        save_hydragnn_energy_convention(
            model_dir,
            model_dir / "ft_model.pk",
            inference_metadata={
                "routed_branches": [7],
                "branch_mlp_checkpoint": branch_mlp.name,
                "branch_mlp_sha256": sha256_file(branch_mlp),
            },
        )
    try:
        calculator = factory.build_hydragnn_calculator(
            HydraGNNConfig(logdir=model_dir, inference_head=7 if newhead else None)
        )
        assert Atoms("NbO2", calculator=calculator).get_potential_energy() == pytest.approx(-23)
        if not newhead:
            weights = torch.softmax(captured["mlp"](torch.zeros((1, 118))), dim=-1)
            assert weights[0, 7].item() == pytest.approx(1)
            assert weights.sum().item() == pytest.approx(1)
    finally:
        torch.set_default_dtype(torch.float32)


def test_training_reference_is_frozen_independently_of_comparison(tmp_path, training_inputs):
    dataset, manifest = training_inputs
    config = TrainerConfig(
        enabled=True,
        train_script=tmp_path / "train.py",
        compare_after_training=False,
        promote_model=False,
        hydragnn_training_references={"manifest": manifest},
    )
    root = tmp_path / "frozen"
    first = resolve_training_references(config, dataset, dft_config=None, reference_root=root)
    second = resolve_training_references(config, dataset, dft_config=None, reference_root=root)
    assert first == second == root / "elemental-references.json"
    metadata = json.loads(manifest.read_text())
    metadata["references"]["Nb"]["energy_eV"] -= 1
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="inputs changed"):
        resolve_training_references(config, dataset, dft_config=None, reference_root=root)


def test_training_without_comparison_still_requires_references(tmp_path, training_inputs):
    dataset, _ = training_inputs
    with pytest.raises(ValueError, match="even when"):
        resolve_training_references(
            TrainerConfig(enabled=True, train_script=tmp_path / "train.py"),
            dataset,
            dft_config=None,
            reference_root=tmp_path / "frozen",
        )


def test_frozen_reference_manifest_cannot_be_relabelled(tmp_path, training_inputs):
    dataset, manifest = training_inputs
    cfg = TrainerConfig(validation_reference_set=manifest)
    root = tmp_path / "frozen"
    frozen = resolve_training_references(cfg, dataset, dft_config=None, reference_root=root)
    metadata = json.loads(frozen.read_text())
    metadata["references"]["Nb"]["energy_eV"] -= 1
    frozen.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        resolve_training_references(cfg, dataset, dft_config=None, reference_root=root)


@pytest.mark.parametrize("built_in", [False, True])
def test_retrain_hook_forwards_references_and_verifies_output(
    tmp_path, training_inputs, monkeypatch, built_in
):
    dataset, manifest = training_inputs
    output = _checkpoint(tmp_path, training_inputs)
    script = tmp_path / ("finetune_hydragnn.py" if built_in else "custom.py")
    script.touch()
    config = TrainerConfig(enabled=True, train_script=script, validation_reference_set=manifest)
    argv = []

    def run(args, **kw):
        argv.extend(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("matsim_agents.active_learning.trainer.subprocess.run", run)
    result = retrain_hydragnn(
        config,
        HydraGNNConfig(logdir=tmp_path, mlp_checkpoint=tmp_path / "mlp.pt"),
        dataset,
        0,
        output,
    )
    assert result == output
    assert "--elemental-reference-manifest" in argv
    assert ("--output-dir" if built_in else "--logdir") in argv
    assert ("--gfm-logdir" if built_in else "--resume_from") in argv
    (output / "ft_model.pk").write_text("broken")
    with pytest.raises(ValueError, match="checkpoint"):
        retrain_hydragnn(
            config,
            HydraGNNConfig(logdir=tmp_path, mlp_checkpoint=tmp_path / "mlp.pt"),
            dataset,
            0,
            output,
        )
