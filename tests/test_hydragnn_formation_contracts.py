"""Physical training and reconstructed inference contracts without real models."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from matsim_agents.active_learning.config import TrainerConfig
from matsim_agents.active_learning.hydragnn_references import resolve_training_references
from matsim_agents.active_learning.trainer import retrain_hydragnn
from matsim_agents.backends.mlip.config import HydraGNNConfig
from matsim_agents.backends.mlip.hydragnn_energy import restore_hydragnn_total_energy
from matsim_agents.datasets.energy_references import predict_elemental_references
from matsim_agents.datasets.formation_training import (
    prepare_hydragnn_training_frames,
    save_hydragnn_energy_convention,
)


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


def _bind_output_to_reference_manifest(output, reference_manifest):
    from matsim_agents.datasets.governance import sha256_file

    for path in (
        output / "energy-convention.json",
        output / "training-reference" / "energy-convention.json",
    ):
        metadata = json.loads(path.read_text())
        metadata["source_elemental_manifest_sha256"] = sha256_file(reference_manifest)
        path.write_text(json.dumps(metadata))


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
    "defect",
    ["checkpoint", "selected_checkpoint", "metadata", "geometry", "data", "reference_manifest"],
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
    elif defect == "reference_manifest":
        path = model / "training-reference/elemental-references.json"
        references = json.loads(path.read_text())
        references["method_signature"] = "tampered"
        path.write_text(json.dumps(references))
    else:
        (model / "training-reference/formation.extxyz").write_text("tampered")
    with pytest.raises(
        ValueError,
        match="reference manifest hash mismatch" if defect == "reference_manifest" else None,
    ):
        restore_hydragnn_total_energy(FormationCalculator(), model, selected)


def test_checkpoint_requires_hash_for_existing_newhead_metadata(tmp_path, training_inputs):
    model = _checkpoint(tmp_path, training_inputs)
    (model / "newhead.json").write_text(json.dumps({"ft_config": {"activation": "relu"}}))
    save_hydragnn_energy_convention(model, model / "ft_model.pk")
    convention = model / "energy-convention.json"
    metadata = json.loads(convention.read_text())
    metadata["inference_artifact_sha256"].pop("newhead.json")
    convention.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="requires verified newhead.json"):
        restore_hydragnn_total_energy(FormationCalculator(), model)


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
    import torch

    import matsim_agents.backends.mlip.calculator as factory

    model = _checkpoint(tmp_path, training_inputs)
    loaded_model = torch.nn.Linear(1, 1)
    loaded_model.num_branches = 16
    monkeypatch.setattr(torch, "load", lambda *a, **kw: loaded_model.state_dict())
    inference = ModuleType("inference_random_structures")
    selected = []

    def load(logdir, checkpoint, precision):
        selected.append(checkpoint)
        return (
            loaded_model,
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

    import matsim_agents.backends.mlip.calculator as factory
    import matsim_agents.backends.mlip.hydragnn_newhead as newhead_surgery
    import matsim_agents.backends.mlip.relaxation as relaxation

    model_dir = _checkpoint(tmp_path, training_inputs)
    captured = {}
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: model.state_dict())
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
        monkeypatch.setattr(newhead_surgery, "apply_newhead_surgery", lambda m, *_args, **kw: m)
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
        (model_dir / "routing.json").write_text(json.dumps({"routed_branches": [7]}))
        from matsim_agents.datasets.governance import sha256_file

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
            assert calculator.active_branch_count == 1
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


def test_manifest_freeze_includes_label_scope_and_filters_superset(
    tmp_path, elemental_manifest, dataset_method_sidecar
):
    manifest = elemental_manifest({"Nb": -10, "O": -5, "Ti": -7})
    atoms = Atoms("Nb2", positions=np.zeros((2, 3)))
    atoms.calc = SinglePointCalculator(atoms, energy=-20, forces=np.zeros((2, 3)))
    dataset = tmp_path / "training.extxyz"
    write(dataset, atoms)
    dataset_method_sidecar(
        dataset,
        manifest,
        method_signature_elements={"Nb", "O"},
    )
    config = TrainerConfig(
        enabled=True,
        train_script=tmp_path / "train.py",
        compare_after_training=False,
        promote_model=False,
        hydragnn_training_references={"manifest": manifest},
    )

    frozen = resolve_training_references(
        config, dataset, dft_config=None, reference_root=tmp_path / "frozen"
    )
    frozen_manifest = json.loads(frozen.read_text())
    assert set(frozen_manifest["references"]) == {"Nb", "O"}
    assert set(
        predict_elemental_references(
            frozen, FormationCalculator(), required_elements={"Nb", "O"}
        ).dft_eV_per_atom
    ) == {"Nb", "O"}


def test_training_reference_manifest_additive_extension_across_iterations(
    tmp_path, elemental_manifest, dataset_method_sidecar, monkeypatch
):
    manifest = elemental_manifest({"Nb": -10, "O": -5, "Ti": -7})

    # Iteration 1: dataset only has Nb
    nb_atoms = Atoms("Nb2", positions=np.zeros((2, 3)))
    nb_atoms.calc = SinglePointCalculator(nb_atoms, energy=-20, forces=np.zeros((2, 3)))
    dataset1 = tmp_path / "train1.extxyz"
    write(dataset1, [nb_atoms])
    dataset_method_sidecar(dataset1, manifest)

    config = TrainerConfig(
        enabled=True,
        train_script=tmp_path / "train.py",
        compare_after_training=False,
        promote_model=False,
        hydragnn_training_references={"manifest": manifest},
    )
    root = tmp_path / "frozen"
    res1 = resolve_training_references(config, dataset1, dft_config=None, reference_root=root)
    assert res1 == root / "elemental-references.json"
    data1 = json.loads(res1.read_text())
    assert set(data1["references"].keys()) == {"Nb"}
    assert (root / "Nb.extxyz").exists()
    assert not (root / "O.extxyz").exists()

    # Iteration 2: dataset adds O (newly labelled species)
    o_atoms = Atoms("NbO2", positions=np.zeros((3, 3)))
    o_atoms.calc = SinglePointCalculator(o_atoms, energy=-23, forces=np.zeros((3, 3)))
    dataset2 = tmp_path / "train2.extxyz"
    write(dataset2, [nb_atoms, o_atoms])
    dataset_method_sidecar(dataset2, manifest)

    original_replace = Path.replace
    interrupted = False

    def interrupt_activation(path, target):
        nonlocal interrupted
        if not interrupted and path.name.startswith(f".{root.name}.next-"):
            interrupted = True
            raise RuntimeError("simulated publication interruption")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupt_activation)
    with pytest.raises(RuntimeError, match="simulated publication interruption"):
        resolve_training_references(config, dataset2, dft_config=None, reference_root=root)
    monkeypatch.setattr(Path, "replace", original_replace)
    res2 = resolve_training_references(config, dataset2, dft_config=None, reference_root=root)
    assert interrupted
    assert res2 == root / "elemental-references.json"
    data2 = json.loads(res2.read_text())
    assert set(data2["references"].keys()) == {"Nb", "O"}
    assert data2["references"]["Nb"] == data1["references"]["Nb"]
    assert (root / "Nb.extxyz").exists()
    assert (root / "O.extxyz").exists()

    # Iteration 3: dataset has Nb and O (no new species), reuses frozen snapshot
    res3 = resolve_training_references(config, dataset2, dft_config=None, reference_root=root)
    assert res3 == root / "elemental-references.json"

    # Iteration 4: dataset adds an element not present in source manifest -> coverage error
    bad_atoms = Atoms("NbFe", positions=np.zeros((2, 3)))
    bad_atoms.calc = SinglePointCalculator(bad_atoms, energy=-15, forces=np.zeros((2, 3)))
    dataset4 = tmp_path / "train4.extxyz"
    write(dataset4, [bad_atoms])
    dataset_method_sidecar(dataset4, manifest)
    with pytest.raises(ValueError, match="lacks coverage"):
        resolve_training_references(config, dataset4, dft_config=None, reference_root=root)


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


@pytest.mark.parametrize("launcher", [False, True])
@pytest.mark.parametrize(
    "script_name", ["custom", "finetune_hydragnn", "finetune_hydragnn_newhead"]
)
def test_retrain_hook_forwards_references_and_verifies_output(
    tmp_path, training_inputs, monkeypatch, script_name, launcher
):
    import matsim_agents.backends.mlip.calculator as factory

    dataset, manifest = training_inputs
    output = _checkpoint(tmp_path, training_inputs)
    built_in = script_name != "custom"
    script = tmp_path / f"{script_name}.py"
    script.touch()
    config = TrainerConfig(
        enabled=True,
        train_script=script,
        train_launcher=tmp_path / "launcher.sh" if launcher else None,
        nodes_for_train=1,
        ranks_per_node=1,
        validation_reference_set=manifest,
    )
    argv = []

    def run(args, **kw):
        argv.extend(args)
        if launcher:
            reference_manifest = Path(args[9])
        else:
            reference_manifest = Path(args[args.index("--elemental-reference-manifest") + 1])
        _bind_output_to_reference_manifest(output, reference_manifest)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("matsim_agents.active_learning.trainer.subprocess.run", run)
    built_calculators = []

    def mock_build_hydragnn_calculator(cfg, logdir_override=None):
        built_calculators.append(cfg)
        logdir = cfg.logdir if logdir_override is None else logdir_override
        # Ensure that checkpoint loading / verification fails if checkpoint is corrupt/broken
        ckpt_name = cfg.checkpoint or "ft_model.pk"
        ckpt_file = logdir / ckpt_name
        if ckpt_file.read_text() == "broken":
            raise ValueError(f"Corrupt checkpoint in {logdir}")
        return FormationCalculator()

    monkeypatch.setattr(factory, "build_hydragnn_calculator", mock_build_hydragnn_calculator)

    result = retrain_hydragnn(
        config,
        HydraGNNConfig(
            logdir=tmp_path,
            checkpoint="incumbent-custom.pk",
            mlp_checkpoint=tmp_path / "mlp.pt",
            ft_repo=tmp_path / "fine tuning repo",
            hydragnn_root=tmp_path / "HydraGNN source",
            charge=-2.5,
            spin=3.0,
        ),
        dataset,
        0,
        output,
    )
    assert result == output
    assert len(built_calculators) == 1
    assert built_calculators[0].logdir == output
    assert built_calculators[0].checkpoint is None
    if launcher:
        assert argv[0:3] == ["bash", str(config.train_launcher), str(script)]
        assert argv[3:10] == [
            str(dataset),
            str(output),
            str(tmp_path),
            str(config.epochs_per_iter),
            str(config.nodes_for_train),
            str(config.ranks_per_node),
            str(output.parent / "elemental-training-reference/elemental-references.json"),
        ]
        assert argv[10:12] == ["incumbent-custom.pk", str(tmp_path / "mlp.pt")]
        assert argv[12:] == (
            [
                str(tmp_path / "fine tuning repo"),
                "-2.5",
                "3.0",
                str(tmp_path / "HydraGNN source"),
            ]
            if built_in
            else []
        )
    else:
        assert "--elemental-reference-manifest" in argv
        assert ("--output-dir" if built_in else "--logdir") in argv
        assert ("--gfm-logdir" if built_in else "--resume_from") in argv
        if built_in:
            assert argv[argv.index("--charge") + 1] == "-2.5"
            assert argv[argv.index("--spin") + 1] == "3.0"
            assert argv[argv.index("--hydragnn-root") + 1] == str(tmp_path / "HydraGNN source")
        else:
            assert "--charge" not in argv
            assert "--spin" not in argv
        if script_name == "finetune_hydragnn_newhead":
            assert argv[argv.index("--ft-repo") + 1] == str(tmp_path / "fine tuning repo")
        else:
            assert "--ft-repo" not in argv
    (output / "ft_model.pk").write_text("broken")
    with pytest.raises(ValueError, match="checkpoint"):
        retrain_hydragnn(
            config,
            HydraGNNConfig(
                logdir=tmp_path,
                checkpoint="incumbent-custom.pk",
                mlp_checkpoint=tmp_path / "mlp.pt",
            ),
            dataset,
            0,
            output,
        )


@pytest.mark.parametrize("defect", ["replay", "missing_digest", "changed_during_training"])
def test_retrain_rejects_bundle_not_bound_to_requested_dataset(
    tmp_path, training_inputs, dataset_method_sidecar, monkeypatch, defect
):
    import matsim_agents.backends.mlip.calculator as factory

    old_dataset, manifest = training_inputs
    output = _checkpoint(tmp_path, training_inputs)
    dataset = old_dataset
    if defect == "replay":
        # The old bundle is internally valid, but the cumulative AL dataset has grown.
        dataset = tmp_path / "cumulative.extxyz"
        frames = read(old_dataset, index=":")
        write(dataset, [*frames, frames[0]])
        dataset_method_sidecar(dataset, manifest)
    elif defect == "missing_digest":
        for path in (
            output / "energy-convention.json",
            output / "training-reference/energy-convention.json",
        ):
            metadata = json.loads(path.read_text())
            metadata.pop("source_dataset_sha256")
            path.write_text(json.dumps(metadata))
    restore_hydragnn_total_energy(FormationCalculator(), output)
    config = TrainerConfig(
        enabled=True,
        train_script=tmp_path / "custom.py",
        validation_reference_set=manifest,
    )

    def run(*args, **kw):
        argv = args[0]
        reference_manifest = Path(argv[argv.index("--elemental-reference-manifest") + 1])
        _bind_output_to_reference_manifest(output, reference_manifest)
        if defect == "changed_during_training":
            with dataset.open("a") as stream:
                stream.write("\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("matsim_agents.active_learning.trainer.subprocess.run", run)

    def unexpected_model_load(*args, **kwargs):
        pytest.fail("replayed bundle must be rejected before model loading")

    monkeypatch.setattr(factory, "build_hydragnn_calculator", unexpected_model_load)
    with pytest.raises(ValueError, match="does not match the requested training dataset"):
        retrain_hydragnn(config, HydraGNNConfig(logdir=tmp_path), dataset, 1, output)


def test_retrain_rejects_bundle_bound_to_different_elemental_manifest(
    tmp_path, training_inputs, elemental_manifest, monkeypatch
):
    import matsim_agents.backends.mlip.calculator as factory

    dataset, _ = training_inputs
    output = _checkpoint(tmp_path, training_inputs)
    different_manifest = elemental_manifest({"Nb": -9, "O": -5})
    config = TrainerConfig(
        enabled=True,
        train_script=tmp_path / "custom.py",
        validation_reference_set=different_manifest,
    )
    monkeypatch.setattr(
        "matsim_agents.active_learning.trainer.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        factory,
        "build_hydragnn_calculator",
        lambda *args, **kwargs: pytest.fail("mismatched reference bundle must be rejected"),
    )

    with pytest.raises(ValueError, match="requested elemental reference manifest"):
        retrain_hydragnn(config, HydraGNNConfig(logdir=tmp_path), dataset, 1, output)


def test_retrain_rejects_multirank_builtin_before_resolving_references(
    tmp_path, training_inputs, monkeypatch
):
    import matsim_agents.active_learning.hydragnn_references as references

    dataset, manifest = training_inputs
    script = tmp_path / "finetune_hydragnn_newhead.py"
    script.touch()
    config = TrainerConfig(
        enabled=True,
        train_script=script,
        train_launcher=tmp_path / "launcher.sh",
        nodes_for_train=2,
        ranks_per_node=1,
        validation_reference_set=manifest,
    )

    def unexpected_reference_resolution(*args, **kwargs):
        pytest.fail("references must not be resolved before process-count validation")

    monkeypatch.setattr(references, "resolve_training_references", unexpected_reference_resolution)

    with pytest.raises(ValueError, match="require exactly one process"):
        retrain_hydragnn(
            config,
            HydraGNNConfig(logdir=tmp_path),
            dataset,
            0,
            tmp_path / "output",
        )


def test_routed_checkpoint_rejects_routed_branch_mismatches(tmp_path, training_inputs):
    from matsim_agents.datasets.governance import sha256_file

    model = _checkpoint(tmp_path, training_inputs)
    branch_mlp = model / "branch-mlp.pt"
    branch_mlp.write_text("routing-checkpoint")
    (model / "routing.json").write_text(json.dumps({"routed_branches": [7]}))
    save_hydragnn_energy_convention(
        model,
        model / "ft_model.pk",
        inference_metadata={
            "routed_branches": [7],
            "branch_mlp_checkpoint": branch_mlp.name,
            "branch_mlp_sha256": sha256_file(branch_mlp),
        },
    )

    # Valid checkpoint succeeds
    restore_hydragnn_total_energy(FormationCalculator(), model)

    # The inference configuration hash is mandatory, not merely an optional map entry.
    convention_file = model / "energy-convention.json"
    convention_data = json.loads(convention_file.read_text())
    convention_data["inference_artifact_sha256"].pop("config.json")
    convention_file.write_text(json.dumps(convention_data))
    with pytest.raises(ValueError, match="requires a verified config.json"):
        restore_hydragnn_total_energy(FormationCalculator(), model)

    # Mismatch 1: energy-convention.json has tampered routed_branches
    convention_data = json.loads(convention_file.read_text())
    convention_data["inference_artifact_sha256"]["config.json"] = sha256_file(model / "config.json")
    convention_data["routed_branches"] = [8]
    convention_file.write_text(json.dumps(convention_data))
    with pytest.raises(ValueError, match="routed branches do not match"):
        restore_hydragnn_total_energy(FormationCalculator(), model)

    # Mismatch 2: routing.json deleted from disk while in artifact hash
    convention_data["routed_branches"] = [7]
    convention_file.write_text(json.dumps(convention_data))
    (model / "routing.json").unlink()
    with pytest.raises(ValueError, match="inference artifact hash mismatch: routing.json"):
        restore_hydragnn_total_energy(FormationCalculator(), model)

    # Mismatch 3: routing.json omitted from artifact hash when routed_branches is declared
    convention_data["inference_artifact_sha256"].pop("routing.json", None)
    convention_file.write_text(json.dumps(convention_data))
    with pytest.raises(ValueError, match="requires verified routing.json"):
        restore_hydragnn_total_energy(FormationCalculator(), model)

    # Mismatch 4: branch_mlp_checkpoint missing or not declared
    routing_json = model / "routing.json"
    routing_json.write_text(json.dumps({"routed_branches": [7]}))
    convention_data["inference_artifact_sha256"]["routing.json"] = sha256_file(routing_json)
    convention_data.pop("branch_mlp_checkpoint", None)
    convention_file.write_text(json.dumps(convention_data))
    with pytest.raises(ValueError, match="requires verified branch MLP checkpoint"):
        restore_hydragnn_total_energy(FormationCalculator(), model)

    # Mismatch 5: branch MLP file tampered on disk
    convention_data["branch_mlp_checkpoint"] = branch_mlp.name
    convention_file.write_text(json.dumps(convention_data))
    branch_mlp.write_text("tampered-routing-mlp")
    with pytest.raises(ValueError, match="routing MLP differs from training artifact"):
        restore_hydragnn_total_energy(FormationCalculator(), model)
