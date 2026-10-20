"""MLIP fine-tuning inputs and keeping surrogate models loaded across AL iterations."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from matsim_agents.datasets.energy_alignment import (
    ENERGY_ALIGNMENT_FILE,
    energy_alignment,
    recorded_alignment,
)
from matsim_agents.execution.allocation import Allocation
from matsim_agents.execution.placement import ResourcePlan

# -- energy alignment -----------------------------------------------------------


@pytest.fixture
def references(elemental_manifest):
    return elemental_manifest({"Nb": -100.0, "O": -50.0})


def _model_energy(per_atom):
    return lambda atoms: sum(per_atom[symbol] for symbol in atoms.get_chemical_symbols())


def test_labels_move_to_the_model_scale_and_keep_formation_energies(references):
    alignment = energy_alignment(
        references, {"Nb", "O"}, _model_energy({"Nb": -10.0, "O": -5.0}), base_model="uma-s-1p1"
    )
    assert alignment.dft_eV_per_atom == {"Nb": -100.0, "O": -50.0}
    assert alignment.model_eV_per_atom == {"Nb": -10.0, "O": -5.0}
    atoms = Atoms("NbO2")
    dft_total = -100.0 - 2 * 50.0 - 3.0  # 3 eV below the elemental references
    aligned = alignment.aligned_energy(atoms, dft_total)
    assert aligned == pytest.approx(-10.0 - 2 * 5.0 - 3.0)
    assert alignment.model_references_from == "uma-s-1p1"


def test_a_fine_tuned_model_keeps_the_scale_of_the_model_it_came_from(tmp_path, references):
    first = energy_alignment(
        references, {"Nb"}, _model_energy({"Nb": -10.0}), base_model="uma-s-1p1"
    )
    model_dir = tmp_path / "iteration_0000" / "model"
    (model_dir / "checkpoints").mkdir(parents=True)
    checkpoint = model_dir / "checkpoints" / "inference_ckpt.pt"
    checkpoint.write_text("weights")
    first.write(model_dir)
    assert recorded_alignment(checkpoint)[1] == model_dir / ENERGY_ALIGNMENT_FILE

    # The fine-tuned model now predicts slightly different elemental energies;
    # the recorded scale is kept, and only the new element is computed.
    second = energy_alignment(
        references,
        {"Nb", "O"},
        _model_energy({"Nb": -10.4, "O": -5.0}),
        base_model=checkpoint,
    )
    assert second.model_eV_per_atom == {"Nb": -10.0, "O": -5.0}
    assert "computed with" in second.model_references_from and "O" in (second.model_references_from)


def test_alignment_needs_references_for_every_element(references):
    with pytest.raises(ValueError, match="lacks coverage"):
        energy_alignment(references, {"Ta"}, _model_energy({}), base_model="m")


# -- fine-tuner inputs ----------------------------------------------------------


def test_uma_loss_is_per_atom_and_the_best_epoch_is_kept(monkeypatch):
    torch = pytest.importorskip("torch")
    from matsim_agents.training import finetune_uma

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.energy_per_atom = torch.nn.Parameter(torch.tensor(0.0))

    model = Model()

    def energy_forces(model, calc, atoms, task_name):
        return model.energy_per_atom * len(atoms), torch.zeros((len(atoms), 3))

    monkeypatch.setattr(finetune_uma, "uma_energy_forces", energy_forces)

    def frame(n):
        atoms = Atoms(f"Si{n}")
        atoms.arrays["forces"] = np.zeros((n, 3))
        return atoms

    train = [frame(1), frame(8)]
    history = finetune_uma._train_uma(
        model,
        None,
        train,
        [-1.0, -8.0],  # -1 eV/atom
        val_atoms=[frame(2)],
        val_energies=[-1.0],  # -0.5 eV/atom: overshooting it raises the validation loss
        task_name="omat",
        epochs=30,
        lr=0.2,
        batch_size=2,
        force_weight=1.0,
        freeze_backbone=False,
        weight_decay=0.0,
        seed=0,
    )
    val = history["validation_loss"]
    assert history["best_epoch"] == int(np.argmin(val)) + 1 < 30
    # Restored weights give the best validation loss, not the last epoch's.
    assert ((model.energy_per_atom.item() * 2 + 1.0) / 2) ** 2 == pytest.approx(min(val))


def test_split_keeps_at_least_one_validation_frame():
    from matsim_agents.training.finetune_uma import split_indices

    train, val = split_indices(3, val_fraction=0.1, seed=0)
    assert len(val) == 1 and sorted(train + val) == [0, 1, 2]
    assert split_indices(5, val_fraction=0.0, seed=0) == ([0, 1, 2, 3, 4], [])


def test_mace_trains_on_aligned_energies_with_foundation_atomic_energies(tmp_path, monkeypatch):
    from matsim_agents.training import finetune_mace

    frames = []
    for index in range(3):
        atoms = Atoms("Si2", positions=[[0, 0, 0], [1.4 + 0.1 * index, 0, 0]])
        atoms.info["energy"] = -200.0
        atoms.arrays["forces"] = np.zeros((2, 3))
        atoms.info["stress"] = np.zeros(6)
        frames.append(atoms)
    train_xyz, val_xyz, n_train, n_val, with_stress = finetune_mace._write_xyz_splits(
        frames, [-10.0, -11.0, -12.0], tmp_path, seed=0
    )
    written = read(train_xyz, index=":") + read(val_xyz, index=":")
    assert (n_train, n_val, with_stress) == (2, 1, True)
    assert sorted(atoms.info["REF_energy"] for atoms in written) == [-12.0, -11.0, -10.0]
    assert all("REF_stress" in atoms.info for atoms in written)

    commands = []
    monkeypatch.setattr(finetune_mace.subprocess, "run", lambda cmd, check: commands.append(cmd))
    finetune_mace._run_mace_train(
        foundation="foundation.model",
        train_xyz=train_xyz,
        val_xyz=val_xyz,
        run_name="ft",
        work_dir=tmp_path,
        model_dir=tmp_path,
        device="cpu",
        dtype="float64",
        epochs=1,
        batch_size=1,
        lr=1e-3,
        weight_decay=0.0,
        lora=True,
        lora_rank=4,
        lora_alpha=1.0,
        with_stress=True,
    )
    cmd = commands[0]

    def value(flag):
        return cmd[cmd.index(flag) + 1]

    assert value("--E0s") == "foundation"
    assert value("--multiheads_finetuning") == "False"
    assert value("--loss") == "stress" and value("--stress_key") == "REF_stress"
    assert value("--compute_stress") == "True" and value("--lora") == "True"


# -- keeping models loaded ------------------------------------------------------


def _alloc():
    return Allocation(scheduler="slurm", nodes=("nid000000",), gpus_per_node=4)


def test_placement_knows_whether_pools_share_devices():
    separate = ResourcePlan.model_validate(
        {
            "preset": "perlmutter-gpu",
            "pools": {
                "mlip": {"devices": [0]},
                "dft": {"devices": [1, 2]},
                "training": {"devices": [3]},
            },
        }
    ).resolve(_alloc())
    assert not separate.shares_devices("mlip", ("dft", "training"))
    shared = ResourcePlan.model_validate(
        {
            "preset": "perlmutter-gpu",
            "pools": {
                "mlip": {"devices": "all", "share_with": ["dft"]},
                "dft": {"devices": "all", "share_with": ["mlip"]},
            },
        }
    ).resolve(_alloc())
    assert shared.shares_devices("mlip", ("dft", "training"))
    assert shared.shares_devices("missing", ("dft",))


@pytest.fixture
def surrogate(monkeypatch):
    import matsim_agents.active_learning.loop as loop_mod

    loads, releases, pools = [], [], []

    def make(mlip_cfg, enable_mc_dropout):
        loads.append((mlip_cfg.hydragnn.logdir, enable_mc_dropout))
        return SimpleNamespace(model=mlip_cfg.hydragnn.logdir)

    class Pool:
        def __init__(self):
            self.closed = False
            pools.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

        def close(self):
            self.closed = True

    monkeypatch.setattr(loop_mod, "make_mlip_calculator", make)
    monkeypatch.setattr(loop_mod, "release_device_memory", lambda: releases.append(1))
    monkeypatch.setattr(loop_mod.MLIPWorkerPool, "from_placement", lambda placement: Pool())
    return loop_mod._SurrogateModels, loads, releases, pools


def _mlip(logdir):
    from matsim_agents.backends.mlip.config import HydraGNNConfig, MLIPConfig

    return MLIPConfig(backend="hydragnn", hydragnn=HydraGNNConfig(logdir=logdir))


def test_models_stay_loaded_until_the_model_changes(tmp_path, surrogate):
    models_type, loads, releases, pools = surrogate
    models = models_type(placement=None, keep=True)
    incumbent = _mlip(tmp_path)
    for _iteration in range(3):
        primary, _ = models.calculators(incumbent, False)
        with models.md_pool():
            pass
        models.after_scoring()
    assert loads == [(tmp_path, False)] and releases == [] and len(pools) == 1
    assert models.incumbent_calculator(incumbent) is primary
    promoted = _mlip(tmp_path / "promoted")
    models.calculators(promoted, False)
    assert loads[-1] == (tmp_path / "promoted", False) and releases == [1]
    models.close()
    assert pools[0].closed


def test_models_are_reloaded_when_their_devices_are_shared(tmp_path, surrogate):
    models_type, loads, releases, pools = surrogate
    models = models_type(placement=None, keep=False)
    incumbent = _mlip(tmp_path)
    for _iteration in range(2):
        models.calculators(incumbent, False)
        with models.md_pool():
            pass
        models.after_scoring()
        assert models.incumbent_calculator(incumbent) is None
    assert len(loads) == 2 and len(releases) == 2 and len(pools) == 2
    assert all(pool.closed for pool in pools)


def test_dropout_models_are_not_reused_for_promotion(tmp_path, surrogate):
    models_type, *_ = surrogate
    models = models_type(placement=None, keep=True)
    incumbent = _mlip(tmp_path)
    models.calculators(incumbent, True)
    assert models.incumbent_calculator(incumbent) is None


def test_promotion_comparison_uses_the_loaded_incumbent(monkeypatch):
    from matsim_agents.active_learning import evaluate

    loaded = object()
    monkeypatch.setattr(
        "matsim_agents.backends.mlip.calculator.make_mlip_calculator",
        lambda cfg: pytest.fail("the loaded incumbent must not be reloaded"),
    )
    seen = []
    monkeypatch.setattr(
        evaluate,
        "predict_elemental_references",
        lambda manifest, calc, required_elements: (
            seen.append(calc) or (_ for _ in ()).throw(RuntimeError("stop"))
        ),
    )
    frame = Atoms("Si", cell=[3, 3, 3], pbc=True)
    with pytest.raises(RuntimeError, match="stop"):
        evaluate.evaluate_frames(
            SimpleNamespace(), [frame], elemental_reference_manifest="m.json", calculator=loaded
        )
    assert seen == [loaded]


def test_fine_tuning_writes_its_alignment(tmp_path, references):
    alignment = energy_alignment(references, {"Nb"}, _model_energy({"Nb": -10.0}), base_model="m")
    alignment.source_dataset_sha256 = "abc"
    path = alignment.write(tmp_path)
    record = json.loads(path.read_text())
    assert record["convention"] == "dft_formation_on_model_elemental_scale"
    assert record["source_dataset_sha256"] == "abc"
