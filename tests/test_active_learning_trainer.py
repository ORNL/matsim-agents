from __future__ import annotations

import json
import socket
import subprocess
import sys

import pytest

from matsim_agents.active_learning import trainer as trainer_module
from matsim_agents.active_learning.config import TrainerConfig
from matsim_agents.active_learning.trainer import _fine_tune_argv, retrain_uma
from matsim_agents.backends.mlip.config import HydraGNNConfig, MACEConfig, UMAConfig
from matsim_agents.execution.placement import ProcessLaunch

# Direct (launcher-less) training must be placed on the driver node.
LAUNCH = ProcessLaunch(
    hosts=(socket.gethostname(),), devices=(0,), ranks_per_node=1, threads_per_rank=1
)


# A stand-in fine-tuner: writes a checkpoint and the energy alignment a real
# fine-tune records (hashes of the dataset and references it used).
_ALIGNMENT = """
import hashlib, json, sys
from pathlib import Path
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def align(out, dataset, manifest):
    (Path(out) / "energy-alignment.json").write_text(json.dumps({
        "manifest_sha256": sha(manifest), "backend": "qe", "method_signature": "m",
        "dft_eV_per_atom": {"Si": -100.0}, "model_eV_per_atom": {"Si": -5.0},
        "model_references_from": "base", "source_dataset_sha256": sha(dataset)}))
"""


def _uma_finetuner(tmp_path, monkeypatch, checkpoint="inference_ckpt.pt"):
    """Run a stand-in script wherever the trainer runs ``python -m finetune_uma``."""
    script = tmp_path / "train.py"
    script.write_text(
        _ALIGNMENT
        + f"""
import argparse
p = argparse.ArgumentParser()
for flag in ("--dataset", "--output-dir", "--base-model", "--task-name", "--epochs",
             "--elemental-reference-manifest", "--device"):
    p.add_argument(flag)
p.add_argument("--lora", action="store_true")
a = p.parse_args()
out = Path(a.output_dir)
(out / {checkpoint!r}).parent.mkdir(parents=True, exist_ok=True)
(out / {checkpoint!r}).write_text("lora" if a.lora else "full")
align(out, a.dataset, a.elemental_reference_manifest)
""",
        encoding="utf-8",
    )
    real_run = subprocess.run

    def run(argv, **kwargs):
        assert argv[:3] == [sys.executable, "-m", "matsim_agents.training.finetune_uma"]
        return real_run([sys.executable, str(script), *argv[3:]], **kwargs)

    monkeypatch.setattr(trainer_module.subprocess, "run", run)
    return script


@pytest.fixture
def references(tmp_path, monkeypatch):
    from matsim_agents.active_learning import trainer

    manifest = tmp_path / "elemental-references.json"
    manifest.write_text("{}")
    monkeypatch.setattr(trainer, "_references_for_training", lambda *a, **k: manifest)
    return manifest


def _trainer(references, **extra):
    return TrainerConfig(
        enabled=True, validation_reference_set=references, **{"epochs_per_iter": 2, **extra}
    )


def test_retrain_uma_returns_verified_checkpoint(tmp_path, monkeypatch, references):
    dataset = tmp_path / "dataset.extxyz"
    dataset.write_text("frames")
    output = tmp_path / "model"
    _uma_finetuner(tmp_path, monkeypatch)

    checkpoint = retrain_uma(
        _trainer(references),
        UMAConfig(model_name="uma-s-1p1"),
        dataset,
        iteration=0,
        out_model_dir=output,
        launch=LAUNCH,
    )

    assert checkpoint == output / "inference_ckpt.pt"
    # LoRA is the default fine-tune mode.
    assert checkpoint.read_text() == "lora"


def test_retrain_uma_finds_nested_fairchem_checkpoint(tmp_path, monkeypatch, references):
    dataset = tmp_path / "dataset.extxyz"
    dataset.write_text("frames")
    output = tmp_path / "model"
    _uma_finetuner(tmp_path, monkeypatch, "run-1/checkpoints/final/inference_ckpt.pt")

    checkpoint = retrain_uma(
        _trainer(references, finetune_mode="full"),
        UMAConfig(model_name="uma-s-1p1"),
        dataset,
        iteration=0,
        out_model_dir=output,
        launch=LAUNCH,
    )

    assert checkpoint == output / "run-1" / "checkpoints" / "final" / "inference_ckpt.pt"
    assert checkpoint.read_text() == "full"


def test_retrain_rejects_a_model_trained_on_other_data(tmp_path, monkeypatch, references):
    dataset = tmp_path / "dataset.extxyz"
    dataset.write_text("frames")
    script = _uma_finetuner(tmp_path, monkeypatch)
    # The fine-tune reads one dataset; the file changes before it is checked.
    script.write_text(
        script.read_text() + "Path(a.dataset).write_text('changed')\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match the requested training dataset"):
        retrain_uma(
            _trainer(references),
            UMAConfig(model_name="uma-s-1p1"),
            dataset,
            iteration=0,
            out_model_dir=tmp_path / "model",
            launch=LAUNCH,
        )


def test_retrain_mace_launcher_gets_references_and_mode(tmp_path, references):
    from matsim_agents.active_learning.trainer import retrain_mace

    dataset = tmp_path / "dataset.extxyz"
    dataset.write_text("frames")
    # The launcher gets `<backend> <nodes> <ranks> -m <module> <args...>` and runs python.
    helper = tmp_path / "finetuner.py"
    helper.write_text(
        _ALIGNMENT
        + """
import argparse
p = argparse.ArgumentParser()
for flag in ("--dataset", "--output-dir", "--family", "--base-model", "--precision",
             "--epochs", "--elemental-reference-manifest", "--device"):
    p.add_argument(flag)
p.add_argument("--lora", action="store_true")
a = p.parse_args(sys.argv[3:])
out = Path(a.output_dir)
(out / "argv.json").write_text(json.dumps(sys.argv[1:]))
(out / "mace_finetuned.model").touch()
align(out, a.dataset, a.elemental_reference_manifest)
""",
        encoding="utf-8",
    )
    launcher = tmp_path / "launcher.sh"
    launcher.write_text(
        f'shift 3\n[ "$1" = -m ] || exit 2\nexec {sys.executable} {helper} "$@"\n',
        encoding="utf-8",
    )
    output = tmp_path / "model"
    model = retrain_mace(
        _trainer(references, train_launcher=launcher),
        MACEConfig(family="mace_mp", model="medium"),
        dataset,
        iteration=0,
        out_model_dir=output,
        launch=LAUNCH,
    )
    assert model == output / "mace_finetuned.model"
    argv = json.loads((output / "argv.json").read_text())
    assert argv[:2] == ["-m", "matsim_agents.training.finetune_mace"]
    flags = dict(zip(argv[2::2], argv[3::2], strict=False))
    assert flags["--family"] == "mace_mp" and flags["--base-model"] == "medium"
    assert flags["--elemental-reference-manifest"] == str(references)
    assert argv[-1] == "--lora"


def test_retrain_mace_requires_the_finetuned_model(tmp_path, references):
    from matsim_agents.active_learning.trainer import retrain_mace

    launcher = tmp_path / "launcher.sh"
    launcher.write_text("exit 0\n", encoding="utf-8")
    (tmp_path / "dataset.extxyz").write_text("frames")
    with pytest.raises(RuntimeError, match="did not write"):
        retrain_mace(
            _trainer(references, train_launcher=launcher),
            MACEConfig(),
            tmp_path / "dataset.extxyz",
            iteration=0,
            out_model_dir=tmp_path / "model",
            launch=LAUNCH,
        )


@pytest.mark.parametrize("backend", ["uma", "mace"])
def test_direct_training_must_be_local_and_single_process(tmp_path, backend):
    from matsim_agents.active_learning.trainer import retrain_mace

    references = tmp_path / "elemental-references.json"
    references.touch()
    config = TrainerConfig(enabled=True, validation_reference_set=references)

    def run(launch):
        if backend == "uma":
            return retrain_uma(
                config, UMAConfig(), tmp_path / "d.extxyz", 0, tmp_path / "o", launch=launch
            )
        return retrain_mace(
            config, MACEConfig(), tmp_path / "d.extxyz", 0, tmp_path / "o", launch=launch
        )

    remote = LAUNCH.model_copy(update={"hosts": ("some-other-node",)})
    with pytest.raises(ValueError, match="place the training pool on the driver's node"):
        run(remote)
    wide = LAUNCH.model_copy(update={"ranks_per_node": 4})
    with pytest.raises(ValueError, match=r"1 node\(s\) x 4 rank"):
        run(wide)
    assert not (tmp_path / "o").exists()


def _hydragnn(tmp_path, **extra):
    return HydraGNNConfig(logdir=tmp_path / "gfm", mlp_checkpoint=tmp_path / "mlp.pt", **extra)


@pytest.mark.parametrize(
    ("backend", "module"),
    [
        ("hydragnn", "matsim_agents.training.finetune_hydragnn"),
        ("hydragnn-newhead", "matsim_agents.training.finetune_hydragnn_newhead"),
        ("uma", "matsim_agents.training.finetune_uma"),
        ("mace", "matsim_agents.training.finetune_mace"),
    ],
)
@pytest.mark.parametrize("launcher", [False, True])
def test_fine_tune_argv_runs_each_backends_module(tmp_path, references, backend, module, launcher):
    mlip = {
        "hydragnn": lambda: _hydragnn(tmp_path),
        "hydragnn-newhead": lambda: _hydragnn(tmp_path, fine_tuner="finetune_hydragnn_newhead"),
        "uma": lambda: UMAConfig(),
        "mace": lambda: MACEConfig(),
    }[backend]()
    script = tmp_path / "launcher.sh"
    script.touch()
    config = _trainer(references, epochs_per_iter=3, train_launcher=script if launcher else None)
    argv = _fine_tune_argv(
        config,
        mlip,
        LAUNCH,
        dataset=tmp_path / "d.extxyz",
        out_dir=tmp_path / "out",
        references=tmp_path / "refs.json",
        seed=11,
    )
    prefix = ["bash", str(script), mlip.backend, "1", "1"] if launcher else [sys.executable]
    assert argv[: len(prefix) + 2] == [*prefix, "-m", module]
    args = argv[len(prefix) + 2 :]
    assert args == mlip.fine_tune_args(
        dataset=tmp_path / "d.extxyz",
        out_dir=tmp_path / "out",
        epochs=3,
        references=tmp_path / "refs.json",
        finetune_mode=config.finetune_mode,
        seed=11,
    )
    flags = dict(zip(args[::2], args[1::2], strict=False))
    assert flags["--dataset"] == str(tmp_path / "d.extxyz")
    assert flags["--output-dir"] == str(tmp_path / "out")
    assert flags["--epochs"] == "3"
    assert flags["--seed"] == "11"
    assert flags["--elemental-reference-manifest"] == str(tmp_path / "refs.json")


def test_swapping_the_mlip_block_swaps_the_fine_tuner(tmp_path, references):
    """The same trainer settings fine-tune whichever backend the mlip block selects."""
    config = _trainer(references)

    def module(mlip):
        argv = _fine_tune_argv(
            config,
            mlip,
            LAUNCH,
            dataset=tmp_path / "d.extxyz",
            out_dir=tmp_path / "out",
            references=tmp_path / "refs.json",
            seed=0,
        )
        return argv[argv.index("-m") + 1]

    hydragnn = _hydragnn(tmp_path)
    assert module(hydragnn) == "matsim_agents.training.finetune_hydragnn"
    assert module(UMAConfig()) == "matsim_agents.training.finetune_uma"
    assert module(MACEConfig()) == "matsim_agents.training.finetune_mace"


def test_routed_hydragnn_fine_tune_requires_a_branch_mlp(tmp_path, references):
    with pytest.raises(ValueError, match="requires mlp_checkpoint"):
        _fine_tune_argv(
            _trainer(references),
            HydraGNNConfig(logdir=tmp_path / "gfm"),
            LAUNCH,
            dataset=tmp_path / "d.extxyz",
            out_dir=tmp_path / "out",
            references=tmp_path / "refs.json",
            seed=0,
        )


@pytest.mark.parametrize(
    ("retrain", "mlip"),
    [
        (trainer_module.retrain_uma, UMAConfig()),
        (trainer_module.retrain_mace, MACEConfig()),
        (trainer_module.retrain_hydragnn, HydraGNNConfig(logdir="/l", mlp_checkpoint="/m.pt")),
    ],
    ids=["uma", "mace", "hydragnn"],
)
def test_multirank_fine_tunes_are_rejected_before_any_setup(
    tmp_path, monkeypatch, references, retrain, mlip
):
    def unexpected(*args, **kwargs):
        pytest.fail("references must not be resolved for a rejected fine-tune")

    monkeypatch.setattr(trainer_module, "_references_for_training", unexpected)
    import matsim_agents.active_learning.training_references as training_references

    monkeypatch.setattr(training_references, "resolve_training_references", unexpected)
    two_ranks = LAUNCH.model_copy(update={"ranks_per_node": 2})
    config = _trainer(references, train_launcher=tmp_path / "launcher.sh")
    with pytest.raises(ValueError, match="run as one process"):
        retrain(config, mlip, tmp_path / "d.extxyz", 0, tmp_path / "out", seed=0, launch=two_ranks)


@pytest.mark.parametrize(
    ("mlip", "supported"),
    [
        (HydraGNNConfig(logdir="/l", mlp_checkpoint="/m.pt", device="xpu"), True),
        (UMAConfig(device="xpu"), False),
        (MACEConfig(device="xpu"), False),
    ],
    ids=["hydragnn", "uma", "mace"],
)
def test_fine_tuning_rejects_devices_the_recipe_cannot_train_on(references, mlip, supported):
    from pathlib import Path

    def argv():
        return _fine_tune_argv(
            _trainer(references),
            mlip,
            LAUNCH,
            dataset=Path("d.extxyz"),
            out_dir=Path("o"),
            references=references,
            seed=0,
        )

    if supported:
        assert argv()[argv().index("--device") + 1] == "xpu"
    else:
        with pytest.raises(ValueError, match="does not support device='xpu'"):
            argv()
