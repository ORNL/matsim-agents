from __future__ import annotations

import socket
import sys

import pytest

from matsim_agents.active_learning.config import TrainerConfig
from matsim_agents.active_learning.trainer import retrain_uma
from matsim_agents.backends.mlip.config import UMAConfig
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


def _uma_script(tmp_path, checkpoint="inference_ckpt.pt"):
    script = tmp_path / "train.py"
    script.write_text(
        _ALIGNMENT
        + f"""
import argparse
p = argparse.ArgumentParser()
for flag in ("--dataset", "--output-dir", "--base-model", "--task-name", "--epochs",
             "--elemental-reference-manifest"):
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
    return script


@pytest.fixture
def references(tmp_path, monkeypatch):
    from matsim_agents.active_learning import trainer

    manifest = tmp_path / "elemental-references.json"
    manifest.write_text("{}")
    monkeypatch.setattr(trainer, "_references_for_training", lambda *a, **k: manifest)
    return manifest


def _trainer(script, references, **extra):
    return TrainerConfig(
        enabled=True,
        train_script=script,
        epochs_per_iter=2,
        validation_reference_set=references,
        **extra,
    )


def test_retrain_uma_returns_verified_checkpoint(tmp_path, references):
    dataset = tmp_path / "dataset.extxyz"
    dataset.write_text("frames")
    output = tmp_path / "model"

    checkpoint = retrain_uma(
        _trainer(_uma_script(tmp_path), references),
        UMAConfig(model_name="uma-s-1p1"),
        dataset,
        iteration=0,
        out_model_dir=output,
        launch=LAUNCH,
    )

    assert checkpoint == output / "inference_ckpt.pt"
    # LoRA is the default fine-tune mode.
    assert checkpoint.read_text() == "lora"


def test_retrain_uma_finds_nested_fairchem_checkpoint(tmp_path, references):
    dataset = tmp_path / "dataset.extxyz"
    dataset.write_text("frames")
    output = tmp_path / "model"

    checkpoint = retrain_uma(
        _trainer(
            _uma_script(tmp_path, "run-1/checkpoints/final/inference_ckpt.pt"),
            references,
            finetune_mode="full",
        ),
        UMAConfig(model_name="uma-s-1p1"),
        dataset,
        iteration=0,
        out_model_dir=output,
        launch=LAUNCH,
    )

    assert checkpoint == output / "run-1" / "checkpoints" / "final" / "inference_ckpt.pt"
    assert checkpoint.read_text() == "full"


def test_retrain_rejects_a_model_trained_on_other_data(tmp_path, references):
    dataset = tmp_path / "dataset.extxyz"
    dataset.write_text("frames")
    script = _uma_script(tmp_path)
    # The fine-tune reads one dataset; the file changes before it is checked.
    script.write_text(
        script.read_text() + "Path(a.dataset).write_text('changed')\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match the requested training dataset"):
        retrain_uma(
            _trainer(script, references),
            UMAConfig(model_name="uma-s-1p1"),
            dataset,
            iteration=0,
            out_model_dir=tmp_path / "model",
            launch=LAUNCH,
        )


def test_retrain_mace_launcher_gets_references_and_mode(tmp_path, references):
    from matsim_agents.active_learning.trainer import retrain_mace
    from matsim_agents.backends.mlip.config import MACEConfig

    dataset = tmp_path / "dataset.extxyz"
    dataset.write_text("frames")
    helper = tmp_path / "align.py"
    helper.write_text(
        _ALIGNMENT + "align(sys.argv[1], sys.argv[2], sys.argv[3])\n", encoding="utf-8"
    )
    launcher = tmp_path / "launcher.sh"
    launcher.write_text(
        'echo "$@" > "$3/argv"\n'
        'touch "$3/mace_finetuned.model"\n'
        f'{sys.executable} {helper} "$3" "$2" "$9"\n',
        encoding="utf-8",
    )
    output = tmp_path / "model"
    model = retrain_mace(
        _trainer(tmp_path / "unused.py", references, train_launcher=launcher),
        MACEConfig(family="mace_mp", model="medium"),
        dataset,
        iteration=0,
        out_model_dir=output,
        launch=LAUNCH,
    )
    assert model == output / "mace_finetuned.model"
    argv = (output / "argv").read_text().split()
    assert argv[3:5] == ["mace_mp", "medium"] and argv[8:] == [str(references), "lora"]


def test_retrain_mace_requires_the_finetuned_model(tmp_path, references):
    from matsim_agents.active_learning.trainer import retrain_mace
    from matsim_agents.backends.mlip.config import MACEConfig

    launcher = tmp_path / "launcher.sh"
    launcher.write_text("exit 0\n", encoding="utf-8")
    (tmp_path / "dataset.extxyz").write_text("frames")
    with pytest.raises(RuntimeError, match="did not write"):
        retrain_mace(
            _trainer(tmp_path / "unused.py", references, train_launcher=launcher),
            MACEConfig(),
            tmp_path / "dataset.extxyz",
            iteration=0,
            out_model_dir=tmp_path / "model",
            launch=LAUNCH,
        )


@pytest.mark.parametrize("backend", ["uma", "mace"])
def test_direct_training_must_be_local_and_single_process(tmp_path, backend):
    from matsim_agents.active_learning.trainer import retrain_mace
    from matsim_agents.backends.mlip.config import MACEConfig

    script = tmp_path / "train.py"
    script.touch()
    references = tmp_path / "elemental-references.json"
    references.touch()
    config = TrainerConfig(enabled=True, train_script=script, validation_reference_set=references)

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
