from __future__ import annotations

import socket

import pytest

from matsim_agents.active_learning.config import TrainerConfig
from matsim_agents.active_learning.trainer import retrain_uma
from matsim_agents.backends.mlip.config import UMAConfig
from matsim_agents.execution.placement import ProcessLaunch

# Direct (launcher-less) training must be placed on the driver node.
LAUNCH = ProcessLaunch(
    hosts=(socket.gethostname(),), devices=(0,), ranks_per_node=1, threads_per_rank=1
)


def test_retrain_uma_returns_verified_checkpoint(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        """from pathlib import Path
import argparse
p = argparse.ArgumentParser()
p.add_argument('--dataset')
p.add_argument('--output-dir')
p.add_argument('--base-model')
p.add_argument('--task-name')
p.add_argument('--epochs')
a = p.parse_args()
Path(a.output_dir).mkdir(parents=True, exist_ok=True)
(Path(a.output_dir) / 'inference_ckpt.pt').write_text('checkpoint')
""",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.extxyz"
    dataset.touch()
    output = tmp_path / "model"

    checkpoint = retrain_uma(
        TrainerConfig(enabled=True, train_script=script, epochs_per_iter=2),
        UMAConfig(model_name="uma-s-1p1"),
        dataset,
        iteration=0,
        out_model_dir=output,
        launch=LAUNCH,
    )

    assert checkpoint == output / "inference_ckpt.pt"
    assert checkpoint.is_file()


def test_retrain_uma_finds_nested_fairchem_checkpoint(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        """from pathlib import Path
import argparse
p = argparse.ArgumentParser()
p.add_argument('--dataset')
p.add_argument('--output-dir')
p.add_argument('--base-model')
p.add_argument('--task-name')
p.add_argument('--epochs')
a = p.parse_args()
checkpoint = Path(a.output_dir) / 'run-1' / 'checkpoints' / 'final' / 'inference_ckpt.pt'
checkpoint.parent.mkdir(parents=True, exist_ok=True)
checkpoint.write_text('checkpoint')
""",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.extxyz"
    dataset.touch()
    output = tmp_path / "model"

    checkpoint = retrain_uma(
        TrainerConfig(enabled=True, train_script=script, epochs_per_iter=2),
        UMAConfig(model_name="uma-s-1p1"),
        dataset,
        iteration=0,
        out_model_dir=output,
        launch=LAUNCH,
    )

    assert checkpoint == output / "run-1" / "checkpoints" / "final" / "inference_ckpt.pt"


@pytest.mark.parametrize("backend", ["uma", "mace"])
def test_direct_training_must_be_local_and_single_process(tmp_path, backend):
    from matsim_agents.active_learning.trainer import retrain_mace
    from matsim_agents.backends.mlip.config import MACEConfig

    script = tmp_path / "train.py"
    script.touch()
    config = TrainerConfig(enabled=True, train_script=script)

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
