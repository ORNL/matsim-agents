from __future__ import annotations

from matsim_agents.active_learning.config import TrainerConfig, UMAConfig
from matsim_agents.active_learning.trainer import retrain_uma


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
    )

    assert checkpoint == output / "inference_ckpt.pt"
    assert checkpoint.is_file()
