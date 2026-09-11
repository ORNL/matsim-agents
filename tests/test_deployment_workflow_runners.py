from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from matsim_agents.active_learning.config import ALConfig
from matsim_agents.workflows.relaxation import ScientificRelaxationConfig

ROOT = Path(__file__).resolve().parents[1]


def _fake_cli(tmp_path: Path) -> Path:
    executable = tmp_path / "matsim-agents"
    executable.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable


def test_shared_relaxation_runner_generates_typed_config(tmp_path: Path) -> None:
    _fake_cli(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "REPO": str(ROOT),
        "RUN_DIR": str(tmp_path / "run"),
        "STRUCTURE": str(ROOT / "tests/integration/data/Si.vasp"),
        "LOGDIR": str(tmp_path / "model"),
        "HYDRAGNN_BRANCH_MLP_CHECKPOINT": str(tmp_path / "weights.pt"),
    }
    Path(env["RUN_DIR"]).mkdir()
    subprocess.run(
        ["bash", str(ROOT / "deployments/common/run-mlip-relaxation.sh")],
        env=env,
        check=True,
    )
    raw = json.loads((Path(env["RUN_DIR"]) / "relaxation-config.json").read_text())
    cfg = ScientificRelaxationConfig.model_validate(raw)
    assert cfg.mode == "mlip"
    assert cfg.structure_path == env["STRUCTURE"]


def test_shared_active_learning_runner_generates_current_schema(tmp_path: Path) -> None:
    _fake_cli(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    seeds = [ROOT / "tests/integration/data/Si.vasp", ROOT / "tests/integration/data/MgO.vasp"]
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "REPO": str(ROOT),
        "RUN_DIR": str(run_dir),
        "MATSIM_SEED_STRUCTURES": ":".join(map(str, seeds)),
        "LOGDIR": str(tmp_path / "model"),
        "HYDRAGNN_BRANCH_MLP_CHECKPOINT": str(tmp_path / "weights.pt"),
        "MATSIM_DFT_BACKEND": "qe",
        "MATSIM_PW_BIN": "/opt/qe/bin/pw.x",
        "MATSIM_PSEUDO_DIR": "/opt/qe/pseudo",
        "MATSIM_DFT_WRAPPER": str(ROOT / "deployments/perlmutter/launchers/_qe-step-perlmutter.sh"),
        "MATSIM_DFT_RANKS_PER_NODE": "4",
    }
    subprocess.run(
        ["bash", str(ROOT / "deployments/common/run-active-learning.sh")],
        env=env,
        check=True,
    )
    raw = json.loads((run_dir / "active-learning-config.json").read_text())
    cfg = ALConfig.model_validate(raw)
    assert cfg.acquisition.strategy == "mc_dropout"
    assert cfg.dft.backend == "qe"
    assert cfg.trainer.enabled is False
    assert list(cfg.md.seed_source.paths) == seeds
