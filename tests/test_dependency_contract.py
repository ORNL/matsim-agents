from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hydragnn_main_dependency_contract() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]

    assert project["requires-python"] == ">=3.11,<3.15"
    assert {"ase==3.26.0", "numpy==2.2.6", "scipy==1.17.1"} <= set(project["dependencies"])
    assert {
        "torch==2.13.0",
        "torchvision==0.28.0",
        "torchaudio==2.11.0",
        "e3nn==0.5.1",
        "torch-ema==0.3",
        "torchmetrics==1.4.0",
        "torch-geometric==2.8.0",
    } <= set(project["optional-dependencies"]["hydragnn"])
    assert "fairchem-core" in project["optional-dependencies"]["uma"]
