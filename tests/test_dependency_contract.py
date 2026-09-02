from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hydragnn_main_dependency_contract() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]

    assert project["requires-python"] == ">=3.11,<3.15"
    assert {
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    } <= set(project["classifiers"])
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
    assert project["optional-dependencies"]["mace"] == ["mace-torch==0.3.16"]


def test_ci_python_matrix_matches_supported_range() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert 'python-version: ["3.11", "3.12", "3.13", "3.14"]' in workflow
    assert '"3.10"' not in workflow


def test_mace_and_hydragnn_e3nn_contracts_are_explicitly_isolated() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = config["project"]["optional-dependencies"]

    assert "e3nn==0.5.1" in extras["hydragnn"]
    installer = (ROOT / "deployments/common/setup/install-mace-compat.sh").read_text(
        encoding="utf-8"
    )
    assert 'E3NN_VERSION="${E3NN_VERSION:-0.4.4}"' in installer
    assert 'MACE_VENV_PATH="${MACE_VENV_PATH:-${MATSIM_DIR}/.venv-mace}"' in installer
