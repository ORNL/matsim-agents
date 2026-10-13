from __future__ import annotations

import shlex
import tomllib
from pathlib import Path

import yaml
from packaging.requirements import Requirement
from packaging.version import Version

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
    assert {"ase==3.26.0", "numpy==2.4.6", "scipy==1.17.1"} <= set(project["dependencies"])
    assert {
        "e3nn==0.5.1",
        "torch-ema==0.3",
        "torchmetrics==1.4.0",
        "torch-geometric==2.8.0",
    } <= set(project["optional-dependencies"]["hydragnn"])
    requirements = {
        requirement.name: requirement
        for requirement in map(Requirement, project["optional-dependencies"]["hydragnn"])
    }
    for version in ("2.13.0a0+gitcf30153", "2.13.0", "2.14.0"):
        assert version in requirements["torch"].specifier
    for version in ("2.12.0", "2.15.0"):
        assert version not in requirements["torch"].specifier
    for version in ("0.28.0a0", "0.28.0", "0.29.0"):
        assert version in requirements["torchvision"].specifier
    for version in ("0.27.0", "0.30.0"):
        assert version not in requirements["torchvision"].specifier
    assert not any(
        dep.startswith("torchaudio") for dep in project["optional-dependencies"]["hydragnn"]
    )
    assert "dscribe>=2.1" in project["dependencies"]
    assert "fairchem-core>=2.20" in project["optional-dependencies"]["uma"]
    assert project["optional-dependencies"]["mace"] == ["mace-torch==0.3.16"]


def test_ci_python_matrix_matches_supported_range() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert 'python-version: ["3.11", "3.12", "3.13", "3.14"]' in workflow
    assert '"3.10"' not in workflow


def test_full_ci_installs_cpu_torch_before_regression_suite() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    torch_requirement = next(
        requirement
        for requirement in map(Requirement, config["project"]["optional-dependencies"]["hydragnn"])
        if requirement.name == "torch"
    )
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["full"]["steps"]
    install_index = next(
        index for index, step in enumerate(steps)
        if "https://download.pytorch.org/whl/cpu" in shlex.split(step.get("run", ""))
    )
    tokens = shlex.split(steps[install_index]["run"])
    assert tokens[tokens.index("--index-url") + 1] == "https://download.pytorch.org/whl/cpu"
    ci_torch = next(Requirement(token) for token in tokens if token.startswith("torch"))
    pins = list(ci_torch.specifier)
    assert ci_torch.name == "torch" and len(pins) == 1 and pins[0].operator == "=="
    version = Version(pins[0].version)
    assert version.local == "cpu"
    assert version in torch_requirement.specifier
    test_index = next(
        index
        for index, step in enumerate(steps)
        if "bash scripts/check.sh coverage" in step.get("run", "")
    )
    assert install_index < test_index


def test_mace_and_hydragnn_e3nn_contracts_are_explicitly_isolated() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = config["project"]["optional-dependencies"]

    assert "e3nn==0.5.1" in extras["hydragnn"]
    installer = (ROOT / "deployments/common/setup/install-mace-compat.sh").read_text(
        encoding="utf-8"
    )
    assert 'E3NN_VERSION="${E3NN_VERSION:-0.4.4}"' in installer
    assert 'MACE_VENV_PATH="${MACE_VENV_PATH:-${MATSIM_DIR}/.venv-mace}"' in installer
