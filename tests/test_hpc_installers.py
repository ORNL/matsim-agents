from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]

INSTALLERS = {
    "frontier": "scripts/hpc/olcf/frontier/installation/install-rocm72.sh",
    "aurora": "scripts/hpc/alcf/aurora/installation/install.sh",
    "perlmutter": "scripts/hpc/nersc/perlmutter/installation/install.sh",
}


def test_hydragnn_extra_accepts_supported_torch_and_aurora_prereleases() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        extras = tomllib.load(stream)["project"]["optional-dependencies"]["hydragnn"]
    requirements = {requirement.name: requirement for requirement in map(Requirement, extras)}
    for version in ("2.13.0a0+gitcf30153", "2.13.0", "2.14.0"):
        assert version in requirements["torch"].specifier
    for version in ("2.12.0", "2.15.0"):
        assert version not in requirements["torch"].specifier
    for version in ("0.28.0a0", "0.28.0", "0.29.0"):
        assert version in requirements["torchvision"].specifier


def test_aurora_detached_runner_records_exit_status(tmp_path) -> None:
    runner = ROOT / "deployments/aurora/setup/run-install-tests.sh"
    local_runner = tmp_path / "run-install-tests.sh"
    local_runner.write_text(runner.read_text())
    environment = os.environ.copy()
    environment.update(MATSIM_DIR=str(ROOT), RUN_DIR=str(tmp_path / "logs"))
    installer = tmp_path / "install.sh"
    for exit_code in (0, 7):
        installer.write_text(
            '#!/bin/bash\n[[ "$RUN_MODEL_TESTS" == "1" ]] || exit 8\n'
            f'exit {exit_code}\n'
        )
        result = subprocess.run(["bash", str(local_runner)], env=environment, capture_output=True)
        assert result.returncode == exit_code
        status = (tmp_path / "logs/status.txt").read_text()
        assert status.startswith("SUCCEEDED" if exit_code == 0 else "FAILED")
        assert f"exit_code={exit_code}" in status


def test_aurora_runs_perlmutter_style_phases_without_switching_checkout(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "commands.log"
    python = fake_bin / "python3"
    python.write_text(
        '#!/bin/bash\nprintf "python %s\\n" "$*" >> "$INSTALL_LOG"\n'
        'if [[ "$*" == "-m pip check" ]]; then exit 1; fi\n'
        'if [[ "$*" == *check-runtime-dependencies.py* ]]; then '
        'exit "${RUNTIME_CHECK_EXIT:-0}"; fi\n'
    )
    python.chmod(0o755)
    git = fake_bin / "git"
    git.write_text('#!/bin/bash\nprintf "git %s\\n" "$*" >> "$INSTALL_LOG"\n'
                   'if [[ "$*" == *rev-parse* ]]; then echo 81d5b88; fi\n')
    git.chmod(0o755)
    startup = tmp_path / "bash-env"
    startup.write_text('module() { printf "module %s\\n" "$*" >> "$INSTALL_LOG"; }\n')
    matsim = tmp_path / "matsim-agents"
    matsim.mkdir()
    (matsim / "pyproject.toml").touch()
    hydra = tmp_path / "HydraGNN"
    (hydra / ".git").mkdir(parents=True)
    hydra_installer = hydra / INSTALLERS["aurora"]
    hydra_installer.parent.mkdir(parents=True)
    hydra_installer.write_text(
        '#!/bin/bash\nprintf "hydra %s %s %s\\n" "$INSTALL_ROOT" "$VENV_PATH" '
        '"$HYDRAGNN_SRC" >> "$INSTALL_LOG"\n'
        'mkdir -p "$VENV_PATH/bin"\ncp "$FAKE_PYTHON" "$VENV_PATH/bin/python"\n'
    )
    for backend in ("uma", "mace"):
        helper = matsim / f"deployments/common/setup/install-{backend}-compat.sh"
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_text(
            '#!/bin/bash\nprintf "compat %s %s %s %s\\n" '
            f'"{backend}" "$BASE_VENV" "${{RECREATE_{backend.upper()}_ENV}}" '
            '"${FACILITY:-}" >> "$INSTALL_LOG"\n'
        )
    vllm = matsim / "deployments/aurora/setup/install-vllm-xpu-aurora.sh"
    vllm.parent.mkdir(parents=True)
    vllm.write_text('#!/bin/bash\necho vllm-xpu >> "$INSTALL_LOG"\n')
    environment = os.environ.copy()
    environment.update(
        PATH=f"{fake_bin}:/usr/bin:/bin", BASH_ENV=str(startup), INSTALL_LOG=str(log),
        FAKE_PYTHON=str(python), MATSIM_DIR=str(matsim), HYDRAGNN_DIR=str(hydra),
        HYDRAGNN_REF="", FRAMEWORKS_MODULE="frameworks/test", RUN_MODEL_TESTS="1",
        VENV_PATH=str(matsim / ".venv"), INSTALL_ROOT=str(matsim / ".hpc-build/aurora"),
        RECREATE_ENV="0", RECREATE_MACE_ENV="1", RECREATE_UMA_ENV="0", INSTALL_VLLM="1",
        HYDRAGNN_INSTALLER=str(hydra_installer),
    )
    installer = ROOT / "deployments/aurora/setup/install.sh"
    subprocess.run(["bash", str(installer)], env=environment, check=True, capture_output=True)
    commands = log.read_text().splitlines()
    assert "module load frameworks/test" in commands
    assert not any("switch" in command or "merge" in command for command in commands)
    assert not any(command.startswith(f"git -C {hydra} fetch") for command in commands)
    hydra_step = next(
        index for index, command in enumerate(commands) if command.startswith("hydra ")
    )
    wheel_step = next(
        index for index, command in enumerate(commands) if "--no-build-isolation" in command
    )
    uma_step = commands.index(f"compat uma {matsim / '.venv'} 0 aurora")
    mace_step = commands.index(f"compat mace {matsim / '.venv'} 1 aurora")
    vllm_step = commands.index("vllm-xpu")
    test_step = commands.index(f"python -m pytest {matsim / 'tests'}")
    assert hydra_step < wheel_step < uma_step < mace_step < vllm_step < test_step
    environment["RUNTIME_CHECK_EXIT"] = "9"
    failed = subprocess.run(["bash", str(installer)], env=environment, capture_output=True)
    assert failed.returncode == 9
    environment.pop("RUNTIME_CHECK_EXIT")
    log.write_text("")
    environment["REUSE_HYDRAGNN_DEPENDENCIES"] = "1"
    subprocess.run(["bash", str(installer)], env=environment, check=True, capture_output=True)
    assert not any(command.startswith("hydra ") for command in log.read_text().splitlines())
    assert "torch_spline_conv" in log.read_text()
    environment.pop("REUSE_HYDRAGNN_DEPENDENCIES")
    hydra_installer.unlink()
    failed = subprocess.run(
        ["bash", str(installer)], env=environment, capture_output=True, text=True
    )
    assert failed.returncode != 0
    assert "selected checkout will not be switched automatically" in failed.stderr


def test_facilities_have_one_canonical_current_hydragnn_installer() -> None:
    for facility, hydragnn_path in INSTALLERS.items():
        installer = ROOT / "deployments" / facility / "setup" / "install.sh"
        text = installer.read_text(encoding="utf-8")
        assert installer.stat().st_mode & 0o111
        assert hydragnn_path in text
        assert "https://github.com/ORNL/HydraGNN.git" in text
        assert 'VENV_PATH="${VENV_PATH:-${MATSIM_DIR}/.venv}"' in text
        assert "${MATSIM_DIR}/.hpc-build/" in text
        assert "${HYDRAGNN_DIR}/installation_DOE_supercomputers" not in text
        assert "hf_transfer" in text
        assert 'INSTALL_UMA="${INSTALL_UMA:-0}"' in text
        assert 'UMA_VENV_PATH="${UMA_VENV_PATH:-${MATSIM_DIR}/.venv-uma}"' in text
        assert 'INSTALL_MACE="${INSTALL_MACE:-0}"' in text
        assert 'MACE_VENV_PATH="${MACE_VENV_PATH:-${MATSIM_DIR}/.venv-mace}"' in text
        assert "deployments/common/setup/install-mace-compat.sh" in text
        assert "deployments/common/setup/install-uma-compat.sh" in text
        assert "pip check" in text
        assert "installation_DOE_supercomputers/hydragnn_installation" not in text
        subprocess.run(["bash", "-n", str(installer)], check=True)


def test_runtime_dependency_audit_checks_selected_extras_and_transitive_requirements(monkeypatch):
    from types import SimpleNamespace

    helper = ROOT / "deployments/common/setup/check-runtime-dependencies.py"
    spec = importlib.util.spec_from_file_location("runtime_dependencies", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    packages = {
        "runtime": SimpleNamespace(version="1.0", requires=[
            'torch>=2.13.0.dev0; extra == "model"',
            'missing-package; extra == "unused"',
            'missing-platform; sys_platform == "never"',
        ]),
        "torch": SimpleNamespace(version="2.13.0a0+git123", requires=["numpy>=2"]),
        "numpy": SimpleNamespace(version="2.4.6", requires=[]),
    }

    def installed(name):
        if name not in packages:
            raise module.PackageNotFoundError(name)
        return packages[name]

    monkeypatch.setattr(module, "distribution", installed)
    assert module.check_dependencies(["runtime[model]"]) == []
    packages["numpy"].version = "1.26.4"
    assert module.check_dependencies(["runtime[model]"]) == [
        "numpy>=2 is required, installed 1.26.4"
    ]
    assert module.check_dependencies(["runtime[unused]"]) == [
        "Missing runtime dependency: missing-package; extra == \"unused\""
    ]


def test_aurora_stages_selected_code_with_upstream_dependencies(tmp_path) -> None:
    helper = ROOT / "deployments/aurora/setup/prepare_hydragnn_source.py"
    spec = importlib.util.spec_from_file_location("prepare_hydragnn_source", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    code, dependencies = tmp_path / "code", tmp_path / "upstream"
    code.mkdir()
    dependencies.mkdir()
    subprocess.run(["git", "init", str(code)], check=True, capture_output=True)
    (code / "model.py").write_text("selected_branch = True\n")
    (code / "requirements-torch.txt").write_text("torch==2.8.0\n")
    utilities = code / "hydragnn/utils/distributed"
    utilities.mkdir(parents=True)
    (utilities / "__init__.py").write_text("selected_utility = True\n")
    subprocess.run(["git", "-C", str(code), "add", "."], check=True)
    (dependencies / "requirements-base.txt").write_text("numpy==2.4.6\nscipy==1.17.1\n")
    (dependencies / "requirements-pyg.txt").write_text("torch_geometric==2.8.0\n")
    (dependencies / "requirements-torch.txt").write_text("torch==2.13.0\ntorchvision==0.28.0\n")
    stage = module.prepare_source(code, dependencies, tmp_path / "build", "2.13.0a0+git123")
    assert (stage / "model.py").read_bytes() == (code / "model.py").read_bytes()
    assert (stage / "requirements-base.txt").read_bytes() == (
        dependencies / "requirements-base.txt"
    ).read_bytes()
    assert "torch==2.13.0a0+git123" in (stage / "requirements-torch.txt").read_text()
    assert (code / "requirements-torch.txt").read_text() == "torch==2.8.0\n"
    from setuptools import find_packages

    assert "hydragnn.utils.distributed" in find_packages(where=str(stage))
    assert (stage / "hydragnn/utils/distributed/__init__.py").read_bytes() == (
        utilities / "__init__.py"
    ).read_bytes()
    assert not (code / "hydragnn/utils/__init__.py").exists()
    try:
        module.prepare_source(code, dependencies, code / "build", "2.13.0")
    except ValueError as error:
        assert "outside the selected" in str(error)
    else:
        raise AssertionError("staging inside selected checkout was accepted")


def test_deployment_scripts_do_not_reference_hydragnn_owned_environment() -> None:
    old_fragment = "HydraGNN/installation_DOE_supercomputers"
    for script in (ROOT / "deployments").rglob("*.sh"):
        assert old_fragment not in script.read_text(encoding="utf-8"), script

    generic = (ROOT / "scripts/setup_env.sh").read_text(encoding="utf-8")
    assert "installation_DOE_supercomputers" not in generic
    assert "hydragnn_installation_bash_script" not in generic
    assert "deployments/${FACILITY}/setup/install.sh" in generic
    assert "frontier|frontier-rocm71|frontier-rocm64)" in generic
    assert "perlmutter|aurora)" in generic


def test_aurora_fairchem_wheel_changes_only_torch_dependency(tmp_path):
    import hashlib
    import json
    from email.parser import BytesParser

    import pytest
    from wheel.wheelfile import WheelFile

    helper = ROOT / "deployments/aurora/setup/prepare_fairchem_wheel.py"
    spec = importlib.util.spec_from_file_location("prepare_fairchem_wheel", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "fairchem_core-2.23.0-py3-none-any.whl"
    metadata_path = "fairchem_core-2.23.0.dist-info/METADATA"
    with WheelFile(source, "w") as wheel:
        wheel.writestr("fairchem/core/__init__.py", b"original_code = True\n")
        wheel.writestr(metadata_path, (
            b"Metadata-Version: 2.1\nName: fairchem-core\nVersion: 2.23.0\n"
            b"Requires-Dist: torch~=2.13.0\nRequires-Dist: e3nn>=0.6\n"
            b"Requires-Dist: torch-sim-atomistic>=0.6.0; "
            b"(python_version >= '3.12') and extra == 'torchsim'\n\n"
        ))
    original = source.read_bytes()
    target = module.prepare_wheel(source, tmp_path / "patched", "2.13.0a0+git123")
    assert "-1aurora-" in target.name
    assert source.read_bytes() == original
    with WheelFile(target) as wheel:
        metadata = BytesParser().parsebytes(wheel.read(metadata_path))
        assert metadata.get_all("Requires-Dist") == [
            "torch==2.13.0a0+git123", "e3nn>=0.6",
            "torch-sim-atomistic>=0.6.0; (python_version >= '3.12') and extra == 'torchsim'",
        ]
        for requirement in metadata.get_all("Requires-Dist"):
            Requirement(requirement)
        assert wheel.read("fairchem/core/__init__.py") == b"original_code = True\n"
        for entry in wheel.namelist():
            wheel.read(entry)
    manifest = json.loads(target.with_suffix(".json").read_text())
    assert manifest["upstream_sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["experimental"] is True
    with pytest.raises(ValueError, match="Torch 2.13"):
        module.prepare_wheel(source, tmp_path / "unsupported", "2.14.0")
    with pytest.raises(ValueError, match="upstream wheel"):
        module.prepare_wheel(target, tmp_path / "repatched", "2.13.0a0+git123")


def test_mace_runtime_paths_use_matsim_owned_compatibility_environment() -> None:
    for script in (ROOT / "deployments").rglob("*.sh"):
        text = script.read_text(encoding="utf-8")
        assert ".hpc-build/perlmutter/mace_venv" not in text, script
        assert "e3nn 0.6.x" not in text, script

    common = ROOT / "deployments/common/setup/install-mace-compat.sh"
    text = common.read_text(encoding="utf-8")
    assert common.stat().st_mode & 0o111
    assert "import dscribe" in text
    assert '--system-site-packages "${MACE_VENV_PATH}"' in text
    assert "MACE replaced Aurora framework Torch" in text
    assert "except Exception" not in text
    subprocess.run(["bash", "-n", str(common)], check=True)


def test_uma_uses_matsim_owned_compatibility_environment() -> None:
    common = ROOT / "deployments/common/setup/install-uma-compat.sh"
    text = common.read_text(encoding="utf-8")
    assert common.stat().st_mode & 0o111
    assert 'UMA_VENV_PATH="${UMA_VENV_PATH:-${MATSIM_DIR}/.venv-uma}"' in text
    assert '"${MATSIM_DIR}[${UMA_MATSIM_EXTRAS}]"' in text
    assert "import dscribe" in text
    assert "from fairchem.core import FAIRChemCalculator, pretrained_mlip" in text
    assert '--system-site-packages "${UMA_VENV_PATH}"' in text
    assert 'export PIP_CONSTRAINT="${CONSTRAINTS}"' in text
    assert "UMA replaced Aurora framework Torch" in text
    assert 'AURORA_FAIRCHEM_VERSION="${AURORA_FAIRCHEM_VERSION:-2.23.0}"' in text
    assert "prepare_fairchem_wheel.py" in text
    assert '"${FAIRCHEM_ARGS[@]}"' in text
    assert "Experimental FairChem override" in text
    subprocess.run(["bash", "-n", str(common)], check=True)


def test_frontier_install_entry_points_are_unambiguous() -> None:
    setup = ROOT / "deployments/frontier/setup"
    installers = sorted(path.name for path in setup.glob("*install*.sh"))
    assert installers == ["install-vllm-rocm72.sh", "install.sh"]

    vllm = (setup / "install-vllm-rocm72.sh").read_text(encoding="utf-8")
    assert 'bash "${REPO}/deployments/frontier/setup/install.sh"' not in vllm
    assert '[[ -x "${VENV}/bin/python" ]]' in vllm
    assert "build-vllm-rocm72.sh" in vllm


def test_installation_guidance_has_no_obsolete_environment_contracts() -> None:
    obsolete = (
        "hydragnn_venv",
        "scripts/perlmutter_venv",
        "third_party/HydraGNN",
        "INSTALL_LLM_EXTRAS=1 bash",
        "numpy==1.26.4",
    )
    roots = (
        ROOT / "README.md",
        ROOT / "docs",
        ROOT / "deployments",
        ROOT / "benchmarks",
        ROOT / "scripts",
    )
    files = [roots[0]]
    for directory in roots[1:]:
        files.extend(directory.rglob("*.md"))
        files.extend(directory.rglob("*.sh"))

    for path in files:
        text = path.read_text(encoding="utf-8")
        for fragment in obsolete:
            assert fragment not in text, f"{path.relative_to(ROOT)} contains {fragment!r}"
        assert re.search(r"bash[^\n`]*install_matsim_", text) is None, path
