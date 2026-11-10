from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from matsim_agents.backends.dft.config import VASPConfig
from matsim_agents.workflows.relaxation import DFTBackendConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("facility", ["aurora", "frontier", "perlmutter"])
@pytest.mark.parametrize("installation", ["facility", "custom"])
def test_explicit_installation_uses_same_placement_wrapper(tmp_path, facility, installation):
    binary = tmp_path / installation / "vasp_std"
    binary.parent.mkdir()
    binary.write_text("#!/bin/bash\nexit 0\n")
    binary.chmod(0o755)
    setup = tmp_path / "runtime.sh"
    setup.write_text("export INSTALLATION_RUNTIME=chosen\n")
    tools = tmp_path / "tools"
    tools.mkdir()
    record = tmp_path / "launch"
    for launcher in ("srun", "mpiexec"):
        tool = tools / launcher
        tool.write_text('#!/bin/bash\nprintf "%s\\n" "$INSTALLATION_RUNTIME" "$@" > "$RECORD"\n')
        tool.chmod(0o755)
    wrapper = ROOT / f"deployments/{facility}/launchers/_vasp-step-{facility}.sh"
    result = subprocess.run(
        ["bash", str(wrapper), str(tmp_path), str(binary), "1", "2", "4"],
        env={
            **os.environ,
            "PATH": f"{tools}:{os.environ['PATH']}",
            "VASP_ENV_SETUP": str(setup),
            "VASP_EXPECTED_SHA256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "MATSIM_ASSIGNED_NODES": "test-node",
            "RECORD": str(record),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = record.read_text().splitlines()
    assert args[0] == "chosen" and args[-1] == str(binary)
    assert "2" in args
    assert "[vasp-runtime]" in result.stdout


@pytest.mark.parametrize("failure", ["setup", "hash", "missing", "setup-exit"])
def test_runtime_errors_abort_before_launch(tmp_path, failure):
    binary = tmp_path / "vasp_std"
    binary.write_text("#!/bin/bash\nexit 0\n")
    binary.chmod(0o755)
    setup = tmp_path / "env.sh"
    setup.write_text("exit 17\n" if failure == "setup-exit" else "export TEST_RUNTIME=1\n")
    env = {
        **os.environ,
        "VASP_BIN": str(binary if failure != "missing" else tmp_path / "missing"),
        "VASP_ENV_SETUP": str(setup if failure != "setup" else tmp_path / "absent"),
    }
    if failure == "hash":
        env["VASP_EXPECTED_SHA256"] = "0" * 64
    result = subprocess.run(
        ["bash", str(ROOT / "deployments/common/vasp-runtime.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0


def test_aurora_default_loads_tested_mpich(tmp_path):
    startup = tmp_path / "bash-env"
    startup.write_text('module() { printf "MODULE %s\\n" "$*"; }\n')
    binary = tmp_path / "vasp_std"
    binary.write_text("#!/bin/bash\n")
    binary.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "deployments/aurora/setup/vasp-runtime-aurora.sh")],
        env={**os.environ, "BASH_ENV": str(startup)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MODULE load mpich/prd/5.0.0.aurora_test.51a9474" in result.stdout


def test_config_runtime_fields_are_optional_and_propagated(tmp_path):
    fields = dict(
        vasp_bin=tmp_path / "vasp_std",
        vasp_wrapper=tmp_path / "wrapper",
        incar_template=tmp_path / "INCAR",
        potcar_dir=tmp_path / "potcars",
    )
    assert VASPConfig(**fields).runtime_environment() == {}
    config = VASPConfig(**fields, vasp_env_setup=tmp_path / "setup", vasp_expected_sha256="a" * 64)
    assert config.runtime_environment() == {
        "VASP_ENV_SETUP": str(tmp_path / "setup"),
        "VASP_EXPECTED_SHA256": "a" * 64,
    }


def test_relaxation_propagates_installation_environment():
    config = DFTBackendConfig(
        backend="vasp",
        potcar_dir="/path/to/potcars",
        environment={"VASP_ENV_SETUP": "/path/to/runtime.sh"},
    )
    env = config.launch_environment("vasp_std")
    assert env is not None
    assert env["VASP_ENV_SETUP"] == "/path/to/runtime.sh"
    assert env["PATH"] == os.environ["PATH"]


def test_invalid_executable_pin_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="vasp_expected_sha256"):
        VASPConfig(
            vasp_bin=tmp_path / "binary",
            vasp_wrapper=tmp_path / "wrapper",
            incar_template=tmp_path / "INCAR",
            potcar_dir=tmp_path / "potcars",
            vasp_expected_sha256="not-a-hash",
        )
