from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INSTALLERS = {
    "frontier": "scripts/hpc/olcf/frontier/installation/install-rocm72.sh",
    "aurora": "scripts/hpc/alcf/aurora/installation/install.sh",
    "perlmutter": "scripts/hpc/nersc/perlmutter/installation/install.sh",
}


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


def test_deployment_scripts_do_not_reference_hydragnn_owned_environment() -> None:
    old_fragment = "HydraGNN/installation_DOE_supercomputers"
    for script in (ROOT / "deployments").rglob("*.sh"):
        assert old_fragment not in script.read_text(encoding="utf-8"), script

    generic = (ROOT / "scripts/setup_env.sh").read_text(encoding="utf-8")
    assert "installation_DOE_supercomputers" not in generic
    assert "hydragnn_installation_bash_script" not in generic
    assert 'deployments/${FACILITY}/setup/install.sh' in generic
    assert "frontier|frontier-rocm71|frontier-rocm64)" in generic
    assert "perlmutter|aurora)" in generic


def test_mace_runtime_paths_use_matsim_owned_compatibility_environment() -> None:
    for script in (ROOT / "deployments").rglob("*.sh"):
        text = script.read_text(encoding="utf-8")
        assert ".hpc-build/perlmutter/mace_venv" not in text, script
        assert "e3nn 0.6.x" not in text, script

    common = ROOT / "deployments/common/setup/install-mace-compat.sh"
    assert common.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", str(common)], check=True)


def test_uma_uses_matsim_owned_compatibility_environment() -> None:
    common = ROOT / "deployments/common/setup/install-uma-compat.sh"
    text = common.read_text(encoding="utf-8")
    assert common.stat().st_mode & 0o111
    assert 'UMA_VENV_PATH="${UMA_VENV_PATH:-${MATSIM_DIR}/.venv-uma}"' in text
    assert '"${MATSIM_DIR}[${UMA_MATSIM_EXTRAS}]"' in text
    assert "from fairchem.core import FAIRChemCalculator, pretrained_mlip" in text
    subprocess.run(["bash", "-n", str(common)], check=True)


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
