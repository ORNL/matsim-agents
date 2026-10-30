from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]

INSTALLERS = {
    "frontier": "scripts/hpc/olcf/frontier/installation/install-rocm72.sh",
    "aurora": "scripts/hpc/alcf/aurora/installation/install.sh",
    "perlmutter": "scripts/hpc/nersc/perlmutter/installation/install.sh",
}


@pytest.mark.parametrize("custom", [False, True])
def test_aurora_vasp_build_installs_makefile_and_loads_modules(tmp_path, custom) -> None:
    repo = tmp_path / "matsim-agents"
    setup = repo / "deployments/aurora/setup"
    setup.mkdir(parents=True)
    (repo / "pyproject.toml").touch()
    helper = setup / "build-vasp-gpu-aurora.sh"
    helper.write_text((ROOT / "deployments/aurora/setup" / helper.name).read_text())
    recipe = setup / "makefile.include.aurora-gpu"
    recipe.write_bytes((ROOT / "deployments/aurora/setup" / recipe.name).read_bytes())
    vasp = tmp_path / "custom-vasp" if custom else repo / "external/vasp6/src/vasp.6.6.1"
    vasp.mkdir(parents=True)
    (vasp / "makefile").touch()
    (vasp / "makefile.include").write_text("stale configuration\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make = fake_bin / "make"
    make.write_text('#!/bin/bash\nprintf "make %s\\n" "$*" >> "$BUILD_LOG"\n')
    make.chmod(0o755)
    startup = tmp_path / "bash-env"
    startup.write_text('module() { printf "module %s\\n" "$*" >> "$BUILD_LOG"; }\n')
    log = tmp_path / "commands.log"
    environment = os.environ.copy()
    for name in ("VASP_ROOT", "MAKEFILE_INCLUDE", "ONEAPI_MODULE", "GPU_UMD_MODULE", "VASP_TARGET"):
        environment.pop(name, None)
    environment.update(
        PATH=f"{fake_bin}:{environment['PATH']}",
        BASH_ENV=str(startup),
        BUILD_LOG=str(log),
        NCORES="2",
        PREFIX="build-test",
        CLEAN_BUILD="0",
    )
    if custom:
        recipe = tmp_path / "custom.include"
        recipe.write_text("FC = custom-wrapper\n")
        environment.update(
            VASP_ROOT=str(vasp),
            MAKEFILE_INCLUDE=str(recipe),
            ONEAPI_MODULE="oneapi/test",
            GPU_UMD_MODULE="umd/test",
            VASP_TARGET="gam",
        )
    result = subprocess.run(["bash", str(helper)], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (vasp / "makefile.include").read_bytes() == recipe.read_bytes()
    commands = log.read_text().splitlines()
    modules = "oneapi/test umd/test" if custom else "oneapi/release/2026.1.0 intel_gpu_umd_aicoe"
    assert f"module load {modules}" in commands
    targets = ["gam"] if custom else ["std", "gam", "ncl"]
    assert [line for line in commands if line.startswith("make ")] == [
        f"make PREFIX=build-test DEPS=1 MODS=1 -j2 {target}" for target in targets
    ]
    if custom:
        environment["MAKEFILE_INCLUDE"] = str(vasp / "makefile.include")
        same_file = subprocess.run(["bash", str(helper)], env=environment, capture_output=True)
        assert same_file.returncode == 0
    environment["MAKEFILE_INCLUDE"] = str(tmp_path / "missing.include")
    before = log.read_text()
    missing = subprocess.run(["bash", str(helper)], env=environment, capture_output=True)
    assert missing.returncode != 0
    assert b"Required path not found" in missing.stderr
    assert log.read_text() == before


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
            f'#!/bin/bash\n[[ "$RUN_MODEL_TESTS" == "1" ]] || exit 8\nexit {exit_code}\n'
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
        'if [[ "$*" == "-m pytest "* ]]; then '
        'printf "pytest-cwd %s\\n" "$PWD" >> "$INSTALL_LOG"; fi\n'
        'if [[ "$*" == "-m pip check" ]]; then exit 1; fi\n'
        'if [[ "$*" == *check-runtime-dependencies.py* ]]; then '
        'exit "${RUNTIME_CHECK_EXIT:-0}"; fi\n'
    )
    python.chmod(0o755)
    git = fake_bin / "git"
    git.write_text(
        '#!/bin/bash\nprintf "git %s\\n" "$*" >> "$INSTALL_LOG"\n'
        'if [[ "$*" == *rev-parse* ]]; then echo 81d5b88; fi\n'
    )
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
        PATH=f"{fake_bin}:/usr/bin:/bin",
        BASH_ENV=str(startup),
        INSTALL_LOG=str(log),
        FAKE_PYTHON=str(python),
        MATSIM_DIR=str(matsim),
        HYDRAGNN_DIR=str(hydra),
        HYDRAGNN_REF="",
        FRAMEWORKS_MODULE="frameworks/test",
        RUN_MODEL_TESTS="1",
        VENV_PATH=str(matsim / ".venv"),
        INSTALL_ROOT=str(matsim / ".hpc-build/aurora"),
        RECREATE_ENV="0",
        RECREATE_MACE_ENV="1",
        RECREATE_UMA_ENV="0",
        INSTALL_VLLM="1",
        REUSE_HYDRAGNN_DEPENDENCIES="0",
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
    assert f"pytest-cwd {matsim}" in commands
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


def _source_runtime_env(
    root: Path, env: dict[str, str], facility: str = "perlmutter"
) -> dict[str, str]:
    script = ROOT / "deployments" / "common" / "setup" / "matsim-runtime-env.sh"
    names = (
        "MATSIM_QE_LAUNCHER",
        "MATSIM_QE_PSEUDO_DIR",
        "MATSIM_VASP_LAUNCHER",
        "MATSIM_VASP_POTCAR_DIR",
        "MATSIM_FAIRCHEM_VENV",
        "MATSIM_MACE_VENV",
    )
    dump = "; ".join(f'echo "{n}=${{{n}:-}}"' for n in names)
    clean = {k: v for k, v in os.environ.items() if not k.startswith("MATSIM_")}
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail; source "{script}" {facility} "{root}" >/dev/null; {dump}',
        ],
        env={**clean, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_runtime_env_exports_only_existing_targets(tmp_path: Path) -> None:
    assert set(_source_runtime_env(tmp_path, {}).values()) == {""}
    pw = tmp_path / "external" / "quantum-espresso" / "install-gpu" / "bin" / "pw.x"
    pw.parent.mkdir(parents=True)
    pw.write_text("", encoding="utf-8")
    pw.chmod(0o755)
    (tmp_path / "external" / "quantum-espresso" / "src" / "pseudo").mkdir(parents=True)
    (tmp_path / ".venv-mace").mkdir()
    launcher = tmp_path / "deployments" / "perlmutter" / "launchers" / "run-pw-gpu-perlmutter.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("", encoding="utf-8")
    got = _source_runtime_env(tmp_path, {"MATSIM_FAIRCHEM_VENV": "/custom/uma"})
    assert got["MATSIM_QE_LAUNCHER"] == str(launcher)
    assert got["MATSIM_QE_PSEUDO_DIR"] == str(
        tmp_path / "external" / "quantum-espresso" / "src" / "pseudo"
    )
    assert got["MATSIM_VASP_LAUNCHER"] == ""
    assert got["MATSIM_FAIRCHEM_VENV"] == "/custom/uma"
    assert got["MATSIM_MACE_VENV"] == str(tmp_path / ".venv-mace")


@pytest.mark.parametrize("facility", sorted(INSTALLERS))
def test_runtime_env_defaults_the_facility_vasp_launcher(tmp_path: Path, facility: str) -> None:
    vasp = tmp_path / "external" / "vasp6" / "src" / "vasp.6.6.1" / "bin" / "vasp_std"
    vasp.parent.mkdir(parents=True)
    vasp.write_text("", encoding="utf-8")
    vasp.chmod(0o755)
    launchers = tmp_path / "deployments" / facility / "launchers"
    launchers.mkdir(parents=True)
    real = ROOT / "deployments" / facility / "launchers" / f"run-vasp-gpu-{facility}.sh"
    (launchers / real.name).symlink_to(real)
    got = _source_runtime_env(tmp_path, {}, facility)
    assert got["MATSIM_VASP_LAUNCHER"] == str(launchers / real.name)
    assert real.stat().st_mode & 0o111
    text = real.read_text(encoding="utf-8")
    assert f"_vasp-step-{facility}.sh" in text or facility == "perlmutter"
    assert '"$1"' not in text and "$#" not in text


def test_installers_print_activation_with_the_installed_venvs() -> None:
    for facility in INSTALLERS:
        text = (ROOT / "deployments" / facility / "setup" / "install.sh").read_text(
            encoding="utf-8"
        )
        hint = next(line for line in text.splitlines() if "Activate with" in line)
        assert f"MATSIM_{facility.upper()}_VENV=${{VENV_PATH}}" in hint
        assert "MATSIM_FAIRCHEM_VENV=${UMA_VENV_PATH}" in hint
        assert "MATSIM_MACE_VENV=${MACE_VENV_PATH}" in hint
        if facility == "frontier":
            assert "--rocm72" in hint


@pytest.mark.parametrize("facility", list(INSTALLERS))
def test_activation_hint_keeps_venv_overrides_after_setup_returns(facility, tmp_path):
    text = (ROOT / "deployments" / facility / "setup" / "install.sh").read_text(encoding="utf-8")
    hint = next(line for line in text.splitlines() if "Activate with" in line)
    command = hint.split("Activate with: ", 1)[1].rstrip('"')
    command = re.sub(r"^module load \S+ && ", "", command)
    stub = tmp_path / "setup.sh"
    stub.write_text("export MATSIM_FAIRCHEM_VENV MATSIM_MACE_VENV\n", encoding="utf-8")
    command = command.replace(f"${{SCRIPT_DIR}}/setup_matsim_{facility}.sh", str(stub))
    values = {"VENV_PATH": "/x/main", "UMA_VENV_PATH": "/x/uma", "MACE_VENV_PATH": "/x/mace"}
    command = re.sub(r"\$\{(\w+)\}", lambda m: values[m.group(1)], command)
    out = subprocess.run(
        ["bash", "-c", f'{command}; echo "$MATSIM_FAIRCHEM_VENV|$MATSIM_MACE_VENV"'],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": os.environ["PATH"]},
    )
    assert out.stdout.strip().splitlines()[-1] == "/x/uma|/x/mace"


def test_facility_setup_scripts_source_runtime_env() -> None:
    for facility in INSTALLERS:
        setup = ROOT / "deployments" / facility / "setup" / f"setup_matsim_{facility}.sh"
        text = setup.read_text(encoding="utf-8")
        assert f'deployments/common/setup/matsim-runtime-env.sh" {facility} ' in text
        launcher = ROOT / "deployments" / facility / "launchers" / f"run-pw-gpu-{facility}.sh"
        assert launcher.is_file()


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
        assert 'INSTALL_UMA="${INSTALL_UMA:-1}"' in text
        assert 'UMA_VENV_PATH="${UMA_VENV_PATH:-${MATSIM_DIR}/.venv-uma}"' in text
        assert 'INSTALL_MACE="${INSTALL_MACE:-1}"' in text
        assert 'MACE_VENV_PATH="${MACE_VENV_PATH:-${MATSIM_DIR}/.venv-mace}"' in text
        assert "deployments/common/setup/install-mace-compat.sh" in text
        assert "deployments/common/setup/install-uma-compat.sh" in text
        assert "pip check" in text
        assert "installation_DOE_supercomputers/hydragnn_installation" not in text
        subprocess.run(["bash", "-n", str(installer)], check=True)


@pytest.mark.parametrize(
    "base,mace,uma,expected",
    [
        ("0", "1", None, "0"),
        ("1", "0", None, "0"),
        ("1", "1", "0", "0"),
        ("0", "0", "1", "1"),
        ("1", "1", "1", "1"),
    ],
)
def test_aurora_recreate_uma_defaults_independently(base, mace, uma, expected):
    installer = (ROOT / "deployments/aurora/setup/install.sh").read_text()
    assignments = "\n".join(line for line in installer.splitlines() if line.startswith("RECREATE_"))
    environment = os.environ.copy()
    environment.update(RECREATE_ENV=base, RECREATE_MACE_ENV=mace)
    environment.pop("RECREATE_UMA_ENV", None)
    if uma is not None:
        environment["RECREATE_UMA_ENV"] = uma
    result = subprocess.run(
        ["bash"],
        input=assignments + '\nprintf "%s" "$RECREATE_UMA_ENV"\n',
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout == expected


@pytest.mark.parametrize("base", ["0", "1"])
def test_aurora_compatibility_recreation_is_opt_in_like_perlmutter(base):
    aurora = (ROOT / "deployments/aurora/setup/install.sh").read_text()
    perlmutter = (ROOT / "deployments/perlmutter/setup/install.sh").read_text()
    for backend in ("UMA", "MACE"):
        assignment = f'RECREATE_{backend}_ENV="${{RECREATE_{backend}_ENV:-0}}"'
        assert assignment in aurora
    assert 'RECREATE_MACE_ENV="${RECREATE_MACE_ENV:-0}"' in perlmutter
    assignments = "\n".join(line for line in aurora.splitlines() if line.startswith("RECREATE_"))
    environment = os.environ.copy()
    environment["RECREATE_ENV"] = base
    for backend in ("UMA", "MACE"):
        environment.pop(f"RECREATE_{backend}_ENV", None)
    result = subprocess.run(
        ["bash"],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        input=assignments + '\nprintf "%s %s" "$RECREATE_UMA_ENV" "$RECREATE_MACE_ENV"\n',
    )
    assert result.stdout == "0 0"


@pytest.mark.parametrize("ref", ["main", "feature/test", "v-test", "commit"])
@pytest.mark.parametrize("checkout_kind", ["selected", "dependencies"])
def test_aurora_refs_work_on_first_install_and_rerun(tmp_path, ref, checkout_kind):
    origin = tmp_path / "origin"
    subprocess.run(["git", "init", "-b", "main", str(origin)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(origin),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "initial",
        ],
        check=True,
        capture_output=True,
    )
    for command in (("branch", "feature/test"), ("tag", "v-test")):
        subprocess.run(["git", "-C", str(origin), *command], check=True, capture_output=True)
    revision = subprocess.check_output(
        ["git", "-C", str(origin), "rev-parse", "HEAD"], text=True
    ).strip()
    selected = revision if ref == "commit" else ref
    checkout = tmp_path / "checkout"
    installer = (ROOT / "deployments/aurora/setup/install.sh").read_text()
    if checkout_kind == "selected":
        prefix = "HYDRAGNN"
        start = installer.index('if [[ ! -d "${HYDRAGNN_DIR}/.git" ]]')
        end = installer.index('[[ "$(realpath -m "${HYDRAGNN_INSTALLER_DIR}")"')
    else:
        prefix = "HYDRAGNN_INSTALLER"
        start = installer.index('if [[ ! -d "${HYDRAGNN_INSTALLER_DIR}/.git" ]]')
        end = installer.index('log "Upstream dependency revision')
    script = "set -euo pipefail\nlog() { :; }\ndie() { exit 1; }\n" + installer[start:end]
    environment = os.environ.copy()
    environment.update(
        {
            f"{prefix}_DIR": str(checkout),
            f"{prefix}_REPO": str(origin),
            f"{prefix}_REF": selected,
        }
    )
    for _attempt in range(2):
        subprocess.run(
            ["bash"], input=script, text=True, env=environment, check=True, capture_output=True
        )
        assert (
            subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
            ).strip()
            == revision
        )
        branch = subprocess.run(
            ["git", "-C", str(checkout), "symbolic-ref", "--short", "HEAD"],
            text=True,
            capture_output=True,
        )
        if checkout_kind == "selected" and ref in ("main", "feature/test"):
            assert branch.returncode == 0 and branch.stdout.strip() == ref
        else:
            assert branch.returncode != 0
    if checkout_kind == "dependencies" and ref == "commit":
        fork = tmp_path / "fork"
        subprocess.run(["git", "clone", str(origin), str(fork)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(fork), "switch", "-c", "audited-fork-only"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(fork),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "--allow-empty",
                "-m",
                "audited fork",
            ],
            check=True,
            capture_output=True,
        )
        expected = subprocess.check_output(
            ["git", "-C", str(fork), "rev-parse", "HEAD"], text=True
        ).strip()
        environment.update(
            HYDRAGNN_INSTALLER_REPO=str(fork), HYDRAGNN_INSTALLER_REF="audited-fork-only"
        )
        subprocess.run(
            ["bash"], input=script, text=True, env=environment, check=True, capture_output=True
        )
        assert (
            subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
            ).strip()
            == expected
        )
        assert subprocess.check_output(
            ["git", "-C", str(checkout), "remote", "get-url", "origin"], text=True
        ).strip() == str(origin)


def test_runtime_dependency_audit_checks_selected_extras_and_transitive_requirements(monkeypatch):
    from types import SimpleNamespace

    helper = ROOT / "deployments/common/setup/check-runtime-dependencies.py"
    spec = importlib.util.spec_from_file_location("runtime_dependencies", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    packages = {
        "runtime": SimpleNamespace(
            version="1.0",
            requires=[
                'torch>=2.13.0.dev0; extra == "model"',
                'missing-package; extra == "unused"',
                'missing-platform; sys_platform == "never"',
            ],
        ),
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
        'Missing runtime dependency: missing-package; extra == "unused"'
    ]


def test_aurora_xpu_qualification_rejects_invalid_predictions(tmp_path):
    import json

    import pytest

    helper = ROOT / "deployments/aurora/jobs/qualify_mlip_xpu.py"
    spec = importlib.util.spec_from_file_location("qualify_mlip_xpu", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    progress = module.Progress(tmp_path / "uma.json", "uma")
    progress.stage("predict_energy")
    record = json.loads((tmp_path / "uma.progress.json").read_text())
    assert record["stage"] == "predict_energy" and record["backend"] == "uma"
    assert record["elapsed_seconds"] >= 0
    module.validate_predictions(-1.0, [[0.0, 0.0, 0.0]], 1)
    for energy, forces in (
        (float("nan"), [[0.0, 0.0, 0.0]]),
        (-1.0, [[float("inf"), 0.0, 0.0]]),
        (-1.0, [[0.0, 0.0]]),
        (-1.0, []),
    ):
        with pytest.raises(ValueError):
            module.validate_predictions(energy, forces, 1)
    job = ROOT / "deployments/aurora/jobs/job-qualify-mlip-xpu-aurora.sh"
    subprocess.run(["bash", "-n", str(job)], check=True)
    assert "mpiexec -n 1 --ppn 1" in job.read_text()
    assert "export HF_HUB_OFFLINE=1" in job.read_text()


def test_aurora_fused_residency_requires_xpu_gnn_and_mlp(monkeypatch):
    from types import SimpleNamespace

    import pytest

    import matsim_agents.backends.mlip.calculator as calculator

    helper = ROOT / "deployments/aurora/jobs/qualify_mlip_xpu.py"
    spec = importlib.util.spec_from_file_location("qualify_mlip_xpu", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        calculator,
        "_find_torch_module",
        lambda root: root if hasattr(root, "parameters") else None,
    )

    def model(device):
        return SimpleNamespace(parameters=lambda: [SimpleNamespace(device=device)])

    fused = SimpleNamespace(model=model("xpu:0"), mlp=model("cpu"))
    with pytest.raises(RuntimeError, match="not exclusively resident"):
        module.verify_model_residency(fused, "hydragnn")
    fused.mlp = None
    with pytest.raises(RuntimeError, match="requires both GNN and MLP"):
        module.verify_model_residency(fused, "hydragnn")
    fused.mlp = model("xpu:0")
    assert module.verify_model_residency(fused, "hydragnn") == ["xpu:0"]


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
    upstream_installer = dependencies / INSTALLERS["aurora"]
    upstream_installer.parent.mkdir(parents=True)
    upstream_installer.write_text("module reset\nmodule load frameworks\nmodule list\n")
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
    staged_installer = stage / INSTALLERS["aurora"]
    assert 'module load "${FRAMEWORKS_MODULE:-frameworks}"' in staged_installer.read_text()
    assert upstream_installer.read_text() == "module reset\nmodule load frameworks\nmodule list\n"
    environment = os.environ.copy()
    environment["FRAMEWORKS_MODULE"] = "frameworks/test"
    result = subprocess.run(
        ["bash"],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        input='module() { printf "%s\\n" "$*"; }\n' + staged_installer.read_text(),
    )
    assert "load frameworks/test" in result.stdout
    try:
        module.prepare_source(code, dependencies, code / "build", "2.13.0")
    except ValueError as error:
        assert "outside the selected" in str(error)
    else:
        raise AssertionError("staging inside selected checkout was accepted")


@pytest.mark.parametrize("mismatch", [None, "interpreter", "version", "path"])
def test_aurora_checks_child_runtime_before_installing_wheels(monkeypatch, tmp_path, mismatch):
    import sys
    from types import SimpleNamespace

    torch = SimpleNamespace(__version__="2.13.0a0+git123", __file__=str(tmp_path / "torch.py"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    arguments = ["-", sys._base_executable, torch.__version__, torch.__file__]
    if mismatch == "interpreter":
        arguments[1] = str(tmp_path / "different-python")
    elif mismatch == "version":
        arguments[2] = "2.13.0"
    elif mismatch == "path":
        arguments[3] = str(tmp_path / "different-torch.py")
    monkeypatch.setattr(sys, "argv", arguments)
    installer = (ROOT / "deployments/aurora/setup/install.sh").read_text()
    start = installer.index('"${PYTHON}" - "${FRAMEWORKS_PYTHON}" "${TORCH_VERSION}"')
    fragment = installer[start:].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    assert start < installer.index('log "Installing HydraGNN and matsim-agents wheels')
    if mismatch:
        with pytest.raises(SystemExit, match="framework"):
            exec(fragment, {})
    else:
        exec(fragment, {})


def test_deployment_scripts_do_not_reference_hydragnn_owned_environment() -> None:
    old_fragment = "HydraGNN/installation_DOE_supercomputers"
    for script in (ROOT / "deployments").rglob("*.sh"):
        assert old_fragment not in script.read_text(encoding="utf-8"), script

    generic = (ROOT / "scripts/setup_env.sh").read_text(encoding="utf-8")
    assert "installation_DOE_supercomputers" not in generic
    assert "hydragnn_installation_bash_script" not in generic
    assert "deployments/${FACILITY}/setup/install.sh" in generic
    assert "frontier|perlmutter|aurora)" in generic
    assert "frontier-rocm" not in generic


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
        wheel.writestr(
            metadata_path,
            (
                b"Metadata-Version: 2.1\nName: fairchem-core\nVersion: 2.23.0\n"
                b"Requires-Dist: torch~=2.13.0\nRequires-Dist: e3nn>=0.6\n"
                b"Requires-Dist: torch-sim-atomistic>=0.6.0; "
                b"(python_version >= '3.12') and extra == 'torchsim'\n\n"
            ),
        )
    original = source.read_bytes()
    target = module.prepare_wheel(source, tmp_path / "patched", "2.13.0a0+git123")
    assert "-1aurora-" in target.name
    assert source.read_bytes() == original
    with WheelFile(target) as wheel:
        metadata = BytesParser().parsebytes(wheel.read(metadata_path))
        assert metadata.get_all("Requires-Dist") == [
            "torch==2.13.0a0+git123",
            "e3nn>=0.6",
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


def test_fairchem_xpu_adapter_requires_xpu_and_preserves_cpu_cuda():
    from types import SimpleNamespace

    import pytest

    helper = ROOT / "deployments/aurora/setup/prepare_fairchem_wheel.py"
    spec = importlib.util.spec_from_file_location("fairchem_xpu_adapter", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = """class MLIPPredictUnit:
    def __init__(self, device):
        self.inference_settings = settings()
        self._load_model()
        self._setup_device(device)
    def _load_model(self):
        self.loaded_mode = self.inference_settings.execution_mode
    def _setup_device(self, device):
        assert device in ["cpu", "cuda"]
        self.device = device
"""
    available = [True]
    namespace = {
        "torch": SimpleNamespace(
            device=lambda value: SimpleNamespace(type=value.split(":")[0]),
            xpu=SimpleNamespace(is_available=lambda: available[0]),
        ),
        "ExecutionMode": SimpleNamespace(GENERAL="general"),
        "settings": lambda: SimpleNamespace(
            compile=True, tf32=True, merge_mole=True, execution_mode="default"
        ),
    }
    exec(module.adapt_xpu_source(source), namespace)
    predictor = namespace["MLIPPredictUnit"]
    for device in ("cpu", "cuda"):
        instance = predictor(device)
        assert instance.device == device and instance.loaded_mode == "default"
    instance = predictor("xpu")
    assert instance.device.type == "xpu" and instance.loaded_mode == "general"
    assert not instance.inference_settings.compile
    assert not instance.inference_settings.tf32
    assert not instance.inference_settings.merge_mole
    available[0] = False
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        predictor("xpu")


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
    assert "guard_mace_optional_backends.py" in text
    assert "except Exception" not in text
    subprocess.run(["bash", "-n", str(common)], check=True)


def test_mace_optional_backend_guard_preserves_supported_backends(tmp_path, monkeypatch):
    import builtins
    import sys
    from types import SimpleNamespace

    import pytest

    helper = ROOT / "deployments/common/setup/guard_mace_optional_backends.py"
    spec = importlib.util.spec_from_file_location("guard_mace_optional_backends", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "wrapper_ops.py"
    backend = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "openequivariance", backend)
    for alias in ("oeq", "_oeq"):
        path.write_text(
            f"import torch\ntry:\n    import openequivariance as {alias}\n"
            "    OEQ_AVAILABLE = True\nexcept ImportError:\n    OEQ_AVAILABLE = False\n"
        )
        assert module.guard_openequivariance(path)
        original = path.read_text()
        assert not module.guard_openequivariance(path)
        assert path.read_text() == original
        for cuda, hip in ((None, None), ("13.0", None), (None, "7.2")):
            monkeypatch.setitem(
                sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda=cuda, hip=hip))
            )
            namespace = {}
            exec(compile(path.read_text(), str(path), "exec"), namespace)
            assert namespace["OEQ_AVAILABLE"] is bool(cuda or hip)
            if cuda or hip:
                assert namespace[alias] is backend
    original_import = builtins.__import__

    def failed_optional_import(name, *args, **kwargs):
        if name == "openequivariance":
            raise RuntimeError("Unexpected accelerator failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failed_optional_import)
    with pytest.raises(RuntimeError, match="Unexpected accelerator failure"):
        exec(compile(path.read_text(), str(path), "exec"), {})


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
    assert '--torch-version "${TORCH_VERSION}" --enable-xpu' in text
    assert "Experimental FairChem override" in text
    subprocess.run(["bash", "-n", str(common)], check=True)


def test_frontier_install_entry_points_are_unambiguous() -> None:
    setup = ROOT / "deployments/frontier/setup"
    installers = sorted(path.name for path in setup.glob("*install*.sh"))
    assert installers == ["install-vllm-rocm72.sh", "install.sh"]

    vllm = (setup / "install-vllm-rocm72.sh").read_text(encoding="utf-8")
    assert 'bash "${REPO}/deployments/frontier/setup/install.sh"' not in vllm
    assert '[[ -x "${BASE_VENV}/bin/python" ]]' in vllm
    assert 'VENV="${VLLM_VENV_PATH:-$REPO/.venv-vllm}"' in vllm
    assert '"${BASE_VENV}/bin/python" -m venv "${VENV}"' in vllm
    assert '"torch==2.14.0" "torchvision==0.29.0"' in vllm
    assert 'export VLLM_VENV_PATH="$VENV"' in vllm
    assert 'source "${VENV}/bin/activate"' in vllm
    assert "conda deactivate" not in vllm
    assert 'SLURM_ACCOUNT="${SLURM_ACCOUNT:-lrn110}"' in vllm
    assert 'sbatch -A "$SLURM_ACCOUNT"' in vllm
    assert "import hydragnn, matsim_agents" not in vllm
    assert "build-vllm-rocm72.sh" in vllm

    build = (setup / "build-vllm-rocm72.sh").read_text(encoding="utf-8")
    launcher = (ROOT / "deployments/frontier/launchers/vllm-server-frontier.sh").read_text(
        encoding="utf-8"
    )
    assert 'VENV="${VLLM_VENV_PATH:-$REPO/.venv-vllm}"' in build
    assert 'VENV="${MATSIM_VLLM_VENV:-$REPO/.venv-vllm}"' in launcher

    frontier = ROOT / "deployments/frontier"
    for relative in (
        "jobs/job-serve-multinode-frontier.sh",
        "smoke-tests/smoke-vllm-singlenode-frontier.sh",
        "smoke-tests/smoke-vllm-multinode-frontier.sh",
    ):
        text = (frontier / relative).read_text(encoding="utf-8")
        assert "MATSIM_VLLM_VENV:-$REPO/.venv-vllm" in text
        assert 'source "$VENV/bin/activate"' in text

    for relative in (
        "jobs/job-discovery-chat-vllm-frontier.sh",
        "jobs/job-heterogeneous-llm-debate-frontier.sh",
        "jobs/job-sequential-benchmark-frontier.sh",
    ):
        text = (frontier / relative).read_text(encoding="utf-8")
        assert ".venv-vllm" in text
        assert "require_tvm_ffi_artifact" in text

    prebuild = (frontier / "setup/prebuild-tvm-ffi-frontier.sh").read_text(encoding="utf-8")
    assert "VLLM_VENV_PATH:-$REPO/.venv-vllm" in prebuild

    benchmark = (frontier / "jobs/job-six-model-benchmark-frontier.sh").read_text(encoding="utf-8")
    assert "VENV=$REPO/.venv" in benchmark
    assert "require_tvm_ffi_artifact" not in benchmark

    frontier_docs = (frontier / "docs/README-frontier.md").read_text(encoding="utf-8")
    assert "`$REPO/.venv-vllm`" in frontier_docs
    assert "Then activates the vLLM-on-ROCm-7.2 venv at:\n`$REPO/.venv`" not in frontier_docs


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
