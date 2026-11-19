from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from matsim_agents.active_learning.config import ALConfig
from matsim_agents.backends.dft.vasp_io import resolve_potcar_paths
from matsim_agents.workflows.relaxation import ScientificRelaxationConfig

ROOT = Path(__file__).resolve().parents[1]


def _python_shim(path: Path) -> None:
    """Run this interpreter (with its own site-packages) under another venv path."""
    path.write_text(f'#!/bin/bash\nexec "{sys.executable}" "$@"\n')
    path.chmod(0o755)


def test_finetune_eval_job_passes_absolute_inputs_and_the_selected_environment(tmp_path):
    """Configs named relative to the submission directory survive the job's cd, and
    the backend read from the mlip block picks the environment."""
    repo = tmp_path / "repo"
    launchers = repo / "deployments/perlmutter/launchers"
    setup = repo / "deployments/perlmutter/setup"
    launchers.mkdir(parents=True)
    setup.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("")
    job = launchers / "job-finetune-eval-perlmutter.sh"
    shutil.copy(ROOT / "deployments/perlmutter/launchers/job-finetune-eval-perlmutter.sh", job)
    shutil.copy(ROOT / "deployments/perlmutter/setup/campaign-naming.sh", setup)
    (repo / "deployments/common").mkdir(parents=True)
    shutil.copy(ROOT / "deployments/common/runtime-env.sh", repo / "deployments/common")
    (setup / "perlmutter-module-stack.sh").write_text("load_perlmutter_modules_gpu() { :; }\n")
    (setup / "model-artifacts-perlmutter.sh").write_text(
        "configure_mlip_model_artifacts() { :; }\n"
    )
    main_venv = tmp_path / "main-venv"
    (main_venv / "bin").mkdir(parents=True)
    _python_shim(main_venv / "bin/python3")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    args_file = tmp_path / "args"
    python = venv / "bin/python"
    python.write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > {args_file}\n')
    python.chmod(0o755)
    (venv / "bin/activate").write_text(f'export PATH="{venv}/bin:$PATH"\n')
    case = tmp_path / "runs" / "case"
    case.mkdir(parents=True)
    (case / "dataset.extxyz").write_text("")
    (case / "elemental_references.json").write_text("{}")
    submit_dir = tmp_path / "submit"
    submit_dir.mkdir()
    (submit_dir / "evaluation.yaml").write_text("objective: interpolation\n")
    (submit_dir / "mace-small.yaml").write_text("mlip:\n  backend: mace\n  model: small\n")
    env = {
        **os.environ,
        "PROJECT_ROOT": str(repo),
        "RUNS_ROOT": str(tmp_path / "runs"),
        "CASE": "case",
        "MATSIM_HYDRAGNN_VENV": str(main_venv),
        "MATSIM_MACE_VENV": str(venv),
        "MLIP_CONFIG": "mace-small.yaml",
        "EVALUATION_CONFIG": "evaluation.yaml",
    }
    proc = subprocess.run(
        ["bash", str(job)], cwd=submit_dir, env=env, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    args = args_file.read_text().splitlines()
    assert args[args.index("--evaluation-config") + 1] == str(submit_dir / "evaluation.yaml")
    assert args[args.index("--mlip-config") + 1] == str(submit_dir / "mace-small.yaml")
    out_dir = tmp_path / "runs" / "finetune-eval" / "mace-small" / "case"
    assert args[args.index("--output-dir") + 1] == str(out_dir)
    assert (out_dir / "mlip.yaml").read_text() == (submit_dir / "mace-small.yaml").read_text()


@pytest.mark.parametrize(
    ("retrain", "config_references", "reference_exists", "blocked"),
    [
        ("1", None, False, True),
        ("1", None, True, False),
        ("1", "training_references", False, False),
        ("1", "validation_reference_set", False, False),
        ("0", None, False, False),
    ],
)
def test_campaign_reference_preflight_precedes_launch(
    tmp_path, retrain, config_references, reference_exists, blocked
):
    """Retraining needs elemental DFT references, checked before anything launches."""
    repo = tmp_path / "repo"
    setup = repo / "deployments/perlmutter/setup"
    setup.mkdir(parents=True)
    marker = tmp_path / "preflight-passed"
    (setup / "campaign-naming.sh").write_text('printf passed > "$TEST_PREFLIGHT_MARKER"\nexit 37\n')
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _python_shim(venv / "bin/python3")
    config = tmp_path / "al.yaml"
    config.write_text(
        {
            "training_references": "trainer:\n  training_references: {manifest: refs.json}\n",
            "validation_reference_set": "trainer:\n  validation_reference_set: refs.json\n",
            None: "trainer: {}\n",
        }[config_references]
    )
    reference = tmp_path / "elemental.json"
    if reference_exists:
        reference.write_text("{}")
    env = {
        **os.environ,
        "PROJECT_ROOT": str(repo),
        "MATSIM_PERLMUTTER_VENV": str(venv),
        "MATSIM_CAMPAIGN_MODE": "dft",
        "MATSIM_CAMPAIGN_RETRAIN": retrain,
        "MATSIM_CAMPAIGN_AL_CONFIG": str(config),
        "MATSIM_CAMPAIGN_PROMOTION_VALIDATION_REFERENCE_SET": str(reference),
        "TEST_PREFLIGHT_MARKER": str(marker),
    }
    result = subprocess.run(
        [
            "bash",
            str(
                ROOT
                / "deployments/perlmutter/jobs"
                / "job-campaign-formula-discovery-all-models-perlmutter.sh"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == (2 if blocked else 37), result.stderr
    assert marker.exists() is not blocked
    if blocked:
        assert "retraining requires elemental DFT references" in result.stderr


@pytest.mark.parametrize("layout", ["nested", "symbol", "mixed", "missing"])
def test_vasp_campaign_preflight_matches_runtime_potcar_layouts(tmp_path, layout):
    repo = tmp_path / "repo"
    jobs = repo / "deployments/perlmutter/jobs"
    jobs.mkdir(parents=True)
    marker = tmp_path / "launched"
    (jobs / "job-campaign-formula-discovery-all-models-perlmutter.sh").write_text(
        '#!/bin/bash\nprintf launched > "$TEST_LAUNCH_MARKER"\n'
    )
    binary = tmp_path / "vasp"
    binary.write_text("#!/bin/bash\nexit 0\n")
    binary.chmod(0o755)
    potcars = tmp_path / "potcars"
    potcars.mkdir()
    for index, symbol in enumerate(("Nb", "Ta", "O")):
        kind = ["nested", "symbol", "nested"][index] if layout == "mixed" else layout
        if layout == "missing" and symbol == "O":
            continue
        path = potcars / symbol / "POTCAR" if kind in {"nested", "missing"} else potcars / symbol
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic POTCAR")
    env = {
        **os.environ,
        "PROJECT_ROOT": str(repo),
        "MATSIM_VASP_BIN": str(binary),
        "MATSIM_VASP_POTCAR_DIR": str(potcars),
        "MATSIM_CAMPAIGN_DFT_METHOD_SIGNATURE": "synthetic-vasp",
        "TEST_LAUNCH_MARKER": str(marker),
    }
    result = subprocess.run(
        [
            "bash",
            str(
                ROOT
                / "deployments/perlmutter/jobs"
                / "job-campaign-formula-discovery-all-models-vasp-perlmutter.sh"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if layout == "missing":
        assert result.returncode == 2
        assert "missing O POTCAR" in result.stderr
        assert not marker.exists()
        with pytest.raises(FileNotFoundError):
            resolve_potcar_paths(["Nb", "Ta", "O"], potcars)
    else:
        assert result.returncode == 0, result.stderr
        assert marker.read_text() == "launched"
        assert len(resolve_potcar_paths(["Nb", "Ta", "O"], potcars)) == 3


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
        "MATSIM_RANDOM_SEED": "17",
        "LOGDIR": str(tmp_path / "model"),
        "HYDRAGNN_BRANCH_MLP_CHECKPOINT": str(tmp_path / "weights.pt"),
        "MATSIM_DFT_BACKEND": "qe",
        "MATSIM_PW_BIN": "/opt/qe/bin/pw.x",
        "MATSIM_PSEUDO_DIR": "/opt/qe/pseudo",
        "MATSIM_DFT_WRAPPER": str(ROOT / "deployments/perlmutter/launchers/_qe-step-perlmutter.sh"),
        "MATSIM_RESOURCES": str(
            ROOT / "deployments/perlmutter/resources/active-learning-sequential.yaml"
        ),
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
    assert cfg.random_seed == 17
    assert cfg.resources.preset == "perlmutter-gpu"
    assert cfg.resources.pools["dft"].share_with == ["training", "mlip"]
    assert "mlip" in cfg.resources.pools


def test_perlmutter_vasp_campaign_config_resolves(tmp_path: Path, monkeypatch) -> None:
    vasp_bin = tmp_path / "vasp_std"
    vasp_bin.touch()
    potcar_dir = tmp_path / "potpaw_PBE.64"
    potcar_dir.mkdir()
    monkeypatch.setenv("PROJECT_ROOT", str(ROOT))
    monkeypatch.setenv("MATSIM_VASP_BIN", str(vasp_bin))
    monkeypatch.setenv("MATSIM_VASP_POTCAR_DIR", str(potcar_dir))

    cfg = ALConfig.from_yaml(
        ROOT / "deployments/perlmutter/jobs/config/campaign-nb-ta-o-uma-vasp.yaml"
    )

    assert cfg.dft.backend == "vasp"
    assert cfg.dft.vasp is not None
    assert cfg.dft.vasp.vasp_bin == vasp_bin
    assert cfg.dft.vasp.potcar_dir == potcar_dir
    assert cfg.resources.pools["dft"].max_concurrent == 1
    assert cfg.dft.vasp.timeout_sec == 14400
    assert cfg.dft.vasp.extra_incar["KSPACING"] == "0.25"
    assert cfg.dft.vasp.extra_incar["KPAR"] == "4"


def _fake_venv(tmp_path: Path, capture: Path, name: str = "venv") -> Path:
    """A venv whose matsim-agents records its arguments, venv name, and job environment."""
    venv = tmp_path / name
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    _python_shim(bin_dir / "python")
    _python_shim(bin_dir / "python3")
    cli = bin_dir / "matsim-agents"
    cli.write_text(
        "#!/bin/bash\n"
        f'printf "%s|%s|%s|%s|%s|%s\\n" "$*" "{name}" "$DFT_BACKEND" '
        f'"${{SCALE_TAG:-}}" "${{AL_SEED:-}}" "${{TRAINER_ENABLED:-}}" >> "{capture}"\n'
    )
    cli.chmod(0o755)
    (bin_dir / "activate").write_text(f'export PATH="{bin_dir}:$PATH"\n')
    return venv


def _fake_aurora_venv(tmp_path: Path, capture: Path, name: str = "venv") -> Path:
    venv = _fake_venv(tmp_path, capture, name)
    (venv / "bash-env").write_text("module() { return 0; }\nml() { return 0; }\n")
    return venv


@pytest.mark.parametrize(
    ("case", "config", "dft"),
    [
        ("hea_bcc", "al_hea_bcc_aurora.yaml", "vasp"),
        ("phosphorene_qe", "al_phosphorene_qe_aurora.yaml", "qe"),
    ],
)
def test_aurora_paper_case_job_runs_aurora_config(tmp_path, case, config, dft):
    capture = tmp_path / "calls.txt"
    venv = _fake_aurora_venv(tmp_path, capture)
    result = subprocess.run(
        ["bash", str(ROOT / "deployments/aurora/jobs/job-active-learning-paper-cases-aurora.sh")],
        env={
            **os.environ,
            "PROJECT_ROOT": str(ROOT),
            "RUNS_ROOT": str(tmp_path / "runs"),
            "MATSIM_AURORA_VENV": str(venv),
            "BASH_ENV": str(venv / "bash-env"),
            "CASE": case,
            "PBS_JOBID": "123.aurora",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = capture.read_text().splitlines()
    assert calls[0].startswith(f"al validate-config {ROOT}/examples/paper_cases/{config}|")
    assert calls[1].startswith(f"al run {ROOT}/examples/paper_cases/{config} --log-level INFO|")
    assert calls[1].split("|")[1:3] == ["venv", dft]


def test_aurora_paper_case_job_forwards_the_mlip_config(tmp_path):
    capture = tmp_path / "calls.txt"
    venv = _fake_aurora_venv(tmp_path, capture)
    mace_venv = _fake_venv(tmp_path, capture, "venv-mace")
    result = subprocess.run(
        ["bash", str(ROOT / "deployments/aurora/jobs/job-active-learning-paper-cases-aurora.sh")],
        env={
            **os.environ,
            "PROJECT_ROOT": str(ROOT),
            "RUNS_ROOT": str(tmp_path / "runs"),
            "MATSIM_AURORA_VENV": str(venv),
            "MATSIM_MACE_VENV": str(mace_venv),
            "BASH_ENV": str(venv / "bash-env"),
            "CASE": "hea_bcc",
            "MLIP_CONFIG": "examples/mlip/mace-mp-medium.yaml",
            "PBS_JOBID": "123.aurora",
        },
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    config = ROOT / "examples/paper_cases/al_hea_bcc_aurora.yaml"
    mlip = ROOT / "examples/mlip/mace-mp-medium.yaml"
    calls = capture.read_text().splitlines()
    assert calls[0].startswith(f"al validate-config {config} --mlip-config {mlip}|venv-mace|")
    assert calls[1].startswith(f"al run {config} --mlip-config {mlip} --log-level INFO|venv-mace|")


def test_aurora_paper_case_job_rejects_unknown_case(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "deployments/aurora/jobs/job-active-learning-paper-cases-aurora.sh")],
        env={**os.environ, "PROJECT_ROOT": str(ROOT), "CASE": "nope"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unknown CASE='nope'" in result.stderr


def test_aurora_scaling_job_tags_run_by_unique_pbs_nodes(tmp_path):
    capture = tmp_path / "calls.txt"
    venv = _fake_aurora_venv(tmp_path, capture)
    nodefile = tmp_path / "nodefile"
    nodefile.write_text("x1\nx1\nx2\nx2\n")
    result = subprocess.run(
        ["bash", str(ROOT / "deployments/aurora/jobs/job-al-dft-scaling-aurora.sh")],
        env={
            **os.environ,
            "PROJECT_ROOT": str(ROOT),
            "RUNS_ROOT": str(tmp_path / "runs"),
            "MATSIM_AURORA_VENV": str(venv),
            "BASH_ENV": str(venv / "bash-env"),
            "PBS_NODEFILE": str(nodefile),
            "REP": "3",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    run = capture.read_text().splitlines()[1].split("|")
    assert (
        run[0]
        == f"al run {ROOT}/examples/paper_cases/al_hea_fcc_scaling_aurora.yaml --log-level INFO"
    )
    assert run[1] == "venv"
    assert run[3:5] == ["hea-fcc-scaling-aurora-N2-r3", "15"]


def _perlmutter_paper_case(tmp_path: Path, **env: str) -> tuple[subprocess.CompletedProcess, list]:
    capture = tmp_path / "calls.txt"
    venvs = {
        name: _fake_venv(tmp_path, capture, name) for name in ("venv", "venv-uma", "venv-mace")
    }
    bash_env = tmp_path / "bash-env"
    bash_env.write_text("module() { return 0; }\nml() { return 0; }\n")
    uma_bundle = tmp_path / "artifacts/uma/uma-s-1p1"
    uma_bundle.mkdir(parents=True)
    (uma_bundle / "checkpoint.pt").write_text("x")
    (uma_bundle / "atom_refs.yaml").write_text("x")
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "deployments/perlmutter/jobs/job-active-learning-paper-cases-perlmutter.sh"),
        ],
        env={
            **os.environ,
            "PROJECT_ROOT": str(ROOT),
            "RUNS_ROOT": str(tmp_path / "runs"),
            "BASH_ENV": str(bash_env),
            "MATSIM_PERLMUTTER_VENV": str(venvs["venv"]),
            "MATSIM_HYDRAGNN_VENV": str(venvs["venv"]),
            "MATSIM_FAIRCHEM_VENV": str(venvs["venv-uma"]),
            "MATSIM_MACE_VENV": str(venvs["venv-mace"]),
            "MATSIM_MODEL_ARTIFACTS_ROOT": str(tmp_path / "artifacts"),
            "SLURM_JOB_ID": "77",
            **env,
        },
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    calls = capture.read_text().splitlines() if capture.exists() else []
    return result, [call.split("|") for call in calls]


def test_perlmutter_zn_formate_selects_the_uma_config_for_the_default_uma_model(tmp_path):
    result, calls = _perlmutter_paper_case(tmp_path, CASE="zn_formate")
    assert result.returncode == 0, result.stderr
    mlip = ROOT / "examples/mlip/uma-s-1p1-omol.yaml"
    config = ROOT / "examples/paper_cases/al_zn_formate_uma.yaml"
    assert calls[1][0] == f"al run {config} --mlip-config {mlip} --log-level INFO"
    assert calls[1][1] == "venv-uma"


def test_perlmutter_paper_case_runs_its_own_model_with_an_empty_mlip_config(tmp_path):
    result, calls = _perlmutter_paper_case(tmp_path, CASE="zn_formate", MLIP_CONFIG="")
    assert result.returncode == 0, result.stderr
    config = ROOT / "examples/paper_cases/al_zn_formate.yaml"
    assert calls[1][0] == f"al run {config} --log-level INFO"
    assert calls[1][1] == "venv"


@pytest.mark.parametrize(
    ("mlip_config", "retrain", "trainer"),
    [
        ("examples/mlip/mace-mp-medium.yaml", None, "false"),
        ("examples/mlip/mace-mp-medium.yaml", "1", "true"),
        ("examples/mlip/uma-s-1p1-omat.yaml", None, "false"),
        ("", None, "true"),
        ("", "0", "false"),
    ],
)
def test_perlmutter_paper_case_retrains_hydragnn_and_freezes_foundation_models(
    tmp_path, mlip_config, retrain, trainer
):
    env = {"CASE": "hea_bcc", "MLIP_CONFIG": mlip_config}
    if retrain is not None:
        env["RETRAIN"] = retrain
    result, calls = _perlmutter_paper_case(tmp_path, **env)
    assert result.returncode == 0, result.stderr
    config = ROOT / "examples/paper_cases/al_hea_bcc.yaml"
    mlip_args = f" --mlip-config {ROOT / mlip_config}" if mlip_config else ""
    venv = {"": "venv", "mace": "venv-mace", "uma": "venv-uma"}
    assert calls[1][0] == f"al run {config}{mlip_args} --log-level INFO"
    assert calls[1][1] == venv[mlip_config.split("/")[-1].split("-")[0]]
    assert calls[1][5] == trainer


def test_bounded_campaign_submission_names_and_forwards_the_mlip_config(tmp_path):
    capture = tmp_path / "sbatch.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        f'#!/bin/bash\nprintf "%s|%s\\n" "$MATSIM_CAMPAIGN_RUN_TAG" "$MLIP_CONFIG" > "{capture}"\n'
    )
    sbatch.chmod(0o755)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _python_shim(venv / "bin/python3")
    references = tmp_path / "elemental.json"
    references.write_text("{}")
    result = subprocess.run(
        ["bash", str(ROOT / "deployments/perlmutter/jobs/submit-nb-ta-o-combined-bounded.sh")],
        env={
            **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "MATSIM_PERLMUTTER_VENV": str(venv),
            "MATSIM_MODEL_ARTIFACTS_ROOT": str(tmp_path / "artifacts"),
            "MATSIM_CAMPAIGN_PROMOTION_VALIDATION_REFERENCE_SET": str(references),
            "MLIP_CONFIG": "examples/mlip/mace-omat-0-medium.yaml",
        },
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    run_tag, mlip_config = capture.read_text().strip().split("|")
    assert run_tag.startswith("nb-ta-o--7llm-mace-al-qe-hull--")
    assert mlip_config == str(ROOT / "examples/mlip/mace-omat-0-medium.yaml")


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("hydragnn", "site-venv"), ("uma", "venv-uma"), ("mace", "venv-mace")],
)
def test_chat_clients_run_in_the_selected_backends_environment(tmp_path, backend, expected):
    config = tmp_path / "mlip.yaml"
    config.write_text(f"mlip:\n  backend: {backend}\n")
    helpers = ROOT / "deployments/common/runtime-env.sh"
    script = (
        f'source "{helpers}"\n'
        f'mlip_backend_venv "{tmp_path}" "$(mlip_backend "{sys.executable}" "{config}")" '
        f'"{tmp_path}/site-venv"\n'
    )
    env = {**os.environ, "MATSIM_FAIRCHEM_VENV": str(tmp_path / "venv-uma")}
    env.pop("MATSIM_MACE_VENV", None)
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    expected_path = tmp_path / (".venv-mace" if backend == "mace" else expected)
    assert result.stdout.strip() == str(expected_path)


def test_chat_default_selection_is_the_hydragnn_foundation_model(tmp_path):
    helpers = ROOT / "deployments/common/runtime-env.sh"
    env = {k: v for k, v in os.environ.items() if k != "MLIP_CONFIG"}
    result = subprocess.run(
        ["bash", "-c", f'source "{helpers}"; default_mlip_config "{tmp_path}" /gfm /gfm/mlp.pt'],
        env=env,
        capture_output=True,
        text=True,
    )
    path = Path(result.stdout.strip())
    assert path == tmp_path / "mlip.yaml"
    assert yaml.safe_load(path.read_text())["mlip"] == {
        "backend": "hydragnn",
        "logdir": "/gfm",
        "mlp_checkpoint": "/gfm/mlp.pt",
    }
    chosen = subprocess.run(
        ["bash", "-c", f'source "{helpers}"; default_mlip_config "{tmp_path}" /gfm /m'],
        env={**env, "MLIP_CONFIG": "/my/uma.yaml"},
        capture_output=True,
        text=True,
    )
    assert chosen.stdout.strip() == "/my/uma.yaml"


@pytest.mark.parametrize("backend", ["hydragnn", "uma", "mace"])
def test_selected_model_artifacts_are_staged(tmp_path, backend):
    """Offline jobs export the selected model's artifact location before loading it."""
    blocks = {
        "hydragnn": "backend: hydragnn\n  logdir: /gfm\n  mlp_checkpoint: /gfm/mlp.pt",
        "uma": "backend: uma\n  model_name: uma-s-1p1",
        "mace": "backend: mace\n  model: small",
    }
    config = tmp_path / "mlip.yaml"
    config.write_text(f"mlip:\n  {blocks[backend]}\n")
    bundle = tmp_path / "artifacts" / "uma" / "uma-s-1p1"
    bundle.mkdir(parents=True)
    (bundle / "checkpoint.pt").write_text("weights")
    (bundle / "atom_refs.yaml").write_text("refs")
    helper = ROOT / "deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
    script = (
        f'source "{helper}"\n'
        f'configure_mlip_model_artifacts "{tmp_path / "repo"}" "{sys.executable}" "{config}"\n'
        'printf "%s|%s\\n" "${MATSIM_UMA_ARTIFACT_DIR:-}" "${MACE_CACHE:-}"\n'
    )
    env = {**os.environ, "MATSIM_MODEL_ARTIFACTS_ROOT": str(tmp_path / "artifacts")}
    for name in ("MATSIM_UMA_ARTIFACT_DIR", "MACE_ARTIFACT_DIR", "MACE_CACHE"):
        env.pop(name, None)
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    uma_dir, mace_cache = result.stdout.strip().split("|")
    assert bool(uma_dir) is (backend == "uma")
    assert bool(mace_cache) is (backend == "mace")
    if backend == "uma":
        assert uma_dir == str(tmp_path / "artifacts" / "uma")
