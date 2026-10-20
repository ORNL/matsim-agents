"""Off-system wiring tests for the Aurora LLM, launcher, and smoke-test scripts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AURORA = ROOT / "deployments" / "aurora"


def _stub(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


def test_aurora_vllm_helper_launches_and_waits_for_exact_model(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_file = tmp_path / "mpiexec-args"
    _stub(bin_dir, "mpiexec", f'printf "%s\\n" "$@" > "{args_file}"\nsleep 30\n')
    _stub(bin_dir, "curl", 'echo \'{"data": [{"id": "org/model-a"}]}\'\n')
    script = f"""
set -e
source "{AURORA}/launchers/aurora-vllm.sh"
aurora_vllm_env "{tmp_path}/run" 4
echo "MASK=$ZE_AFFINITY_MASK TMPDIR=$TMPDIR FLAT=$ZE_FLAT_DEVICE_HIERARCHY"
aurora_vllm_start /models/a org/model-a 4 8123 4096 "{tmp_path}/vllm.log"
aurora_vllm_wait 8123 org/model-a 30
aurora_vllm_stop
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "MASK=0,1,2,3 TMPDIR=/tmp FLAT=FLAT" in result.stdout
    assert "org/model-a ready" in result.stdout
    args = args_file.read_text().splitlines()
    assert args[:4] == ["-n", "1", "--ppn", "1"]
    assert "-u" in args and "PMI_RANK" in args
    assert str(AURORA / "smoke-tests/aurora_vllm_entrypoint.py") in args
    for flag, value in (
        ("--model", "/models/a"),
        ("--served-model-name", "org/model-a"),
        ("--tensor-parallel-size", "4"),
        ("--port", "8123"),
        ("--max-model-len", "4096"),
        ("--distributed-executor-backend", "mp"),
    ):
        assert args[args.index(flag) + 1] == value


def test_aurora_vllm_wait_fails_when_server_exits(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "mpiexec", "exit 1\n")
    _stub(bin_dir, "curl", "exit 7\n")
    script = f"""
source "{AURORA}/launchers/aurora-vllm.sh"
aurora_vllm_start /models/a a 1 8000 4096 "{tmp_path}/vllm.log"
sleep 1
aurora_vllm_wait 8000 a 30
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1
    assert "server exited before readiness" in result.stderr


@pytest.mark.parametrize(
    ("launcher", "models", "expected"),
    [
        (
            "launch-test-all-models-aurora.sh",
            ["Qwen2.5-14B-Instruct", "Mixtral-8x22B-Instruct-v0.1"],
            [
                "SMOKE_MODEL_PATH={models}/Qwen2.5-14B-Instruct,"
                "SMOKE_MODEL_NAME=Qwen/Qwen2.5-14B-Instruct,TP_SIZE=2",
                "SMOKE_MODEL_PATH={models}/Mixtral-8x22B-Instruct-v0.1,"
                "SMOKE_MODEL_NAME=mistralai/Mixtral-8x22B-Instruct-v0.1,TP_SIZE=8",
            ],
        ),
        (
            "launch-test-multinode-aurora.sh",
            ["Llama-3.3-70B-Instruct"],
            [
                "-l select=2 -v SERVE_MODEL_PATH={models}/Llama-3.3-70B-Instruct,"
                "SERVE_MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct,SERVE_TP_SIZE=16,"
                "SERVE_SMOKE=1"
            ],
        ),
    ],
)
def test_aurora_test_launchers_submit_available_models(tmp_path, launcher, models, expected):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    submissions = tmp_path / "qsub.log"
    _stub(bin_dir, "qsub", f'echo "$*" >> "{submissions}"\necho "100.aurora"\n')
    _stub(
        bin_dir,
        "qstat",
        'if [[ "$1" == "-xf" ]]; then echo "    Exit_status = 0"; exit 0; fi\nexit 1\n',
    )
    model_root = tmp_path / "models"
    for name in models:
        (model_root / name).mkdir(parents=True)
    result = subprocess.run(
        ["bash", str(AURORA / "launchers" / launcher)],
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "MODEL_ROOT": str(model_root),
            "RUNS_ROOT": str(tmp_path / "runs"),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    lines = submissions.read_text().splitlines()
    assert len(lines) == len(expected)
    for line, fragment in zip(lines, expected, strict=True):
        assert fragment.format(models=model_root) in line
    summary = next((tmp_path / "runs").glob("test-*.tsv")).read_text()
    assert summary.count("100.aurora\t0\t") == len(expected)


def test_aurora_resume_launcher_honors_model_whitelist(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    submissions = tmp_path / "qsub.log"
    _stub(bin_dir, "qsub", f'echo "$*" >> "{submissions}"\necho "7.aurora"\n')
    _stub(bin_dir, "qstat", 'if [[ "$1" == "-xf" ]]; then echo "Exit_status = 0"; fi\nexit 1\n')
    model_root = tmp_path / "models"
    for name in ("Qwen3.6-27B", "gemma-4-31B-it"):
        (model_root / name).mkdir(parents=True)
    result = subprocess.run(
        ["bash", str(AURORA / "launchers/launch-test-singlenode-resume-aurora.sh")],
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "MODEL_ROOT": str(model_root),
            "RUNS_ROOT": str(tmp_path / "runs"),
            "RESUME_MODELS": "gemma-4-31B-it",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    (line,) = submissions.read_text().splitlines()
    assert "SMOKE_MODEL_NAME=google/gemma-4-31B-it" in line


@pytest.mark.parametrize(
    ("script", "env", "message"),
    [
        ("jobs/job-llm-check-aurora.sh", {"MODEL_ROOT": "/nonexistent"}, "model directory"),
        ("jobs/job-sequential-benchmark-aurora.sh", {}, "BENCHMARK_PROMPT is required"),
        ("smoke-tests/smoke-transformers-aurora.sh", {"MATSIM_MODEL_DIR": "/nonexistent"}, "model"),
    ],
)
def test_aurora_llm_jobs_fail_fast_on_missing_inputs(tmp_path, script, env, message):
    environment = {k: v for k, v in os.environ.items() if k != "BENCHMARK_PROMPT"}
    environment.update(
        {"PROJECT_ROOT": str(ROOT), "RUNS_ROOT": str(tmp_path / "runs"), "PBS_JOBID": "1"}
    )
    environment.update(env)
    result = subprocess.run(
        ["bash", str(AURORA / script)],
        env=environment,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )
    assert result.returncode == 2
    assert message in result.stderr


def test_aurora_multinode_serve_job_has_opt_in_smoke_mode():
    text = (AURORA / "jobs/job-serve-multinode-vllm-aurora.sh").read_text()
    smoke = text.index('if [[ "${SERVE_SMOKE:-0}" == "1" ]]; then')
    assert smoke < text.index('wait "$VLLM_PID"\necho "[serve] vLLM process exited."')
    assert "/v1/chat/completions" in text[smoke:]


SHARED_ENTRY_POINTS = [
    "launchers/_hydragnn-train-step-{f}.sh",
    "launchers/_qe-step-{f}.sh",
    "launchers/_vasp-step-{f}.sh",
    "launchers/launch-test-all-models-{f}.sh",
    "launchers/launch-test-multinode-{f}.sh",
    "launchers/launch-test-singlenode-resume-{f}.sh",
    "jobs/job-active-learning-paper-cases-{f}.sh",
    "jobs/job-active-learning-uq-{f}.sh",
    "jobs/job-al-dft-scaling-{f}.sh",
    "jobs/job-discovery-chat-{f}.sh",
    "jobs/job-llm-check-{f}.sh",
    "jobs/job-portability-benchmark-{f}.sh",
    "jobs/job-qe-warmstart-{f}.sh",
    "jobs/job-sequential-benchmark-{f}.sh",
    "jobs/job-serve-multinode-{f}.sh",
    "jobs/job-single-relaxation-{f}.sh",
    "jobs/submit-al-dft-scaling-sweep.sh",
    "setup/{f}-module-stack.sh",
    "setup/install.sh",
    "setup/setup_matsim_{f}.sh",
    "smoke-tests/smoke-transformers-{f}.sh",
]


@pytest.mark.parametrize("facility", ["frontier", "perlmutter", "aurora"])
def test_facilities_share_core_entry_points(facility):
    missing = [
        pattern.format(f=facility)
        for pattern in SHARED_ENTRY_POINTS
        if not (ROOT / "deployments" / facility / pattern.format(f=facility)).is_file()
    ]
    assert missing == []


def test_aurora_scripts_load_modules_through_the_shared_helper():
    inline = {
        str(path.relative_to(ROOT))
        for path in AURORA.rglob("*.sh")
        if path.parent.name != "setup"
        and any(line.strip().startswith("module load") for line in path.read_text().splitlines())
    }
    # The vLLM serve job reloads frameworks inside a per-node `bash -c` worker command.
    assert inline == {"deployments/aurora/jobs/job-serve-multinode-vllm-aurora.sh"}
