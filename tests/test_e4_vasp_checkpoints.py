from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "deployments/aurora/jobs/job-e4-vasp-ab-aurora.pbs"


@pytest.fixture
def checkpoint_job(tmp_path):
    case = tmp_path / "case-1"
    case.mkdir()
    for name in ("POSCAR", "POTCAR", "KPOINTS"):
        (case / name).write_text(f"input {name}\n")
    (case / "START").write_text(
        "ISTART = 0\nICHARG = 2\nPREC = PRECISION\nKPAR = 8\nNELM = 170\n"
        "nsw=240; LWAVE=F; LCHARG=F; LH5=T; ENCUT=350 # comment; NSW=99\n"
        "ISIF=3\nIBRION=2\nEDIFF=1E-6\n"
    )
    run_root = case / "custom/repeat-01"
    step = tmp_path / "fake-vasp.sh"
    step.write_text(
        """#!/bin/bash
set -euo pipefail
cd "$1"
touch started
attempt=${PWD##*-}
if [[ "$attempt" != 01 ]]; then
  grep -q 'wave checkpoint' WAVECAR
  grep -q 'charge checkpoint' CHGCAR
  grep -q 'geometry checkpoint' POSCAR
  grep -qx 'ISTART = 1' INCAR
  grep -qx 'ICHARG = 1' INCAR
  [[ ! -e STOPCAR ]]
fi
if [[ "$MODE" == stop || "$MODE" == signal || ("$MODE" == timed && "$attempt" == 01) ]]; then
  while [[ ! -e STOPCAR ]]; do sleep 0.05; done
elif [[ "$MODE" == external-stop || "$MODE" == stopped-scf-failure ||
        ("$MODE" == normal-stop && "$attempt" == 02) ]]; then
  echo 'LSTOP = .TRUE.' > STOPCAR
fi
echo 'geometry checkpoint' > CONTCAR
echo 'wave checkpoint' > WAVECAR
echo 'charge checkpoint' > CHGCAR
if [[ "$MODE" == empty ]]; then : > WAVECAR; fi
if [[ "$MODE" == missing ]]; then rm CHGCAR; fi
if [[ "$MODE" != abnormal ]]; then echo 'General timing and accounting' > OUTCAR; fi
nion=1
if [[ ("$MODE" == normal && "$attempt" == 01) || "$MODE" == between-segments ]]; then
  nion=$(awk '/^NSW =/{print $3}' INCAR)
elif [[ "$MODE" == normal || "$MODE" == before-normal ||
        ("$MODE" == timed && "$attempt" != 01) ||
        ("$MODE" == normal-stop && "$attempt" == 01) ]]; then
  echo 'reached required accuracy' >> OUTCAR
fi
if [[ "$MODE" == between-segments || "$MODE" == before-normal ]]; then
  echo soft-walltime-limit > "$STOP_MARKER"
fi
electronic=3
if [[ "$MODE" == scf-failure || "$MODE" == stopped-scf-failure ]]; then electronic=170; fi
if [[ "$MODE" == zero-steps ]]; then nion=0; touch OSZICAR; fi
for ((i=1; i<=nion; i++)); do
  echo "DAV: $electronic -1" >> OSZICAR
  echo "$i F= -1 E0= -1" >> OSZICAR
done
if [[ "$MODE" == exit-failure ]]; then exit 7; fi
"""
    )
    text = JOB.read_text()
    settings = text.split('NSW_MAX="${E4_NSW:-30}"', 1)[1].split("NELM_MAX=170", 1)[0]
    directories = text.split("umask 077", 1)[1].split("log() {", 1)[0]
    lifecycle = text.split("log() {", 1)[1].split('\nlog "job=', 1)[0]
    workflow = text.split("initialize_recovery() {", 1)[1]
    script = tmp_path / "job.sh"
    script.write_text(
        "set -euo pipefail\n"
        "CASE_DIR=$1\nVARIANT=custom\nREPEAT=repeat-01\nVASP_SHA256=test-binary\n"
        'NSW_MAX="${E4_NSW:-30}"'
        + settings
        + "\nNELM_MAX=170\nRANKS=8\nTHREADS=8\nVASP_BIN=unused\nATTEMPT=0\nSTATUS=\n"
        + "umask 077"
        + directories
        + "log() {"
        + lifecycle
        + "\nexport STOP_MARKER\n"
        + '\nif [[ "$MODE" == expired ]]; then SOFT_STOP_DEADLINE=0; fi\n'
        + "initialize_recovery() {"
        + workflow
    )
    env = {
        **os.environ,
        "VASP_STEP": str(step),
        "VASP_ENV_SETUP": str(step),
        "MODE": "normal",
        "E4_SOFT_STOP_SECONDS": "100",
    }
    env.pop("E4_NSW", None)
    env.pop("E4_RESUME", None)
    env["E4_CHECKPOINT_SECONDS"] = "50"
    return script, case, run_root, env


def run_job(checkpoint_job, **overrides):
    script, case, run_root, env = checkpoint_job
    result = subprocess.run(
        ["bash", str(script), str(case)],
        env={**env, **overrides},
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, run_root


@pytest.mark.parametrize("segment_steps", [None, "2"])
def test_segments_restart_at_same_precision_then_switch(checkpoint_job, segment_steps):
    overrides = {} if segment_steps is None else {"E4_NSW": segment_steps}
    result, run_root = run_job(checkpoint_job, **overrides)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "final=converged:N0" in result.stdout
    attempts = sorted((run_root / "attempts").iterdir())
    assert len(attempts) == 3
    for attempt, precision in zip(attempts, ["Low", "Low", "Normal"], strict=True):
        incar = (attempt / "INCAR").read_text()
        assert f"PREC = {precision}" in incar
        assert f"NSW = {segment_steps or '30'}\n" in incar
        for tag, value in [("LWAVE", ".TRUE."), ("LCHARG", ".TRUE."), ("LH5", ".FALSE.")]:
            assert f"{tag} = {value}\n" in incar
        assert "ENCUT=350" in incar and "ISIF=3" in incar and "EDIFF=1E-6" in incar
        active = [line for line in incar.splitlines() if not line.lstrip().startswith("#")]
        assert sum(line.lstrip().upper().startswith("NSW") for line in active) == 1
    for tag in ("0", "1", "N0"):
        checked = subprocess.run(
            ["sha256sum", "-c", f"{tag}.checkpoint.sha256"],
            cwd=run_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "NSW=99" in (checkpoint_job[1] / "START").read_text()


@pytest.mark.parametrize("mode", ["stop", "external-stop"])
def test_graceful_stop_is_archived_without_relaunch(checkpoint_job, mode):
    result, run_root = run_job(checkpoint_job, MODE=mode, E4_SOFT_STOP_SECONDS="1")
    assert result.returncode == 75, result.stdout + result.stderr
    assert "final=Low:interrupted:stopcar" in result.stdout
    assert len(list((run_root / "attempts").iterdir())) == 1
    assert (run_root / "0.checkpoint.sha256").is_file()
    assert (run_root / "0.STOPCAR").read_text() == "LSTOP = .TRUE.\n"
    if mode == "stop":
        marker = next((run_root / "allocations").glob("*/soft-stop-requested"))
        assert marker.read_text() == "soft-walltime-limit\n"


@pytest.mark.parametrize("stop_signal", [signal.SIGUSR1, signal.SIGTERM])
def test_scheduler_signal_waits_for_vasp_and_checkpoints(checkpoint_job, stop_signal):
    script, case, run_root, env = checkpoint_job
    process = subprocess.Popen(
        ["bash", str(script), str(case)],
        env={**env, "MODE": "signal"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not (run_root / "attempts/attempt-01/started").exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        process.send_signal(stop_signal)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 75, stdout + stderr
        assert (run_root / "0.checkpoint.sha256").is_file()
        reason = signal.Signals(stop_signal).name.removeprefix("SIG")
        marker = next((run_root / "allocations").glob("*/soft-stop-requested"))
        assert marker.read_text() == f"scheduler-signal-{reason}\n"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


@pytest.mark.parametrize(
    ("mode", "status"),
    [
        ("empty", "invalid-checkpoint"),
        ("missing", "invalid-checkpoint"),
        ("abnormal", "abnormal-termination"),
        ("scf-failure", "electronic-nonconvergence"),
        ("stopped-scf-failure", "electronic-nonconvergence"),
        ("zero-steps", "no-completed-ionic-step"),
        ("exit-failure", "exit-7"),
    ],
)
def test_invalid_results_are_not_accepted_as_checkpoints(checkpoint_job, mode, status):
    result, run_root = run_job(checkpoint_job, MODE=mode)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"final=Low:failed:{status}" in result.stdout
    assert not list(run_root.glob("*.checkpoint.sha256"))
    assert len(list((run_root / "attempts").iterdir())) == 1


def test_expired_deadline_prevents_launch(checkpoint_job):
    result, run_root = run_job(checkpoint_job, MODE="expired")
    assert result.returncode == 75, result.stdout + result.stderr
    assert "soft-stop-before-launch" in result.stdout
    assert not list((run_root / "attempts").iterdir())


@pytest.mark.parametrize("mode", ["between-segments", "before-normal"])
def test_stop_between_attempts_preserves_checkpoint_and_prevents_launch(checkpoint_job, mode):
    result, run_root = run_job(checkpoint_job, MODE=mode)
    assert result.returncode == 75, result.stdout + result.stderr
    assert "soft-stop-before-launch" in result.stdout
    assert len(list((run_root / "attempts").iterdir())) == 1
    assert (run_root / "0.checkpoint.sha256").is_file()


@pytest.mark.parametrize(
    "overrides",
    [
        {"E4_NSW": "0"},
        {"E4_NSW": "abc"},
        {"E4_SOFT_STOP_SECONDS": "0"},
        {"E4_SOFT_STOP_SECONDS": "28800"},
        {"E4_CHECKPOINT_SECONDS": "0"},
        {"E4_RESUME": "yes"},
    ],
)
def test_invalid_checkpoint_settings_fail_explicitly(checkpoint_job, overrides):
    result, run_root = run_job(checkpoint_job, **overrides)
    assert result.returncode == 2
    assert result.stderr
    assert not run_root.exists()


def test_time_checkpoint_continues_without_allocation_stop(checkpoint_job):
    result, run_root = run_job(checkpoint_job, MODE="timed", E4_CHECKPOINT_SECONDS="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "status=checkpoint:time" in result.stdout
    assert "final=converged:N0" in result.stdout
    assert not list((run_root / "allocations").glob("*/soft-stop-requested"))
    assert (run_root / "0.STOPCAR").is_file()


@pytest.mark.parametrize("mode", ["external-stop", "before-normal"])
def test_explicit_resume_uses_checkpoint_and_preserves_old_attempts(checkpoint_job, mode):
    first, run_root = run_job(checkpoint_job, MODE=mode)
    assert first.returncode == 75, first.stdout + first.stderr
    original = (run_root / "0.WAVECAR").read_bytes()
    # Simulate an abandoned attempt with a reserved tag and no accepted checkpoint.
    abandoned = run_root / "attempts/attempt-02"
    abandoned.mkdir()
    (abandoned / "snapshot-tag").write_text("1\n" if mode == "external-stop" else "N0\n")
    result, _ = run_job(checkpoint_job, E4_RESUME="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "resume verified tag=0" in result.stdout
    assert (run_root / "0.WAVECAR").read_bytes() == original
    resumed = run_root / "attempts/attempt-03"
    assert resumed.is_dir()
    if mode == "external-stop":
        assert (resumed / "snapshot-tag").read_text() == "2\n"
        assert "PREC = Low" in (resumed / "INCAR").read_text()
    else:
        assert (resumed / "snapshot-tag").read_text() == "N1\n"
        assert "PREC = Normal" in (resumed / "INCAR").read_text()
    assert len(list((run_root / "allocations").iterdir())) == 2


def test_resume_of_completed_normal_does_not_launch(checkpoint_job):
    first, run_root = run_job(checkpoint_job)
    assert first.returncode == 0
    count = len(list((run_root / "attempts").iterdir()))
    result, _ = run_job(checkpoint_job, E4_RESUME="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "already complete" in result.stdout
    assert len(list((run_root / "attempts").iterdir())) == count


def test_resume_interrupted_normal_stays_normal(checkpoint_job):
    first, run_root = run_job(checkpoint_job, MODE="normal-stop")
    assert first.returncode == 75, first.stdout + first.stderr
    result, _ = run_job(checkpoint_job, E4_RESUME="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "resume verified tag=N0 prec=Normal" in result.stdout
    resumed = run_root / "attempts/attempt-03"
    assert "PREC = Normal" in (resumed / "INCAR").read_text()
    assert (resumed / "snapshot-tag").read_text() == "N1\n"


def test_failed_resume_attempt_preserves_previous_accepted_checkpoint(checkpoint_job):
    first, run_root = run_job(checkpoint_job, MODE="external-stop")
    assert first.returncode == 75
    saved = (run_root / "latest-checkpoint").read_bytes()
    failed, _ = run_job(checkpoint_job, E4_RESUME="1", MODE="empty")
    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert (run_root / "latest-checkpoint").read_bytes() == saved
    result, _ = run_job(checkpoint_job, E4_RESUME="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run_root / "attempts/attempt-03/snapshot-tag").read_text() == "2\n"


@pytest.mark.parametrize("modified", ["0.WAVECAR", "0.CONTCAR", "0.checkpoint.sha256", "START"])
def test_resume_rejects_modified_inputs_or_checkpoint(checkpoint_job, modified):
    first, run_root = run_job(checkpoint_job, MODE="external-stop")
    assert first.returncode == 75
    target = checkpoint_job[1] / modified if modified == "START" else run_root / modified
    with target.open("a") as handle:
        handle.write("modified\n")
    result, _ = run_job(checkpoint_job, E4_RESUME="1")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "mismatch" in result.stdout or "changed" in result.stdout
    assert len(list((run_root / "attempts").iterdir())) == 1


def test_resume_rejects_policy_change(checkpoint_job):
    first, _ = run_job(checkpoint_job, MODE="external-stop")
    assert first.returncode == 75
    result, _ = run_job(checkpoint_job, E4_RESUME="1", E4_NSW="10")
    assert result.returncode == 2
    assert "policy changed" in result.stdout


def test_resume_rejects_legacy_metadata(checkpoint_job):
    run_root = checkpoint_job[2]
    (run_root / "attempts").mkdir(parents=True)
    result, _ = run_job(checkpoint_job, E4_RESUME="1")
    assert result.returncode == 2
    assert "legacy runs require separate validation" in result.stdout


def test_lock_rejects_concurrent_resume(checkpoint_job):
    first, run_root = run_job(checkpoint_job, MODE="external-stop")
    assert first.returncode == 75
    with (run_root / ".allocation.lock").open("w") as lock:
        import fcntl

        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, _ = run_job(checkpoint_job, E4_RESUME="1")
    assert result.returncode == 2
    assert "already active" in result.stderr
