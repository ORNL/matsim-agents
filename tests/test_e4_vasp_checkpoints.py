from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "deployments/aurora/jobs/job-e4-vasp-ab-aurora.pbs"
QUALIFICATION_PBS = ROOT / "deployments/aurora/jobs/job-e4-bcc-restart-qualification.pbs"


@pytest.fixture
def checkpoint_job(tmp_path):
    case = tmp_path / "case-1"
    case.mkdir()
    nodefile = tmp_path / "pbs-nodefile"
    nodefile.write_text("aurora-node-01\n" * 208 + "aurora-node-02\n" * 208)
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
[[ "$4" == "${E4_MPI_RANKS:-8}" ]]
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
if [[ "$MODE" == stop || "$MODE" == signal ||
      ("$MODE" == timed && "$attempt" == 01) ||
      ("$MODE" == checkpoint-deletes-stopcar && "$attempt" == 01) ]]; then
  while [[ ! -e STOPCAR ]]; do sleep 0.05; done
elif [[ "$MODE" == external-stop || "$MODE" == stopped-scf-failure ||
        ("$MODE" == normal-stop && "$attempt" == 02) ||
        ("$MODE" == checkpoint-deletes-stopcar && "$attempt" == 02) ]]; then
  echo 'LSTOP = .TRUE.' > STOPCAR
fi
echo 'geometry checkpoint' > CONTCAR
echo 'wave checkpoint' > WAVECAR
echo 'charge checkpoint' > CHGCAR
if [[ "$MODE" == checkpoint-deletes-stopcar && "$attempt" == 01 ]]; then
  rm STOPCAR
fi
if [[ "$MODE" == empty ]]; then : > WAVECAR; fi
if [[ "$MODE" == missing ]]; then rm CHGCAR; fi
if [[ "$MODE" != abnormal ]]; then
  {
    echo 'NIONS = 1'
    echo 'General timing and accounting'
    echo ' TOTAL-FORCE (eV/Angst)'
    echo ' ------------------------------------------------'
    force=0.005
    [[ "$MODE" != force-unconverged ]] || force=0.02
    printf ' 0.0 0.0 0.0 %s 0.0 0.0\\n' "$force"
    echo ' ------------------------------------------------'
  } > OUTCAR
fi
nion=1
if [[ ("$MODE" == normal && "$attempt" == 01) || "$MODE" == between-segments ]]; then
  nion=$(awk '/^NSW =/{print $3}' INCAR)
elif [[ "$MODE" == normal || "$MODE" == before-normal ||
        "$MODE" == force-unconverged ||
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
        ': "${E4_CALCULATION_ID:?E4_CALCULATION_ID is required}"\n'
        'STAGE_MODE="${E4_STAGE_MODE:-low-normal}"\n'
        'SINGLE_PRECISION="${E4_SINGLE_PRECISION:-Accurate}"\n'
        'SINGLE_PREFIX="${E4_SINGLE_PREFIX:-A}"\n'
        'RANKS="${E4_MPI_RANKS:-8}"\n'
        'THREADS="${E4_OMP_THREADS:-8}"\n'
        'NSW_MAX="${E4_NSW:-30}"'
        + settings
        + "\nNELM_MAX=170\nVASP_BIN=unused\nATTEMPT=0\nSTATUS=\n"
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
        "E4_CALCULATION_ID": "test-calculation-01",
        "E4_REQUESTED_NODES": "2",
        "E4_REQUESTED_WALLTIME_SECONDS": "900",
        "PBS_JOBID": "fixture-job-0001",
        "PBS_NODEFILE": str(nodefile),
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
    assert "stage=Normal completed_ionic_steps=0 maximum_ionic_steps=240" in result.stdout
    attempts = sorted((run_root / "attempts").iterdir())
    assert len(attempts) == 3
    for attempt, precision in zip(attempts, ["Low", "Low", "Normal"], strict=True):
        incar = (attempt / "INCAR").read_text()
        assert f"PREC = {precision}" in incar
        assert f"NSW = {segment_steps or '30'}\n" in incar
        assert "EDIFFG = -0.01\n" in incar
        assert "EDIFF=1E-6" in incar
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


def test_checkpoint_is_accepted_when_vasp_removes_stopcar(checkpoint_job):
    result, run_root = run_job(
        checkpoint_job,
        MODE="checkpoint-deletes-stopcar",
        E4_CHECKPOINT_SECONDS="1",
        E4_SOFT_STOP_SECONDS="100",
    )
    assert result.returncode == 75, result.stdout + result.stderr
    assert "status=checkpoint:time" in result.stdout
    assert "final=Low:interrupted:stopcar" in result.stdout
    assert len(list((run_root / "attempts").iterdir())) == 2
    assert (run_root / "0.checkpoint.sha256").is_file()
    assert (run_root / "1.checkpoint.sha256").is_file()


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
        ("force-unconverged", "force-threshold-not-met"),
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
    expected_stop = (
        "interrupted:soft-stop-before-launch"
        if mode == "before-normal"
        else "final=Low:interrupted:stopcar"
    )
    assert expected_stop in result.stdout
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
        {"E4_STAGE_NSW_MAX": "abc"},
        {"E4_STAGE_NSW_MAX": "241"},
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


@pytest.mark.parametrize(
    "modified",
    ["0.WAVECAR", "0.CHGCAR", "0.CONTCAR", "0.checkpoint.sha256", "START"],
)
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


def test_stage_limit_caps_segment_and_is_not_reset_on_resume(checkpoint_job):
    first, run_root = run_job(
        checkpoint_job,
        MODE="normal",
        E4_NSW="30",
        E4_STAGE_NSW_MAX="2",
    )
    assert first.returncode == 1, first.stdout + first.stderr
    assert "final=Low:failed:stage-step-limit" in first.stdout
    attempts = sorted((run_root / "attempts").iterdir())
    assert len(attempts) == 1
    assert "NSW = 2\n" in (attempts[0] / "INCAR").read_text()

    resumed, _ = run_job(
        checkpoint_job,
        MODE="normal",
        E4_NSW="30",
        E4_STAGE_NSW_MAX="2",
        E4_RESUME="1",
    )
    assert resumed.returncode == 1, resumed.stdout + resumed.stderr
    assert "resume verified tag=0" in resumed.stdout
    assert "completed_ionic_steps=2 maximum_ionic_steps=2" in resumed.stdout
    assert "stage-step-limit" in resumed.stdout
    assert len(list((run_root / "attempts").iterdir())) == 1


def test_resume_rejects_mpi_runtime_change(checkpoint_job):
    first, _ = run_job(checkpoint_job, MODE="external-stop")
    assert first.returncode == 75
    result, _ = run_job(
        checkpoint_job,
        E4_RESUME="1",
        VASP_MPICH_MODULE="different-mpich-module",
    )
    assert result.returncode == 2
    assert "policy changed" in result.stdout


def test_resume_rejects_calculation_identity_change(checkpoint_job):
    first, _ = run_job(checkpoint_job, MODE="external-stop", E4_CALCULATION_ID="case-original")
    assert first.returncode == 75
    result, _ = run_job(
        checkpoint_job,
        E4_RESUME="1",
        E4_CALCULATION_ID="different-calculation",
    )
    assert result.returncode == 2
    assert "policy changed" in result.stdout


def test_calculation_id_is_required(checkpoint_job):
    result, run_root = run_job(checkpoint_job, E4_CALCULATION_ID="")
    assert result.returncode != 0
    assert "E4_CALCULATION_ID is required" in result.stderr
    assert not run_root.exists()


def test_fresh_run_refuses_to_overwrite_existing_calculation(checkpoint_job):
    first, run_root = run_job(checkpoint_job, MODE="external-stop")
    assert first.returncode == 75
    original_checkpoint = (run_root / "latest-checkpoint").read_bytes()
    original_record = (run_root / "run-record.log").read_bytes()

    second, _ = run_job(checkpoint_job, E4_RESUME="0")
    assert second.returncode != 0
    assert (run_root / "latest-checkpoint").read_bytes() == original_checkpoint
    assert (run_root / "run-record.log").read_bytes() == original_record


def test_run_record_accounts_for_each_resume_allocation(checkpoint_job):
    first, run_root = run_job(
        checkpoint_job,
        MODE="external-stop",
        PBS_JOBID="fixture-job-0001",
    )
    assert first.returncode == 75
    second, _ = run_job(
        checkpoint_job,
        E4_RESUME="1",
        PBS_JOBID="fixture-job-0002",
    )
    assert second.returncode == 0, second.stdout + second.stderr

    accounting = [
        line
        for line in (run_root / "run-record.log").read_text().splitlines()
        if "allocation_accounting " in line
    ]
    assert len(accounting) == 2
    for line, job_id in zip(accounting, ["fixture-job-0001", "fixture-job-0002"], strict=True):
        assert f"pbs_job_id={job_id}" in line
        assert "calculation_id=test-calculation-01" in line
        assert "allocated_nodes=2" in line
        assert "requested_nodes=2" in line
        assert "walltime_limit_seconds=900" in line
        assert "requested_node_hours=0.500000" in line
        assert "elapsed_seconds=" in line
        assert "elapsed_node_hours=" in line


def test_bcc_restart_qualification_pbs_is_bounded_and_non_submitting():
    text = QUALIFICATION_PBS.read_text()
    assert "#PBS -A HydraGNN" in text
    assert "#PBS -q capacity" in text
    assert "#PBS -l select=1" in text
    assert "#PBS -l walltime=00:15:00" in text
    assert "E4_SOFT_STOP_SECONDS=680" in text
    assert "E4_REQUESTED_WALLTIME_SECONDS=900" in text
    assert "REPEAT=repeat-03" in text
    assert "E4_CALCULATION_ID=E4-BCC-AB-Nb22Ta22V21Hf21Zr21Ti21-case-1" in text
    assert "E4_RESTART_PHASE:?set E4_RESTART_PHASE=first or resume" in text
    assert "qsub " not in text


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


def test_single_precision_checkpoint_tag_resumes_with_same_protocol(checkpoint_job):
    first, run_root = run_job(
        checkpoint_job,
        MODE="external-stop",
        E4_STAGE_MODE="single",
        E4_SINGLE_PRECISION="Accurate",
    )
    assert first.returncode == 75, first.stdout + first.stderr
    assert "final=Accurate:interrupted:stopcar" in first.stdout
    assert (run_root / "A0.checkpoint.sha256").is_file()

    second, _ = run_job(
        checkpoint_job,
        MODE="normal",
        E4_RESUME="1",
        E4_STAGE_MODE="single",
        E4_SINGLE_PRECISION="Accurate",
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert "final=converged:A1" in second.stdout
    assert (run_root / "A1.checkpoint.sha256").is_file()
    assert "ISTART = 1" in (run_root / "A1.INCAR").read_text()
    assert "ICHARG = 1" in (run_root / "A1.INCAR").read_text()


def test_li_fepo4_kpar_one_and_ncore_twelve_use_twelve_ranks(checkpoint_job):
    script, case, run_root, env = checkpoint_job
    start = (case / "START").read_text().replace("KPAR = 8", "KPAR = 1\nNCORE = 12")
    (case / "START").write_text(start)
    first, _ = run_job(
        checkpoint_job,
        MODE="external-stop",
        E4_STAGE_MODE="single",
        E4_SINGLE_PRECISION="Accurate",
        E4_MPI_RANKS="12",
    )
    assert first.returncode == 75, first.stdout + first.stderr
    assert "KPAR = 1" in (run_root / "A0.INCAR").read_text()
    assert "NCORE = 12" in (run_root / "A0.INCAR").read_text()
    second, _ = run_job(
        checkpoint_job,
        MODE="normal",
        E4_RESUME="1",
        E4_STAGE_MODE="single",
        E4_SINGLE_PRECISION="Accurate",
        E4_MPI_RANKS="12",
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert "final=converged:A1" in second.stdout


def test_ncore_cannot_exceed_ranks_per_kpoint_group(checkpoint_job):
    _, case, run_root, _ = checkpoint_job
    (case / "START").write_text(
        (case / "START").read_text().replace("KPAR = 8", "KPAR = 1\nNCORE = 12")
    )
    result, _ = run_job(
        checkpoint_job,
        MODE="normal",
        E4_STAGE_MODE="single",
        E4_SINGLE_PRECISION="Accurate",
        E4_MPI_RANKS="8",
    )
    assert result.returncode == 1
    assert "invalid NCORE=12 for 8 MPI ranks per k-point group" in result.stdout
    assert not (run_root / "attempts/attempt-01/started").exists()
    assert not list(run_root.glob("A*.checkpoint.sha256"))
