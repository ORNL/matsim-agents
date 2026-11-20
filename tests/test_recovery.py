"""Failure classification and the recovery policy's setting adjustments."""

from __future__ import annotations

import pytest

from matsim_agents.execution.recovery import (
    FailureClass,
    adjusted_settings,
    can_retry,
    classify_failure,
)


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ({"status": "failed", "return_code": 124, "failure_reason": "x"}, "walltime"),
        ({"status": "failed", "failure_type": "FileNotFoundError"}, "launcher"),
        ({"status": "failed", "failure_reason": "pw.x: command not found"}, "launcher"),
        (
            {"status": "failed", "return_code": 127, "failure_reason": "did not converge"},
            "launcher",
        ),
        (
            {"status": "failed", "return_code": 126, "failure_reason": "did not converge"},
            "launcher",
        ),
        (
            {
                "status": "failed",
                "return_code": 2,
                "failure_reason": "VASP did not converge (return code 2): "
                "ERROR: vasp binary not executable: /opt/vasp/bin/vasp_std",
            },
            "launcher",
        ),
        ({"status": "failed", "failure_type": "PermissionError"}, "approval_or_input"),
        (
            {
                "status": "failed",
                "return_code": 1,
                "failure_reason": "QE did not converge (return code 1): Error in routine "
                "read_namelists (1): bad line in namelist &electrons",
            },
            "approval_or_input",
        ),
        (
            {
                "status": "failed",
                "return_code": 1,
                "failure_reason": "VASP did not converge (return code 1): Error reading item "
                "'NELM' from file INCAR.",
            },
            "approval_or_input",
        ),
        ({"status": "failed", "scf_converged": False, "return_code": 2}, "scf_not_converged"),
        ({"status": "failed", "failure_reason": "non-finite output"}, "non_finite_output"),
        ({"status": "failed", "failure_reason": "produced no energy, force"}, "no_result"),
        ({"status": "done", "scf_converged": True, "steps": 50}, "ionic_step_limit"),
        ({"status": "done", "steps": 10}, "unknown"),
        ({"status": "failed", "failure_reason": "segfault"}, "unknown"),
    ],
)
def test_failures_are_classified_deterministically(outcome, expected):
    assert classify_failure(outcome, max_steps=50) == FailureClass(expected)


def test_scf_recovery_softens_mixing_without_editing_the_input():
    settings = {"ecutwfc_ry": 60, "extra_electrons": {"electron_maxstep": 300}}
    qe, _ = adjusted_settings(FailureClass.SCF_NOT_CONVERGED, "qe", settings, None)
    assert qe["mixing_beta"] == 0.2 and qe["extra_electrons"]["electron_maxstep"] == 600
    assert settings["extra_electrons"]["electron_maxstep"] == 300
    vasp, _ = adjusted_settings(FailureClass.SCF_NOT_CONVERGED, "vasp", {}, None)
    assert vasp == {"nelm": 200, "algo": "All"}


def test_scf_recovery_reaches_the_written_pw_input(tmp_path):
    from ase.build import bulk

    from matsim_agents.backends.dft.qe_relax import QESettings, write_pw_input

    def namelist_value(text, key):
        for line in text.splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == key:
                return float(value.strip().replace("d", "e"))
        raise AssertionError(f"{key} missing from pw.in")

    (tmp_path / "Si.upf").write_text("placeholder")
    atoms = bulk("Si", "diamond", a=5.43)
    settings: dict = {}
    written = []
    for attempt in range(3):
        qe = QESettings(
            calculation="relax",
            pseudo_dir=str(tmp_path),
            nstep=50,
            forc_conv_thr_ry_au=0.001,
            **settings,
        )
        path = tmp_path / f"pw{attempt}.in"
        write_pw_input(atoms, qe, str(path))
        text = path.read_text()
        written.append(
            (namelist_value(text, "mixing_beta"), namelist_value(text, "electron_maxstep"))
        )
        settings, _ = adjusted_settings(FailureClass.SCF_NOT_CONVERGED, "qe", settings, None)
    assert written == [(0.4, 200), (0.2, 400), (0.1, 800)]


def test_walltime_is_retried_only_with_a_timeout_to_extend():
    assert adjusted_settings(FailureClass.WALLTIME, "qe", {}, 600.0) == ({}, 1200.0)
    assert can_retry(FailureClass.WALLTIME, timeout_sec=600.0)
    assert not can_retry(FailureClass.WALLTIME, timeout_sec=None)
    assert not can_retry(FailureClass.LAUNCHER, timeout_sec=600.0)
    assert not can_retry(FailureClass.APPROVAL_OR_INPUT, timeout_sec=None)
