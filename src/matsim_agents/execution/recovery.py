"""Controlled recovery from failed DFT relaxations.

A failed attempt is classified by type, and a versioned policy says per type
whether, and with which adjusted settings, it is retried. MLIP relaxations are
not retried: they are deterministic, so a rerun gives the same failure.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

RECOVERY_POLICY_VERSION = "recovery-v1"
WALLTIME_RETURN_CODE = 124

_QE_MIXING_BETA = 0.4
_QE_ELECTRON_MAXSTEP = 200
_VASP_NELM = 100


class FailureClass(StrEnum):
    SCF_NOT_CONVERGED = "scf_not_converged"
    IONIC_STEP_LIMIT = "ionic_step_limit"
    WALLTIME = "walltime"
    LAUNCHER = "launcher"
    APPROVAL_OR_INPUT = "approval_or_input"
    NON_FINITE_OUTPUT = "non_finite_output"
    NO_RESULT = "no_result"
    UNKNOWN = "unknown"


class RecoveryRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_retries: int
    action: str


class RecoveryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str = RECOVERY_POLICY_VERSION
    rules: dict[FailureClass, RecoveryRule]

    def rule(self, failure: FailureClass) -> RecoveryRule:
        return self.rules.get(failure, RecoveryRule(max_retries=0, action="give up"))


DEFAULT_POLICY = RecoveryPolicy(
    rules={
        FailureClass.SCF_NOT_CONVERGED: RecoveryRule(
            max_retries=2,
            action="QE: halve mixing_beta, double electron_maxstep; VASP: double NELM, ALGO=All",
        ),
        FailureClass.IONIC_STEP_LIMIT: RecoveryRule(
            max_retries=2, action="restart from the last optimized structure"
        ),
        FailureClass.WALLTIME: RecoveryRule(max_retries=1, action="double timeout_sec"),
        FailureClass.NON_FINITE_OUTPUT: RecoveryRule(max_retries=1, action="retry unchanged"),
        FailureClass.NO_RESULT: RecoveryRule(max_retries=1, action="retry unchanged"),
    }
)

_LAUNCHER_ERRORS = {"FileNotFoundError", "NotADirectoryError"}
_INPUT_ERRORS = {"PermissionError", "ValueError", "ValidationError", "KeyError", "TypeError"}


def classify_failure(outcome: dict[str, Any], *, max_steps: int | None = None) -> FailureClass:
    """The failure class of a relaxation outcome that is failed or not converged."""
    error = outcome.get("failure_type")
    reason = str(outcome.get("failure_reason") or "").lower()
    if outcome.get("return_code") == WALLTIME_RETURN_CODE or "timed out" in reason:
        return FailureClass.WALLTIME
    if error in _LAUNCHER_ERRORS or "not found" in reason or "no such file" in reason:
        return FailureClass.LAUNCHER
    if error in _INPUT_ERRORS:
        return FailureClass.APPROVAL_OR_INPUT
    if outcome.get("scf_converged") is False:
        return FailureClass.SCF_NOT_CONVERGED
    if outcome["status"] != "done":
        if "non-finite" in reason:
            return FailureClass.NON_FINITE_OUTPUT
        if "no energy" in reason or "no stage result" in reason or "did not converge" in reason:
            return FailureClass.NO_RESULT
        return FailureClass.UNKNOWN
    steps = outcome.get("steps")
    if max_steps is not None and steps is not None and steps >= max_steps:
        return FailureClass.IONIC_STEP_LIMIT
    return FailureClass.UNKNOWN


def adjusted_settings(
    failure: FailureClass, backend: str, settings: dict[str, Any], timeout_sec: float | None
) -> tuple[dict[str, Any], float | None]:
    """The DFT settings and timeout for the next attempt after ``failure``."""
    settings = dict(settings)
    if failure == FailureClass.SCF_NOT_CONVERGED:
        if backend == "vasp":
            settings["nelm"] = 2 * int(settings.get("nelm", _VASP_NELM))
            settings["algo"] = "All"
        else:
            electrons = dict(settings.get("extra_electrons", {}))
            settings["mixing_beta"] = 0.5 * float(settings.get("mixing_beta", _QE_MIXING_BETA))
            electrons["electron_maxstep"] = 2 * int(
                electrons.get("electron_maxstep", _QE_ELECTRON_MAXSTEP)
            )
            settings["extra_electrons"] = electrons
    elif failure == FailureClass.WALLTIME and timeout_sec is not None:
        timeout_sec = 2 * timeout_sec
    return settings, timeout_sec


def can_retry(failure: FailureClass, *, timeout_sec: float | None) -> bool:
    """Whether the policy has a remedy for ``failure`` at all."""
    if failure == FailureClass.WALLTIME:
        return timeout_sec is not None
    return DEFAULT_POLICY.rule(failure).max_retries > 0


__all__ = [
    "DEFAULT_POLICY",
    "RECOVERY_POLICY_VERSION",
    "FailureClass",
    "RecoveryPolicy",
    "RecoveryRule",
    "adjusted_settings",
    "can_retry",
    "classify_failure",
]
