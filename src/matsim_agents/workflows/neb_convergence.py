"""Convergence checks that gate DFT/NEB qualification.

Pure functions over numbers a caller already has. They run no DFT and promote nothing: a
passing check is an input to ``qualification_matrix``, which still needs a hashed artifact.

Rules:

* a convergence series needs at least ``MIN_POINTS`` settings and a strictly increasing
  parameter; a barrier that is missing or non-finite makes the series ``invalid``;
* a series is ``converged`` only if the last step changes the barrier by less than the
  tolerance and the step before it did too; one small step can be a coincidence;
* a barrier that is still moving by more than the tolerance at the end is ``not_converged``;
* endpoint relaxation, the climbing image and the saddle mode are checked separately and all
  must hold.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

MIN_POINTS = 3
DEFAULT_TOLERANCE_EV = 0.02
DEFAULT_FORCE_THRESHOLD_EV_PER_A = 0.05
SeriesVerdict = Literal["converged", "not_converged", "invalid"]


@dataclass(frozen=True)
class SeriesCheck:
    parameter: str
    verdict: SeriesVerdict
    last_change_eV: float | None
    previous_change_eV: float | None
    tolerance_eV: float
    reasons: tuple[str, ...] = ()


def check_barrier_series(
    parameter: str,
    settings: Sequence[float],
    barriers_eV: Sequence[float | None],
    *,
    tolerance_eV: float = DEFAULT_TOLERANCE_EV,
) -> SeriesCheck:
    """Whether the barrier has stopped changing as ``parameter`` increases."""

    def invalid(reason: str) -> SeriesCheck:
        return SeriesCheck(parameter, "invalid", None, None, tolerance_eV, (reason,))

    if tolerance_eV <= 0 or not math.isfinite(tolerance_eV):
        raise ValueError("tolerance_eV must be positive and finite")
    if len(settings) != len(barriers_eV):
        return invalid("settings and barriers differ in length")
    if len(settings) < MIN_POINTS:
        return invalid(f"{len(settings)} points, need {MIN_POINTS}")
    if any(b is None or not math.isfinite(b) for b in barriers_eV):
        return invalid("a barrier is missing or not finite; missing is not zero")
    if any(not math.isfinite(s) for s in settings) or any(
        later <= earlier for earlier, later in zip(settings, settings[1:], strict=False)
    ):
        return invalid("settings must be finite and strictly increasing")
    values = [float(b) for b in barriers_eV if b is not None]
    last = abs(values[-1] - values[-2])
    previous = abs(values[-2] - values[-3])
    ok = last < tolerance_eV and previous < tolerance_eV
    reasons = () if ok else (f"{parameter}: last changes {previous:.4f}, {last:.4f} eV",)
    return SeriesCheck(
        parameter, "converged" if ok else "not_converged", last, previous, tolerance_eV, reasons
    )


@dataclass(frozen=True)
class NebInputs:
    """What a finished NEB run reports, as plain numbers."""

    endpoint_max_force_eV_per_A: tuple[float, float]
    interior_max_error_eV_per_A: float | None
    climbing_image_used: bool
    n_imaginary_modes_at_saddle: int | None
    n_images: int


@dataclass(frozen=True)
class NebQualificationCheck:
    passed: bool
    failures: tuple[str, ...] = field(default_factory=tuple)
    endpoints_ok: bool = False
    climbing_image_ok: bool = False


def check_neb_run(
    run: NebInputs,
    *,
    force_threshold_eV_per_A: float = DEFAULT_FORCE_THRESHOLD_EV_PER_A,
    min_images: int = 5,
) -> NebQualificationCheck:
    """Endpoint relaxation, path convergence, climbing image and saddle mode."""
    failures: list[str] = []
    for label, force in zip(("initial", "final"), run.endpoint_max_force_eV_per_A, strict=True):
        if not math.isfinite(force) or force > force_threshold_eV_per_A:
            failures.append(f"{label} endpoint not relaxed (max force {force} eV/A)")
    endpoints_ok = not failures
    err = run.interior_max_error_eV_per_A
    if err is None or not math.isfinite(err):
        failures.append("path error missing; missing is not zero")
    elif err > force_threshold_eV_per_A:
        failures.append(f"path not converged (max error {err} eV/A)")
    climbing_ok = run.climbing_image_used and err is not None and math.isfinite(err)
    if not run.climbing_image_used:
        failures.append("no climbing image")
    if run.n_imaginary_modes_at_saddle is None:
        failures.append("saddle mode not assessed")
    elif run.n_imaginary_modes_at_saddle != 1:
        failures.append(f"saddle has {run.n_imaginary_modes_at_saddle} imaginary modes, need 1")
    if run.n_images < min_images:
        failures.append(f"{run.n_images} images, need {min_images}")
    climbing_ok = climbing_ok and run.n_imaginary_modes_at_saddle == 1
    return NebQualificationCheck(not failures, tuple(failures), endpoints_ok, climbing_ok)


def qualifying_criteria(
    cutoff: SeriesCheck, kpoints: SeriesCheck, run: NebQualificationCheck
) -> dict[str, bool]:
    """The ``dft-neb`` matrix criteria these checks support. Evidence is still required."""
    return {
        "relaxed-endpoints": run.endpoints_ok,
        "cutoff-convergence": cutoff.verdict == "converged",
        "kpoint-convergence": kpoints.verdict == "converged",
        "climbing-image": run.climbing_image_ok,
    }


__all__ = [
    "DEFAULT_FORCE_THRESHOLD_EV_PER_A",
    "DEFAULT_TOLERANCE_EV",
    "MIN_POINTS",
    "NebInputs",
    "NebQualificationCheck",
    "SeriesCheck",
    "check_barrier_series",
    "check_neb_run",
    "qualifying_criteria",
]
