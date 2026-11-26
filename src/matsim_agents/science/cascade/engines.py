"""Engine interface for cascades and a size-convergence helper (WP1.10).

The in-repo integrator is the only cascade engine that runs here. ``LammpsEngine`` only checks
that LAMMPS is present and fails clearly when it is not; it never emulates LAMMPS. Running
LAMMPS goes through ``backends.external.lammps``; no LAMMPS cascade script exists yet.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class EngineUnavailableError(RuntimeError):
    pass


class CascadeEngine(Protocol):
    name: str

    def check_available(self) -> None: ...


class InRepoEngine:
    name = "matsim-agents-integrator"

    def check_available(self) -> None:
        return None


class LammpsEngine:
    name = "lammps"

    def check_available(self) -> None:
        explicit = os.environ.get("LAMMPS_EXECUTABLE")
        if importlib.util.find_spec("lammps") is None and not (
            (explicit and shutil.which(explicit))
            or shutil.which("lmp")
            or shutil.which("lmp_serial")
            or shutil.which("lmp_mpi")
        ):
            raise EngineUnavailableError(
                "LAMMPS is not installed (no python module and no lmp binary); "
                "cascade runs with this engine are not available in this environment"
            )


@dataclass(frozen=True)
class SizeConvergence:
    sizes: tuple[int, ...]
    values: tuple[float, ...]
    converged: bool
    relative_change: float | None
    tolerance: float
    statement: str


def size_convergence(
    values_by_size: dict[int, list[float]], *, tolerance: float, min_realizations: int = 3
) -> SizeConvergence:
    """Compare the mean of an observable between the two largest cells.

    Converged only if each of those two cells has ``min_realizations`` realizations and the
    relative change of the means is within ``tolerance``. Fewer than two sizes is never converged.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    sizes = tuple(sorted(values_by_size))
    means = tuple(float(np.mean(values_by_size[n])) for n in sizes)
    if len(sizes) < 2:
        return SizeConvergence(sizes, means, False, None, tolerance, "need at least two cell sizes")
    big, prev = sizes[-1], sizes[-2]
    short = [n for n in (big, prev) if len(values_by_size[n]) < min_realizations]
    if short:
        return SizeConvergence(
            sizes,
            means,
            False,
            None,
            tolerance,
            f"cells {short} have fewer than {min_realizations} realizations",
        )
    ref = means[-1]
    if ref == 0.0:
        return SizeConvergence(sizes, means, False, None, tolerance, "largest-cell mean is zero")
    rel = abs(means[-1] - means[-2]) / abs(ref)
    ok = rel <= tolerance
    return SizeConvergence(
        sizes,
        means,
        ok,
        float(rel),
        tolerance,
        f"relative change {rel:.3g} between {prev} and {big} atoms "
        + ("within" if ok else "exceeds")
        + f" tolerance {tolerance:g}",
    )
