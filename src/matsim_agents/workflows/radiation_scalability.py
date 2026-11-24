"""Toy-scale cost measurements, seeded reproducibility and an environment record (E5).

Everything here is CPU-only and tiny. The scaling workload is the synthetic KMC chain from
``radiation_kinetics_qualification``, sized in *states*, not atoms, so the result says how
this kinetic engine's cost grows and nothing about DFT, MLIP or large-cell cost. Timing and
memory are measured, never assumed; they vary run to run, so no pass/fail threshold is set on
them. ``supports`` therefore stays empty for scaling: a human decides what the numbers allow.

Reproducibility is stricter: the same seed must give a byte-identical canonical JSON result,
and a different seed must not (a control that catches a check which compares nothing).

The environment record lists versions that were *observed*. ``pinned`` is true only when a
lock file exists; the repository currently has none, and the record says so.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from matsim_agents.kinetics.ensemble import run_ensemble
from matsim_agents.workflows.radiation_kinetics_qualification import (
    TEMPERATURE_K,
    chain_network,
)

LOCK_FILE_NAMES = ("uv.lock", "poetry.lock", "requirements.lock", "conda-lock.yml")
TRACKED_PACKAGES = ("numpy", "pydantic", "scipy", "torch", "ase")


@dataclass(frozen=True)
class ScalingPoint:
    n_states: int
    wall_s: float
    peak_bytes: int
    mean_events: float


@dataclass(frozen=True)
class ScalingReport:
    points: tuple[ScalingPoint, ...]
    time_exponent: float | None
    events_exponent: float | None
    supports: tuple[str, ...]
    caveats: tuple[str, ...]


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Least-squares slope of ``log y`` on ``log x``; None when it is undefined."""
    if len(xs) < 2 or any(v <= 0 or not math.isfinite(v) for v in (*xs, *ys)):
        return None
    lx, ly = [math.log(v) for v in xs], [math.log(v) for v in ys]
    mx, my = math.fsum(lx) / len(lx), math.fsum(ly) / len(ly)
    den = math.fsum((a - mx) ** 2 for a in lx)
    if den == 0:
        return None
    return math.fsum((a - mx) * (b - my) for a, b in zip(lx, ly, strict=True)) / den


def measure_scaling(sizes: Sequence[int], *, seed: int, n_trajectories: int = 20) -> ScalingReport:
    """Wall time, peak Python allocation and event count per chain size."""
    if len(sizes) < 3 or list(sizes) != sorted(set(sizes)):
        raise ValueError("sizes must be at least three distinct increasing values")
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be at least 1")
    points = []
    for n in sizes:
        net, start = chain_network(n)
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            res = run_ensemble(
                net,
                start,
                TEMPERATURE_K,
                n_trajectories=n_trajectories,
                seed=seed,
                max_events=10 * n,
                require="exploratory",
            )
            wall = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        points.append(ScalingPoint(n, wall, peak, res.mean_events))
    xs = [float(p.n_states) for p in points]
    return ScalingReport(
        points=tuple(points),
        time_exponent=_slope(xs, [p.wall_s for p in points]),
        events_exponent=_slope(xs, [p.mean_events for p in points]),
        supports=(),
        caveats=(
            "CPU-only toy chain sized in states, not atoms",
            "timing and memory are measured and vary between runs; no threshold is applied",
            "says nothing about DFT, MLIP or large-cell cost",
        ),
    )


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_encode)


def _encode(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"not serialisable: {type(value).__name__}")


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


@dataclass(frozen=True)
class ReproducibilityReport:
    identical: bool
    seed_sensitive: bool
    digests: tuple[str, ...]
    control_digest: str
    verdict: str
    supports: tuple[str, ...]


def check_reproducibility(
    run: Callable[[int], Any], *, seed: int, repeats: int = 2
) -> ReproducibilityReport:
    """``run(seed)`` repeated must hash identically; ``run(seed + 1)`` must differ."""
    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    digests = tuple(digest(run(seed)) for _ in range(repeats))
    control = digest(run(seed + 1))
    identical = len(set(digests)) == 1
    sensitive = control != digests[0]
    ok = identical and sensitive
    return ReproducibilityReport(
        identical,
        sensitive,
        digests,
        control,
        "pass" if ok else "fail",
        ("bitwise-seeded",) if ok else (),
    )


@dataclass(frozen=True)
class EnvironmentRecord:
    python: str
    platform: str
    packages: dict[str, str | None]
    lock_file: str | None
    pinned: bool
    supports: tuple[str, ...]
    caveats: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def environment_record(root: Path | str = ".") -> EnvironmentRecord:
    """Observed versions; ``pinned`` needs a lock file under ``root``."""
    base = Path(root)
    lock = next((n for n in LOCK_FILE_NAMES if (base / n).is_file()), None)
    packages: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    caveats = ["versions are observed, not resolved from a lock"]
    if lock is None:
        caveats.append("no lock file found; the environment is not pinned")
    return EnvironmentRecord(
        python=sys.version.split()[0],
        platform=platform.platform(),
        packages=packages,
        lock_file=lock,
        pinned=lock is not None,
        supports=("environment-pinned",) if lock is not None else (),
        caveats=tuple(caveats),
    )


__all__ = [
    "EnvironmentRecord",
    "ReproducibilityReport",
    "ScalingPoint",
    "ScalingReport",
    "canonical_json",
    "check_reproducibility",
    "digest",
    "environment_record",
    "measure_scaling",
]
