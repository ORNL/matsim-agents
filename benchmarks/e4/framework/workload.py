"""Deterministic E4 workload manifests.

A workload is a list of task specifications derived only from a JSON config and
its seed. Strong scaling uses one task pool for every node count; weak scaling
uses ``tasks_per_node * nodes`` tasks whose first entries are identical across
node counts. The manifest hash covers every field that affects execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TASK_KINDS = ("controlled", "dft", "mlip-relax")
DURATION_MODELS = ("fixed", "uniform", "lognormal")
FAILURE_MODES = ("exit", "nonfinite", "no-result")


@dataclass(frozen=True)
class TaskSpec:
    """One logical task; ``task_id`` is unique and stable across runs."""

    task_id: str
    kind: str
    task_class: str
    pool: str
    seconds: float = 0.0
    artifact_bytes: int = 0
    seed: int = 0
    fail_attempts: int = 0
    failure_mode: str = "exit"
    structure_path: str | None = None
    structure_sha256: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unit(seed: int, *parts: object) -> float:
    """Deterministic value in [0, 1) from a seed and labels."""

    digest = hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _duration(spec: Mapping[str, Any], seed: int, index: int, task_class: str) -> float:
    model = spec.get("model", "fixed")
    if model not in DURATION_MODELS:
        raise ValueError(f"unknown duration model {model!r}; expected {DURATION_MODELS}")
    if model == "fixed":
        value = float(spec["seconds"])
    elif model == "uniform":
        low, high = float(spec["min_seconds"]), float(spec["max_seconds"])
        if not 0 <= low <= high:
            raise ValueError("uniform durations need 0 <= min_seconds <= max_seconds")
        value = low + (high - low) * _unit(seed, "duration", task_class, index)
    else:
        median, sigma = float(spec["median_seconds"]), float(spec["sigma"])
        if median <= 0 or sigma < 0:
            raise ValueError("lognormal durations need median_seconds > 0 and sigma >= 0")
        u1 = max(_unit(seed, "duration-a", task_class, index), 1e-12)
        u2 = _unit(seed, "duration-b", task_class, index)
        normal = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        value = median * math.exp(sigma * normal)
        if "max_seconds" in spec:
            value = min(value, float(spec["max_seconds"]))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"duration must be finite and nonnegative, got {value}")
    return round(value, 6)


def _structures(task_class_spec: Mapping[str, Any], root: Path) -> list[tuple[str, str]]:
    paths = task_class_spec.get("structures") or []
    if not paths:
        raise ValueError(f"task class {task_class_spec.get('name')!r} needs structures")
    resolved = []
    for raw in paths:
        path = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"structure not found: {path}")
        _validate_chemistry(path)
        resolved.append((str(path), sha256_file(path)))
    return resolved


_ZN_FORMATE_SPECIES = {"Zn", "C", "H", "O"}


def _validate_chemistry(path: Path) -> None:
    """Reject a Zn-formate seed that is not an intact Zn(HCOO)2 framework."""

    from ase.io import read

    species = set(read(path).get_chemical_symbols())
    if "Zn" in species and species <= _ZN_FORMATE_SPECIES:
        from benchmarks.e4.aurora_dft.six_class_inputs import validate_zn_formate

        try:
            validate_zn_formate(read(path))
        except ValueError as exc:
            raise ValueError(f"{path}: not a valid Zn(HCOO)2 structure: {exc}") from exc


def _class_tasks(
    spec: Mapping[str, Any], count: int, seed: int, root: Path, offset: int = 0
) -> list[TaskSpec]:
    name = spec["name"]
    kind = spec.get("kind", "controlled")
    if kind not in TASK_KINDS:
        raise ValueError(f"task class {name!r} has unknown kind {kind!r}")
    pool = spec.get("pool", "dft" if kind in {"controlled", "dft"} else "mlip")
    faults = spec.get("faults") or {}
    fraction = float(faults.get("fraction", 0.0))
    if not 0 <= fraction <= 1:
        raise ValueError("faults.fraction must be in [0, 1]")
    failure_mode = faults.get("mode", "exit")
    if failure_mode not in FAILURE_MODES:
        raise ValueError(f"unknown failure mode {failure_mode!r}")
    structures = _structures(spec, root) if kind in {"dft", "mlip-relax"} else []
    tasks = []
    for local in range(count):
        index = offset + local
        failing = fraction > 0 and _unit(seed, "fault", name, index) < fraction
        structure = structures[index % len(structures)] if structures else (None, None)
        tasks.append(
            TaskSpec(
                task_id=f"{name}-{index:06d}",
                kind=kind,
                task_class=name,
                pool=pool,
                seconds=_duration(spec.get("duration", {"seconds": 0}), seed, index, name)
                if kind == "controlled"
                else 0.0,
                artifact_bytes=int(spec.get("artifact_bytes", 0)),
                seed=seed,
                fail_attempts=int(faults.get("attempts", 1)) if failing else 0,
                failure_mode=failure_mode,
                structure_path=structure[0],
                structure_sha256=structure[1],
                params=dict(spec.get("params", {})),
            )
        )
    return tasks


@dataclass(frozen=True)
class Workload:
    workload_id: str
    experiment: str
    nodes: int
    tasks: tuple[TaskSpec, ...]
    config_sha256: str

    @property
    def workload_hash(self) -> str:
        return sha256_text(canonical_json([task.to_json() for task in self.tasks]))

    @property
    def planned_tasks(self) -> int:
        return len(self.tasks)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "workload_id": self.workload_id,
            "experiment": self.experiment,
            "nodes": self.nodes,
            "planned_tasks": self.planned_tasks,
            "workload_hash": self.workload_hash,
            "config_sha256": self.config_sha256,
            "tasks": [task.to_json() for task in self.tasks],
        }


def task_counts(config: Mapping[str, Any], experiment: str, nodes: int) -> dict[str, int]:
    """Tasks per class: fixed for strong scaling, proportional to nodes for weak scaling."""

    classes = config["task_classes"]
    if experiment == "strong":
        counts = {spec["name"]: int(spec["strong_tasks"]) for spec in classes}
    elif experiment == "weak":
        counts = {spec["name"]: int(spec["tasks_per_node"]) * nodes for spec in classes}
    elif experiment == "fixed":
        counts = {spec["name"]: int(spec["tasks"]) for spec in classes}
    else:
        raise ValueError(f"unknown experiment {experiment!r}")
    if any(count < 0 for count in counts.values()) or not sum(counts.values()):
        raise ValueError("a workload needs a positive number of tasks")
    return counts


def build_workload(
    config: Mapping[str, Any],
    *,
    experiment: str,
    nodes: int,
    root: str | Path | None = None,
    classes: Sequence[str] | None = None,
) -> Workload:
    """Build the deterministic workload for one experiment and node count."""

    if nodes < 1:
        raise ValueError("nodes must be positive")
    names = [spec["name"] for spec in config["task_classes"]]
    if len(names) != len(set(names)):
        raise ValueError("task class names must be unique")
    base = Path(root) if root is not None else Path.cwd()
    seed = int(config.get("seed", 0))
    counts = task_counts(config, experiment, nodes)
    tasks: list[TaskSpec] = []
    for spec in config["task_classes"]:
        if classes is not None and spec["name"] not in classes:
            continue
        tasks.extend(_class_tasks(spec, counts[spec["name"]], seed, base))
    if not tasks:
        raise ValueError("no task classes selected")
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("task IDs are not unique")
    return Workload(
        workload_id=str(config["workload_id"]),
        experiment=experiment,
        nodes=nodes,
        tasks=tuple(tasks),
        config_sha256=sha256_text(canonical_json(config)),
    )


def _expand(value: Any, path: str | Path) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item, path) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, path) for item in value]
    if isinstance(value, str) and "$" in value:
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ValueError(f"{path}: unset environment variable in {value!r}")
        return expanded
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config; ``${VAR}`` strings are expanded so configs carry no site paths."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return _expand(value, path)


__all__ = [
    "TaskSpec",
    "Workload",
    "build_workload",
    "canonical_json",
    "load_config",
    "sha256_file",
    "sha256_text",
    "task_counts",
]
