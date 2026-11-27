"""Staged ``neb.x`` runs: no-CI first, climbing image later, explicit restarts.

A stage history lives in ``<workdir>/neb_stages.json`` (schema ``matsim-neb-stages/1``). Every
stage has its own input (``neb.stage<N>.in``) and log (``neb.stage<N>.out``); nothing from an
earlier stage is overwritten or deleted. Before a restart stage is prepared, the previous
stage's ``prefix.path/.dat/.int/.crd/.xyz/.axsf`` files are copied (hash-verified) into
``history/stage<N>/``, because ``neb.x`` rewrites those files in place.

Stage kinds:

``initial``   ``restart_mode='from_scratch'``, ``ci_scheme='no-CI'``.
``continue``  explicit resume with the same CI settings and a larger ``nstep_path``; allowed only
              after an iteration limit, an incomplete output or a process failure.
``ci``        ``restart_mode='restart'`` with ``auto`` or ``manual`` climbing images; allowed
              only when the previous stage converged, or hit the iteration limit with every
              interior error at most ``ci_entry_threshold_ev_ang`` and an interior highest image.
              The threshold is a workflow setting, not a QE default.

Restart facts from the QE sources: a ``restart`` run reads ``<prefix>.path`` from the working
directory and, when it is missing, silently starts from scratch; that is why the file's presence
and unchanged hash are verified here. A run whose ``nstep_path`` is not above the iterations
already done stops at once; that is rejected too.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.backends.dft.process import run_launcher
from matsim_agents.backends.dft.qe_neb import QENebConfig
from matsim_agents.backends.dft.qe_neb_input import write_neb_input
from matsim_agents.backends.dft.qe_neb_parse import parse_neb_output
from matsim_agents.backends.dft.qe_neb_run import NebRunRecord, run_neb
from matsim_agents.core.fileio import atomic_write_text
from matsim_agents.core.hashing import sha256_file as _sha256

SCHEMA = "matsim-neb-stages/1"
HISTORY_FILE = "neb_stages.json"
DEFAULT_CI_ENTRY_THRESHOLD = 0.1  # eV/Angstrom; workflow choice, not a QE default
SNAPSHOT_SUFFIXES = (".path", ".dat", ".int", ".crd", ".xyz", ".axsf")

StageKind = Literal["initial", "continue", "ci"]
StageStatus = Literal["prepared", "run"]

# Settings a restart may change; every other config field must equal the first stage's.
_RESTART_MAY_CHANGE = frozenset(
    {
        "ci_scheme",
        "climbing_images",
        "restart_mode",
        "nstep_path",
        "path_thr_ev_ang",
        "launcher",
        "nimage",
        "npool",
        "extra_args",
        "neb_executable",
    }
)
_CONTINUABLE = ("iteration_limit", "incomplete_output", "process_failure")


class StageError(ValueError):
    """A stage cannot be prepared or run; the message says why."""


class StageEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=1)
    kind: StageKind
    status: StageStatus
    created_at: str
    config: dict[str, Any]
    config_hash: str
    images_digest: str
    input_path: str
    input_sha256: str
    log_name: str
    snapshot_dir: str | None = None
    snapshot_sha256: dict[str, str] = Field(default_factory=dict)
    path_sha256_before: str | None = None
    # filled after the run
    run_status: str | None = None
    exit_code: int | None = None
    outcome: str | None = None
    iterations_completed: int | None = None
    max_interior_error_eV_per_A: float | None = None
    forward_barrier_eV: float | None = None
    reverse_barrier_eV: float | None = None
    climbing_images: tuple[int, ...] = ()
    path_sha256_after: str | None = None
    record_path: str | None = None
    warnings: tuple[str, ...] = ()


class StageHistory(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = Field(default=SCHEMA, alias="schema")
    stages: tuple[StageEntry, ...] = ()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def images_digest(images: Sequence[Atoms]) -> str:
    """Hash of species, cell and positions of every image (repr floats, no rounding)."""
    digest = hashlib.sha256()
    for image in images:
        digest.update(repr(image.get_chemical_symbols()).encode())
        digest.update(repr(image.cell.array.tolist()).encode())
        digest.update(repr(image.positions.tolist()).encode())
    return digest.hexdigest()


def history_path(workdir: str | os.PathLike) -> Path:
    return Path(workdir) / HISTORY_FILE


def load_history(workdir: str | os.PathLike) -> StageHistory:
    """The stage history, or an empty one; a corrupt file raises ``StageError``."""
    path = history_path(workdir)
    if not path.exists():
        return StageHistory()
    try:
        data = json.loads(path.read_text())
        if data.get("schema") != SCHEMA:
            raise StageError(f"{path} has schema {data.get('schema')!r}, expected {SCHEMA!r}")
        return StageHistory.model_validate(data)
    except (ValueError, TypeError) as exc:
        if isinstance(exc, StageError):
            raise
        raise StageError(f"{path} is not a valid stage history: {exc}") from exc


def _save_history(workdir: str | os.PathLike, history: StageHistory) -> None:
    text = json.dumps(history.model_dump(mode="json", by_alias=True), indent=2) + "\n"
    atomic_write_text(history_path(workdir), text)


def derive_stage_config(
    base: QENebConfig,
    kind: StageKind,
    *,
    nstep_path: int | None = None,
    path_thr_ev_ang: float | None = None,
    ci_scheme: Literal["auto", "manual"] = "auto",
    climbing_images: tuple[int, ...] = (),
) -> QENebConfig:
    """A validated config for a later stage of the same calculation."""
    data = base.model_dump(mode="json")
    if kind == "initial":
        data.update(restart_mode="from_scratch", ci_scheme="no-CI", climbing_images=())
    elif kind == "continue":
        data.update(restart_mode="restart")
    else:
        data.update(restart_mode="restart", ci_scheme=ci_scheme, climbing_images=climbing_images)
    if nstep_path is not None:
        data["nstep_path"] = nstep_path
    if path_thr_ev_ang is not None:
        data["path_thr_ev_ang"] = path_thr_ev_ang
    return QENebConfig.model_validate(data)


def check_restart_compatible(first: Mapping[str, Any], later: QENebConfig) -> None:
    """Reject a stage whose physics, structure counts or file layout differ from stage 1."""
    new = later.model_dump(mode="json")
    differing = sorted(
        key for key in new if key not in _RESTART_MAY_CHANGE and new[key] != first.get(key)
    )
    if differing:
        raise StageError(f"restart config differs from the first stage in: {differing}")


def ci_entry_problems(
    entry: StageEntry, *, threshold: float = DEFAULT_CI_ENTRY_THRESHOLD
) -> list[str]:
    """Why the previous stage is not converged enough to switch on a climbing image."""
    if entry.status != "run":
        return ["the previous stage has not been run"]
    if entry.outcome == "converged":
        pass
    elif entry.outcome == "iteration_limit":
        error = entry.max_interior_error_eV_per_A
        if error is None or error > threshold:
            return [f"interior error {error} exceeds the CI entry threshold {threshold}"]
    else:
        return [f"previous stage outcome is {entry.outcome!r}"]
    if entry.forward_barrier_eV is None:
        return ["the highest sampled image is an endpoint; there is no interior maximum to climb"]
    return []


def _snapshot(workdir: Path, prefix: str, source: StageEntry) -> tuple[str, dict[str, str]]:
    """Copy the previous stage's output files; verify hashes; never replace a snapshot."""
    target = workdir / "history" / f"stage{source.index}"
    wanted = [workdir / f"{prefix}{suffix}" for suffix in SNAPSHOT_SUFFIXES]
    wanted += [workdir / source.log_name, workdir / (source.log_name + ".record.json")]
    sources = [p for p in wanted if p.is_file()]
    hashes = {p.name: _sha256(p) for p in sources}
    if target.exists():
        for name, value in hashes.items():
            copy = target / name
            if not copy.is_file() or _sha256(copy) != value:
                raise StageError(f"{target} already exists with different content")
        return str(target), hashes
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".stage{source.index}-", dir=target.parent))
    try:
        for path in sources:
            shutil.copy2(path, staging / path.name)
            if _sha256(staging / path.name) != hashes[path.name]:
                raise StageError(f"copy of {path.name} does not match the original")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return str(target), hashes


def prepare_stage(
    config: QENebConfig,
    images: Sequence[Atoms],
    *,
    kind: StageKind,
    ci_entry_threshold_ev_ang: float = DEFAULT_CI_ENTRY_THRESHOLD,
) -> StageEntry:
    """Write the next stage's input, snapshot earlier output and record the stage."""
    workdir = Path(config.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    history = load_history(workdir)
    digest = images_digest(images)
    index = len(history.stages) + 1
    previous = history.stages[-1] if history.stages else None
    snapshot_dir: str | None = None
    snapshot: dict[str, str] = {}
    path_before: str | None = None

    if kind == "initial":
        if history.stages:
            raise StageError("an initial stage already exists; resume explicitly instead")
        if config.restart_mode != "from_scratch" or config.ci_scheme != "no-CI":
            raise StageError("the initial stage must be from_scratch with ci_scheme 'no-CI'")
        existing = [
            p.name for s in SNAPSHOT_SUFFIXES if (p := workdir / f"{config.prefix}{s}").exists()
        ]
        if existing:
            raise StageError(f"earlier NEB output would be overwritten: {existing}")
    else:
        if previous is None:
            raise StageError(f"no earlier stage to {kind}; prepare the initial stage first")
        if previous.status != "run":
            raise StageError("the previous stage has not been run")
        if config.restart_mode != "restart":
            raise StageError(f"a {kind} stage needs restart_mode 'restart'")
        check_restart_compatible(history.stages[0].config, config)
        if digest != history.stages[0].images_digest:
            raise StageError("the images differ from those of the first stage")
        if previous.iterations_completed is not None and config.nstep_path <= (
            previous.iterations_completed
        ):
            raise StageError(
                f"nstep_path {config.nstep_path} is not above the {previous.iterations_completed} "
                "iterations already done; neb.x would stop at once"
            )
        path_file = workdir / f"{config.prefix}.path"
        if not path_file.is_file():
            raise StageError(
                f"{path_file.name} is missing; neb.x would silently restart from scratch"
            )
        path_before = _sha256(path_file)
        if previous.path_sha256_after is not None and path_before != previous.path_sha256_after:
            raise StageError(f"{path_file.name} changed since the previous stage finished")
        if kind == "ci":
            if previous.config["ci_scheme"] != "no-CI" or config.ci_scheme == "no-CI":
                raise StageError("a ci stage goes from no-CI to auto or manual climbing images")
            problems = ci_entry_problems(previous, threshold=ci_entry_threshold_ev_ang)
            if problems:
                raise StageError("not ready for climbing images: " + "; ".join(problems))
        else:
            if previous.outcome not in _CONTINUABLE:
                raise StageError(f"cannot continue after outcome {previous.outcome!r}")
            if (config.ci_scheme, config.climbing_images) != (
                previous.config["ci_scheme"],
                tuple(previous.config["climbing_images"]),
            ):
                raise StageError("a continue stage must keep the CI settings of the previous one")
        snapshot_dir, snapshot = _snapshot(workdir, config.prefix, previous)

    input_path = workdir / f"neb.stage{index}.in"
    written = write_neb_input(config, images, input_path, overwrite=False)
    entry = StageEntry(
        index=index,
        kind=kind,
        status="prepared",
        created_at=_now(),
        config=config.model_dump(mode="json"),
        config_hash=config.config_hash(),
        images_digest=digest,
        input_path=str(input_path),
        input_sha256=written.sha256,
        log_name=f"neb.stage{index}.out",
        snapshot_dir=snapshot_dir,
        snapshot_sha256=snapshot,
        path_sha256_before=path_before,
    )
    _save_history(workdir, StageHistory(stages=(*history.stages, entry)))
    return entry


def run_stage(
    workdir: str | os.PathLike,
    *,
    timeout_sec: float | None = None,
    env: Mapping[str, str] | None = None,
    launcher=run_launcher,
) -> StageEntry:
    """Run the prepared stage, parse its output and record the outcome."""
    root = Path(workdir).resolve()
    history = load_history(root)
    if not history.stages or history.stages[-1].status != "prepared":
        raise StageError("there is no prepared stage to run")
    entry = history.stages[-1]
    input_path = Path(entry.input_path)
    if not input_path.is_file() or _sha256(input_path) != entry.input_sha256:
        raise StageError(f"{input_path.name} is missing or changed since it was prepared")
    config = QENebConfig.model_validate(entry.config)
    record: NebRunRecord = run_neb(
        config,
        input_path,
        timeout_sec=timeout_sec,
        env=env,
        stdout_name=entry.log_name,
        launcher=launcher,
    )
    parse = parse_neb_output(
        Path(record.stdout_path),
        record=record,
        expected_images=config.n_images,
        path_thr_ev_ang=config.path_thr_ev_ang,
        dat_path=root / f"{config.prefix}.dat",
    )
    path_file = root / f"{config.prefix}.path"
    done = entry.model_copy(
        update={
            "status": "run",
            "run_status": record.status,
            "exit_code": record.exit_code,
            "outcome": parse.outcome,
            "iterations_completed": parse.iterations_completed,
            "max_interior_error_eV_per_A": parse.max_interior_error_eV_per_A,
            "forward_barrier_eV": parse.forward_barrier_eV,
            "reverse_barrier_eV": parse.reverse_barrier_eV,
            "climbing_images": parse.climbing_images,
            "path_sha256_after": _sha256(path_file) if path_file.is_file() else None,
            "record_path": record.stdout_path + ".record.json",
            "warnings": parse.warnings,
        }
    )
    _save_history(root, StageHistory(stages=(*history.stages[:-1], done)))
    return done


__all__ = [
    "DEFAULT_CI_ENTRY_THRESHOLD",
    "HISTORY_FILE",
    "StageEntry",
    "StageError",
    "StageHistory",
    "check_restart_compatible",
    "ci_entry_problems",
    "derive_stage_config",
    "history_path",
    "images_digest",
    "load_history",
    "prepare_stage",
    "run_stage",
]
