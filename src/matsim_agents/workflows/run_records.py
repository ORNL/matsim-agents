"""Run records shared by every way of starting work: config files, objectives and resumes."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.core.contracts import EvidenceLevel, ProvenanceRecord
from matsim_agents.core.hashing import sha256_file
from matsim_agents.execution.run_directory import ScientificRunDirectory

RECORD_FILE = "run-record.json"
RunState = Literal["running", "complete", "partial", "blocked", "failed", "interrupted"]
DEFAULT_RUN_ROOTS = ("runs", "outputs")


class RunRecord(BaseModel):
    """What was asked, what happened, and where the native evidence lives."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    kind: str
    source: str | None = None
    source_sha256: str | None = None
    objective: str | None = None
    state: RunState = "running"
    native_run_directory: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    resumable: bool = False


def start_run(
    root: str | Path,
    *,
    kind: str,
    source: Path | None = None,
    objective: str | None = None,
    resolved_config: dict[str, Any] | None = None,
) -> tuple[ScientificRunDirectory, RunRecord]:
    """Create the run directory and write the initial ``running`` record."""
    run = ScientificRunDirectory.create(
        root,
        workflow=f"run:{kind}",
        request={"kind": kind, "source": str(source) if source else None, "objective": objective},
        resolved_config=resolved_config or {},
        # The record claims no scientific evidence; native runs carry their own levels.
        provenance=ProvenanceRecord(
            workflow=f"run:{kind}", evidence_level=EvidenceLevel.HYPOTHESIS
        ),
        prefix=kind,
    )
    record = RunRecord(
        run_id=run.run_id,
        kind=kind,
        source=str(source) if source else None,
        source_sha256=sha256_file(source) if source else None,
        objective=objective,
    )
    write_record(run, record)
    return run, record


def write_record(run: ScientificRunDirectory, record: RunRecord) -> None:
    run.write_json(RECORD_FILE, record)
    lines = [
        f"# {record.kind} run {record.run_id}",
        "",
        f"- State: **{record.state}**",
        f"- Source: `{record.source or record.objective}`",
        f"- Native run directory: `{record.native_run_directory or 'none'}`",
        "",
        "A finished run is not a scientific conclusion: read the native evidence and its",
        "convergence criteria before claiming a result.",
    ]
    if record.blockers:
        lines += ["", "## Blocked by", "", *[f"- {item}" for item in record.blockers]]
    if record.failure_reason:
        lines += ["", "## Failure", "", record.failure_reason]
    run.write_text("report.md", "\n".join(lines) + "\n")


def load_record(directory: str | Path) -> RunRecord:
    path = Path(directory) / RECORD_FILE
    if not path.is_file():
        raise ValueError(f"{directory} is not a matsim-agents run (no {RECORD_FILE})")
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))


def find_run(reference: str | Path, roots: Iterable[str | Path] | None = None) -> Path:
    """Resolve a run directory from its path or its run ID under the run roots."""
    candidate = Path(reference)
    if candidate.is_dir() and (candidate / RECORD_FILE).is_file():
        return candidate.resolve()
    if os.sep in str(reference):
        raise ValueError(f"{reference} is not a matsim-agents run directory")
    searched = [Path(root) for root in (roots or DEFAULT_RUN_ROOTS)]
    for root in searched:
        found = root / str(reference)
        if (found / RECORD_FILE).is_file():
            return found.resolve()
    where = ", ".join(str(root) for root in searched)
    raise ValueError(f"No run {reference!r} under {where}; pass --runs-dir or the run directory")


def list_runs(roots: Iterable[str | Path]) -> list[tuple[Path, RunRecord]]:
    found = []
    for root in map(Path, roots):
        if root.is_dir():
            for record_path in sorted(root.glob(f"*/{RECORD_FILE}")):
                found.append((record_path.parent, load_record(record_path.parent)))
    return found


__all__ = [
    "DEFAULT_RUN_ROOTS",
    "RECORD_FILE",
    "RunRecord",
    "RunState",
    "find_run",
    "list_runs",
    "load_record",
    "start_run",
    "write_record",
]
