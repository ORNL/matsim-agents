"""Content-addressed candidate records and a resumable append-only manifest.

A candidate's identity is the hash of what defines it (structure, magnetic state, defect and
generation recipe), never of a run-specific name, so the same candidate generated twice is the
same record. The manifest is an append-only JSONL event log that survives interruption: it is
replayed on open, a torn final line is ignored, and status changes are events, not rewrites.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.core.fileio import append_line
from matsim_agents.science.magnetism.config import MagneticConfiguration

Status = Literal["generated", "queued", "running", "done", "failed", "skipped"]
TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "failed", "skipped"})
POSITION_DECIMALS = 5


def structure_hash(atoms: Atoms) -> str:
    """Hash independent of atom order and periodic wrapping: fractional positions are wrapped
    and rounded, then rows are sorted. Rigid translations of the whole cell are not removed."""
    cell = np.round(atoms.cell.array, 6)
    scaled = np.round(atoms.get_scaled_positions(wrap=True), POSITION_DECIMALS) % 1.0
    rows = sorted(
        (sym, *[f"{x:.{POSITION_DECIMALS}f}" for x in row])
        for sym, row in zip(atoms.get_chemical_symbols(), scaled, strict=True)
    )
    payload = json.dumps({"cell": cell.tolist(), "rows": rows}, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def candidate_identity(
    atoms: Atoms,
    *,
    magnetic: MagneticConfiguration | None = None,
    defect: Mapping[str, Any] | None = None,
    recipe: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "structure": structure_hash(atoms),
        "magnetic": magnetic.fingerprint() if magnetic is not None else None,
        "defect": dict(defect or {}),
        "recipe": dict(recipe or {}),
    }
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


class CandidateRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    structure_hash: str
    composition: str
    n_atoms: int = Field(gt=0)
    magnetic_fingerprint: str | None = None
    magnetic_mode: str | None = None
    defect: dict[str, Any] = Field(default_factory=dict)
    parent_ids: tuple[str, ...] = ()
    generation_method: str
    seed: int | None = None
    stratum: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_atoms(
        cls,
        atoms: Atoms,
        *,
        generation_method: str,
        magnetic: MagneticConfiguration | None = None,
        defect: Mapping[str, Any] | None = None,
        recipe: Mapping[str, Any] | None = None,
        parent_ids: Iterable[str] = (),
        seed: int | None = None,
        stratum: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CandidateRecord:
        return cls(
            candidate_id=candidate_identity(
                atoms,
                magnetic=magnetic,
                defect=defect,
                recipe={"method": generation_method, **dict(recipe or {})},
            ),
            structure_hash=structure_hash(atoms),
            composition=atoms.get_chemical_formula(),
            n_atoms=len(atoms),
            magnetic_fingerprint=magnetic.fingerprint() if magnetic is not None else None,
            magnetic_mode=magnetic.mode if magnetic is not None else None,
            defect=dict(defect or {}),
            parent_ids=tuple(parent_ids),
            generation_method=generation_method,
            seed=seed,
            stratum=stratum,
            metadata=dict(metadata or {}),
        )


class CandidateManifest:
    """Append-only, resumable store of candidates and their status history."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: dict[str, CandidateRecord] = {}
        self.status: dict[str, Status] = {}
        self.details: dict[str, dict[str, Any]] = {}
        self.torn_lines = 0
        if self.path.exists():
            self._replay()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _replay(self) -> None:
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                self._apply(event)
            except (ValueError, KeyError):
                self.torn_lines += 1

    def _apply(self, event: dict[str, Any]) -> None:
        if event["event"] == "add":
            record = CandidateRecord.model_validate(event["record"])
            if record.candidate_id not in self.records:
                self.records[record.candidate_id] = record
                self.status[record.candidate_id] = "generated"
        elif event["event"] == "status":
            cid = event["candidate_id"]
            if cid not in self.records:
                raise KeyError(cid)
            self.status[cid] = event["status"]
            if event.get("detail"):
                self.details.setdefault(cid, {}).update(event["detail"])
        else:
            raise KeyError(event["event"])

    def _append(self, event: dict[str, Any]) -> None:
        append_line(
            self.path, json.dumps(event, sort_keys=True, default=str, separators=(",", ":"))
        )

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, candidate_id: str) -> bool:
        return candidate_id in self.records

    def add(self, record: CandidateRecord) -> bool:
        """Register ``record``; ``False`` (and no write) when its identity already exists."""
        if record.candidate_id in self.records:
            return False
        event = {"event": "add", "record": record.model_dump(mode="json")}
        self._append(event)
        self._apply(event)
        return True

    def set_status(
        self, candidate_id: str, status: Status, detail: Mapping[str, Any] | None = None
    ) -> None:
        if candidate_id not in self.records:
            raise KeyError(f"unknown candidate {candidate_id}")
        if self.status[candidate_id] in TERMINAL_STATUSES and status not in TERMINAL_STATUSES:
            raise ValueError(
                f"candidate {candidate_id} is {self.status[candidate_id]}; "
                "terminal states cannot be reopened"
            )
        event = {
            "event": "status",
            "candidate_id": candidate_id,
            "status": status,
            "detail": dict(detail or {}),
        }
        self._append(event)
        self._apply(event)

    def pending(self) -> list[CandidateRecord]:
        """Candidates not in a terminal state, in insertion order (running ones included so
        an interrupted run re-queues them)."""
        return [r for cid, r in self.records.items() if self.status[cid] not in TERMINAL_STATUSES]

    def counts(self) -> dict[str, int]:
        return dict(Counter(self.status.values()))

    def recover_interrupted(self) -> list[str]:
        """Return ``running`` candidates to ``queued`` after a crash; returns their ids."""
        stale = [cid for cid, s in self.status.items() if s == "running"]
        for cid in stale:
            self.set_status(cid, "queued", {"recovered_from": "running"})
        return stale
