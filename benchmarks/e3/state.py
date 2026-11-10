"""Persistent E3 workflow state.

The state file is a derived, resumable view.  The per-strategy evidence
ledgers remain the authoritative record; every state field that matters for
scientific claims points back at ledger record IDs.
"""
from __future__ import annotations

import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "e3-workflow-state-v2"


class E3State(str, Enum):
    SELECTED = "selected"
    AWAITING_DFT = "awaiting_dft"
    EVIDENCE_QUALIFIED = "evidence_qualified"
    EVIDENCE_INGESTED = "evidence_ingested"
    REJECTED = "rejected"
    REASSESSED = "reassessed"
    ADAPTATION_ASSESSED = "adaptation_assessed"
    NEXT_ROUND_READY = "next_round_ready"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    SCIENTIFICALLY_UNRESOLVED = "scientifically_unresolved"
    AUTHORIZATION_REQUIRED = "authorization_required"


class GateOutcome(str, Enum):
    INSUFFICIENT_VALIDATION_EVIDENCE = "insufficient_validation_evidence"
    NO_VALIDATION_DEFICIT = "no_validation_deficit"
    INSUFFICIENT_QUALIFIED_DATA = "insufficient_qualified_training_data"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    AUTHORIZATION_REQUIRED = "authorization_required"


class E3Attempt(BaseModel):
    """One logical DFT attempt as seen in the run record."""

    tag: str
    prec: str | None = None
    attempt_dir: str | None = None
    status: str | None = None
    rc: int | None = None


class E3CalculationRecord(BaseModel):
    candidate_id: str
    calculation_id: str
    round: int
    requested_by: list[str]
    decision_ids: dict[str, str]
    structure_sha256: str
    method_signature: str
    group: str | None = None
    state: E3State = E3State.AWAITING_DFT
    pbs_job_id: str | None = None
    normal_attempt_dir: str | None = None
    attempts: list[E3Attempt] = Field(default_factory=list)
    final_status: str | None = None
    qualified: bool | None = None
    rejection_reasons: list[str] = Field(default_factory=list)
    execution_warnings: list[str] = Field(default_factory=list)
    allocation_history: list[dict[str, Any]] = Field(default_factory=list)
    values: dict[str, Any] = Field(default_factory=dict)
    elapsed_seconds: float | None = None
    node_hours: float | None = None
    nodes: int | None = None
    evidence_ids: dict[str, str] = Field(default_factory=dict)
    ingest_decision_ids: dict[str, str] = Field(default_factory=dict)


class E3StrategyState(BaseModel):
    strategy: str
    ledger_path: str
    logical_attempts: int = 0
    remaining_attempts: int = 0
    counterfactual_node_hours: float = 0.0
    counterfactual_dft_calculations: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_seconds: float = 0.0
    hypothesis_revision_ids: list[str] = Field(default_factory=list)
    reassessment_status: str | None = None
    rejected_reassessments: list[str] = Field(default_factory=list)
    gate_outcome: GateOutcome | None = None
    gate_reasons: list[str] = Field(default_factory=list)
    state: E3State = E3State.AWAITING_DFT


class E3RoundRecord(BaseModel):
    round: int
    selections: dict[str, list[str]] = Field(default_factory=dict)
    calculations: dict[str, E3CalculationRecord] = Field(default_factory=dict)
    strategies: dict[str, E3StrategyState] = Field(default_factory=dict)
    unique_physical_node_hours: float = 0.0
    unique_physical_calculations: int = 0
    next_round_manifest: str | None = None
    state: E3State = E3State.AWAITING_DFT


class E3WorkflowState(BaseModel):
    schema_version: str = SCHEMA_VERSION
    material: str
    method_signature: str
    protocol_sha256: str
    batch_manifest_sha256: str | None = None
    current_round: int = 1
    rounds: list[E3RoundRecord] = Field(default_factory=list)

    def round(self, number: int) -> E3RoundRecord:
        for record in self.rounds:
            if record.round == number:
                return record
        record = E3RoundRecord(round=number)
        self.rounds.append(record)
        self.rounds.sort(key=lambda r: r.round)
        return record

    def acquired_structures(self) -> set[str]:
        """Structure hashes already requested in any round (never re-acquire)."""
        return {
            calc.structure_sha256
            for record in self.rounds
            for calc in record.calculations.values()
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "E3WorkflowState":
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported E3 state schema: {data.get('schema_version')}")
        return cls.model_validate(data)
