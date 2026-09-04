"""Scientific workflow contracts shared by every matsim-agents entry point."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class EvidenceLevel(StrEnum):
    """Fidelity of the evidence supporting a result."""

    HYPOTHESIS = "hypothesis"
    MLIP_PREDICTION = "mlip_prediction"
    MLIP_RELAXATION = "mlip_relaxation"
    LOW_FIDELITY_DFT = "low_fidelity_dft"
    CONVERGED_DFT = "converged_dft"
    HIGHER_ACCURACY_DFT = "higher_accuracy_dft"
    EXPERIMENTAL = "experimental"


class WorkflowStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    PARTIAL = "partial"
    REJECTED = "rejected"


class ValidationRecord(BaseModel):
    """One numerical, physical, uncertainty, or scientific validation."""

    stage: Literal["numerical", "physical", "uncertainty", "scientific"]
    name: str
    passed: bool
    message: str
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class ComputeBudget(BaseModel):
    """Hard workflow limits; ``None`` means that the user set no limit."""

    max_candidates: int | None = Field(None, ge=1)
    max_mlip_relaxations: int | None = Field(None, ge=1)
    max_dft_calculations: int | None = Field(None, ge=1)
    max_active_learning_iterations: int | None = Field(None, ge=1)
    max_node_hours: float | None = Field(None, gt=0)


class ApprovalPolicy(BaseModel):
    """Human approval gates for expensive or model-changing actions."""

    before_dft: bool = True
    before_retraining: bool = True
    before_model_promotion: bool = True


class ProvenanceRecord(BaseModel):
    """Immutable lineage and numerical-method metadata for an artifact."""

    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    workflow: str
    evidence_level: EvidenceLevel
    software_versions: dict[str, str] = Field(default_factory=dict)
    backend: str | None = None
    backend_version: str | None = None
    model_identifier: str | None = None
    model_checkpoint_hash: str | None = None
    parent_run_id: str | None = None
    parent_dataset_id: str | None = None
    random_seed: int | None = None
    numerical_settings: dict[str, Any] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    energy_reference: str | None = None


class WorkflowResult(BaseModel):
    """Standard envelope returned by scientific workflows."""

    run_id: str
    workflow: str
    status: WorkflowStatus
    evidence_level: EvidenceLevel
    converged: bool | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    validations: list[ValidationRecord] = Field(default_factory=list)
    failure_reason: str | None = None
    provenance: ProvenanceRecord

    @model_validator(mode="after")
    def _failed_results_explain_why(self) -> WorkflowResult:
        if (
            self.status in {WorkflowStatus.FAILED, WorkflowStatus.REJECTED}
            and not self.failure_reason
        ):
            raise ValueError("failed or rejected workflow results require failure_reason")
        return self


__all__ = [
    "ApprovalPolicy",
    "ComputeBudget",
    "EvidenceLevel",
    "ProvenanceRecord",
    "ValidationRecord",
    "WorkflowResult",
    "WorkflowStatus",
]
