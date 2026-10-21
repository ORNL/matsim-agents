"""Scientific workflow contracts shared by every matsim-agents entry point."""

from __future__ import annotations

from datetime import UTC, datetime
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
    """Workflow budgets; enforcement boundaries depend on the resource.

    MLIP limits count candidate relaxation attempts (including failures and
    post-promotion reevaluation), not generation, MD, or reference qualification.
    Node-hour thresholds are checked between formula runs.
    ``None`` means that the user set no limit.
    """

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

    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
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


# Evidence levels from DFT; only these can support thermodynamic claims.
DFT_EVIDENCE = frozenset(
    {
        EvidenceLevel.LOW_FIDELITY_DFT,
        EvidenceLevel.CONVERGED_DFT,
        EvidenceLevel.HIGHER_ACCURACY_DFT,
    }
)
# Fidelity order: a result at a level satisfies requirements at or below it.
EVIDENCE_RANK = {level: rank for rank, level in enumerate(EvidenceLevel)}


def at_least(level: EvidenceLevel, required: EvidenceLevel) -> bool:
    """Whether evidence at ``level`` meets a ``required`` fidelity."""
    return EVIDENCE_RANK[level] >= EVIDENCE_RANK[required]


class CharacterizationStatus(StrEnum):
    """Outcome of a characterization; failing to compute is never evidence against."""

    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    SUFFICIENTLY_CHARACTERIZED = "sufficiently_characterized"
    INCONCLUSIVE = "inconclusive"
    INVALID_PROTOCOL = "invalid_protocol"
    FAILED_INFRASTRUCTURE = "failed_infrastructure"
    CANCELLED = "cancelled"


# Canonical unit of every observable a characterization can require.
OBSERVABLE_UNITS = {
    "relaxed_structure": "structure",
    "energy_per_atom": "eV/atom",
    "residual_max_force": "eV/A",
    "relative_energy_per_atom": "eV/atom",
    "formation_energy_per_atom": "eV/atom",
    "energy_above_hull": "eV/atom",
}


class ObservableRequirement(BaseModel):
    """An observable a characterization must produce, its unit, and its fidelity."""

    name: Literal[
        "relaxed_structure",
        "energy_per_atom",
        "residual_max_force",
        "relative_energy_per_atom",
        "formation_energy_per_atom",
        "energy_above_hull",
    ]
    unit: str
    minimum_evidence_level: EvidenceLevel = EvidenceLevel.MLIP_RELAXATION

    @model_validator(mode="after")
    def _canonical_unit(self) -> ObservableRequirement:
        expected = OBSERVABLE_UNITS[self.name]
        if self.unit != expected:
            raise ValueError(
                f"observable {self.name!r} is reported in {expected!r}, not {self.unit!r}"
            )
        return self


class EvidenceItem(BaseModel):
    """One numerical artifact a characterization relies on."""

    evidence_id: str
    kind: str
    evidence_level: EvidenceLevel
    method: str
    artifact: str | None = None
    artifact_sha256: str | None = None
    converged: bool | None = None


class ObservableValue(BaseModel):
    """A computed observable, for one structure or for the system."""

    name: str
    value: float | str
    unit: str
    evidence_level: EvidenceLevel
    structure: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class CharacterizationReport(BaseModel):
    """Machine-checkable answer to a characterization contract."""

    schema_version: Literal[1] = 1
    contract_id: str
    contract_sha256: str
    recipe: str
    status: CharacterizationStatus
    conclusion: str | None = Field(
        None, description="The answer, scoped to the evidence; none unless characterized."
    )
    observables: list[ObservableValue] = Field(default_factory=list)
    methods: dict[str, str] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommended_next_test: str | None = None

    @model_validator(mode="after")
    def _explained(self) -> CharacterizationReport:
        if self.status == CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED:
            if not self.conclusion or not self.evidence:
                raise ValueError("a characterization needs a conclusion and its evidence")
        elif self.conclusion is not None:
            raise ValueError(f"a {self.status.value} characterization draws no conclusion")
        elif not (self.missing or self.limitations):
            raise ValueError(f"a {self.status.value} characterization must say what is missing")
        cited = {item.evidence_id for item in self.evidence}
        unknown = sorted({i for value in self.observables for i in value.evidence_ids} - cited)
        if unknown:
            raise ValueError(f"observables cite unknown evidence {unknown}")
        return self


__all__ = [
    "DFT_EVIDENCE",
    "EVIDENCE_RANK",
    "OBSERVABLE_UNITS",
    "ApprovalPolicy",
    "CharacterizationReport",
    "CharacterizationStatus",
    "ComputeBudget",
    "EvidenceItem",
    "EvidenceLevel",
    "ObservableRequirement",
    "ObservableValue",
    "ProvenanceRecord",
    "ValidationRecord",
    "WorkflowResult",
    "WorkflowStatus",
    "at_least",
]
