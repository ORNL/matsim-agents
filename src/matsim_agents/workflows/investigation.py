"""Persistent hypothesis, evidence, and revision contracts for agentic studies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from matsim_agents.execution.contracts import (
    ComputeBudget,
    EvidenceLevel,
    ProvenanceRecord,
    WorkflowStatus,
)
from matsim_agents.execution.run_directory import ScientificRunDirectory
from matsim_agents.workflows.phase_exploration import (
    PhaseExplorationPolicy,
    PhaseExplorationWorkflowResult,
)


class PropertyTask(BaseModel):
    property_name: str
    method: str
    success_criterion: str
    required_fidelity: EvidenceLevel
    rationale: str


class ScientificHypothesis(BaseModel):
    objective: str
    hypothesis: str
    proposed_compositions: list[str]
    scientific_rationale: str
    assumptions: list[str] = Field(default_factory=list)
    property_tasks: list[PropertyTask] = Field(default_factory=list)
    estimated_cost: str | None = None
    parent_run_id: str | None = None


class HypothesisRevision(BaseModel):
    parent_run_id: str
    disposition: Literal["supported", "rejected", "partially_supported", "inconclusive"]
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    remaining_gaps: list[str] = Field(default_factory=list)
    revised_hypothesis: ScientificHypothesis


class InvestigationConfig(BaseModel):
    objective: str
    output_root: str = "./runs"
    parent_run: str | None = None
    phase_policy: PhaseExplorationPolicy = Field(default_factory=PhaseExplorationPolicy)
    budget: ComputeBudget = Field(default_factory=ComputeBudget)


class InvestigationResult(BaseModel):
    run_id: str
    run_directory: str
    status: WorkflowStatus
    hypothesis: ScientificHypothesis
    explorations: list[PhaseExplorationWorkflowResult] = Field(default_factory=list)
    report_path: str


def run_investigation(
    cfg: InvestigationConfig,
    *,
    hypothesis_builder: Callable[[str, dict[str, Any] | None], ScientificHypothesis],
    phase_runner: Callable[[str, PhaseExplorationPolicy, str], PhaseExplorationWorkflowResult],
) -> InvestigationResult:
    """Compose hypothesis generation with the reusable phase workflow."""

    previous = None
    parent_id = None
    if cfg.parent_run:
        parent = ScientificRunDirectory.open(cfg.parent_run)
        parent_id = parent.run_id
        results_path = parent.path / "results.json"
        previous = (
            __import__("json").loads(results_path.read_text()) if results_path.exists() else None
        )
    hypothesis = hypothesis_builder(cfg.objective, previous)
    hypothesis.parent_run_id = parent_id
    provenance = ProvenanceRecord(
        workflow="agentic_investigation",
        evidence_level=EvidenceLevel.HYPOTHESIS,
        parent_run_id=parent_id,
    )
    run = ScientificRunDirectory.create(
        cfg.output_root,
        workflow="agentic_investigation",
        request={"objective": cfg.objective, "parent_run": cfg.parent_run},
        resolved_config=cfg.model_dump(mode="json"),
        provenance=provenance,
    )
    compositions = hypothesis.proposed_compositions
    if cfg.budget.max_candidates is not None:
        compositions = compositions[: cfg.budget.max_candidates]
    explorations = [
        phase_runner(comp, cfg.phase_policy, str(run.path / "calculations" / comp))
        for comp in compositions
    ]
    report = run.path / "report.md"
    lines = [
        f"# Investigation: {cfg.objective}",
        "",
        hypothesis.hypothesis,
        "",
        "## Candidates",
        "",
    ]
    for result in explorations:
        lines.append(
            f"- {result.composition}: {len(result.initial.phase_candidates)} phase candidate(s)"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = InvestigationResult(
        run_id=run.run_id,
        run_directory=str(run.path),
        status=WorkflowStatus.COMPLETE,
        hypothesis=hypothesis,
        explorations=explorations,
        report_path=str(report),
    )
    run.write_json("hypothesis.json", hypothesis)
    run.write_json("results.json", result)
    return result


__all__ = [
    "HypothesisRevision",
    "InvestigationConfig",
    "InvestigationResult",
    "PropertyTask",
    "ScientificHypothesis",
    "run_investigation",
]
