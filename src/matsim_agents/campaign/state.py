"""Campaign-wide state: the formula registry and accumulating hull references.

A ``CampaignState`` tracks every formula considered across a multi-iteration
discovery campaign (see the element-set-to-formula generation module,
:mod:`matsim_agents.discovery.formula`) plus the growing set of reference
energies used for convex-hull ranking
(:class:`matsim_agents.discovery.stability.ReferenceEnergySet`). Structure
generation, relaxation, and stability scoring for any one active formula
reuse the existing single-composition pipeline
(:mod:`matsim_agents.discovery.wrapper`); this module does not reimplement
any of that.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from matsim_agents.discovery.formula import FormulaCandidate, FormulaGenerationPolicy
from matsim_agents.discovery.stability import RankingMode, ReferenceEnergySet, StabilityReport
from matsim_agents.execution.contracts import ComputeBudget, WorkflowStatus


class FormulaRunRecord(BaseModel):
    """Durable execution state for one formula in a campaign."""

    formula: str
    status: WorkflowStatus = WorkflowStatus.PLANNED
    iteration: int = 0
    attempts: int = 0
    output_dir: str | None = None
    failure_reason: str | None = None
    n_mlip_relaxations: int = 0
    n_dft_calculations: int = 0
    n_active_learning_iterations: int = 0
    node_hours: float = 0.0
    model_promoted: bool = False
    evidence: dict[str, Any] = Field(default_factory=dict)


class CampaignReviewRecord(BaseModel):
    """Auditable outcome of one campaign evidence review."""

    iteration: int
    debate_run_id: str | None = None
    deactivate_formulas: list[str] = Field(default_factory=list)
    reactivate_formulas: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CampaignState(BaseModel):
    """Persistent record of a multi-formula discovery campaign."""

    campaign_id: str
    element_set: list[str]
    formula_policy: FormulaGenerationPolicy
    formulas: dict[str, FormulaCandidate] = Field(default_factory=dict)
    reference_energies: ReferenceEnergySet | None = None
    stability_reports: dict[str, StabilityReport] = Field(default_factory=dict)
    formula_runs: dict[str, FormulaRunRecord] = Field(default_factory=dict)
    review_history: list[CampaignReviewRecord] = Field(default_factory=list)
    debate_run_ids: list[str] = Field(default_factory=list)
    iteration: int = 0
    budget: ComputeBudget = Field(default_factory=ComputeBudget)
    status: WorkflowStatus = WorkflowStatus.PLANNED

    @classmethod
    def load(cls, path: str | Path) -> CampaignState:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def upsert_formulas(self, candidates: Iterable[FormulaCandidate]) -> None:
        for candidate in candidates:
            self.formulas[candidate.reduced_formula] = candidate

    def active_formulas(self) -> list[FormulaCandidate]:
        return [candidate for candidate in self.formulas.values() if candidate.active]

    def record_stability(self, report: StabilityReport) -> None:
        """Record a formula's stability report and, if it is a hull-consistent
        new ground state, fold it into the campaign's growing reference set so
        the *next* formula's hull ranking accounts for it."""
        self.stability_reports[report.formula] = report
        ground_state = report.ground_state
        if (
            report.ranking_mode == RankingMode.CONVEX_HULL
            and report.chemically_stable_proxy
            and ground_state.formation_energy_eV_per_atom is not None
            and self.reference_energies is not None
        ):
            self.reference_energies.competing_phases[report.formula] = (
                ground_state.formation_energy_eV_per_atom
            )
