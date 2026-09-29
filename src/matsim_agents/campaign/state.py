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

from matsim_agents.campaign.acquisition import CampaignAcquisitionState
from matsim_agents.campaign.registry import CandidateRegistry
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
    acquisition_branch: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class CampaignReviewRecord(BaseModel):
    """Auditable outcome of one campaign evidence review."""

    iteration: int
    debate_run_id: str | None = None
    deactivate_formulas: list[str] = Field(default_factory=list)
    reactivate_formulas: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HullState(BaseModel):
    """Versioned thermodynamic state derived from compatible DFT reports."""

    version: int
    parent_version: int | None = None
    reference_set_id: str
    method_signature: str
    hull_vertices: dict[str, str] = Field(default_factory=dict)
    near_hull_phases: dict[str, float] = Field(default_factory=dict)
    energy_above_hull_eV_per_atom: dict[str, float] = Field(default_factory=dict)
    new_hull_vertices: list[str] = Field(default_factory=list)
    removed_hull_vertices: list[str] = Field(default_factory=list)
    uncompetitive_formulas: list[str] = Field(default_factory=list)
    iteration: int


class CampaignState(BaseModel):
    """Persistent record of a multi-formula discovery campaign."""

    campaign_id: str
    element_set: list[str]
    formula_policy: FormulaGenerationPolicy
    formulas: dict[str, FormulaCandidate] = Field(default_factory=dict)
    reference_energies: ReferenceEnergySet | None = None
    stability_reports: dict[str, StabilityReport] = Field(default_factory=dict)
    current_hull: HullState | None = None
    hull_history: list[HullState] = Field(default_factory=list)
    formula_runs: dict[str, FormulaRunRecord] = Field(default_factory=dict)
    review_history: list[CampaignReviewRecord] = Field(default_factory=list)
    acquisition: CampaignAcquisitionState = Field(default_factory=CampaignAcquisitionState)
    candidate_registry: CandidateRegistry = Field(default_factory=CandidateRegistry)
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
            self._snapshot_hull()

    def _snapshot_hull(self, near_hull_threshold_eV_per_atom: float = 0.05) -> None:
        assert self.reference_energies is not None
        energies: dict[str, float] = {}
        vertices: dict[str, str] = {}
        near_hull: dict[str, float] = {}
        uncompetitive: list[str] = []
        for formula, report in self.stability_reports.items():
            energy = report.ground_state.energy_above_hull_eV_per_atom
            if report.ranking_mode != RankingMode.CONVEX_HULL or energy is None:
                continue
            energies[formula] = energy
            if energy <= 1e-8:
                vertices[formula] = report.ground_state.optimized_structure_path
            elif energy <= near_hull_threshold_eV_per_atom:
                near_hull[formula] = energy
            else:
                uncompetitive.append(formula)
        previous = self.current_hull
        previous_vertices = set(previous.hull_vertices) if previous is not None else set()
        current_vertices = set(vertices)
        snapshot = HullState(
            version=len(self.hull_history) + 1,
            parent_version=previous.version if previous is not None else None,
            reference_set_id=self.reference_energies.identifier,
            method_signature=self.reference_energies.method_signature,
            hull_vertices=vertices,
            near_hull_phases=near_hull,
            energy_above_hull_eV_per_atom=energies,
            new_hull_vertices=sorted(current_vertices - previous_vertices),
            removed_hull_vertices=sorted(previous_vertices - current_vertices),
            uncompetitive_formulas=sorted(uncompetitive),
            iteration=self.iteration,
        )
        self.current_hull = snapshot
        self.hull_history.append(snapshot)
