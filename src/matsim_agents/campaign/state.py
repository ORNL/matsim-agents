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

from collections.abc import Iterable

from pydantic import BaseModel, Field

from matsim_agents.discovery.formula import FormulaCandidate, FormulaGenerationPolicy
from matsim_agents.discovery.stability import RankingMode, ReferenceEnergySet, StabilityReport
from matsim_agents.execution.contracts import ComputeBudget, WorkflowStatus


class CampaignState(BaseModel):
    """Persistent record of a multi-formula discovery campaign."""

    campaign_id: str
    element_set: list[str]
    formula_policy: FormulaGenerationPolicy
    formulas: dict[str, FormulaCandidate] = Field(default_factory=dict)
    reference_energies: ReferenceEnergySet | None = None
    stability_reports: dict[str, StabilityReport] = Field(default_factory=dict)
    debate_run_ids: list[str] = Field(default_factory=list)
    iteration: int = 0
    budget: ComputeBudget = Field(default_factory=ComputeBudget)
    status: WorkflowStatus = WorkflowStatus.PLANNED

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
