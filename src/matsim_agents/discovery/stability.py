"""Stability scoring from a batch of relaxed candidate structures.

We score two aspects:

* **Chemical stability** (relative): for a fixed composition, the lowest
  total energy per atom across the relaxed seeds is the candidate
  ground state. All other seeds are reported as ``ΔE/atom`` above it.
  Absolute formation energies vs. elemental references would require a
  curated reference set; we expose hooks but do not require it.

* **Dynamical stability** (proxy): a relaxed structure is considered
  *dynamically plausible* if the residual maximum atomic force is below
  a small threshold (default 0.05 eV/Å). A full phonon spectrum check
  (no imaginary modes at the Γ-point) is left as an optional follow-up
  because it requires either finite-difference Hessians or a phonopy
  workflow.

A seed's ``source`` ("prototype" vs "random") is propagated into the
report so that downstream agents can flag candidates that arose from
the pyXtal random-structure path: those are novel topologies that have
not been observed crystallographically and should be DFT-validated
before any stability claim is published.
"""

from __future__ import annotations

from enum import StrEnum
from itertools import combinations
from typing import Iterable, Sequence

from pydantic import BaseModel, Field, model_validator

from matsim_agents.discovery.seeds import PhaseCandidate
from matsim_agents.orchestration.state import RelaxationResult


class PhaseStability(BaseModel):
    """Per-phase stability summary."""

    structure_path: str
    optimized_structure_path: str
    composition: dict[str, float] | None = None
    final_energy_eV: float
    energy_per_atom_eV: float
    delta_e_above_min_eV_per_atom: float
    final_max_force_eV_per_A: float
    converged: bool
    dynamically_stable_proxy: bool = Field(
        ...,
        description="True if max residual force is below `force_tol_eV_per_A`.",
    )
    # Seed-provenance fields (populated from the matching PhaseCandidate when
    # provided; ``source`` defaults to "prototype" for legacy callers).
    source: str = "prototype"
    prototype_id: str | None = None
    space_group: int | None = None
    needs_dft_verification: bool = False
    eligible_for_ranking: bool = True
    exclusion_reason: str | None = None
    formation_energy_eV_per_atom: float | None = None
    energy_above_hull_eV_per_atom: float | None = None
    decomposition: dict[str, float] = Field(default_factory=dict)


class RankingMode(StrEnum):
    """Scientific meaning of a phase ranking."""

    RELATIVE = "relative_phase_ranking"
    CONVEX_HULL = "convex_hull_ranking"


class ReferencePhaseEntry(BaseModel):
    """One method-compatible competing polymorph with auditable provenance."""

    phase_id: str = Field(min_length=1)
    formula: str = Field(min_length=1)
    formation_energy_eV_per_atom: float = Field(allow_inf_nan=False)
    method_signature: str = Field(min_length=1)
    backend: str | None = None
    structure_path: str | None = None
    structure_hash: str | None = None
    total_energy_eV: float | None = Field(None, allow_inf_nan=False)
    energy_per_atom_eV: float | None = Field(None, allow_inf_nan=False)
    source: str = "user_supplied"
    provenance: dict[str, str] = Field(default_factory=dict)
    corrections: dict[str, float] = Field(default_factory=dict)


class ElementalReferenceEntry(BaseModel):
    """Selected elemental chemical potential and its computational provenance."""

    element: str = Field(min_length=1)
    phase_id: str = Field(min_length=1)
    reference_formula: str = Field(min_length=1)
    energy_eV_per_atom: float = Field(allow_inf_nan=False)
    method_signature: str = Field(min_length=1)
    backend: str | None = None
    structure_path: str | None = None
    structure_hash: str | None = None
    total_energy_eV: float | None = Field(None, allow_inf_nan=False)
    source: str = "user_supplied"
    provenance: dict[str, str] = Field(default_factory=dict)
    corrections: dict[str, float] = Field(default_factory=dict)


class ReferenceCompletenessPolicy(BaseModel):
    """Minimum chemical coverage required before treating a hull as complete."""

    required_formulas: list[str] = Field(default_factory=list)
    require_binary_subsystems: bool = True
    require_ternary_competitor: bool = False


class ReferenceCompletenessReport(BaseModel):
    """Auditable explanation of whether a reference hull is provisional."""

    target_elements: list[str]
    missing_elemental_references: list[str] = Field(default_factory=list)
    missing_required_formulas: list[str] = Field(default_factory=list)
    missing_binary_subsystems: list[str] = Field(default_factory=list)
    missing_ternary_competitor: bool = False
    provisional: bool


class ReferenceEnergySet(BaseModel):
    """Compatible elemental/competing-phase references for hull analysis."""

    identifier: str
    method_signature: str
    backend: str | None = None
    elemental_energies_eV_per_atom: dict[str, float]
    elemental_entries: dict[str, ElementalReferenceEntry] = Field(default_factory=dict)
    competing_phases: dict[str, float] = Field(
        default_factory=dict,
        description="Formation energies in eV/atom keyed by composition formula.",
    )
    phase_entries: list[ReferencePhaseEntry] = Field(default_factory=list)
    completeness_policy: ReferenceCompletenessPolicy = Field(
        default_factory=ReferenceCompletenessPolicy
    )

    @model_validator(mode="after")
    def _validate_phase_entries(self) -> ReferenceEnergySet:
        phase_ids = [entry.phase_id for entry in self.phase_entries]
        if len(phase_ids) != len(set(phase_ids)):
            raise ValueError("reference phase IDs must be unique")
        for entry in self.phase_entries:
            if entry.method_signature != self.method_signature:
                raise ValueError(
                    f"reference phase {entry.phase_id!r} uses method signature "
                    f"{entry.method_signature!r}, expected {self.method_signature!r}"
                )
            if self.backend is not None and entry.backend not in {None, self.backend}:
                raise ValueError(
                    f"reference phase {entry.phase_id!r} uses backend {entry.backend!r}, "
                    f"expected {self.backend!r}"
                )
        for element, entry in self.elemental_entries.items():
            if entry.element != element:
                raise ValueError(
                    f"elemental reference key {element!r} does not match entry element "
                    f"{entry.element!r}"
                )
            if entry.method_signature != self.method_signature:
                raise ValueError(
                    f"elemental reference {entry.phase_id!r} uses method signature "
                    f"{entry.method_signature!r}, expected {self.method_signature!r}"
                )
            if self.backend is not None and entry.backend not in {None, self.backend}:
                raise ValueError(
                    f"elemental reference {entry.phase_id!r} uses backend {entry.backend!r}, "
                    f"expected {self.backend!r}"
                )
            if self.elemental_energies_eV_per_atom.get(element) != entry.energy_eV_per_atom:
                raise ValueError(f"elemental energy for {element!r} does not match its typed entry")
        return self

    def competing_phase_records(self) -> list[tuple[str, str, float]]:
        """Return ``(phase_id, formula, formation_energy)`` for all references."""
        records = [
            (entry.phase_id, entry.formula, entry.formation_energy_eV_per_atom)
            for entry in self.phase_entries
        ]
        records.extend(
            (f"legacy:{formula}", formula, energy)
            for formula, energy in self.competing_phases.items()
        )
        return records

    def audit_completeness(self, target_elements: Iterable[str]) -> ReferenceCompletenessReport:
        """Check elemental, required-formula, and chemical-subsystem coverage."""
        from pymatgen.core import Composition as PMGComposition

        elements = sorted(set(target_elements))
        missing_elements = sorted(set(elements) - set(self.elemental_energies_eV_per_atom))
        formulas = {
            PMGComposition(formula).reduced_formula
            for _phase_id, formula, _energy in self.competing_phase_records()
        }
        required = {
            PMGComposition(formula).reduced_formula
            for formula in self.completeness_policy.required_formulas
        }
        missing_required = sorted(required - formulas)
        covered_element_sets = {
            frozenset(PMGComposition(formula).as_dict())
            for _phase_id, formula, _energy in self.competing_phase_records()
        }
        missing_binary = []
        if self.completeness_policy.require_binary_subsystems:
            missing_binary = [
                "-".join(pair)
                for pair in combinations(elements, 2)
                if frozenset(pair) not in covered_element_sets
            ]
        missing_ternary = bool(
            self.completeness_policy.require_ternary_competitor
            and len(elements) >= 3
            and frozenset(elements) not in covered_element_sets
        )
        provisional = bool(
            missing_elements or missing_required or missing_binary or missing_ternary
        )
        return ReferenceCompletenessReport(
            target_elements=elements,
            missing_elemental_references=missing_elements,
            missing_required_formulas=missing_required,
            missing_binary_subsystems=missing_binary,
            missing_ternary_competitor=missing_ternary,
            provisional=provisional,
        )


class StabilityReport(BaseModel):
    """Outcome of comparing a batch of relaxed structures."""

    formula: str
    ground_state: PhaseStability
    ranking: list[PhaseStability]
    chemically_stable_proxy: bool = Field(
        ...,
        description="True if ground-state phase is dynamically stable AND no "
        "other phase is within `degeneracy_tol_eV_per_atom`.",
    )
    summary: str
    ranking_mode: RankingMode = RankingMode.RELATIVE
    reference_set_id: str | None = None
    degeneracy_tolerance_eV_per_atom: float = 0.01
    degeneracy_reference_structure_path: str | None = None
    near_degenerate_structure_paths: list[str] = Field(default_factory=list)


def recalibrate_hull_reports(
    reports: Iterable[StabilityReport],
    reference_energies: ReferenceEnergySet,
) -> None:
    """Recompute stored phase-diagram results against one reference set."""
    try:
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        from pymatgen.core import Composition as PMGComposition
        from pymatgen.entries.computed_entries import ComputedEntry
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("convex-hull ranking requires pymatgen") from exc

    compatible_reports = [
        report
        for report in reports
        if report.ranking_mode == RankingMode.CONVEX_HULL
        and report.reference_set_id == reference_energies.identifier
    ]
    entries = [
        ComputedEntry(element, energy)
        for element, energy in reference_energies.elemental_energies_eV_per_atom.items()
    ]
    for phase_id, phase_formula, formation_per_atom in reference_energies.competing_phase_records():
        composition = PMGComposition(phase_formula)
        reference_total = sum(
            amount * reference_energies.elemental_energies_eV_per_atom[element]
            for element, amount in composition.as_dict().items()
        )
        entries.append(
            ComputedEntry(
                composition,
                reference_total + formation_per_atom * composition.num_atoms,
                entry_id=None if phase_id.startswith("legacy:") else phase_id,
            )
        )

    candidate_entries: list[tuple[StabilityReport, PhaseStability, object]] = []
    for report in compatible_reports:
        for index, phase in enumerate(report.ranking):
            composition = PMGComposition(
                phase.composition or _composition_from_path(phase.optimized_structure_path)
            )
            if (
                composition.reduced_composition
                != PMGComposition(report.formula).reduced_composition
            ):
                raise ValueError(
                    f"optimized composition {composition.formula} does not match "
                    f"reported formula {report.formula}"
                )
            entry = ComputedEntry(
                composition,
                phase.final_energy_eV,
                entry_id=f"campaign-{report.formula}-{index}",
            )
            entries.append(entry)
            candidate_entries.append((report, phase, entry))

    if not candidate_entries:
        return
    diagram = PhaseDiagram(entries)
    for _report, phase, entry in candidate_entries:
        decomposition, energy_above_hull = diagram.get_decomp_and_e_above_hull(entry)
        phase.formation_energy_eV_per_atom = diagram.get_form_energy_per_atom(entry)
        phase.energy_above_hull_eV_per_atom = float(energy_above_hull)
        phase.decomposition = {
            str(product.entry_id or product.composition.reduced_formula): float(fraction)
            for product, fraction in decomposition.items()
        }
    for report in compatible_reports:
        report.ranking.sort(
            key=lambda phase: (
                phase.energy_above_hull_eV_per_atom
                if phase.energy_above_hull_eV_per_atom is not None
                else float("inf")
            )
        )
        report.ground_state = report.ranking[0]


def _composition_from_path(path: str) -> dict[str, float]:
    from collections import Counter

    from ase.io import read

    return {element: float(amount) for element, amount in Counter(read(path).symbols).items()}


def score_stability(
    formula: str,
    relaxations: Iterable[RelaxationResult],
    force_tol_eV_per_A: float = 0.05,
    degeneracy_tol_eV_per_atom: float = 0.01,
    *,
    candidates: Sequence[PhaseCandidate] | None = None,
    ranking_mode: RankingMode = RankingMode.RELATIVE,
    reference_energies: ReferenceEnergySet | None = None,
    method_signature: str | None = None,
) -> StabilityReport:
    """Rank relaxations of the same composition and report stability.

    Parameters
    ----------
    candidates:
        Optional seed candidates to join against by ``structure_path`` so
        that the report can carry the ``source`` / ``prototype_id`` /
        ``space_group`` / ``needs_dft_verification`` provenance. When
        omitted, all entries default to a "prototype" source.
    """
    if ranking_mode == RankingMode.CONVEX_HULL:
        if reference_energies is None:
            raise ValueError("convex_hull_ranking requires a compatible reference-energy set")
        if not method_signature or method_signature != reference_energies.method_signature:
            raise ValueError(
                "convex-hull candidate and reference energies must share method_signature"
            )

    cand_by_path: dict[str, PhaseCandidate] = {}
    if candidates is not None:
        cand_by_path = {c.structure_path: c for c in candidates}

    items: list[PhaseStability] = []
    for r in relaxations:
        composition = _composition_from_path(r.optimized_structure_path)
        n_atoms = int(sum(composition.values()))
        e_per_atom = r.final_energy_eV / max(n_atoms, 1)
        cand = cand_by_path.get(r.structure_path)
        items.append(
            PhaseStability(
                structure_path=r.structure_path,
                optimized_structure_path=r.optimized_structure_path,
                composition=composition,
                final_energy_eV=r.final_energy_eV,
                energy_per_atom_eV=e_per_atom,
                delta_e_above_min_eV_per_atom=0.0,  # filled in below
                final_max_force_eV_per_A=r.final_max_force_eV_per_A,
                converged=r.converged,
                dynamically_stable_proxy=r.final_max_force_eV_per_A <= force_tol_eV_per_A,
                source=cand.source if cand is not None else "prototype",
                prototype_id=cand.prototype_id if cand is not None else None,
                space_group=cand.space_group if cand is not None else None,
                needs_dft_verification=(cand.needs_dft_verification if cand is not None else False),
                eligible_for_ranking=(
                    r.converged and r.final_max_force_eV_per_A <= force_tol_eV_per_A
                ),
                exclusion_reason=(
                    None
                    if r.converged and r.final_max_force_eV_per_A <= force_tol_eV_per_A
                    else "unconverged or residual force exceeds tolerance"
                ),
            )
        )

    if not items:
        raise ValueError("score_stability requires at least one relaxation result.")

    eligible = [it for it in items if it.eligible_for_ranking]
    if not eligible:
        raise ValueError("no converged candidates satisfy the force tolerance for ranking")
    e_min = min(it.energy_per_atom_eV for it in eligible)
    for it in items:
        it.delta_e_above_min_eV_per_atom = it.energy_per_atom_eV - e_min

    ranking = sorted(eligible, key=lambda it: it.energy_per_atom_eV)
    if ranking_mode == RankingMode.CONVEX_HULL:
        assert reference_energies is not None
        try:
            from pymatgen.analysis.phase_diagram import PhaseDiagram
            from pymatgen.core import Composition as PMGComposition
            from pymatgen.entries.computed_entries import ComputedEntry
        except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
            raise RuntimeError("convex-hull ranking requires pymatgen") from exc

        target_comp = PMGComposition(formula)
        missing = set(target_comp.as_dict()) - set(
            reference_energies.elemental_energies_eV_per_atom
        )
        if missing:
            raise ValueError(
                f"reference-energy set lacks elemental references for {sorted(missing)}"
            )
        entries = [
            ComputedEntry(element, energy)
            for element, energy in reference_energies.elemental_energies_eV_per_atom.items()
        ]
        for (
            phase_id,
            phase_formula,
            formation_per_atom,
        ) in reference_energies.competing_phase_records():
            comp = PMGComposition(phase_formula)
            ref_total = sum(
                amount * reference_energies.elemental_energies_eV_per_atom[element]
                for element, amount in comp.as_dict().items()
            )
            entries.append(
                ComputedEntry(
                    comp,
                    ref_total + formation_per_atom * comp.num_atoms,
                    entry_id=None if phase_id.startswith("legacy:") else phase_id,
                )
            )
        candidate_entries = []
        for index, item in enumerate(ranking):
            candidate_comp = PMGComposition(item.composition)
            if candidate_comp.reduced_composition != target_comp.reduced_composition:
                raise ValueError(
                    f"optimized composition {candidate_comp.formula} does not match "
                    f"target formula {formula}"
                )
            entry = ComputedEntry(
                candidate_comp,
                item.final_energy_eV,
                entry_id=f"candidate-{index}",
            )
            entries.append(entry)
            candidate_entries.append((item, entry))
        diagram = PhaseDiagram(entries)
        for item, entry in candidate_entries:
            decomposition, e_hull = diagram.get_decomp_and_e_above_hull(entry)
            item.formation_energy_eV_per_atom = diagram.get_form_energy_per_atom(entry)
            item.energy_above_hull_eV_per_atom = float(e_hull)
            item.decomposition = {
                str(phase.entry_id or phase.composition.reduced_formula): float(fraction)
                for phase, fraction in decomposition.items()
            }
        ranking = sorted(
            ranking,
            key=lambda item: (
                item.energy_above_hull_eV_per_atom
                if item.energy_above_hull_eV_per_atom is not None
                else float("inf")
            ),
        )
    ground = ranking[0]
    degeneracy_reference = min(eligible, key=lambda item: item.energy_per_atom_eV)
    near_degenerate = [
        item
        for item in eligible
        if item is not degeneracy_reference
        and item.energy_per_atom_eV - degeneracy_reference.energy_per_atom_eV
        < degeneracy_tol_eV_per_atom
    ]
    chem_stable = ground.dynamically_stable_proxy and not near_degenerate

    n_prototype = sum(1 for it in ranking if it.source == "prototype")
    n_random = sum(1 for it in ranking if it.source == "random")

    summary_lines = [
        f"Composition {formula}: {len(ranking)} candidate seed(s) relaxed "
        f"({n_prototype} prototype + {n_random} random).",
        f"Predicted ground state: {ground.optimized_structure_path} "
        f"(E/atom = {ground.energy_per_atom_eV:.4f} eV, "
        f"|F|max = {ground.final_max_force_eV_per_A:.4f} eV/Å, "
        f"dynamically_stable_proxy = {ground.dynamically_stable_proxy}, "
        f"source = {ground.source}"
        + (f", prototype = {ground.prototype_id}" if ground.prototype_id else "")
        + (f", SG = {ground.space_group}" if ground.space_group else "")
        + ").",
    ]
    if ground.needs_dft_verification:
        summary_lines.append(
            "NOVELTY ALERT: ground-state candidate originated from the pyXtal "
            "random-structure search (no matching known crystal prototype). "
            "Treat this as a HYPOTHESIS — DFT verification is required before "
            "any stability claim can be published."
        )
    if near_degenerate:
        tolerance_meV_per_atom = degeneracy_tol_eV_per_atom * 1000.0
        summary_lines.append(
            "**Energetically near-degenerate polymorphs within "
            f"{tolerance_meV_per_atom:g} meV/atom**: {len(near_degenerate)} other phase(s) "
            "fall within this energy window relative to the lowest-energy eligible "
            f"polymorph, {degeneracy_reference.optimized_structure_path} "
            f"(E/atom = {degeneracy_reference.energy_per_atom_eV:.6f} eV). "
            "Ground-state assignment is uncertain."
        )
    summary_lines.append(f"Chemical-stability proxy: {'PASS' if chem_stable else 'INCONCLUSIVE'}.")

    return StabilityReport(
        formula=formula,
        ground_state=ground,
        ranking=ranking,
        chemically_stable_proxy=chem_stable,
        summary="\n".join(summary_lines),
        ranking_mode=ranking_mode,
        reference_set_id=(reference_energies.identifier if reference_energies else None),
        degeneracy_tolerance_eV_per_atom=degeneracy_tol_eV_per_atom,
        degeneracy_reference_structure_path=degeneracy_reference.optimized_structure_path,
        near_degenerate_structure_paths=[item.optimized_structure_path for item in near_degenerate],
    )
