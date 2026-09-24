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
from typing import Iterable, Sequence

from pydantic import BaseModel, Field

from matsim_agents.discovery.seeds import PhaseCandidate
from matsim_agents.orchestration.state import RelaxationResult


class PhaseStability(BaseModel):
    """Per-phase stability summary."""

    structure_path: str
    optimized_structure_path: str
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


class ReferenceEnergySet(BaseModel):
    """Compatible elemental/competing-phase references for hull analysis."""

    identifier: str
    method_signature: str
    elemental_energies_eV_per_atom: dict[str, float]
    competing_phases: dict[str, float] = Field(
        default_factory=dict,
        description="Formation energies in eV/atom keyed by composition formula.",
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


def _atoms_count_from_path(path: str) -> int:
    from ase.io import read

    return len(read(path))


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
        n_atoms = _atoms_count_from_path(r.optimized_structure_path)
        e_per_atom = r.final_energy_eV / max(n_atoms, 1)
        cand = cand_by_path.get(r.structure_path)
        items.append(
            PhaseStability(
                structure_path=r.structure_path,
                optimized_structure_path=r.optimized_structure_path,
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
        for phase_formula, formation_per_atom in reference_energies.competing_phases.items():
            comp = PMGComposition(phase_formula)
            ref_total = sum(
                amount * reference_energies.elemental_energies_eV_per_atom[element]
                for element, amount in comp.as_dict().items()
            )
            entries.append(ComputedEntry(comp, ref_total + formation_per_atom * comp.num_atoms))
        candidate_entries = []
        for index, item in enumerate(ranking):
            entry = ComputedEntry(target_comp, item.final_energy_eV, entry_id=f"candidate-{index}")
            entries.append(entry)
            candidate_entries.append((item, entry))
        diagram = PhaseDiagram(entries)
        for item, entry in candidate_entries:
            decomposition, e_hull = diagram.get_decomp_and_e_above_hull(entry)
            item.formation_energy_eV_per_atom = diagram.get_form_energy_per_atom(entry)
            item.energy_above_hull_eV_per_atom = float(e_hull)
            item.decomposition = {
                phase.composition.reduced_formula: float(fraction)
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

    near_degenerate = [
        it for it in ranking[1:] if it.delta_e_above_min_eV_per_atom < degeneracy_tol_eV_per_atom
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
        summary_lines.append(
            f"WARNING: {len(near_degenerate)} other phase(s) within "
            f"{degeneracy_tol_eV_per_atom:.3f} eV/atom; ground-state assignment is uncertain."
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
    )
