"""Stability scoring from a batch of relaxed candidate structures.

We score two aspects:

* **Chemical stability** (relative): for a fixed composition, the lowest
  total energy per atom across the relaxed prototypes is the candidate
  ground state. All other phases are reported as ``ΔE/atom`` above it.
  Absolute formation energies vs. elemental references would require a
  curated reference set; we expose hooks but do not require it.

* **Dynamical stability** (proxy): a relaxed structure is considered
  *dynamically plausible* if the residual maximum atomic force is below
  a small threshold (default 0.05 eV/Å). A full phonon spectrum check
  (no imaginary modes at the Γ-point) is left as an optional follow-up
  because it requires either finite-difference Hessians or a phonopy
  workflow.
"""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, Field

from matsim_agents.state import RelaxationResult


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


def _atoms_count_from_path(path: str) -> int:
    from ase.io import read

    return len(read(path))


def score_stability(
    formula: str,
    relaxations: Iterable[RelaxationResult],
    force_tol_eV_per_A: float = 0.05,
    degeneracy_tol_eV_per_atom: float = 0.01,
) -> StabilityReport:
    """Rank relaxations of the same composition and report stability."""
    items: list[PhaseStability] = []
    for r in relaxations:
        n_atoms = _atoms_count_from_path(r.optimized_structure_path)
        e_per_atom = r.final_energy_eV / max(n_atoms, 1)
        items.append(PhaseStability(
            structure_path=r.structure_path,
            optimized_structure_path=r.optimized_structure_path,
            final_energy_eV=r.final_energy_eV,
            energy_per_atom_eV=e_per_atom,
            delta_e_above_min_eV_per_atom=0.0,  # filled in below
            final_max_force_eV_per_A=r.final_max_force_eV_per_A,
            converged=r.converged,
            dynamically_stable_proxy=r.final_max_force_eV_per_A <= force_tol_eV_per_A,
        ))

    if not items:
        raise ValueError("score_stability requires at least one relaxation result.")

    e_min = min(it.energy_per_atom_eV for it in items)
    for it in items:
        it.delta_e_above_min_eV_per_atom = it.energy_per_atom_eV - e_min

    ranking = sorted(items, key=lambda it: it.energy_per_atom_eV)
    ground = ranking[0]

    near_degenerate = [it for it in ranking[1:]
                       if it.delta_e_above_min_eV_per_atom < degeneracy_tol_eV_per_atom]
    chem_stable = ground.dynamically_stable_proxy and not near_degenerate

    summary_lines = [
        f"Composition {formula}: {len(ranking)} candidate phase(s) relaxed.",
        f"Predicted ground state: {ground.optimized_structure_path} "
        f"(E/atom = {ground.energy_per_atom_eV:.4f} eV, "
        f"|F|max = {ground.final_max_force_eV_per_A:.4f} eV/Å, "
        f"dynamically_stable_proxy = {ground.dynamically_stable_proxy}).",
    ]
    if near_degenerate:
        summary_lines.append(
            f"WARNING: {len(near_degenerate)} other phase(s) within "
            f"{degeneracy_tol_eV_per_atom:.3f} eV/atom; ground-state assignment is uncertain."
        )
    summary_lines.append(
        f"Chemical-stability proxy: {'PASS' if chem_stable else 'INCONCLUSIVE'}."
    )

    return StabilityReport(
        formula=formula,
        ground_state=ground,
        ranking=ranking,
        chemically_stable_proxy=chem_stable,
        summary="\n".join(summary_lines),
    )
