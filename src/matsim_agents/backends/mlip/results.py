"""Result types produced by MLIP relaxations."""

from __future__ import annotations

from pydantic import BaseModel


class RelaxationResult(BaseModel):
    """Outcome of a single atomistic structure relaxation."""

    structure_path: str
    optimized_structure_path: str
    trajectory_path: str
    log_csv_path: str
    final_energy_eV: float
    final_max_force_eV_per_A: float
    num_steps: int
    converged: bool
    top_branch: int | None = None
    top_branch_weight: float | None = None
    notes: str | None = None
