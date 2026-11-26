"""Cell-size and boundary diagnostics for a cascade run.

Heuristics, reported as findings and never as proof of adequacy. Everything is measured from
a displacement trajectory against the starting structure: how far atoms moved from their
start (the damaged region), whether that region reaches the periodic images or the
thermostatted layer, and whether kinetic energy reappears at the PKA's far side
(recirculation). Thresholds are caller-set because they depend on the material.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms


@dataclass(frozen=True)
class BoundaryReport:
    ok: bool
    findings: tuple[str, ...]
    max_displacement_A: float
    damaged_extent_A: float
    min_cell_length_A: float
    n_in_boundary_displaced: int
    reservoir_fraction: float


def _mic_displacement(start: Atoms, end: Atoms) -> np.ndarray:
    d = end.positions - start.positions
    cell = np.asarray(start.cell.array)
    frac = np.linalg.solve(cell.T, d.T).T
    frac -= np.round(frac) * np.asarray(start.pbc, dtype=float)
    return frac @ cell


def assess_cascade_boundaries(
    start: Atoms,
    end: Atoms,
    *,
    pka_index: int,
    displaced_threshold_A: float = 0.5,
    extent_cell_fraction: float = 0.5,
    boundary_mask: np.ndarray | None = None,
    min_reservoir_fraction: float = 0.2,
    far_side_ke_eV: float | None = None,
    far_side_ke_limit_eV: float | None = None,
) -> BoundaryReport:
    if len(start) != len(end):
        raise ValueError("start and end must have the same atoms")
    disp = np.linalg.norm(_mic_displacement(start, end), axis=1)
    displaced = disp > displaced_threshold_A
    findings: list[str] = []
    lengths = np.asarray(start.cell.lengths())
    pbc = np.asarray(start.pbc, dtype=bool)
    min_len = float(lengths[pbc].min()) if pbc.any() else float("inf")
    extent = 0.0
    if displaced.any():
        p = start.positions - start.positions[pka_index]
        frac = np.linalg.solve(np.asarray(start.cell.array).T, p.T).T
        frac -= np.round(frac) * pbc.astype(float)
        dist = np.linalg.norm(frac @ np.asarray(start.cell.array), axis=1)
        extent = float(dist[displaced].max())
    if np.isfinite(min_len) and extent > extent_cell_fraction * min_len:
        findings.append(
            f"damaged region reaches {extent:.1f} A from the PKA, more than "
            f"{extent_cell_fraction:.0%} of the shortest periodic length {min_len:.1f} A: "
            "interaction with periodic images is likely"
        )
    n_b = 0
    reservoir = 0.0
    if boundary_mask is not None:
        reservoir = float(np.mean(boundary_mask))
        n_b = int(np.sum(displaced & boundary_mask))
        if n_b:
            findings.append(f"{n_b} atoms in the thermostatted layer were displaced")
        if reservoir < min_reservoir_fraction:
            findings.append(
                f"thermal reservoir holds {reservoir:.0%} of the atoms, "
                f"below the {min_reservoir_fraction:.0%} minimum"
            )
    if (
        far_side_ke_eV is not None
        and far_side_ke_limit_eV is not None
        and far_side_ke_eV > far_side_ke_limit_eV
    ):
        findings.append("kinetic energy recirculates to the far side of the cell")
    return BoundaryReport(
        ok=not findings,
        findings=tuple(findings),
        max_displacement_A=float(disp.max()) if len(disp) else 0.0,
        damaged_extent_A=extent,
        min_cell_length_A=min_len,
        n_in_boundary_displaced=n_b,
        reservoir_fraction=reservoir,
    )


def minimum_cell_length_A(
    damaged_extent_A: float, *, safety: float = 2.0, extent_cell_fraction: float = 0.5
) -> float:
    """Shortest periodic length for which a damaged region of this extent passes the check."""
    if damaged_extent_A < 0 or safety < 1:
        raise ValueError("need a non-negative extent and safety >= 1")
    return safety * damaged_extent_A / (2 * extent_cell_fraction)


__all__ = ["BoundaryReport", "assess_cascade_boundaries", "minimum_cell_length_A"]
