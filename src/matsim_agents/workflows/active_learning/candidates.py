"""Sampling-method-independent candidate contract shared by MD, NEB and later samplers.

Uncertainty scoring, acquisition, DFT labelling and provenance consume only this type, so a new
sampler plugs in by emitting ``Candidate`` objects with its method-specific facts in
``metadata``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ase import Atoms


@dataclass
class Candidate:
    """One configuration proposed for labelling.

    ``seed_path`` and ``md_step`` keep the original MD meaning and stay valid for MD candidates;
    other methods leave them at their defaults and describe themselves in ``metadata``.
    """

    candidate_id: str
    atoms: Atoms
    seed_path: str = ""
    md_step: int | None = None
    predicted_energy_eV: float | None = None
    predicted_max_force_eV_per_A: float | None = None
    predicted_forces_eV_per_A: list[list[float]] | None = None
    sampling_method: str = "md"
    parent_id: str | None = None
    sampling_run_id: str | None = None
    geometry_fingerprint: str | None = None
    model_id: str | None = None
    uncertainty: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["Candidate"]
