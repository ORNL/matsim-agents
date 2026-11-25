"""PKA-informed candidate ranking (no per-PKA DFT jobs, no surviving-defect claims).

A species-resolved PKA spectrum says which recoiling species matter. This module uses that
only as one input to a transparent greedy ranking of candidates (for example defect
environments to compute with DFT). Terms:

* ``production``: the species' share of total PKA production (isotopes or elements);
* ``representativeness``: diminishing return for covering a species that is already selected;
* ``kinetic_importance``: user-supplied weight of the candidate in the kinetic model (default 1);
* ``uncertainty``: user-supplied model uncertainty (default 1);
* cost: node-hours, so the score is information per node-hour.

The weights are a policy, not a fitted result: nothing here is shown to beat random
selection. Surviving-defect production is reported as ``NOT_ESTABLISHED``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.science.irradiation.contracts import DefectStatus, PKASpectrum


class PKACandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    species: str = Field(min_length=1)  # isotope ("W182") or element ("W") key
    node_hours: float = Field(gt=0.0)
    kinetic_importance: float = Field(default=1.0, ge=0.0)
    uncertainty: float = Field(default=1.0, ge=0.0)


class PKAScoredSelection(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    species: str
    production: float
    representativeness: float
    score: float
    node_hours: float


class PKAAcquisitionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    selected: tuple[PKAScoredSelection, ...]
    unmatched_candidates: tuple[str, ...]
    uncovered_species: tuple[str, ...]
    budget_node_hours: float
    used_node_hours: float
    surviving_defect_production: DefectStatus = DefectStatus.NOT_ESTABLISHED
    note: str = (
        "ranking policy only; not shown to outperform random selection; PKA production is not "
        "surviving-defect production"
    )


def _production_by_key(spectrum: PKASpectrum) -> dict[str, float]:
    out = dict(spectrum.species_probabilities())
    for element, rate in spectrum.element_rates().items():
        out.setdefault(element, rate / spectrum.total_rate)
    return out


def select_pka_informed(
    spectrum: PKASpectrum,
    candidates: Sequence[PKACandidate],
    budget_node_hours: float,
    *,
    batch_size: int | None = None,
) -> PKAAcquisitionResult:
    """Greedy, deterministic ranking. Ties break by ``candidate_id``."""
    if not budget_node_hours > 0:
        raise ValueError("budget_node_hours must be positive")
    production = _production_by_key(spectrum)
    unmatched = tuple(sorted(c.candidate_id for c in candidates if c.species not in production))
    pool = [c for c in candidates if c.species in production]
    taken: dict[str, int] = {}
    selected: list[PKAScoredSelection] = []
    used = 0.0
    while pool and (batch_size is None or len(selected) < batch_size):
        best = None
        for c in pool:
            if used + c.node_hours > budget_node_hours:
                continue
            rep = 1.0 / (1.0 + taken.get(c.species, 0))
            value = production[c.species] * rep * c.kinetic_importance * c.uncertainty
            score = value / c.node_hours
            key = (-score, c.candidate_id)
            if best is None or key < best[0]:
                best = (key, c, rep, score)
        if best is None:
            break
        _, c, rep, score = best
        selected.append(
            PKAScoredSelection(
                candidate_id=c.candidate_id,
                species=c.species,
                production=production[c.species],
                representativeness=rep,
                score=score,
                node_hours=c.node_hours,
            )
        )
        taken[c.species] = taken.get(c.species, 0) + 1
        used += c.node_hours
        pool.remove(c)
    uncovered = tuple(
        sorted(
            s.species.key
            for s in spectrum.spectra
            if s.species.key not in taken and s.species.element not in taken
        )
    )
    return PKAAcquisitionResult(
        selected=tuple(selected),
        unmatched_candidates=unmatched,
        uncovered_species=uncovered,
        budget_node_hours=budget_node_hours,
        used_node_hours=math.fsum(s.node_hours for s in selected),
    )


__all__ = [
    "PKAAcquisitionResult",
    "PKACandidate",
    "PKAScoredSelection",
    "select_pka_informed",
]
