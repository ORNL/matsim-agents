"""Turn analysed cascade ensembles into the irradiation ``SurvivingDefects`` contract.

The contract's default answer is ``NOT_ESTABLISHED``. This adapter returns ``ESTABLISHED`` only
when the caller names where the structures came from (a cascade simulation at the PKA's species
and energy) and the ensemble has at least two realizations. It does not check that provenance, so
``ESTABLISHED`` here means "computed from the stated cascades", not validated.
"""

from __future__ import annotations

from collections.abc import Sequence

from matsim_agents.science.defects.ensemble import weighted_estimate
from matsim_agents.science.defects.survival import SurvivingDefectSummary
from matsim_agents.science.irradiation.contracts import DefectStatus, SurvivingDefects


def surviving_defects_from_ensemble(
    summaries: Sequence[SurvivingDefectSummary],
    weights: Sequence[float] | None = None,
    *,
    cascade_provenance: str | None,
    confidence: float = 0.95,
) -> SurvivingDefects:
    if not cascade_provenance or not cascade_provenance.strip():
        return SurvivingDefects(
            status=DefectStatus.NOT_ESTABLISHED,
            reason="no cascade provenance given; structures are not tied to a cascade simulation",
        )
    if len(summaries) < 2:
        return SurvivingDefects(
            status=DefectStatus.NOT_ESTABLISHED,
            reason=f"{len(summaries)} cascade realization(s); at least 2 are needed",
        )
    est = weighted_estimate([s.n_frenkel_pairs for s in summaries], weights, confidence=confidence)
    return SurvivingDefects(
        status=DefectStatus.ESTABLISHED,
        reason=(
            f"weighted mean over {est.n} cascade realizations (effective n {est.effective_n:.1f}), "
            f"{est.confidence:.0%} {est.method} interval [{est.ci_low:.3g}, {est.ci_high:.3g}], "
            f"assuming independent realizations; provenance: {cascade_provenance.strip()}; "
            "not validated against reference data"
        ),
        n_frenkel_pairs=est.mean,
    )


__all__ = ["surviving_defects_from_ensemble"]
