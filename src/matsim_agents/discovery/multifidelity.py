"""Multifidelity screening of magnetic candidates (M15).

F0 descriptors, F1 MLIP (geometry and energy only), F2 spin-polarized DFT, F3 NEB.
A candidate is promoted by recorded evidence, never by fidelity order alone. Magnetic claims need
F2 or higher, because a standard MLIP carries no spin degree of freedom here. Cost accounting
records only what was actually spent.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

Fidelity = Literal["F0", "F1", "F2", "F3"]
ORDER: tuple[Fidelity, ...] = ("F0", "F1", "F2", "F3")
MAGNETIC_CAPABLE: frozenset[str] = frozenset({"F2", "F3"})
F1_LIMITATION = (
    "MLIP evidence is geometry and energy only; it cannot confirm, rank or rule out magnetic states"
)


@dataclass(frozen=True)
class FidelityEvidence:
    candidate_id: str
    fidelity: Fidelity
    value: float | None  # e.g. energy per atom, descriptor distance, barrier; None = no result
    converged: bool
    cost: float  # in the campaign's cost unit, as measured (not estimated)
    uncertainty: float | None = None
    magnetic_state_resolved: bool = False
    note: str = ""


@dataclass(frozen=True)
class PromotionRule:
    """Thresholds are per source fidelity and are inputs, not defaults claimed as qualified."""

    source: Fidelity
    target: Fidelity
    max_value: float | None = None  # promote when value <= max_value (lower is better)
    max_uncertainty: float | None = None  # reject (hold) when uncertainty is larger
    top_fraction: float | None = None  # alternatively keep the best fraction


@dataclass(frozen=True)
class PromotionDecision:
    candidate_id: str
    source: Fidelity
    target: Fidelity | None
    promoted: bool
    reason: str


def _allowed(rule: PromotionRule) -> None:
    if ORDER.index(rule.target) <= ORDER.index(rule.source):
        raise ValueError("a promotion must go to a higher fidelity")


def decide_promotions(
    evidence: Sequence[FidelityEvidence], rule: PromotionRule
) -> list[PromotionDecision]:
    """Evidence-based decisions for candidates evaluated at ``rule.source``."""
    _allowed(rule)
    at_source = [e for e in evidence if e.fidelity == rule.source]
    usable = [e for e in at_source if e.converged and e.value is not None]
    cutoff: float | None = rule.max_value
    if rule.top_fraction is not None and usable:
        ranked = sorted(usable, key=lambda e: (e.value, e.candidate_id))
        keep = max(1, int(round(rule.top_fraction * len(ranked))))
        fraction_cutoff = ranked[keep - 1].value
        cutoff = fraction_cutoff if cutoff is None else min(cutoff, fraction_cutoff)
    decisions: list[PromotionDecision] = []
    for e in sorted(at_source, key=lambda x: x.candidate_id):
        if not e.converged or e.value is None:
            decisions.append(
                PromotionDecision(e.candidate_id, rule.source, None, False, "no converged result")
            )
        elif (
            rule.max_uncertainty is not None
            and e.uncertainty is not None
            and e.uncertainty > rule.max_uncertainty
        ):
            decisions.append(
                PromotionDecision(
                    e.candidate_id,
                    rule.source,
                    rule.target,
                    True,
                    "uncertainty above limit; escalated for a better estimate",
                )
            )
        elif cutoff is None or e.value <= cutoff:
            decisions.append(
                PromotionDecision(e.candidate_id, rule.source, rule.target, True, "within cutoff")
            )
        else:
            decisions.append(
                PromotionDecision(e.candidate_id, rule.source, None, False, "above cutoff")
            )
    return decisions


def magnetic_claims_allowed(evidence: Iterable[FidelityEvidence]) -> tuple[bool, str]:
    """Magnetic conclusions need converged F2+ evidence that resolved the magnetic state."""
    rows = list(evidence)
    good = [
        e
        for e in rows
        if e.fidelity in MAGNETIC_CAPABLE and e.converged and e.magnetic_state_resolved
    ]
    if good:
        return True, f"{len(good)} converged spin-polarized evaluation(s)"
    if any(e.fidelity == "F1" for e in rows):
        return False, F1_LIMITATION
    return False, "no spin-polarized evidence"


@dataclass
class CostLedger:
    """Sums measured cost per fidelity; never estimates or back-fills."""

    unit: str = "node_hours"
    spent: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    wasted: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def record(self, e: FidelityEvidence) -> None:
        if e.cost < 0:
            raise ValueError("cost cannot be negative")
        self.spent[e.fidelity] += e.cost
        self.counts[e.fidelity] += 1
        if not e.converged:
            self.wasted[e.fidelity] += e.cost

    def total(self) -> float:
        return float(sum(self.spent.values()))

    def summary(self) -> Mapping[str, Mapping[str, float]]:
        return {
            f: {
                "evaluations": float(self.counts[f]),
                "cost": self.spent[f],
                "wasted_on_unconverged": self.wasted[f],
                "mean_cost": self.spent[f] / self.counts[f],
            }
            for f in ORDER
            if self.counts[f]
        }


def budget_remaining(ledger: CostLedger, budget: float) -> float:
    return budget - ledger.total()


def savings_versus_all_f2(
    ledger: CostLedger, *, n_candidates: int, measured_f2_mean_cost: float
) -> float | None:
    """Cost saved relative to sending every candidate to F2 at the measured mean F2 cost.

    ``None`` when no F2 cost has been measured, since any figure would be invented.
    """
    if measured_f2_mean_cost <= 0 or ledger.counts.get("F2", 0) == 0:
        return None
    return n_candidates * measured_f2_mean_cost - ledger.total()
