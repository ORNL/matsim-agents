"""Hard node-hour budget ledger for the hierarchical DFT campaign.

Accounting states of one calculation attempt:

* ``reserved``  -- approved for dispatch, not yet submitted (counts predicted cost)
* ``committed`` -- submitted and running (counts predicted cost)
* ``consumed``  -- finished, converged or not (counts the actual cost)
* ``released``  -- cancelled before any node time was used (counts nothing)

Failed and unconverged jobs are ``consumed``. A retry needs a reason and an authorizer and is a
separate attempt. A non-zero validation reserve may only be drawn by validation categories.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.core.fileio import atomic_write_text

Category = Literal[
    "lower_foundations",
    "bridges",
    "six_component_validation",
    "magnetic",
    "defect",
    "neb",
    "convergence_validation",
    "property",
]
CATEGORIES: tuple[str, ...] = (
    "lower_foundations",
    "bridges",
    "six_component_validation",
    "magnetic",
    "defect",
    "neb",
    "convergence_validation",
    "property",
)
VALIDATION_CATEGORIES = frozenset({"six_component_validation", "convergence_validation"})
State = Literal["reserved", "committed", "consumed", "released"]
Outcome = Literal["converged", "failed", "unconverged"]


class BudgetError(RuntimeError):
    """Base class for refused budget operations."""


class BudgetExceeded(BudgetError):
    """The request would exceed the uncommitted budget (total, category or reserve rules)."""


class DuplicateCalculation(BudgetError):
    """An active attempt of this calculation already exists."""


class RetryNotAuthorized(BudgetError):
    """A retry lacks a reason or an authorizer."""


class LedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calc_id: str
    attempt: int = Field(1, ge=1)
    category: Category
    state: State
    predicted_node_hours: float = Field(ge=0.0)
    actual_node_hours: float | None = Field(None, ge=0.0)
    outcome: Outcome | None = None
    job_id: str | None = None
    retry_reason: str | None = None
    authorized_by: str | None = None

    @property
    def key(self) -> str:
        return f"{self.calc_id}#{self.attempt}"

    @property
    def amount(self) -> float:
        if self.state == "released":
            return 0.0
        if self.state == "consumed":
            return float(self.actual_node_hours or 0.0)
        return self.predicted_node_hours


class BudgetSummary(BaseModel):
    total: float
    consumed: float
    reserved: float
    committed: float
    predicted: float
    actual: float
    remaining: float
    validation_reserve: float
    validation_reserve_remaining: float
    by_category: dict[str, dict[str, float]]


class BudgetLedger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_node_hours: float = Field(gt=0.0)
    validation_reserve_fraction: float = Field(0.15, gt=0.0, lt=1.0)
    allocation: dict[str, float] = Field(default_factory=dict)
    entries: list[LedgerEntry] = Field(default_factory=list)
    reallocation_log: list[dict[str, float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> BudgetLedger:
        if not self.allocation:
            self.allocation = self.default_allocation()
        unknown = set(self.allocation) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"unknown budget categories: {sorted(unknown)}")
        if any(v < 0 for v in self.allocation.values()):
            raise ValueError("allocations must be non-negative")
        budget = self.total_node_hours - self.validation_reserve
        if sum(self.allocation.values()) > budget + 1e-9:
            raise ValueError("category allocations exceed the budget net of the validation reserve")
        for cat in CATEGORIES:
            self.allocation.setdefault(cat, 0.0)
        return self

    # --- allocation -----------------------------------------------------------------------
    @property
    def validation_reserve(self) -> float:
        return self.total_node_hours * self.validation_reserve_fraction

    def default_allocation(self) -> dict[str, float]:
        weights = {
            "lower_foundations": 0.35,
            "bridges": 0.25,
            "six_component_validation": 0.0,
            "magnetic": 0.10,
            "defect": 0.15,
            "neb": 0.10,
            "convergence_validation": 0.05,
            "property": 0.0,  # property characterization is funded only by explicit allocation
        }
        pool = self.total_node_hours - self.validation_reserve
        return {k: pool * v for k, v in weights.items()}

    # --- accounting -----------------------------------------------------------------------
    def _used(self, category: str) -> float:
        return sum(e.amount for e in self.entries if e.category == category)

    def _reserve_drawn(self) -> float:
        return sum(max(0.0, self._used(c) - self.allocation[c]) for c in VALIDATION_CATEGORIES)

    def uncommitted(self) -> float:
        """Total node-hours not yet reserved, committed or consumed."""
        return self.total_node_hours - sum(e.amount for e in self.entries)

    def available(self, category: str) -> float:
        """Most a new request in ``category`` may cost (category, reserve and total rules)."""
        own = max(0.0, self.allocation[category] - self._used(category))
        if category in VALIDATION_CATEGORIES:
            own += max(0.0, self.validation_reserve - self._reserve_drawn())
        return max(0.0, min(own, self.uncommitted()))

    def can_dispatch(self, category: str, predicted_node_hours: float) -> bool:
        return predicted_node_hours <= self.available(category) + 1e-9

    def _active(self, calc_id: str) -> LedgerEntry | None:
        for e in self.entries:
            if e.calc_id == calc_id and e.state in ("reserved", "committed"):
                return e
        return None

    def _latest(self, calc_id: str) -> LedgerEntry | None:
        found = [e for e in self.entries if e.calc_id == calc_id]
        return max(found, key=lambda e: e.attempt) if found else None

    def reserve(self, calc_id: str, category: str, predicted_node_hours: float) -> LedgerEntry:
        if category not in CATEGORIES:
            raise BudgetError(f"unknown category {category!r}")
        if predicted_node_hours < 0:
            raise BudgetError("predicted cost must be non-negative")
        if self._latest(calc_id) is not None:
            raise DuplicateCalculation(
                f"{calc_id} already has a ledger entry; a rerun must use retry() with a reason"
            )
        self._guard(category, predicted_node_hours, calc_id)
        entry = LedgerEntry(
            calc_id=calc_id,
            category=category,  # type: ignore[arg-type]
            state="reserved",
            predicted_node_hours=predicted_node_hours,
        )
        self.entries.append(entry)
        return entry

    def _guard(self, category: str, cost: float, calc_id: str) -> None:
        if cost > self.available(category) + 1e-9:
            raise BudgetExceeded(
                f"{calc_id}: {cost:.1f} node-hours exceeds available "
                f"{self.available(category):.1f} in {category} "
                f"(uncommitted total {self.uncommitted():.1f})"
            )

    def commit(self, calc_id: str, job_id: str | None = None) -> LedgerEntry:
        entry = self._active(calc_id)
        if entry is None or entry.state != "reserved":
            raise BudgetError(f"{calc_id} has no reserved attempt to submit")
        entry.state = "committed"
        entry.job_id = job_id
        return entry

    def record_result(
        self, calc_id: str, actual_node_hours: float, outcome: Outcome = "converged"
    ) -> LedgerEntry:
        """Failed and unconverged jobs are consumed exactly like converged ones."""
        entry = self._active(calc_id)
        if entry is None:
            raise BudgetError(f"{calc_id} has no active attempt")
        entry.state = "consumed"
        entry.actual_node_hours = actual_node_hours
        entry.outcome = outcome
        return entry

    def release(self, calc_id: str) -> LedgerEntry:
        entry = self._active(calc_id)
        if entry is None or entry.state != "reserved":
            raise BudgetError(f"{calc_id} has no unsubmitted reservation to release")
        entry.state = "released"
        return entry

    def retry(
        self, calc_id: str, predicted_node_hours: float, *, reason: str, authorized_by: str
    ) -> LedgerEntry:
        if not reason.strip() or not authorized_by.strip():
            raise RetryNotAuthorized("a retry needs both a reason and an authorizer")
        last = self._latest(calc_id)
        if last is None:
            raise BudgetError(f"{calc_id} has never been attempted")
        if last.state in ("reserved", "committed"):
            raise DuplicateCalculation(f"{calc_id} still has an active attempt")
        if last.state == "consumed" and last.outcome == "converged":
            raise RetryNotAuthorized(f"{calc_id} converged; there is nothing to retry")
        self._guard(last.category, predicted_node_hours, calc_id)
        entry = LedgerEntry(
            calc_id=calc_id,
            attempt=last.attempt + 1,
            category=last.category,
            state="reserved",
            predicted_node_hours=predicted_node_hours,
            retry_reason=reason,
            authorized_by=authorized_by,
        )
        self.entries.append(entry)
        return entry

    def reallocate(self, weights: dict[str, float]) -> dict[str, float]:
        """Redistribute unspent category allocation by ``weights``; the reserve is untouched."""
        if any(c not in CATEGORIES for c in weights) or any(v < 0 for v in weights.values()):
            raise BudgetError("weights must be non-negative and use known categories")
        norm = sum(weights.values())
        if norm <= 0:
            raise BudgetError("at least one positive weight is required")
        base = {c: min(self._used(c), self.allocation[c]) for c in CATEGORIES}
        pool = self.total_node_hours - self.validation_reserve - sum(base.values())
        new = {c: base[c] + pool * weights.get(c, 0.0) / norm for c in CATEGORIES}
        self.reallocation_log.append({c: new[c] - self.allocation[c] for c in CATEGORIES})
        self.allocation = new
        return new

    def summary(self) -> BudgetSummary:
        def total(state: str) -> float:
            return sum(e.amount for e in self.entries if e.state == state)

        by_cat = {
            c: {
                "allocation": self.allocation[c],
                "used": self._used(c),
                "available": self.available(c),
            }
            for c in CATEGORIES
        }
        live = [e for e in self.entries if e.state != "released"]
        return BudgetSummary(
            total=self.total_node_hours,
            consumed=total("consumed"),
            reserved=total("reserved"),
            committed=total("committed"),
            predicted=sum(e.predicted_node_hours for e in live),
            actual=sum(e.actual_node_hours or 0.0 for e in live),
            remaining=self.uncommitted(),
            validation_reserve=self.validation_reserve,
            validation_reserve_remaining=max(0.0, self.validation_reserve - self._reserve_drawn()),
            by_category=by_cat,
        )

    # --- persistence ----------------------------------------------------------------------
    def save(self, path: str | os.PathLike) -> None:
        atomic_write_text(path, self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | os.PathLike) -> BudgetLedger:
        return cls.model_validate(json.loads(Path(path).read_text()))
