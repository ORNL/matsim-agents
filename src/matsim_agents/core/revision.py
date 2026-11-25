"""Shared budget for evidence-gated revisions."""

from __future__ import annotations

from dataclasses import dataclass

MAX_RATIONALE_CHARS = 2000


class RevisionBudgetError(RuntimeError):
    """The revision budget is spent."""


@dataclass
class RevisionBudget:
    """Bounds on revisions per hypothesis and on optional LLM calls."""

    max_revisions: int = 5
    max_llm_calls: int = 0
    llm_calls_used: int = 0

    def __post_init__(self) -> None:
        if self.max_revisions < 1 or self.max_llm_calls < 0 or self.llm_calls_used < 0:
            raise ValueError("budgets must be positive integers (LLM calls may be zero)")

    @property
    def llm_calls_left(self) -> int:
        return max(self.max_llm_calls - self.llm_calls_used, 0)


__all__ = ["MAX_RATIONALE_CHARS", "RevisionBudget", "RevisionBudgetError"]
