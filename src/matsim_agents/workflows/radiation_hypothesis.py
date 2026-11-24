"""Hypothesis revision for the radiation-damage workflow, with deterministic gates.

The status of a revision is decided by rules, never by a language model:

* no kinetics result: ``proposed`` (nothing has been tested);
* an exploratory ensemble from unqualified barriers: ``unresolved``;
* ``supported`` and ``contradicted`` need a qualified run, which this workflow cannot
  produce, so they are unreachable here.

Evidence must already be in the ledger as measured ``migration_barrier`` records for this
run's transition; a failed or unconverged calculation is not cited. An optional rationale
writer (an LLM callable) may draft the rationale text only. It cannot change the status, its
output is labelled, and it is bounded by the budget. A failing writer falls back to the
deterministic text.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from matsim_agents.execution.ledger import (
    EvidenceLedger,
    EvidenceOutcome,
    HypothesisRecord,
    HypothesisStatus,
)
from matsim_agents.workflows.radiation_damage import RadiationDamageConfig, RadiationDamageRun

RationaleWriter = Callable[[str], str]
MAX_RATIONALE_CHARS = 2000


class RevisionBudgetError(RuntimeError):
    """The revision budget is spent."""


class RevisionGateError(RuntimeError):
    """The inputs fail a deterministic gate."""


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


@dataclass(frozen=True)
class RevisionOutcome:
    record: HypothesisRecord
    used_llm: bool
    notes: tuple[str, ...]


def _claim(config: RadiationDamageConfig, transition_id: str) -> str:
    return (
        f"Vacancy-hop recovery of a seeded {'-'.join(sorted(config.counts))} BCC alloy at "
        f"{config.temperature_K:g} K, transition {transition_id}, follows the HTST kinetics of "
        "the supplied barrier."
    )


def _summary(run: RadiationDamageRun, status: HypothesisStatus) -> str:
    stages = ", ".join(f"{s.name}={s.status}" for s in run.state.stages)
    lines = [f"status={status.value}; stages: {stages}"]
    ensemble = run.ensemble
    if ensemble is not None:
        outcomes = {k: round(v.probability, 4) for k, v in ensemble.outcomes.items()}
        lines.append(
            f"{ensemble.n_trajectories} trajectories at {ensemble.temperature_K:g} K, "
            f"outcomes {outcomes or 'none absorbed'}, censored fraction "
            f"{ensemble.censored_fraction:.3f}"
        )
    lines.extend(f"blocker: {b}" for b in run.blockers)
    return "\n".join(lines)


def _cited_evidence(ledger: EvidenceLedger, transition_id: str) -> list[str]:
    return [
        e.evidence_id
        for e in ledger.evidence(subject=transition_id)
        if e.kind == "migration_barrier" and e.outcome == EvidenceOutcome.MEASURED
    ]


def revise_hypothesis(
    ledger: EvidenceLedger,
    hypothesis_id: str,
    config: RadiationDamageConfig,
    run: RadiationDamageRun,
    *,
    budget: RevisionBudget | None = None,
    rationale_writer: RationaleWriter | None = None,
    campaign_id: str | None = None,
) -> RevisionOutcome:
    """Append one revision of ``hypothesis_id`` reflecting ``run``.

    Raises ``RevisionBudgetError`` when the hypothesis already has ``max_revisions`` revisions
    and ``RevisionGateError`` when the run has no transition.
    """
    budget = budget or RevisionBudget()
    transition_id = run.state.transition_id
    if transition_id is None:
        raise RevisionGateError("the run has no transition; the structure stage did not finish")
    history = ledger.history(hypothesis_id)
    if len(history) >= budget.max_revisions:
        raise RevisionBudgetError(
            f"{hypothesis_id} already has {len(history)} of {budget.max_revisions} revisions"
        )

    notes: list[str] = []
    cited = _cited_evidence(ledger, transition_id)
    if run.ensemble is None:
        status = HypothesisStatus.PROPOSED
    else:
        status = HypothesisStatus.UNRESOLVED
        if not cited:
            notes.append("kinetics ran but no measured barrier evidence is in the ledger")

    summary = _summary(run, status)
    rationale = f"Deterministic summary (no qualification): {summary}"
    used_llm = False
    if rationale_writer is not None:
        if budget.llm_calls_left == 0:
            notes.append("LLM budget exhausted; deterministic rationale used")
        else:
            budget.llm_calls_used += 1
            try:
                draft = rationale_writer(summary).strip()
            except Exception as exc:  # a writer failure must not block the record
                notes.append(f"rationale writer failed ({type(exc).__name__}); deterministic")
            else:
                if draft:
                    used_llm = True
                    rationale += f"\n[LLM-drafted, unverified] {draft[:MAX_RATIONALE_CHARS]}"

    tests = ["qualified NEB barrier with relaxed endpoints and a climbing image"]
    tests.extend(b for b in run.blockers if "not tested" in b or "not assessed" in b)
    record = HypothesisRecord(
        hypothesis_id=hypothesis_id,
        prior_revision_id=history[-1].revision_id if history else None,
        campaign_id=campaign_id,
        claim_type="property",
        scientific_claim=_claim(config, transition_id),
        falsifiable_prediction=(
            "A qualified barrier changes the exploratory outcome probabilities beyond their "
            "Wilson intervals."
            if run.ensemble is not None
            else None
        ),
        status=status,
        evidence_ids=cited,
        support_level=ledger.support_level(cited),
        scientific_rationale=rationale,
        proposed_tests=tests,
    )
    stored = ledger.append(record)
    assert isinstance(stored, HypothesisRecord)
    return RevisionOutcome(record=stored, used_llm=used_llm, notes=tuple(notes))


__all__ = [
    "MAX_RATIONALE_CHARS",
    "RationaleWriter",
    "RevisionBudget",
    "RevisionBudgetError",
    "RevisionGateError",
    "RevisionOutcome",
    "revise_hypothesis",
]
