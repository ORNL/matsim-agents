"""Seeded random, stratified and adaptive candidate sampling.

Every selection carries the strategy, seed, round and inclusion probability, so a campaign can be
audited and re-weighted. The validation set is drawn first, uniformly at random, and is disjoint
from anything the samplers can return, so adaptive selection can never leak into it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from matsim_agents.core.seeding import numpy_rng

Strategy = Literal["random", "stratified", "adaptive", "adaptive_exploration"]


@dataclass(frozen=True)
class Selection:
    candidate_id: str
    strategy: Strategy
    seed: int
    round: int
    stratum: str
    inclusion_probability: float | None


@dataclass(frozen=True)
class PoolSplit:
    validation: tuple[str, ...]
    available: tuple[str, ...]
    seed: int


def split_validation(candidate_ids: Sequence[str], fraction: float, seed: int) -> PoolSplit:
    """Uniform random validation draw; independent of strata, scores and later selection."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation fraction must be in (0, 1)")
    ids = sorted(set(candidate_ids))
    n_val = min(max(1, round(fraction * len(ids))), len(ids) - 1) if len(ids) > 1 else 0
    chosen = set(numpy_rng(seed).choice(len(ids), size=n_val, replace=False).tolist())
    validation = tuple(ids[i] for i in sorted(chosen))
    available = tuple(c for i, c in enumerate(ids) if i not in chosen)
    return PoolSplit(validation, available, seed)


def sample_random(
    candidates: Sequence[str], n: int, *, seed: int, round_index: int = 0
) -> list[Selection]:
    ids = sorted(candidates)
    k = min(n, len(ids))
    rng = numpy_rng(seed, round_index)
    picks = sorted(rng.choice(len(ids), size=k, replace=False).tolist())
    p = k / len(ids) if ids else None
    return [Selection(ids[i], "random", seed, round_index, "all", p) for i in picks]


def sample_stratified(
    candidates: Mapping[str, str],
    n: int,
    *,
    seed: int,
    round_index: int = 0,
    allocation: Literal["proportional", "equal"] = "proportional",
) -> list[Selection]:
    """``candidates`` maps id -> stratum. Every non-empty stratum gets at least one pick when
    ``n`` allows, so rare strata are not lost to proportional rounding."""
    strata: dict[str, list[str]] = {}
    for cid, stratum in sorted(candidates.items()):
        strata.setdefault(stratum, []).append(cid)
    names = sorted(strata)
    total = len(candidates)
    if total == 0 or n <= 0:
        return []
    n = min(n, total)
    if allocation == "equal":
        weights = {s: 1.0 / len(names) for s in names}
    else:
        weights = {s: len(strata[s]) / total for s in names}
    quota = {s: min(len(strata[s]), max(1, int(round(weights[s] * n)))) for s in names}
    if n >= len(names):
        order = sorted(names, key=lambda s: (-weights[s], s))
        while sum(quota.values()) > n:
            donor = next((s for s in reversed(order) if quota[s] > 1), None)
            if donor is None:
                break
            quota[donor] -= 1
        while sum(quota.values()) < n:
            taker = next((s for s in order if quota[s] < len(strata[s])), None)
            if taker is None:
                break
            quota[taker] += 1
    else:  # fewer picks than strata: keep the largest strata
        keep = sorted(names, key=lambda s: (-len(strata[s]), s))[:n]
        quota = {s: (1 if s in keep else 0) for s in names}
    out: list[Selection] = []
    for index, stratum in enumerate(names):
        members = strata[stratum]
        k = quota[stratum]
        if k == 0:
            continue
        rng = numpy_rng(seed, round_index, index)
        picks = sorted(rng.choice(len(members), size=k, replace=False).tolist())
        for i in picks:
            out.append(
                Selection(members[i], "stratified", seed, round_index, stratum, k / len(members))
            )
    return sorted(out, key=lambda s: s.candidate_id)


def sample_adaptive(
    scores: Mapping[str, float],
    strata: Mapping[str, str],
    n: int,
    *,
    seed: int,
    round_index: int,
    exploration_fraction: float = 0.2,
) -> list[Selection]:
    """Highest-score picks (e.g. surrogate uncertainty) plus a random exploration share.

    Exploration picks are uniform over the remainder, so the sample keeps a known-probability part
    for unbiased estimates. Ties break by id for determinism.
    """
    if not 0.0 <= exploration_fraction <= 1.0:
        raise ValueError("exploration_fraction must be in [0, 1]")
    ids = sorted(scores)
    n = min(n, len(ids))
    n_explore = int(round(exploration_fraction * n))
    n_exploit = n - n_explore
    ranked = sorted(ids, key=lambda c: (-scores[c], c))
    exploit = ranked[:n_exploit]
    rest = ranked[n_exploit:]
    rng = numpy_rng(seed, round_index, 99)
    explore_idx = sorted(rng.choice(len(rest), size=min(n_explore, len(rest)), replace=False))
    explore = [rest[i] for i in explore_idx]
    p_explore = len(explore) / len(rest) if rest else None
    out = [Selection(c, "adaptive", seed, round_index, strata.get(c, "all"), None) for c in exploit]
    out += [
        Selection(c, "adaptive_exploration", seed, round_index, strata.get(c, "all"), p_explore)
        for c in explore
    ]
    return sorted(out, key=lambda s: s.candidate_id)


@dataclass(frozen=True)
class StopRule:
    max_evaluations: int | None = None
    max_rounds: int | None = None
    stable_rounds: int | None = None  # stop after this many rounds with no new stratum
    target_uncertainty: float | None = None  # stop when the max score is below this
    plateau_rounds: int | None = None  # stop when the tracked metric stops improving
    plateau_tolerance: float = 0.0


@dataclass
class CampaignState:
    evaluations: int = 0
    round_index: int = 0
    strata_seen: set[str] = field(default_factory=set)
    rounds_without_new_stratum: int = 0
    max_score: float | None = None
    metric_history: list[float] = field(default_factory=list)  # lower is better


def record_round(state: CampaignState, selections: Sequence[Selection], *, max_score=None) -> None:
    new = {s.stratum for s in selections} - state.strata_seen
    state.rounds_without_new_stratum = 0 if new else state.rounds_without_new_stratum + 1
    state.strata_seen |= new
    state.evaluations += len(selections)
    state.round_index += 1
    state.max_score = max_score


def should_stop(rule: StopRule, state: CampaignState) -> tuple[bool, str | None]:
    """First satisfied rule wins; the reason is reported, never silently swallowed."""
    checks: list[tuple[bool, str]] = [
        (
            rule.max_evaluations is not None and state.evaluations >= rule.max_evaluations,
            "evaluation budget reached",
        ),
        (
            rule.max_rounds is not None and state.round_index >= rule.max_rounds,
            "round limit reached",
        ),
        (
            rule.stable_rounds is not None
            and state.rounds_without_new_stratum >= rule.stable_rounds,
            "no new stratum discovered",
        ),
        (
            rule.target_uncertainty is not None
            and state.max_score is not None
            and state.max_score <= rule.target_uncertainty,
            "uncertainty target met",
        ),
        (_plateaued(rule, state), "tracked metric has plateaued"),
    ]
    for hit, reason in checks:
        if hit:
            return True, reason
    return False, None


def _plateaued(rule: StopRule, state: CampaignState) -> bool:
    k = rule.plateau_rounds
    history = state.metric_history
    if k is None or len(history) <= k:
        return False
    return min(history[-k:]) >= min(history[:-k]) - rule.plateau_tolerance


def stratum_counts(selections: Sequence[Selection]) -> dict[str, int]:
    return dict(Counter(s.stratum for s in selections))
