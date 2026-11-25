"""Seeded sampling, validation independence and stopping rules (M14)."""

from __future__ import annotations

import pytest

from matsim_agents.discovery.exploration.sampling import (
    CampaignState,
    StopRule,
    record_round,
    sample_adaptive,
    sample_random,
    sample_stratified,
    should_stop,
    split_validation,
    stratum_counts,
)

IDS = [f"c{i:03d}" for i in range(100)]
STRATA = {c: ("rare" if i < 3 else "mid" if i < 23 else "common") for i, c in enumerate(IDS)}


def test_random_is_reproducible_and_seed_sensitive_with_metadata():
    a = sample_random(IDS, 10, seed=1)
    assert a == sample_random(IDS, 10, seed=1)
    assert [s.candidate_id for s in a] != [s.candidate_id for s in sample_random(IDS, 10, seed=2)]
    assert all(s.strategy == "random" and s.seed == 1 and s.inclusion_probability == 0.1 for s in a)
    assert len({s.candidate_id for s in a}) == 10
    assert sample_random(IDS, 1000, seed=1)[0].inclusion_probability == 1.0


def test_stratified_covers_rare_strata_and_respects_n():
    picks = sample_stratified(STRATA, 10, seed=4)
    counts = stratum_counts(picks)
    assert len(picks) == 10 and counts["rare"] >= 1 and counts["mid"] >= 1
    assert counts["common"] > counts["rare"]
    assert picks == sample_stratified(STRATA, 10, seed=4)
    for s in picks:
        members = [c for c, st in STRATA.items() if st == s.stratum]
        assert s.inclusion_probability == pytest.approx(counts[s.stratum] / len(members))


def test_equal_allocation_and_fewer_picks_than_strata():
    equal = stratum_counts(sample_stratified(STRATA, 9, seed=0, allocation="equal"))
    assert equal == {"rare": 3, "mid": 3, "common": 3}
    two = stratum_counts(sample_stratified(STRATA, 2, seed=0))
    assert sum(two.values()) == 2 and "rare" not in two  # largest strata kept
    assert sample_stratified({}, 5, seed=0) == []


def test_adaptive_takes_top_scores_plus_known_probability_exploration():
    scores = {c: float(i) for i, c in enumerate(IDS)}
    picks = sample_adaptive(scores, STRATA, 10, seed=5, round_index=2, exploration_fraction=0.2)
    exploit = {s.candidate_id for s in picks if s.strategy == "adaptive"}
    explore = [s for s in picks if s.strategy == "adaptive_exploration"]
    assert exploit == set(IDS[-8:])
    assert len(explore) == 2 and all(s.candidate_id not in exploit for s in explore)
    assert all(s.inclusion_probability == pytest.approx(2 / 92) for s in explore)
    assert picks == sample_adaptive(scores, STRATA, 10, seed=5, round_index=2)
    assert all(s.round == 2 for s in picks)
    with pytest.raises(ValueError):
        sample_adaptive(scores, STRATA, 5, seed=0, round_index=0, exploration_fraction=1.5)


def test_validation_split_is_disjoint_from_every_sampler():
    split = split_validation(IDS, 0.2, seed=9)
    assert len(split.validation) == 20 and not set(split.validation) & set(split.available)
    assert set(split.validation) | set(split.available) == set(IDS)
    assert split == split_validation(list(reversed(IDS)), 0.2, seed=9)
    avail_strata = {c: STRATA[c] for c in split.available}
    drawn = {s.candidate_id for s in sample_random(split.available, 30, seed=1)}
    drawn |= {s.candidate_id for s in sample_stratified(avail_strata, 30, seed=1)}
    scores = {c: 1.0 for c in split.available}
    drawn |= {
        s.candidate_id for s in sample_adaptive(scores, avail_strata, 30, seed=1, round_index=0)
    }
    assert not drawn & set(split.validation)
    with pytest.raises(ValueError):
        split_validation(IDS, 1.0, seed=0)


def test_stopping_rules_report_their_reason():
    state = CampaignState()
    assert should_stop(StopRule(max_evaluations=5), state) == (False, None)
    picks = sample_random(IDS, 6, seed=0)
    record_round(state, picks, max_score=0.9)
    assert should_stop(StopRule(max_evaluations=5), state) == (True, "evaluation budget reached")
    assert should_stop(StopRule(max_rounds=1), state)[1] == "round limit reached"
    assert should_stop(StopRule(target_uncertainty=0.5), state) == (False, None)
    record_round(state, picks, max_score=0.4)
    assert should_stop(StopRule(target_uncertainty=0.5), state)[1] == "uncertainty target met"
    assert should_stop(StopRule(stable_rounds=1), state)[1] == "no new stratum discovered"


def test_plateau_rule_needs_history_and_detects_no_improvement():
    state = CampaignState(metric_history=[0.5, 0.4])
    rule = StopRule(plateau_rounds=2)
    assert should_stop(rule, state) == (False, None)  # not enough history
    state.metric_history = [0.5, 0.3, 0.31, 0.32]
    assert should_stop(rule, state) == (True, "tracked metric has plateaued")
    state.metric_history = [0.5, 0.4, 0.3, 0.2]
    assert should_stop(rule, state) == (False, None)
