from __future__ import annotations

import pytest
from pydantic import ValidationError

from matsim_agents.science.acquisition.acquisition import (
    CandidateAcquisitionInputs,
    CandidateAcquisitionPolicy,
    expected_transfer_gain,
    select_candidate_batch,
)


def cand(cid, cost=10.0, **kw):
    kw.setdefault("calculation_id", f"calc-{cid}")
    return CandidateAcquisitionInputs(candidate_id=cid, cost_node_hours=cost, **kw)


MISSING = {f"pair1:A-{x}" for x in "BCDE"}


def pol(**kw):
    return CandidateAcquisitionPolicy(**{"batch_size": 3, **kw})


def test_selection_is_deterministic_and_order_independent():
    pool = [
        cand(c, relevance=r, uncertainty=u, transfer_gain=0.5)
        for c, r, u in [("a", 0.9, 0.2), ("b", 0.5, 0.8), ("c", 0.6, 0.6), ("d", 0.6, 0.6)]
    ]
    one = select_candidate_batch(pool, pol())
    two = select_candidate_batch(list(reversed(pool)), pol())
    assert one.selected == two.selected
    assert one.model_dump() == two.model_dump()


def test_equal_value_ties_break_on_candidate_id():
    pool = [cand("z", relevance=0.5), cand("y", relevance=0.5)]
    assert select_candidate_batch(pool, pol(batch_size=1)).selected == ["y"]


def test_cost_ranks_equal_value_candidates():
    pool = [cand("cheap", cost=5, relevance=0.8), cand("dear", cost=50, relevance=0.8)]
    sel = select_candidate_batch(pool, pol(batch_size=2, cost_floor_node_hours=1.0))
    assert sel.selected == ["cheap", "dear"]
    by = sel.by_id()
    assert by["cheap"].value_per_node_hour > by["dear"].value_per_node_hour


def test_a_cheap_but_valueless_calculation_is_never_selected_for_being_cheap():
    pool = [
        cand("valuable", cost=100, relevance=0.9, uncertainty=0.9, transfer_gain=0.9),
        cand("cheap", cost=1, relevance=0.01),
    ]
    sel = select_candidate_batch(pool, pol(batch_size=2))
    assert sel.selected == ["valuable"]
    assert sel.by_id()["cheap"].reason == "below_value_floor"
    zero = select_candidate_batch([cand("free", cost=1)], pol())
    assert zero.selected == []


def test_duplicates_are_rejected_against_history_and_within_the_batch():
    pool = [cand("a", relevance=0.9), cand("b", relevance=0.9, calculation_id="calc-a")]
    sel = select_candidate_batch(pool, pol())
    assert sel.selected == ["a"] and sel.by_id()["b"].reason == "duplicate_calculation"
    done = select_candidate_batch(pool, pol(), existing_calculation_ids={"calc-a"})
    assert done.selected == []
    assert {s.reason for s in done.scores} == {"duplicate_calculation"}


def test_incompatible_poorly_connected_and_saturated_are_rejected():
    pool = [
        cand("ok", region="r1", relevance=0.5),
        cand(
            "bad", region="r1", relevance=1.0, compatible=False, incompatibility_reason="PBE vs LDA"
        ),
        cand("iso", region="r1", relevance=1.0, connectivity=0.05),
        cand("full", region="sat", relevance=1.0),
    ]
    sel = select_candidate_batch(pool, pol(), region_counts={"sat": 25})
    reasons = {s.candidate_id: s.reason for s in sel.scores}
    assert sel.selected == ["ok"]
    assert reasons["bad"] == "incompatible: PBE vs LDA"
    assert reasons["iso"] == "poorly_connected" and reasons["full"] == "saturated_region"


def test_relevance_and_missing_interaction_coverage_drive_selection():
    pool = [
        cand("generic", relevance=0.2),
        cand("relevant", relevance=0.9),
        cand("bridge", relevance=0.3, newly_covered=frozenset(MISSING)),
    ]
    sel = select_candidate_batch(pool, pol(batch_size=1), missing_interactions=MISSING)
    assert sel.selected == ["bridge"]
    assert sel.by_id()["bridge"].newly_covered == sorted(MISSING)
    again = select_candidate_batch(pool[:2], pol(batch_size=1), missing_interactions=MISSING)
    assert again.selected == ["relevant"]


def test_repeated_coverage_is_not_rewarded_twice():
    shared = frozenset({"pair1:A-B"})
    pool = [
        cand("one", relevance=0.4, newly_covered=shared),
        cand("two", relevance=0.4, newly_covered=shared),
        cand("other", relevance=0.4, newly_covered=frozenset({"pair1:A-C"})),
    ]
    sel = select_candidate_batch(pool, pol(batch_size=2), missing_interactions=MISSING)
    assert "other" in sel.selected and len(sel.selected) == 2


def test_diversity_penalises_repeated_regions():
    pool = [cand(f"s{i}", region="same", relevance=0.5, diversity=0.8) for i in range(3)]
    pool.append(cand("new", region="new", relevance=0.5, diversity=0.8))
    sel = select_candidate_batch(pool, pol(batch_size=2))
    assert "new" in sel.selected


def test_unknown_transfer_gain_uses_a_visible_coverage_uncertainty_fallback():
    unknown = cand("u", relevance=0.5, uncertainty=0.9, newly_covered=frozenset(MISSING))
    known = cand("k", relevance=0.5, uncertainty=0.9, transfer_gain=0.0)
    sel = select_candidate_batch([unknown, known], pol(), missing_interactions=MISSING)
    by = sel.by_id()
    assert (
        by["u"].basis == "coverage+uncertainty fallback"
        and "transfer_gain" not in by["u"].components
    )
    assert by["k"].basis == "transfer-guided"
    assert by["u"].value > by["k"].value


def test_transfer_gain_mapping_never_invents_evidence():
    assert expected_transfer_gain("NOT_YET_ESTABLISHED") is None
    assert expected_transfer_gain("pending") is None
    assert expected_transfer_gain("unsupported") == 0.0
    assert expected_transfer_gain("validated", 0.3, scale=0.6) == pytest.approx(0.5)
    assert expected_transfer_gain("validated", 9.0) == 1.0
    assert expected_transfer_gain("validated") is None


def test_magnetic_and_defect_relevance_raise_value():
    base = cand("base", relevance=0.5)
    mag = cand("mag", relevance=0.5, magnetic_relevance=1.0, defect_relevance=1.0)
    assert select_candidate_batch([base, mag], pol(batch_size=1)).selected == ["mag"]


def test_batch_node_hour_cap_is_respected():
    pool = [cand(f"c{i}", cost=40, relevance=0.5 + i / 100) for i in range(5)]
    sel = select_candidate_batch(pool, pol(batch_size=5, max_batch_node_hours=100))
    assert len(sel.selected) == 2 and sel.total_node_hours == 80


def test_invalid_inputs_fail_loudly():
    with pytest.raises(ValidationError):
        cand("x", cost=0)
    with pytest.raises(ValidationError):
        cand("x", relevance=1.5)
    with pytest.raises(ValidationError):
        CandidateAcquisitionPolicy(unknown_switch=True)
