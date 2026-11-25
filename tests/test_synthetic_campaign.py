"""Synthetic scalability and recovery demonstrations (M19). Software evidence only."""

from __future__ import annotations

import numpy as np

from matsim_agents.discovery.exploration.candidates import CandidateManifest
from matsim_agents.discovery.exploration.demo import (
    compare_strategies,
    evaluate_pending,
    ingest_synthetic_candidates,
    make_candidate,
    run_scale_demo,
)


def test_candidates_are_reproducible_and_carry_their_index():
    a, atoms_a = make_candidate(7, seed=1)
    b, atoms_b = make_candidate(7, seed=1)
    assert a == b and atoms_a.get_chemical_formula() == atoms_b.get_chemical_formula()
    assert a.metadata["index"] == 7 and a.stratum.startswith("Cr")
    assert make_candidate(8, seed=1)[0].candidate_id != a.candidate_id


def test_ingest_is_idempotent(tmp_path):
    manifest = CandidateManifest(tmp_path / "m.jsonl")
    added, dup = ingest_synthetic_candidates(manifest, 50, seed=3)
    again_added, again_dup = ingest_synthetic_candidates(manifest, 50, seed=3)
    assert added + dup == 50 and again_added == 0 and again_dup == 50


def test_interrupted_run_recovers_without_repeating_finished_work(tmp_path):
    path = tmp_path / "m.jsonl"
    manifest = CandidateManifest(path)
    added, _ = ingest_synthetic_candidates(manifest, 40, seed=5)
    calls: dict[str, int] = {}
    finished = evaluate_pending(manifest, 5, crash_after=15, calls=calls)
    assert finished == 15
    reopened = CandidateManifest(path)
    assert reopened.counts()["done"] == 15 and reopened.counts()["running"] == 1
    assert reopened.recover_interrupted() != []
    evaluate_pending(reopened, 5, calls=calls)
    assert reopened.counts() == {"done": added}
    assert max(calls.values()) == 1  # no candidate evaluated twice
    assert len(calls) == added


def test_scale_demo_records_measured_numbers(tmp_path):
    stats = run_scale_demo(tmp_path / "scale.jsonl", 600, seed=2)
    assert stats.added + stats.duplicates == 600
    assert stats.counts == {"done": stats.added}
    assert stats.seconds > 0 and stats.peak_mib > 0
    assert set(stats.evaluations.values()) == {1}


def test_results_are_labelled_synthetic(tmp_path):
    manifest = CandidateManifest(tmp_path / "m.jsonl")
    ingest_synthetic_candidates(manifest, 5, seed=1)
    evaluate_pending(manifest, 1)
    results = [d["result"] for d in manifest.details.values()]
    assert results and all(r["label_source"] == "synthetic" for r in results)


def test_adaptive_vs_random_comparison_is_deterministic_and_reports_both():
    first = compare_strategies(seed=11)
    assert first == compare_strategies(seed=11)
    assert {r.strategy for r in first} == {"random", "adaptive"}
    assert all(r.budget == 80 and np.isfinite(r.mean_abs_error) for r in first)


def test_comparison_across_seeds_is_reported_without_assuming_a_winner():
    rows = [compare_strategies(seed=s) for s in range(5)]
    by = {name: [r.mean_abs_error for row in rows for r in row if r.strategy == name]
          for name in ("random", "adaptive")}  # fmt: skip
    assert len(by["random"]) == len(by["adaptive"]) == 5
    assert all(e >= 0 for errs in by.values() for e in errs)
