import json
from pathlib import Path

import pytest

from benchmarks.e4.aurora_dft import scaling

REPO = Path(__file__).resolve().parents[1]
PLANS = REPO / "research/paper/e4/aurora-dft-scaling/priority-scaling"


def test_strong_tiers_respect_queue_walltime_and_segment_eight_tasks():
    manifest = scaling.validate_manifest(PLANS / "strong-manifest.json")
    tiers = {tier["nodes"]: tier for tier in manifest["tiers"]}
    one_node = tiers[1]
    assert one_node["walltime_cap_hours"] == 96
    assert one_node["node_hours_cap"] == 192
    assert [segment["segment_id"] for segment in one_node["segments"]] == [
        "segment-1",
        "segment-2",
    ]
    assert scaling.dispatch_plan(manifest, 1, "segment-1") == [
        [task_id] for task_id in scaling.STRONG_IDS[:4]
    ]
    assert scaling.dispatch_plan(manifest, 1, "segment-2") == [
        [task_id] for task_id in scaling.STRONG_IDS[4:]
    ]
    for nodes in (2, 4, 8):
        assigned = [
            task_id for wave in scaling.dispatch_plan(manifest, nodes, None) for task_id in wave
        ]
        assert sorted(assigned) == sorted(scaling.STRONG_IDS)
        assert len(set(assigned)) == 8
    assert all(tier["walltime_cap_hours"] <= 168 for tier in tiers.values())


def test_weak_tiers_use_unique_frozen_tasks_sequentially():
    manifest = scaling.validate_manifest(PLANS / "weak-manifest.json")
    tiers = {tier["nodes"]: tier for tier in manifest["tiers"]}
    for nodes, tier in tiers.items():
        assigned = [task_id for node in tier["node_assignments"] for task_id in node]
        assert len(assigned) == 2 * nodes
        assert len(set(assigned)) == len(assigned)
        assert tier["walltime_cap_hours"] == 48
        assert tier["node_hours_cap"] == 48 * nodes
    assert len(manifest["available_unique_case_ids"]) == 16
    assert len(manifest["additional_case_ids"]) == 8
    assert scaling.dispatch_plan(manifest, 8, None) == [
        [pair[0] for pair in scaling.WEAK_PAIRS],
        [pair[1] for pair in scaling.WEAK_PAIRS],
    ]


def test_dispatch_requires_separate_authorization_before_pbs(monkeypatch, tmp_path):
    manifest_path = PLANS / "strong-manifest.json"
    with pytest.raises(PermissionError, match="separate E4 scaling authorization"):
        scaling.dispatch(manifest_path, 1, "segment-1", dry_run=False)

    manifest = json.loads(manifest_path.read_text())
    manifest["execution_authorized"] = True
    manifest["authorization"] = {"id": "approved-resource-request"}
    authorized_path = tmp_path / "authorized-test-manifest.json"
    authorized_path.write_text(json.dumps(manifest))
    monkeypatch.setenv("E4_SCALING_AUTHORIZATION", "approved-resource-request")
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("PBS_NODEFILE", raising=False)
    with pytest.raises(RuntimeError, match="inside its PBS allocation"):
        scaling.dispatch(authorized_path, 1, "segment-1", dry_run=False)
