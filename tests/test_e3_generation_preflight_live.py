"""Opt-in CPU-only, full production-shaped generation (no mocked generators or inference)."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from benchmarks.e3 import production

pytestmark = pytest.mark.skipif(
    os.environ.get("MATSIM_E3_REAL_GENERATION_TEST") != "1",
    reason="set MATSIM_E3_REAL_GENERATION_TEST=1 for actual 64-candidate CPU generation",
)


@pytest.mark.parametrize("material", list(production.MATERIALS))
def test_actual_full_round_generation_for_both_policy_paths(tmp_path, material):
    manifest = production.build_campaign_manifest()
    spec = next(row for row in manifest["rounds"] if row["material"] == material)
    rule_spec = {**spec, "strategy": "rule_based_adaptation"}
    _, rule_tasks, rule_round = production._materialize_round(
        manifest,
        rule_spec,
        tmp_path,
        [],
        None,
    )
    deterministic = production._read_tasks(rule_tasks)
    assert len(deterministic) == 64
    proposal = json.dumps(
        {
            "allocations": [
                {"action_id": row["action"]["action_id"], "count": row["count"]}
                for row in rule_round["allocations"]
            ],
            "cited_evidence_ids": ["campaign-design"],
            "rationale": "Replay validated allocation through the LLM path without inference",
        }
    )
    _, llm_tasks, _ = production._materialize_round(
        manifest,
        spec,
        tmp_path,
        [],
        SimpleNamespace(
            model="offline-allocation-replay", invoke=lambda _: SimpleNamespace(content=proposal)
        ),
    )
    llm = production._read_tasks(llm_tasks)
    assert len(llm) == 64
    assert len({row["structure_sha256"] for row in llm}) == 64
    assert {row["structure_sha256"] for row in llm} == {
        row["structure_sha256"] for row in deterministic
    }
    assert not list(tmp_path.glob("**/tasks/**/result.json"))
