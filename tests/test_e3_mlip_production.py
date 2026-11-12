from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.e3.production import (
    E3_STRATEGIES,
    MATERIALS,
    REPLICATE_SEEDS,
    _action_space,
    _execute_round_tasks,
    _llm_allocation,
    _materialize_round,
    _rule_allocation,
    _task_result,
    build_campaign_manifest,
    release_gpu_hours,
    reserve_gpu_hours,
    validate_campaign_manifest,
)


def test_campaign_manifest_has_all_matched_rounds_and_budgets():
    manifest = build_campaign_manifest()

    validate_campaign_manifest(manifest)
    assert len(manifest["rounds"]) == 6 * 2 * 3 * 10
    assert manifest["total_logical_optimizations"] == 61120
    assert manifest["resource_request"]["account"] == "m5216"
    assert manifest["resource_request"]["billing_account"] == "m5216_g"
    assert manifest["resource_request"]["qos"] == "regular"
    assert manifest["resource_request"]["billing_qos"] == "gpu_regular"
    assert manifest["resource_request"]["requested_gpu_hour_cap"] == 2000
    assert manifest["resource_request"]["maximum_scheduled_gpu_hours"] == pytest.approx(1999.2)
    assert manifest["resource_request"]["maximum_scheduled_gpu_node_hours"] == pytest.approx(499.8)
    assert manifest["llm_guided_configuration"] == {
        "provider": "mag",
        "model": "gemini-4-argon@google",
        "mag_provider": "google",
        "temperature": 0.0,
        "request_limits": {"max_calls_per_round": 1, "max_tokens": 8192},
        "environment_overrides": "prohibited",
    }

    for material in MATERIALS:
        for strategy in E3_STRATEGIES:
            rows = [
                row
                for row in manifest["rounds"]
                if row["material"] == material and row["strategy"] == strategy
            ]
            assert len(rows) == 30
            assert sum(row["main_target"] for row in rows) == 5000
            assert sum(row["finite_size_control_target"] for row in rows) == (
                20 * len(MATERIALS[material]["tiers"])
            )
            assert sum(len(row["finite_size_control_families"]) for row in rows) == 20
            assert {row["seed"] for row in rows} == set(REPLICATE_SEEDS)
            for row in rows:
                assert sum(row["minimum_main_count_by_tier"].values()) <= row["main_target"]


def test_perlmutter_launcher_targets_the_authorized_gpu_resources():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "deployments"
        / "perlmutter"
        / "jobs"
        / "job-e3-mlip-production-perlmutter.sh"
    )
    text = launcher.read_text(encoding="utf-8")

    assert 'SLURM_JOB_ACCOUNT:-}" == "m5216_g"' in text
    assert "#SBATCH -p gpu_ss11" not in text
    assert "#SBATCH -q regular" in text
    assert "MATSIM_E3_MLIP_RESOURCE_AUTHORIZED=m5216:regular:2000" in text
    assert "MATSIM_E3_MLIP_BASE_PYTHON" in text
    assert "MATSIM_E3_MLIP_WORKER_PYTHON" in text


def test_class_actions_are_material_specific_and_mof_never_uses_pyxtal():
    assert {action["generator"] for action in _action_space("bcc_refractory")} == {
        "native_lattice_occupations"
    }
    assert {action["generator"] for action in _action_space("fcc_cantor")} == {
        "native_lattice_occupations"
    }
    assert {action["generator"] for action in _action_space("phosphorene")} == {"2d_template"}
    assert {action["generator"] for action in _action_space("zn_formate_mof")} == {"cod_7205949"}
    assert "pyxtal" not in {action["generator"] for action in _action_space("zn_formate_mof")}
    assert any(action["space_groups"] for action in _action_space("nb_ta_o"))


def test_rule_policy_is_deterministic_and_satisfies_tier_floors():
    actions = _action_space("bcc_refractory")
    floors = {"4": 2, "8": 2, "16": 2, "32": 2, "64": 2}

    first = _rule_allocation(actions, [], target=25, minimum_by_tier=floors)
    second = _rule_allocation(actions, [], target=25, minimum_by_tier=floors)

    assert first == second
    assert sum(item["count"] for item in first) == 25
    by_tier = {}
    for item in first:
        tier = str(item["action"]["size_tier"])
        by_tier[tier] = by_tier.get(tier, 0) + item["count"]
    assert all(by_tier[tier] >= floor for tier, floor in floors.items())


def test_lifepo4_parent_action_capacity_is_shared_across_allocators():
    actions = _action_space("lifepo4")
    parent_actions = [
        action
        for action in actions
        if action["family_label"] == "olivine-pristine" and action["family"] == "olivine"
    ]
    assert parent_actions

    allocation = _rule_allocation(
        actions,
        [],
        target=20,
        minimum_by_tier={"4": 2, "8": 2, "16": 2, "32": 2},
    )
    parent_counts = {}
    for row in allocation:
        action = row["action"]
        if action["family_label"] == "olivine-pristine":
            tier = str(action["size_tier"])
            parent_counts[tier] = parent_counts.get(tier, 0) + row["count"]
    assert all(count <= 1 for count in parent_counts.values())

    exhausted = parent_actions[0]
    prior_history = [
        {
            "action_id": exhausted["action_id"],
            "action": exhausted,
            "attempted": 1,
            "converged": 1,
        }
    ]
    allocation = _rule_allocation(
        actions,
        prior_history,
        target=20,
        minimum_by_tier={"4": 2, "8": 2, "16": 2, "32": 2},
    )
    assert not any(
        row["action"]["family_label"] == "olivine-pristine"
        and row["action"]["size_tier"] == exhausted["size_tier"]
        for row in allocation
    )

    class FakeChat:
        def invoke(self, prompt):
            catalog = prompt.split("Hierarchical action catalog: ", maxsplit=1)[1]
            assert exhausted["action_id"] not in catalog
            return type(
                "Response",
                (),
                {
                    "content": json.dumps(
                        {
                            "allocations": [{"action_id": exhausted["action_id"], "count": 1}],
                            "cited_evidence_ids": ["campaign-design"],
                            "rationale": "test finite-capacity rejection",
                        }
                    )
                },
            )()

    with pytest.raises(ValueError, match="finite generation-action capacity"):
        _llm_allocation(
            FakeChat(),
            "lifepo4",
            actions,
            prior_history,
            target=1,
            minimum_by_tier={},
        )


def test_round_materialization_preserves_control_generation_family(monkeypatch, tmp_path):
    from benchmarks.e3 import production
    from benchmarks.e3.structure_generation import _FAMILIES

    manifest = build_campaign_manifest()
    spec = next(
        row
        for row in manifest["rounds"]
        if row["material"] == "bcc_refractory"
        and row["strategy"] == E3_STRATEGIES[0]
        and row["seed"] == REPLICATE_SEEDS[0]
        and row["round"] == 1
    )
    spec = {
        **spec,
        "main_target": 1,
        "minimum_main_count_by_tier": {"4": 1},
        "finite_size_control_families": [0],
    }
    manifest["action_spaces"]["bcc_refractory"] = _action_space("bcc_refractory")
    monkeypatch.setattr(
        production,
        "_verified_checkpoint",
        lambda: (tmp_path / "checkpoint", "checkpoint-sha256"),
    )

    _, tasks_path, result = _materialize_round(manifest, spec, tmp_path, [], None)

    tasks = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    controls = [task for task in tasks if task["candidate_kind"] == "finite_size_control"]
    assert result["finite_size_controls"] == len(MATERIALS["bcc_refractory"]["tiers"])
    assert len(controls) == len(MATERIALS["bcc_refractory"]["tiers"])
    for control in controls:
        assert control["control_family_id"] == 0
        assert control["action"]["family"] in _FAMILIES["bcc_refractory"]
        assert control["generation"]["family"] in _FAMILIES["bcc_refractory"]


def test_llm_policy_validates_exact_budget_allowlist_and_evidence():
    actions = _action_space("phosphorene")
    allowed = actions[:2]

    class FakeChat:
        model_id = "test-model"

        def invoke(self, prompt):
            assert "no DFT labels" in prompt
            assert "material class(m)" in prompt
            return type(
                "Response",
                (),
                {
                    "content": json.dumps(
                        {
                            "allocations": [{"action_id": allowed[0]["action_id"], "count": 2}],
                            "cited_evidence_ids": ["campaign-design"],
                            "rationale": "The allotrope tier has not yet been sampled.",
                        }
                    )
                },
            )()

    allocations, citations, rationale = _llm_allocation(
        FakeChat(),
        "phosphorene",
        actions,
        [],
        target=2,
        minimum_by_tier={str(allowed[0]["size_tier"]): 2},
    )
    assert allocations == [{"action": allowed[0], "count": 2}]
    assert citations == ["campaign-design"]
    assert rationale

    with pytest.raises(ValueError, match="outside the list|invalid or repeated action"):
        _llm_allocation(
            type(
                "FakeChat",
                (),
                {
                    "invoke": lambda self, prompt: type(
                        "Response",
                        (),
                        {
                            "content": json.dumps(
                                {
                                    "allocations": [{"action_id": "not-allowed", "count": 2}],
                                    "cited_evidence_ids": ["campaign-design"],
                                    "rationale": "invalid",
                                }
                            )
                        },
                    )()
                },
            )(),
            "phosphorene",
            actions,
            [],
            target=2,
            minimum_by_tier={},
        )


def test_llm_configuration_cannot_be_overridden_by_environment(monkeypatch):
    from benchmarks.e3 import production

    calls = []

    def fake_get_chat_model(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("matsim_agents.backends.llm.provider.get_chat_model", fake_get_chat_model)
    monkeypatch.setenv("MATSIM_E3_MLIP_LLM_PROVIDER", "unapproved")
    monkeypatch.setenv("MATSIM_E3_MLIP_LLM_MODEL", "unapproved-model")
    monkeypatch.setenv("MATSIM_E3_MLIP_MAG_PROVIDER", "unapproved")

    production._chat_model()

    assert calls == [
        {
            "provider": "mag",
            "model": "gemini-4-argon@google",
            "temperature": 0.0,
            "max_tokens": 8192,
            "mag_provider": "google",
        }
    ]


def test_multi_node_worker_command_uses_one_gpu_per_rank(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(
        "benchmarks.e3.production.subprocess.run",
        lambda command, check: calls.append((command, check)),
    )

    _execute_round_tasks(
        tmp_path / "tasks.jsonl",
        tmp_path,
        worker_count=8,
        workers_per_node=4,
        worker_python="/project/.venv-mace/bin/python",
    )

    command, check = calls[0]
    assert check is True
    assert command[0] == "srun"
    assert command[command.index("--nodes") + 1] == "2"
    assert command[command.index("--ntasks") + 1] == "8"
    assert command[command.index("--gpus-per-task") + 1] == "1"


def test_task_result_only_accepts_a_terminal_result(tmp_path):
    task = {"candidate_id": "candidate-1", "structure_sha256": "geometry-hash"}
    attempt = tmp_path / "candidate-1" / "attempt-0001"
    attempt.mkdir(parents=True)
    path = attempt / "result.json"
    path.write_text(
        json.dumps(
            {
                "structure_sha256": "geometry-hash",
                "terminal": False,
            }
        )
    )
    assert _task_result(tmp_path / "candidate-1", task) is None

    path.write_text(
        json.dumps(
            {
                "structure_sha256": "geometry-hash",
                "terminal": True,
            }
        )
    )
    assert _task_result(tmp_path / "candidate-1", task) == path


def test_resource_ledger_enforces_total_gpu_hour_cap_and_releases_unused_time(tmp_path):
    state = tmp_path / "resource-budget.json"
    cap = 10_000

    reserved = reserve_gpu_hours(
        state,
        "job-1",
        reserved_gpu_seconds=6_000,
        cap_gpu_seconds=cap,
    )
    assert reserved["reserved_gpu_seconds"] == 6_000
    with pytest.raises(RuntimeError, match="only 4000 remain"):
        reserve_gpu_hours(
            state,
            "job-2",
            reserved_gpu_seconds=4_001,
            cap_gpu_seconds=cap,
        )

    released = release_gpu_hours(
        state,
        "job-1",
        actual_gpu_seconds=3_000,
    )
    assert released == {
        "cap_gpu_seconds": cap,
        "spent_gpu_seconds": 3_000,
        "reserved_gpu_seconds": 0,
    }
    assert (
        reserve_gpu_hours(
            state,
            "job-2",
            reserved_gpu_seconds=7_000,
            cap_gpu_seconds=cap,
        )["reserved_gpu_seconds"]
        == 7_000
    )


def test_checkpoint_path_is_the_verified_local_artifact(tmp_path, monkeypatch):
    from benchmarks.e3 import mlip_exploration

    artifact = tmp_path / "mace" / "maceomat0mediummodel"
    artifact.parent.mkdir()
    artifact.write_bytes(b"frozen model fixture")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.setenv("MACE_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(mlip_exploration, "E3_MLIP_CHECKPOINT_SHA256", digest)

    checkpoint_path, actual_digest = mlip_exploration._verified_checkpoint()

    assert checkpoint_path == artifact.resolve()
    assert actual_digest == digest
