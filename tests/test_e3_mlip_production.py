from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.e3.production import (
    E3_MLIP_CHECKPOINT_SHA256,
    MATERIALS,
    OPTIMIZER_SETTINGS,
    REPLICATE_SEEDS,
    _action_space,
    _distinct_relaxed_minima,
    _execute_round_tasks,
    _execute_task,
    _finite_capacity_group,
    _llm_allocation,
    _materialize_round,
    _mof_bond_graph,
    _mof_connectivity_guard,
    _rule_allocation,
    _task_result,
    build_campaign_manifest,
    release_gpu_hours,
    reserve_gpu_hours,
    run_replicate,
    run_worker,
    validate_campaign_manifest,
)


def test_spooled_slurm_launcher_resolves_submission_checkout(tmp_path, monkeypatch):
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    script = repo / "deployments/perlmutter/jobs/job-e3-mlip-production-perlmutter.sh"
    spooled = tmp_path / "slurm_script"
    spooled.write_text(script.read_text())
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    for key, value in {
        "SLURM_SUBMIT_DIR": str(repo),
        "SLURM_JOB_ID": "test-not-an-allocation",
        "SLURM_JOB_ACCOUNT": "m5216_g",
        "SLURM_JOB_QOS": "gpu_premium",
        "SLURM_NNODES": "16",
        "SLURM_NTASKS": "64",
        "MATSIM_E3_MLIP_RESOURCE_AUTHORIZED": "m5216:premium:768",
        "MATSIM_E3_MLIP_RESERVATION_ID": "pre-reserved-test-allocation",
        "MATSIM_E3_MLIP_BASE_PYTHON": str(tmp_path / "missing-python"),
    }.items():
        monkeypatch.setenv(key, value)
    result = subprocess.run(
        ["bash", str(spooled)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "prepared base and MACE Python environments are required" in result.stderr
    assert "not a matsim-agents checkout" not in result.stderr


def test_separate_rerun_uses_explicit_shared_resource_ledger():
    repo = Path(__file__).resolve().parents[1]
    script = (repo / "deployments/perlmutter/jobs/job-e3-mlip-production-perlmutter.sh").read_text()
    assert (
        'RESOURCE_STATE="${MATSIM_E3_MLIP_RESOURCE_STATE:-${OUTPUT_ROOT}/resource-budget.json}"'
        in script
    )
    assert '--resource-state "${RESOURCE_STATE}"' in script
    assert script.count('--state "${RESOURCE_STATE}"') == 1
    helper = (
        repo / "deployments/perlmutter/jobs/submit-e3-mlip-production-perlmutter.sh"
    ).read_text()
    assert "MATSIM_E3_MLIP_RESOURCE_STATE:-${OUTPUT_ROOT}/resource-budget.json" in helper
    assert '--state "${MATSIM_E3_MLIP_RESOURCE_STATE}"' in helper


def test_worker_step_waits_for_slow_ranks_after_first_success(monkeypatch, tmp_path):
    import time

    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")
    monkeypatch.setattr(
        "benchmarks.e3.production.subprocess.run", lambda command, **_: calls.append(command)
    )
    _execute_round_tasks(
        tmp_path / "tasks.jsonl",
        tmp_path,
        64,
        4,
        "worker-python",
        hard_optimization_seconds=300,
        round_deadline=time.monotonic() + 1200,
    )
    assert "--wait=0" in calls[0]
    assert "--wait=10" not in calls[0]
    assert "--kill-on-bad-exit=1" in calls[0]
    assert "--time" in calls[0]


def test_hard_deadline_records_nonconvergence_and_is_not_retried(tmp_path, monkeypatch):
    from matsim_agents.execution.deadline_worker import TaskDeadlineExceeded

    tasks_path = tmp_path / "tasks.jsonl"
    task = {
        "candidate_id": "test",
        "structure_sha256": "test-sha",
        "candidate_kind": "unique_configuration",
        "structure_path": "test.extxyz",
    }
    tasks_path.write_text(json.dumps(task) + "\n")
    monkeypatch.setattr(
        "benchmarks.e3.production._verified_checkpoint",
        lambda: (Path("unused"), E3_MLIP_CHECKPOINT_SHA256),
    )
    calls = []
    startup_deadlines = []

    class Worker:
        def __init__(self, *args, **kwargs):
            startup_deadlines.append(kwargs["startup_seconds"])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self, request, *, seconds):
            calls.append(seconds)
            candidate, root = request
            attempt = root / candidate["candidate_id"] / "attempt-0001"
            attempt.mkdir(parents=True)
            (attempt / "partial.traj").write_bytes(b"preserved")
            raise TaskDeadlineExceeded("stalled optimizer step")

    monkeypatch.setattr("matsim_agents.execution.deadline_worker.DeadlineWorker", Worker)
    result = run_worker(
        tasks_path,
        tmp_path / "results",
        rank=0,
        world_size=1,
        hard_optimization_seconds=300,
    )
    assert result == {"nonconverged": 1, "attempted": 1}
    path = tmp_path / "results/tasks/test/attempt-0001/result.json"
    payload = json.loads(path.read_text())
    assert payload["converged"] is False
    assert payload["terminal"] is True
    assert payload["termination_reason"] == "optimization_time_limit"
    assert payload["timeout_enforcement"] == "parent_process_kill"
    assert payload["final_energy_eV"] is None
    assert payload["optimizer"]["max_optimization_seconds"] == 300
    assert (path.parent / "partial.traj").read_bytes() == b"preserved"
    assert run_worker(
        tasks_path,
        tmp_path / "results",
        rank=0,
        world_size=1,
        hard_optimization_seconds=300,
    ) == {"resumed": 1}
    assert calls == [300]
    assert startup_deadlines == [15 * 60, 15 * 60]
    assert _task_result(path.parent.parent, task) is None
    task["optimizer"] = payload["optimizer"]
    assert _task_result(path.parent.parent, task) == path


def test_hard_deadline_is_forwarded_to_all_gpu_workers(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setattr(
        "benchmarks.e3.production.subprocess.run",
        lambda command, **kwargs: calls.append(command),
    )
    _execute_round_tasks(
        tmp_path / "tasks.jsonl",
        tmp_path / "results",
        64,
        4,
        "python",
        hard_optimization_seconds=300,
    )
    assert calls[0][calls[0].index("--nodes") + 1] == "16"
    assert calls[0][-2:] == ["--hard-optimization-seconds", "300"]


def test_existing_stream_cannot_change_optimization_deadline(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(build_campaign_manifest()))
    stream = tmp_path / "runs/nb_ta_o/llm_guided_adaptation/seed-20261113"
    stream.mkdir(parents=True)
    (stream / "execution-policy.json").write_text(json.dumps({"hard_optimization_seconds": 300}))
    with pytest.raises(ValueError, match="64 workers and a 300-second hard cutoff"):
        run_replicate(
            manifest_path,
            tmp_path / "runs",
            "nb_ta_o",
            "llm_guided_adaptation",
            20261113,
            worker_count=64,
            hard_optimization_seconds=60,
            round_index=1,
        )


@pytest.mark.parametrize("seconds", [0, -1, 301, float("nan"), float("inf")])
def test_hard_deadline_rejects_invalid_or_longer_than_five_minutes(seconds):
    from benchmarks.e3.production import _hard_optimizer_policy

    with pytest.raises(ValueError, match="within"):
        _hard_optimizer_policy(seconds)


def test_hard_timeout_is_counted_as_nonconverged_without_usable_minimum(tmp_path):
    import time

    from benchmarks.e3.production import (
        _hard_optimizer_policy,
        _round_action_outcomes,
        _timeout_result,
    )

    task = {
        "candidate_id": "timeout",
        "structure_sha256": "sha",
        "candidate_kind": "unique_configuration",
        "structure_path": "initial.extxyz",
        "optimizer": _hard_optimizer_policy(300),
        "action_id": "action",
        "action": {
            "size_tier": "small",
            "family": "test",
            "generator": "test",
        },
        "formula": "Cu",
        "formula_units": 1,
        "num_atoms": 1,
    }
    _timeout_result(task, tmp_path, time.monotonic(), time.time())
    outcome = _round_action_outcomes([task], tmp_path)[0]
    assert outcome["attempted"] == 1
    assert outcome["nonconverged"] == 1
    assert outcome["converged"] == outcome["failed"] == 0
    assert outcome["mean_energy_eV_per_atom"] is None
    assert outcome["minimum_energy_eV_per_atom"] is None


def test_wrapper_constructs_explicit_five_minute_deadline(monkeypatch, tmp_path):
    from matsim_agents.cli.workflow_catalog import WorkflowConfig
    from matsim_agents.cli.workflows import _command

    resource = tmp_path / "resource.json"
    resource.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cap_gpu_seconds": 2764800,
                "reservations": {"test": {"reserved_gpu_seconds": 460800}},
            }
        )
    )
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    cfg = WorkflowConfig.model_validate(
        {
            "workflow": "e3-mlip",
            "settings": {
                "manifest": str(tmp_path / "manifest"),
                "output_root": str(tmp_path / "runs"),
                "resource_state": str(resource),
                "material": "nb_ta_o",
                "hard_optimization_seconds": 300,
            },
        }
    )
    command = _command(cfg, approved=True)
    assert command[-2:] == ["--hard-optimization-seconds", "300"]


def test_campaign_manifest_has_six_classes_and_bounded_budgets():
    manifest = build_campaign_manifest()

    validate_campaign_manifest(manifest)
    assert manifest["optimizer"] == OPTIMIZER_SETTINGS
    assert manifest["optimizer"]["relax_cell"] is True
    assert manifest["optimizer"]["relative_increase_threshold"] is None
    assert manifest["optimizer"]["stress_tolerance_eV_per_A3"] is None
    assert manifest["optimizer"]["max_cell_step_strain"] is None
    assert manifest["optimizer"]["max_cell_total_strain"] is None
    assert manifest["optimizer"]["max_optimization_seconds"] == 60
    assert manifest["optimizer"]["optimization_timeout_policy"] == "between_steps_v1"
    assert manifest["optimizer"]["mof_connectivity_version"] == "atom-indexed-v1"
    assert manifest["optimizer"]["phosphorene_cell_mask"] == [True, True, False, False, False, True]
    assert len(manifest["rounds"]) == 6 * 3
    assert manifest["total_logical_optimizations"] == 1152
    assert manifest["resource_request"]["account"] == "m5216"
    assert manifest["resource_request"]["billing_account"] == "m5216_g"
    assert manifest["resource_request"]["qos"] == "premium"
    assert manifest["resource_request"]["billing_qos"] == "gpu_premium"
    assert manifest["resource_request"]["requested_gpu_hour_cap"] == 768
    assert manifest["resource_request"]["maximum_scheduled_gpu_hours"] == pytest.approx(768)
    assert manifest["resource_request"]["maximum_scheduled_gpu_node_hours"] == pytest.approx(192)
    assert manifest["llm_guided_configuration"] == {
        "provider": "copilot",
        "model": "gemini-4-carbon",
        "decoding": "provider_default",
        "request_limits": {"max_calls_per_round": 1, "max_tokens": 8192},
        "environment_overrides": "prohibited",
    }

    for material in MATERIALS:
        rows = [row for row in manifest["rounds"] if row["material"] == material]
        assert len(rows) == 3
        assert sum(row["main_target"] for row in rows) == 192
        assert all(row["finite_size_control_target"] == 0 for row in rows)
        assert {row["strategy"] for row in rows} == {"llm_guided_adaptation"}
        assert {row["seed"] for row in rows} == set(REPLICATE_SEEDS)
        for row in rows:
            assert row["main_target"] == 64
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

    assert 'SLURM_JOB_ACCOUNT:-}" == "${MATSIM_E3_MLIP_ACCOUNT}_g"' in text
    assert "#SBATCH -p gpu_ss11" not in text
    assert "#SBATCH -q premium" in text
    assert "#SBATCH -N 16" in text
    assert "#SBATCH -t 02:00:00" in text
    assert "#SBATCH --array=0-5%5" in text
    assert "MATSIM_E3_MLIP_RESOURCE_AUTHORIZED=${MATSIM_E3_MLIP_ACCOUNT}:premium:768" in text
    assert "MATSIM_E3_MLIP_BASE_PYTHON" in text
    assert "MATSIM_E3_MLIP_WORKER_PYTHON" in text
    assert text.index("export PYTHONPATH=") < text.index("budget-bind")
    assert "budget-release" not in text


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

    with pytest.raises(ValueError, match="unavailable or repeated action ID"):
        _llm_allocation(
            FakeChat(),
            "lifepo4",
            actions,
            prior_history,
            target=1,
            minimum_by_tier={},
        )


@pytest.mark.parametrize("family_label", ["black-phosphorene", "blue-phosphorene"])
def test_pristine_phosphorene_capacity_is_shared_across_allocators_and_rounds(family_label):
    actions = _action_space("phosphorene")
    pristine = [
        action
        for action in actions
        if action["family_label"] == family_label
        and action["operation"] == "allotrope"
        and action["defect"] is None
        and action["strain"] is None
    ]
    assert pristine
    assert all(_finite_capacity_group(action) is not None for action in pristine)

    allocation = _rule_allocation(actions, [], target=10, minimum_by_tier={})
    allocated = {}
    for row in allocation:
        group = _finite_capacity_group(row["action"])
        if group is not None:
            allocated[group] = allocated.get(group, 0) + row["count"]
    assert all(count <= 1 for count in allocated.values())

    exhausted = pristine[0]
    prior_history = [
        {
            "action_id": exhausted["action_id"],
            "action": exhausted,
            "attempted": 1,
            "converged": 1,
        }
    ]
    allocation = _rule_allocation(actions, prior_history, target=1, minimum_by_tier={})
    assert not any(row["action"]["action_id"] == exhausted["action_id"] for row in allocation)

    remaining = next(action for action in pristine if action["action_id"] != exhausted["action_id"])

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
                            "allocations": [{"action_id": remaining["action_id"], "count": 1}],
                            "cited_evidence_ids": ["campaign-design"],
                            "rationale": "sample an unused pristine allotrope tier",
                        }
                    )
                },
            )()

    allocations, _, _ = _llm_allocation(
        FakeChat(),
        "phosphorene",
        actions,
        prior_history,
        target=1,
        minimum_by_tier={str(remaining["size_tier"]): 1},
    )
    assert allocations[0]["action"]["action_id"] == remaining["action_id"]
    assert allocations[0]["count"] == 1
    assert allocations[0]["action"]["feasibility"]["remaining_capacity"] == 1


@pytest.mark.parametrize(
    ("material", "minimum_by_tier"),
    [
        ("bcc_refractory", {"4": 1}),
        ("phosphorene", {"4": 1}),
    ],
)
def test_round_materialization_replicates_control_families_across_tiers(
    monkeypatch, tmp_path, material, minimum_by_tier
):
    from ase.io import read

    from benchmarks.e3 import production
    from benchmarks.e3.structure_generation import _FAMILIES

    manifest = build_campaign_manifest()
    spec = next(
        row
        for row in manifest["rounds"]
        if row["material"] == material
        and row["strategy"] == "llm_guided_adaptation"
        and row["seed"] == REPLICATE_SEEDS[0]
        and row["round"] == 1
    )
    spec = {
        **spec,
        "strategy": "rule_based_adaptation",
        "main_target": 1,
        "minimum_main_count_by_tier": minimum_by_tier,
        "finite_size_control_families": [0],
    }
    manifest["action_spaces"][material] = _action_space(material)
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
    assert result["finite_size_controls"] == len(MATERIALS[material]["tiers"])
    assert len(controls) == len(MATERIALS[material]["tiers"])
    base = read(controls[0]["structure_path"], index=0, format="extxyz")
    for control in controls:
        assert control["control_family_id"] == 0
        assert control["action"]["family"] in _FAMILIES[material]
        assert control["generation"]["family"] in _FAMILIES[material]
        control_metadata = control["generation"]["finite_size_control"]
        assert control_metadata["control_family_id"] == 0
        expected = base.repeat(tuple(control_metadata["supercell_repetitions"]))
        actual = read(control["structure_path"], index=0, format="extxyz")
        assert actual.get_chemical_symbols() == expected.get_chemical_symbols()
        assert list(actual.cell.array.flat) == pytest.approx(list(expected.cell.array.flat))
        assert list(actual.positions.flat) == pytest.approx(list(expected.positions.flat))


def test_llm_policy_validates_exact_budget_allowlist_and_evidence():
    actions = _action_space("phosphorene")
    allowed = next(action for action in actions if _finite_capacity_group(action) is None)

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
                            "allocations": [{"action_id": allowed["action_id"], "count": 2}],
                            "cited_evidence_ids": ["campaign-design"],
                            "rationale": "The bounded perturbation has not yet been sampled.",
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
        minimum_by_tier={str(allowed["size_tier"]): 2},
    )
    assert allocations[0]["action"]["action_id"] == allowed["action_id"]
    assert allocations[0]["count"] == 2
    assert allocations[0]["action"]["feasibility"]["generator_available"]
    assert citations == ["campaign-design"]
    assert rationale

    with pytest.raises(ValueError, match="unavailable or repeated action"):
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
            "provider": "copilot",
            "model": "gemini-4-carbon",
            "max_tokens": 8192,
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
    assert _task_result(tmp_path / "candidate-1", task) is None
    path.write_text(
        json.dumps(
            {
                "structure_sha256": "geometry-hash",
                "terminal": True,
                "optimizer": OPTIMIZER_SETTINGS,
            }
        )
    )
    assert _task_result(tmp_path / "candidate-1", task) == path


def test_manifest_rejects_old_fixed_cell_protocol():
    manifest = build_campaign_manifest()
    manifest["optimizer"] = {**OPTIMIZER_SETTINGS, "relax_cell": False}
    with pytest.raises(ValueError, match="optimizer"):
        validate_campaign_manifest(manifest)


def test_manifest_rejects_old_mof_guard_protocol():
    manifest = build_campaign_manifest()
    manifest["optimizer"] = dict(manifest["optimizer"])
    manifest["optimizer"].pop("mof_connectivity_version")
    with pytest.raises(ValueError, match="optimizer"):
        validate_campaign_manifest(manifest)


def test_manifest_rejects_previous_llm_provider():
    manifest = build_campaign_manifest()
    manifest["llm_guided_configuration"] = {
        **manifest["llm_guided_configuration"],
        "provider": "mag",
    }
    with pytest.raises(ValueError, match="scientific objectives"):
        validate_campaign_manifest(manifest)


def test_timed_relaxations_reuse_calculator_and_save_completed_step(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ase.build import bulk
    from ase.calculators.emt import EMT
    from ase.io import read, write

    import matsim_agents.backends.mlip.relaxation as relaxation

    source = tmp_path / "input.extxyz"
    write(source, bulk("Cu", "fcc", a=3.8, cubic=True))
    clock_values = iter([100.0, 159.9, 160.0, 200.0, 260.0])
    monkeypatch.setattr(relaxation, "time", SimpleNamespace(monotonic=lambda: next(clock_values)))
    monkeypatch.setattr(
        relaxation,
        "_relaxation_calculator",
        lambda args: relaxation._RelaxCalculator(EMT(), 0, 0, None),
    )
    monkeypatch.setattr(
        "benchmarks.e3.production._verified_checkpoint",
        lambda: (source, E3_MLIP_CHECKPOINT_SHA256),
    )
    monkeypatch.setattr("benchmarks.e3.production._space_group_number", lambda atoms: None)
    cache = relaxation.CalculatorCache()
    task = {
        "candidate_kind": "unique_configuration",
        "structure_sha256": "hash",
        "structure_path": str(source),
        "material": "bcc_refractory",
    }
    results = [
        _execute_task({**task, "candidate_id": candidate}, tmp_path / "tasks", cache)
        for candidate in ("first", "second")
    ]
    assert cache.loads == 1
    assert [result["steps"] for result in results] == [2, 1]
    for result in results:
        assert result["status"] == "nonconverged"
        assert result["termination_reason"] == "optimization_time_limit"
        assert result["converged"] is False
        assert len(read(result["trajectory_path"], index=":")) == result["steps"] + 1
        assert Path(result["optimized_structure_path"]).is_file()
        assert _task_result(tmp_path / "tasks" / result["candidate_id"], task)


def test_timed_out_structures_are_not_counted_as_relaxed_minima():
    tasks = [{"candidate_kind": "unique_configuration", "formula": "Cu"}]
    results = [{"status": "nonconverged", "termination_reason": "optimization_time_limit"}]
    assert _distinct_relaxed_minima(tasks, results) == {}


def test_prior_fixed_cell_evidence_is_not_replayed(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(build_campaign_manifest()))
    round_dir = (
        tmp_path
        / "runs"
        / "bcc_refractory"
        / "llm_guided_adaptation"
        / "seed-20261113"
        / "round-01"
    )
    round_dir.mkdir(parents=True)
    (round_dir / "round-summary.json").write_text(json.dumps({"status": "complete"}))
    (round_dir / "round-manifest.json").write_text(json.dumps({"optimizer": {"relax_cell": False}}))
    with pytest.raises(ValueError, match="requires a fresh stream"):
        run_replicate(
            manifest_path,
            tmp_path / "runs",
            "bcc_refractory",
            "llm_guided_adaptation",
            20261113,
            worker_count=64,
            hard_optimization_seconds=300,
            round_index=1,
        )


def test_mof_bond_graph_handles_periodic_images_and_pair_cutoffs():
    from ase import Atoms

    atoms = Atoms(
        ["Zn", "O", "C", "O", "H"],
        positions=[[0.2, 1, 1], [9.5, 1, 1], [5, 5, 5], [6.5, 5, 5], [5, 6.2, 5]],
        cell=[10, 10, 10],
        pbc=True,
    )
    expected = frozenset({(0, 1), (2, 3), (2, 4)})
    assert _mof_bond_graph(atoms) == expected
    atoms.positions[1, 0] -= 10
    assert _mof_bond_graph(atoms) == expected
    atoms.positions[3, 0] = 6.7
    atoms.positions[4, 1] = 6.3
    assert _mof_bond_graph(atoms) == frozenset({(0, 1)})


def test_mof_bond_swap_preserving_coordination_is_rejected():
    from ase import Atoms

    from benchmarks.e3.structure_generation import _mof_connectivity_signature

    atoms = Atoms(
        ["C", "C", "O", "O"],
        positions=[[2, 5, 5], [8, 5, 5], [3, 5, 5], [9, 5, 5]],
        cell=[20, 20, 20],
        pbc=True,
    )
    guard = _mof_connectivity_guard(atoms)
    swapped = atoms.copy()
    swapped.positions[[2, 3]] = swapped.positions[[3, 2]]
    assert _mof_connectivity_signature(atoms) == _mof_connectivity_signature(swapped)
    assert guard(atoms) is None
    assert guard(swapped) == "mof_connectivity_changed"


def test_mof_bond_swap_rolls_back_relaxation(tmp_path, monkeypatch):
    from ase import Atoms
    from ase.calculators.emt import EMT
    from ase.io import read, write
    from ase.optimize import FIRE

    import matsim_agents.backends.mlip.relaxation as relaxation

    initial = Atoms(
        ["C", "C", "O", "O"],
        positions=[[2, 5, 5], [8, 5, 5], [3, 5, 5], [9, 5, 5]],
        cell=[20, 20, 20],
        pbc=True,
    )
    source = tmp_path / "input.extxyz"
    write(source, initial)

    def swap_step(optimizer, *args, **kwargs):
        atoms = optimizer.atoms.atoms
        atoms.positions[[2, 3]] = atoms.positions[[3, 2]]

    monkeypatch.setattr(FIRE, "step", swap_step)
    monkeypatch.setattr(
        relaxation,
        "_relaxation_calculator",
        lambda args: relaxation._RelaxCalculator(EMT(), 0, 0, None),
    )
    result = relaxation.relax_with_mlip(
        relaxation.RelaxStructureInput(
            structure_path=str(source),
            output_dir=str(tmp_path),
            mlip_backend="mace",
            relax_cell=True,
            maxiter=2,
        ),
        step_guard=_mof_connectivity_guard(initial),
    )
    restored = read(result.optimized_structure_path)
    assert not result.converged
    assert result.termination_reason == "mof_connectivity_changed"
    assert restored.positions == pytest.approx(initial.positions)
    assert restored.cell.array == pytest.approx(initial.cell.array)


@pytest.mark.parametrize("material", sorted(MATERIALS))
@pytest.mark.parametrize("hard_deadline", [False, True])
def test_worker_applies_cell_policy_for_each_family(tmp_path, monkeypatch, material, hard_deadline):
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from ase.io import read, write

    import matsim_agents.backends.mlip.relaxation as relaxation
    from matsim_agents.backends.mlip.results import RelaxationResult

    source = tmp_path / "input.extxyz"
    write(source, bulk("Cu", "fcc", a=3.8, cubic=True))
    task = {
        "candidate_id": "candidate",
        "candidate_kind": "main",
        "structure_sha256": "geometry",
        "structure_path": str(source),
        "material": material,
    }
    if hard_deadline:
        from benchmarks.e3.production import _hard_optimizer_policy

        task["optimizer"] = _hard_optimizer_policy(300)

    class Cache:
        def __init__(self):
            self.calculator = EMT()

        def get(self, args):
            return relaxation._RelaxCalculator(self.calculator, 0, 0, "")

    def fake_relax(args, *, calculators, step_guard):
        assert args.relax_cell
        assert args.require_native_stress
        assert args.stress_tolerance_eV_per_A3 is None
        assert args.relative_increase_threshold is None
        assert args.max_cell_step_strain is None
        assert args.max_cell_total_strain is None
        assert args.max_optimization_seconds == (300 if hard_deadline else 60)
        assert args.cell_mask == (
            (True, True, False, False, False, True) if material == "phosphorene" else None
        )
        assert (step_guard is not None) == (material == "zn_formate_mof")
        if step_guard is not None:
            atoms = read(args.structure_path)
            assert step_guard(atoms) is None
            atoms.positions[0, 0] += 1
            assert step_guard(atoms) == "mof_connectivity_changed"
        return RelaxationResult(
            structure_path=args.structure_path,
            optimized_structure_path=args.structure_path,
            trajectory_path="trajectory",
            log_csv_path="log",
            final_energy_eV=0.0,
            final_max_force_eV_per_A=0.0,
            num_steps=1,
            converged=True,
        )

    monkeypatch.setattr(relaxation, "relax_with_mlip", fake_relax)
    if material == "zn_formate_mof":
        monkeypatch.setattr(
            "benchmarks.e3.production._mof_bond_graph",
            lambda atoms: float(atoms.positions[0, 0]),
        )
    monkeypatch.setattr("benchmarks.e3.production._space_group_number", lambda atoms: None)
    monkeypatch.setattr(
        "benchmarks.e3.production._verified_checkpoint",
        lambda: (source, E3_MLIP_CHECKPOINT_SHA256),
    )
    result = _execute_task(task, tmp_path / "tasks", Cache())
    assert result["status"] == "converged", result
    assert result["optimizer"]["relative_increase_threshold"] is None
    assert result["optimizer"]["stress_tolerance_eV_per_A3"] is None
    assert result["optimizer"]["max_cell_step_strain"] is None
    assert result["optimizer"]["max_cell_total_strain"] is None
    assert len(result["final_stress_eV_per_A3"]) == 6


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
