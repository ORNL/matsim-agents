from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.e3.production import (
    E3_MLIP_CHECKPOINT_SHA256,
    E3_STRATEGIES,
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
    validate_campaign_manifest,
)


def test_campaign_manifest_has_all_matched_rounds_and_budgets():
    manifest = build_campaign_manifest()

    validate_campaign_manifest(manifest)
    assert manifest["optimizer"] == OPTIMIZER_SETTINGS
    assert manifest["optimizer"]["relax_cell"] is True
    assert manifest["optimizer"]["relative_increase_threshold"] == 0.2
    assert manifest["optimizer"]["max_cell_step_strain"] == 0.1
    assert manifest["optimizer"]["max_cell_total_strain"] == 0.5
    assert manifest["optimizer"]["max_optimization_seconds"] == 60
    assert manifest["optimizer"]["optimization_timeout_policy"] == "between_steps_v1"
    assert manifest["optimizer"]["mof_connectivity_version"] == "atom-indexed-v1"
    assert manifest["optimizer"]["phosphorene_cell_mask"] == [True, True, False, False, False, True]
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
        "provider": "copilot",
        "model": "gemini-4-carbon",
        "decoding": "provider_default",
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
    assert text.index("export PYTHONPATH=") < text.index("budget-reserve")


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
    assert allocations == [{"action": remaining, "count": 1}]


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
        and row["strategy"] == E3_STRATEGIES[0]
        and row["seed"] == REPLICATE_SEEDS[0]
        and row["round"] == 1
    )
    spec = {
        **spec,
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
    assert allocations == [{"action": allowed, "count": 2}]
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


def test_completed_fixed_cell_round_cannot_be_resumed(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(build_campaign_manifest()))
    round_dir = (
        tmp_path
        / "runs"
        / "bcc_refractory"
        / "rule_based_adaptation"
        / "seed-20261111"
        / "round-01"
    )
    round_dir.mkdir(parents=True)
    (round_dir / "round-summary.json").write_text(json.dumps({"status": "complete"}))
    (round_dir / "round-manifest.json").write_text(json.dumps({"optimizer": {"relax_cell": False}}))
    with pytest.raises(ValueError, match="different optimization protocol"):
        run_replicate(
            manifest_path,
            tmp_path / "runs",
            "bcc_refractory",
            "rule_based_adaptation",
            20261111,
            worker_count=1,
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
def test_worker_applies_cell_policy_for_each_family(tmp_path, monkeypatch, material):
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

    class Cache:
        def __init__(self):
            self.calculator = EMT()

        def get(self, args):
            return relaxation._RelaxCalculator(self.calculator, 0, 0, "")

    def fake_relax(args, *, calculators, step_guard):
        assert args.relax_cell
        assert args.require_native_stress
        assert args.stress_tolerance_eV_per_A3 == 0.01
        assert args.relative_increase_threshold == 0.2
        assert args.max_cell_step_strain == 0.1
        assert args.max_cell_total_strain == 0.5
        assert args.max_optimization_seconds == 60
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
