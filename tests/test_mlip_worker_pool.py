"""MLIP worker pool, calculator reuse, and pooled phase exploration (resource dispatch phase 3)."""

from __future__ import annotations

import operator
import os
import socket

import pytest

from matsim_agents.backends.mlip import relaxation
from matsim_agents.backends.mlip.pool import MLIPWorkerPool, WorkerOutcome, WorkerSlot
from matsim_agents.backends.mlip.relaxation import CalculatorCache, RelaxStructureInput
from matsim_agents.backends.mlip.results import RelaxationResult
from matsim_agents.discovery.seeds import PhaseCandidate
from matsim_agents.execution.allocation import Allocation
from matsim_agents.execution.placement import ResourcePlan


def _uma_args(**overrides) -> RelaxStructureInput:
    values = {"structure_path": "Si.vasp", "mlip_backend": "uma", "mlp_device": "cpu"}
    return RelaxStructureInput(**{**values, **overrides})


class _Calculator:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


def test_calculator_cache_reuses_and_evicts(monkeypatch):
    built = []

    def build(args):
        built.append(args.uma_model_name)
        return relaxation._RelaxCalculator(_Calculator(), 0, 0, None)

    monkeypatch.setattr(relaxation, "_relaxation_calculator", build)
    cache = CalculatorCache(capacity=2)
    first = cache.get(_uma_args())
    again = cache.get(_uma_args(structure_path="Other.vasp", maxiter=5))
    assert again.calculator is first.calculator
    assert first.calculator.resets == 1
    cache.get(_uma_args(uma_model_name="b"))
    cache.get(_uma_args(uma_model_name="c"))
    cache.get(_uma_args())
    assert built == ["uma-s-1p1", "b", "c", "uma-s-1p1"]
    assert cache.loads == 4


def test_calculator_cache_capacity_must_be_positive():
    with pytest.raises(ValueError, match="capacity"):
        CalculatorCache(capacity=0)


def test_local_pool_returns_outcomes_in_order():
    def maybe_fail(value):
        if value == 2:
            raise ValueError("bad item")
        return value * 10

    outcomes = MLIPWorkerPool().map(maybe_fail, [1, 2, 3])
    assert [outcome.value for outcome in outcomes] == [10, None, 30]
    assert isinstance(outcomes[1].error, ValueError)
    assert not MLIPWorkerPool().uses_workers


def test_local_pool_relaxations_share_one_model(monkeypatch):
    loads = []

    def build(args):
        loads.append(args)
        return relaxation._RelaxCalculator(_Calculator(), 0, 0, None)

    def fake_relax(args, *, calculators=None):
        calculators.get(args)
        return RelaxationResult(
            structure_path=args.structure_path,
            optimized_structure_path="opt",
            trajectory_path="traj",
            log_csv_path="log",
            final_energy_eV=-1.0,
            final_max_force_eV_per_A=0.01,
            num_steps=1,
            converged=True,
        )

    monkeypatch.setattr(relaxation, "_relaxation_calculator", build)
    monkeypatch.setattr(relaxation, "relax_with_mlip", fake_relax)
    pool = MLIPWorkerPool()
    outcomes = pool.relax([_uma_args(structure_path=f"{i}.vasp") for i in range(3)])
    assert [outcome.value.structure_path for outcome in outcomes] == ["0.vasp", "1.vasp", "2.vasp"]
    assert len(loads) == 1
    assert pool.relax_one(_uma_args()).converged


def test_worker_processes_get_one_device_each():
    slots = [
        WorkerSlot(device=1, env={"CUDA_VISIBLE_DEVICES": "1"}),
        WorkerSlot(device=3, env={"CUDA_VISIBLE_DEVICES": "3"}),
    ]
    with MLIPWorkerPool(slots) as pool:
        assert pool.uses_workers and pool.concurrency == 2
        masks = {outcome.value for outcome in pool.map(os.getenv, ["CUDA_VISIBLE_DEVICES"] * 6)}
        assert masks == {"1", "3"}
        ordered = pool.map(operator.neg, range(8))
        assert [outcome.value for outcome in ordered] == [-index for index in range(8)]


def test_crashed_worker_fails_only_its_task_and_is_replaced():
    with MLIPWorkerPool([WorkerSlot(device=0, env={"CUDA_VISIBLE_DEVICES": "0"})]) as pool:
        crashed = pool.map(os._exit, [3])
        assert "exited" in str(crashed[0].error)
        after = pool.map(os.getenv, ["CUDA_VISIBLE_DEVICES"])
        assert after[0].value == "0"


def _placement(pools: dict, nodes=None, scheduler="local", preset="perlmutter-gpu"):
    nodes = nodes or (socket.gethostname(),)
    plan = ResourcePlan.model_validate({"preset": preset, "pools": pools})
    return plan.resolve(Allocation(scheduler, tuple(nodes), 4))


def test_from_placement_builds_one_worker_per_mlip_device():
    placement = _placement(
        {
            "llm": {
                "devices": [0],
                "servers": [{"name": "p", "kind": "managed-vllm", "model": "/models/x"}],
            },
            "mlip": {"devices": [1, 2, 3], "max_concurrent": 2, "memory_fraction": 0.5},
        }
    )
    pool = MLIPWorkerPool.from_placement(placement)
    assert [slot.device for slot in pool.slots] == [1, 2]
    assert pool.slots[0].env == {"CUDA_VISIBLE_DEVICES": "1"}
    assert pool.slots[1].memory_fraction == 0.5


def test_from_placement_without_mlip_pool_runs_locally():
    assert not MLIPWorkerPool.from_placement(None).uses_workers
    placement = _placement({"dft": {"nodes": "all"}})
    assert not MLIPWorkerPool.from_placement(placement).uses_workers


def test_from_placement_rejects_multi_node_pools():
    placement = _placement({"mlip": {"nodes": 2}}, nodes=("a", "b"), scheduler="slurm")
    with pytest.raises(ValueError, match="set resources.pools.mlip.nodes to 1"):
        MLIPWorkerPool.from_placement(placement)


def test_from_placement_rejects_pool_on_another_node():
    placement = _placement({"mlip": {}}, nodes=("not-this-host",), scheduler="slurm")
    with pytest.raises(ValueError, match="place the mlip pool on the node"):
        MLIPWorkerPool.from_placement(placement)


class _StubPool:
    def __init__(self, fail_index: int):
        self.fail_index = fail_index
        self.inputs = []

    def relax(self, inputs):
        self.inputs = list(inputs)
        outcomes = []
        for index, args in enumerate(inputs):
            if index == self.fail_index:
                outcomes.append(WorkerOutcome(error=RuntimeError("diverged")))
                continue
            outcomes.append(
                WorkerOutcome(
                    value=RelaxationResult(
                        structure_path=args.structure_path,
                        optimized_structure_path=args.structure_path,
                        trajectory_path="traj",
                        log_csv_path="log",
                        final_energy_eV=-float(index),
                        final_max_force_eV_per_A=0.01,
                        num_steps=1,
                        converged=True,
                    )
                )
            )
        return outcomes


def test_pooled_exploration_keeps_candidate_order_and_isolates_failures(tmp_path, monkeypatch):
    from matsim_agents.discovery import wrapper

    candidates = [
        PhaseCandidate(
            formula="Si",
            structure_path=str(tmp_path / f"seed{index}.vasp"),
            prototype_id=f"proto{index}",
        )
        for index in range(4)
    ]
    monkeypatch.setattr(wrapper, "generate_seeds", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(wrapper, "score_stability", lambda *args, **kwargs: None)
    events = []
    pool = _StubPool(fail_index=1)
    result = wrapper.explore_composition(
        "Si",
        mlip_backend="uma",
        output_dir=str(tmp_path),
        max_relaxations=3,
        mlip_pool=pool,
        on_phase_start=lambda cand: events.append(("start", cand.prototype_id)),
        on_relaxation_attempt=lambda: events.append(("attempt",)),
        on_phase_done=lambda cand, res: events.append(("done", cand.prototype_id)),
    )
    assert [args.structure_path for args in pool.inputs] == [
        c.structure_path for c in candidates[:3]
    ]
    assert [item.structure_path for item in result.relaxations] == [
        candidates[0].structure_path,
        candidates[2].structure_path,
    ]
    assert result.failures == ["proto1: diverged"]
    assert result.unrelaxed_candidate_paths == [candidates[3].structure_path]
    assert events == [
        ("start", "proto0"),
        ("attempt",),
        ("start", "proto1"),
        ("attempt",),
        ("start", "proto2"),
        ("attempt",),
        ("done", "proto0"),
        ("done", "proto2"),
    ]


def test_exploration_rejects_both_relax_fn_and_pool(tmp_path):
    from matsim_agents.discovery.wrapper import explore_composition

    with pytest.raises(ValueError, match="either relax_fn or mlip_pool"):
        explore_composition(
            "Si",
            output_dir=str(tmp_path),
            relax_fn=lambda args: None,
            mlip_pool=MLIPWorkerPool(),
        )


def test_worker_masks_follow_the_driver_mask(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    placement = _placement({"mlip": {"devices": [0, 1]}})
    pool = MLIPWorkerPool.from_placement(placement)
    assert [slot.env["CUDA_VISIBLE_DEVICES"] for slot in pool.slots] == ["2", "3"]


def test_worker_masks_reject_devices_the_driver_cannot_see(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    placement = _placement({"mlip": {"devices": [0, 1]}})
    with pytest.raises(ValueError, match="give the driver every device of pool 'mlip'"):
        MLIPWorkerPool.from_placement(placement)


# -- review fixes: driver device and model release -------------------------------------


class _FakeDeviceModule:
    def __init__(self):
        self.current = 0
        self.calls = []

    def is_available(self):
        return True

    def current_device(self):
        return self.current

    def set_device(self, ordinal):
        self.calls.append(("set_device", ordinal))
        self.current = ordinal

    def set_per_process_memory_fraction(self, fraction, ordinal):
        self.calls.append(("fraction", fraction, ordinal))

    def empty_cache(self):
        self.calls.append(("empty_cache",))


def test_driver_runs_on_the_mlip_device_with_its_budget(monkeypatch):
    from matsim_agents.backends.mlip import pool as pool_module

    fake = _FakeDeviceModule()
    monkeypatch.setattr(pool_module, "_torch_device_module", lambda vendor: fake)
    placement = _placement(
        {
            "llm": {
                "devices": [0],
                "servers": [{"name": "p", "kind": "managed-vllm", "model": "org/x"}],
            },
            "mlip": {"devices": [1, 2], "memory_fraction": 0.4},
        }
    )
    with pool_module.driver_on_mlip_device(placement) as ordinal:
        assert ordinal == 1
        assert fake.current == 1
    assert fake.calls == [
        ("set_device", 1),
        ("fraction", 0.4, 1),
        ("fraction", 1.0, 1),
        ("set_device", 0),
    ]


def test_driver_binding_is_a_no_op_without_an_mlip_pool(monkeypatch):
    from matsim_agents.backends.mlip import pool as pool_module

    monkeypatch.setattr(
        pool_module, "_torch_device_module", lambda vendor: pytest.fail("no device change")
    )
    with pool_module.driver_on_mlip_device(None) as ordinal:
        assert ordinal is None
    with pool_module.driver_on_mlip_device(_placement({"dft": {"nodes": "all"}})) as ordinal:
        assert ordinal is None


def test_close_drops_local_models_and_restarts_lazily(monkeypatch):
    built = []

    def build(args):
        built.append(args)
        return relaxation._RelaxCalculator(_Calculator(), 0, 0, None)

    def fake_relax(args, *, calculators=None):
        calculators.get(args)
        return RelaxationResult(
            structure_path=args.structure_path,
            optimized_structure_path="opt",
            trajectory_path="traj",
            log_csv_path="log",
            final_energy_eV=-1.0,
            final_max_force_eV_per_A=0.01,
            num_steps=1,
            converged=True,
        )

    monkeypatch.setattr(relaxation, "_relaxation_calculator", build)
    monkeypatch.setattr(relaxation, "relax_with_mlip", fake_relax)
    pool = MLIPWorkerPool()
    pool.relax([_uma_args()])
    pool.close()
    assert pool._local_calculators is None
    pool.relax([_uma_args()])
    assert len(built) == 2


def test_chat_releases_exploration_models_only_for_real_active_learning(monkeypatch):
    from matsim_agents.orchestration import interactive_chat
    from matsim_agents.orchestration.interactive_chat import (
        DiscoveryChatConfig,
        DiscoveryChatSession,
    )

    released = []
    monkeypatch.setattr(interactive_chat, "release_device_memory", lambda: released.append(1))

    class Pool:
        closed = 0

        def close(self):
            Pool.closed += 1

    dry = DiscoveryChatSession(config=DiscoveryChatConfig(llm_only=True), mlip_pool=Pool())
    dry.release_mlip_models()
    assert Pool.closed == 0 and not released
    real = DiscoveryChatSession(
        config=DiscoveryChatConfig(llm_only=True, active_learning_dry_run=False),
        mlip_pool=Pool(),
    )
    real.release_mlip_models()
    assert Pool.closed == 1 and released == [1]


def test_cache_eviction_frees_memory_before_loading(monkeypatch):
    from matsim_agents.backends.mlip import pool as pool_module

    events = []
    monkeypatch.setattr(pool_module, "release_device_memory", lambda: events.append("free"))

    def build(args):
        events.append(f"load {args.uma_model_name}")
        return relaxation._RelaxCalculator(_Calculator(), 0, 0, None)

    monkeypatch.setattr(relaxation, "_relaxation_calculator", build)
    cache = CalculatorCache(capacity=1)
    cache.get(_uma_args(uma_model_name="a"))
    cache.get(_uma_args(uma_model_name="a"))
    cache.get(_uma_args(uma_model_name="b"))
    assert events == ["load a", "free", "load b"]


def test_local_training_mask_follows_the_driver_mask(monkeypatch):
    from matsim_agents.execution.placement import ProcessLaunch

    launch = ProcessLaunch(
        hosts=("n0",),
        devices=(1,),
        ranks_per_node=1,
        threads_per_rank=1,
        env={"CUDA_VISIBLE_DEVICES": "1"},
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert launch.environment(local=True)["CUDA_VISIBLE_DEVICES"] == "5"
    # Launcher-started steps keep node-wide indices for srun/mpiexec.
    assert launch.environment()["CUDA_VISIBLE_DEVICES"] == "1"
