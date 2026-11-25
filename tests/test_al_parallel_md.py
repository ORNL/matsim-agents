"""Per-seed random streams and concurrent active-learning MD."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write

from matsim_agents.backends.mlip.config import UMAConfig
from matsim_agents.backends.mlip.pool import WorkerOutcome
from matsim_agents.core.seeding import numpy_rng
from matsim_agents.workflows.active_learning import md_sampler as candidates_module
from matsim_agents.workflows.active_learning.config import MDConfig, SeedSourceConfig
from matsim_agents.workflows.active_learning.md_sampler import _SeedTask, sample_md_candidates


def _seeds(tmp_path: Path) -> list[Path]:
    paths = []
    for index, scale in enumerate((1.0, 1.01, 0.99)):
        atoms = bulk("Cu", "fcc", a=3.6 * scale, cubic=True).repeat((2, 1, 1))
        path = tmp_path / f"seed{index}.extxyz"
        write(path, atoms)
        paths.append(path)
    return paths


def _md(seed_paths: list[Path]) -> MDConfig:
    return MDConfig(
        seed_source=SeedSourceConfig(kind="paths", paths=seed_paths),
        n_steps=20,
        sample_every=5,
        temperature_K=600,
    )


def _signature(candidates) -> list[tuple[str, bytes]]:
    return [
        (item.candidate_id, np.round(item.atoms.positions, 10).tobytes()) for item in candidates
    ]


class _ReverseOrderPool:
    """Runs tasks in this process in reverse order; returns outcomes in submission order."""

    uses_workers = True

    def __init__(self):
        self.calls = 0

    def map(self, fn, items):
        self.calls += 1
        items = list(items)
        values = {index: fn(item) for index, item in reversed(list(enumerate(items)))}
        return [WorkerOutcome(value=values[index]) for index in range(len(items))]


def test_md_is_reproducible_and_seed_dependent(tmp_path):
    seeds = _seeds(tmp_path)

    def sample(name, seed):
        return sample_md_candidates(
            _md(seeds), EMT(), tmp_path / name, seed_paths=seeds, random_seed=seed
        )

    first = sample("a", 7)
    assert first and _signature(first) == _signature(sample("b", 7))
    assert _signature(first) != _signature(sample("c", 8))


def test_a_seed_structure_trajectory_does_not_depend_on_later_structures(tmp_path):
    seeds = _seeds(tmp_path)
    alone = sample_md_candidates(_md(seeds[:1]), EMT(), tmp_path / "a", seeds[:1], random_seed=7)
    all_ = sample_md_candidates(_md(seeds), EMT(), tmp_path / "b", seeds, random_seed=7)
    assert _signature(alone) == _signature(all_)[: len(alone)]


def test_concurrent_md_matches_sequential_md(tmp_path, monkeypatch):
    seeds = _seeds(tmp_path)
    sequential = sample_md_candidates(
        _md(seeds), EMT(), tmp_path / "seq", seed_paths=seeds, random_seed=7
    )

    import matsim_agents.backends.mlip.calculator as calculator_module

    monkeypatch.setattr(calculator_module, "make_mlip_calculator", lambda cfg, **kwargs: EMT())
    monkeypatch.setattr(candidates_module, "_WORKER_CALCULATORS", {})
    pool = _ReverseOrderPool()
    pooled = sample_md_candidates(
        _md(seeds),
        None,
        tmp_path / "pool",
        seed_paths=seeds,
        random_seed=7,
        mlip_pool=pool,
        mlip_cfg=UMAConfig(),
    )
    assert pool.calls == 1
    assert _signature(pooled) == _signature(sequential)
    assert [item.seed_path for item in pooled] == sorted(item.seed_path for item in pooled)


def test_concurrent_md_requires_mlip_config(tmp_path):
    seeds = _seeds(tmp_path)
    with pytest.raises(ValueError, match="needs mlip_cfg"):
        sample_md_candidates(
            _md(seeds),
            None,
            tmp_path,
            seed_paths=seeds,
            random_seed=7,
            mlip_pool=_ReverseOrderPool(),
        )


def test_failed_seed_names_the_seed(tmp_path):
    seeds = _seeds(tmp_path)

    class FailingPool:
        uses_workers = True

        def map(self, fn, items):
            return [WorkerOutcome(error=RuntimeError("worker died")) for _ in items]

    with pytest.raises(RuntimeError, match="seed0.extxyz failed: worker died"):
        sample_md_candidates(
            _md(seeds),
            None,
            tmp_path,
            seed_paths=seeds,
            random_seed=7,
            mlip_pool=FailingPool(),
            mlip_cfg=UMAConfig(),
        )


def test_seed_task_pickles_with_its_random_stream(tmp_path):
    rng = numpy_rng(3, 0)
    task = _SeedTask("{}", "{}", False, "seed.extxyz", str(tmp_path), rng)
    restored = pickle.loads(pickle.dumps(task))
    assert np.array_equal(restored.rng.standard_normal(3), rng.standard_normal(3))
