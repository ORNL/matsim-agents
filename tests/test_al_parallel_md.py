"""Per-seed random streams and concurrent active-learning MD."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write

from matsim_agents.active_learning import candidates as candidates_module
from matsim_agents.active_learning.candidates import (
    _SeedTask,
    sample_md_candidates,
    seed_rngs,
)
from matsim_agents.active_learning.config import MDConfig, SeedSourceConfig
from matsim_agents.backends.mlip.config import MLIPConfig
from matsim_agents.backends.mlip.pool import WorkerOutcome


def _seeds(tmp_path: Path) -> list[Path]:
    paths = []
    for index, scale in enumerate((1.0, 1.01, 0.99)):
        atoms = bulk("Cu", "fcc", a=3.6 * scale, cubic=True).repeat((2, 1, 1))
        path = tmp_path / f"seed{index}.extxyz"
        write(path, atoms)
        paths.append(path)
    return paths


def _md(seed_paths: list[Path], random_seed: int | None = 7) -> MDConfig:
    return MDConfig(
        seed_source=SeedSourceConfig(kind="paths", paths=seed_paths),
        n_steps=20,
        sample_every=5,
        temperature_K=600,
        random_seed=random_seed,
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


def test_seed_rngs_are_reproducible_and_independent():
    first = [rng.standard_normal(4) for rng in seed_rngs(11, 3)]
    again = [rng.standard_normal(4) for rng in seed_rngs(11, 3)]
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    assert not np.array_equal(first[0], first[1])
    assert seed_rngs(None, 2) == [None, None]


def test_seed_stream_does_not_depend_on_later_seeds():
    two = seed_rngs(5, 2)[0].standard_normal(3)
    five = seed_rngs(5, 5)[0].standard_normal(3)
    assert np.array_equal(two, five)


def test_md_is_reproducible(tmp_path):
    seeds = _seeds(tmp_path)
    first = sample_md_candidates(_md(seeds), EMT(), tmp_path / "a", seed_paths=seeds)
    second = sample_md_candidates(_md(seeds), EMT(), tmp_path / "b", seed_paths=seeds)
    assert first and _signature(first) == _signature(second)


def test_concurrent_md_matches_sequential_md(tmp_path, monkeypatch):
    seeds = _seeds(tmp_path)
    sequential = sample_md_candidates(_md(seeds), EMT(), tmp_path / "seq", seed_paths=seeds)

    import matsim_agents.backends.mlip.calculator as calculator_module

    monkeypatch.setattr(calculator_module, "make_mlip_calculator", lambda cfg, **kwargs: EMT())
    monkeypatch.setattr(candidates_module, "_WORKER_CALCULATORS", {})
    pool = _ReverseOrderPool()
    pooled = sample_md_candidates(
        _md(seeds),
        None,
        tmp_path / "pool",
        seed_paths=seeds,
        mlip_pool=pool,
        mlip_cfg=MLIPConfig.model_validate({"backend": "uma", "uma": {}}),
    )
    assert pool.calls == 1
    assert _signature(pooled) == _signature(sequential)
    assert [item.seed_path for item in pooled] == sorted(item.seed_path for item in pooled)


def test_concurrent_md_requires_mlip_config(tmp_path):
    seeds = _seeds(tmp_path)
    with pytest.raises(ValueError, match="needs mlip_cfg"):
        sample_md_candidates(
            _md(seeds), None, tmp_path, seed_paths=seeds, mlip_pool=_ReverseOrderPool()
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
            mlip_pool=FailingPool(),
            mlip_cfg=MLIPConfig.model_validate({"backend": "uma", "uma": {}}),
        )


def test_seed_task_pickles_with_its_random_stream(tmp_path):
    rng = seed_rngs(3, 1)[0]
    task = _SeedTask("{}", "{}", False, "seed.extxyz", str(tmp_path), rng)
    restored = pickle.loads(pickle.dumps(task))
    assert np.array_equal(restored.rng.standard_normal(3), rng.standard_normal(3))
