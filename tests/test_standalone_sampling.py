from __future__ import annotations

import json
from contextlib import nullcontext

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import write

from matsim_agents.backends.mlip.config import MACEConfig
from matsim_agents.execution.placement import ResourcePlan
from matsim_agents.workflows.active_learning.config import (
    MDConfig,
    MDSamplingStrategy,
    NebSamplingConfig,
    NEBSamplingStrategy,
    SamplingConfig,
    SamplingRunConfig,
    SeedSourceConfig,
)
from matsim_agents.workflows.active_learning.standalone_sampling import run_sampling


def _patch_sampling_runtime(monkeypatch):
    import matsim_agents.workflows.active_learning.loop as loop
    import matsim_agents.workflows.active_learning.standalone_sampling as standalone

    class FakeModels:
        def __init__(self, placement, *, keep):
            pass

        def calculators(self, mlip_config, enable_dropout):
            return EMT(), []

        def md_pool(self):
            return nullcontext(None)

        def close(self):
            pass

    monkeypatch.setattr(loop, "_SurrogateModels", FakeModels)
    monkeypatch.setattr(ResourcePlan, "place", lambda self, run_dir=None: object())
    monkeypatch.setattr(standalone, "driver_on_mlip_device", lambda placement: nullcontext())


def test_standalone_neb_sampling_runs_without_dft_or_active_learning(tmp_path, monkeypatch):
    initial = Atoms(
        "Cu2",
        positions=[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    initial.set_array("atom_id", np.array([0, 1]))
    final = initial.copy()
    final.positions[0, 1] += 0.3
    initial_path, final_path = tmp_path / "initial.extxyz", tmp_path / "final.extxyz"
    write(initial_path, initial)
    write(final_path, final)

    _patch_sampling_runtime(monkeypatch)

    config = SamplingRunConfig(
        random_seed=1,
        mlip=MACEConfig(model="small"),
        sampling=SamplingConfig(
            strategies=[
                NEBSamplingStrategy(
                    method="neb",
                    parameters=NebSamplingConfig(
                        initial_path=initial_path,
                        final_path=final_path,
                        n_images=4,
                        max_steps=1,
                        min_distance_A=0.5,
                    ),
                )
            ]
        ),
        resources=ResourcePlan(pools={"mlip": {}}),
    )

    result = run_sampling(config, tmp_path / "run")
    saved = json.loads((tmp_path / "run" / "sampling-results.json").read_text())

    assert result["n_candidates"] == 2
    assert result["neb_pathways"][0]["result"]["state"] in {
        "converged_unqualified",
        "unconverged",
    }
    assert (tmp_path / "run" / "candidates.extxyz").is_file()
    assert saved["state"] == "complete"


def test_standalone_md_sampling_runs_without_active_learning(tmp_path, monkeypatch):
    structure = Atoms(
        "Cu2",
        positions=[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    seed = tmp_path / "seed.extxyz"
    write(seed, structure)
    _patch_sampling_runtime(monkeypatch)
    config = SamplingRunConfig(
        random_seed=2,
        mlip=MACEConfig(model="small"),
        sampling=SamplingConfig(
            strategies=[
                MDSamplingStrategy(
                    method="md",
                    parameters=MDConfig(
                        seed_source=SeedSourceConfig(paths=[seed]),
                        n_steps=2,
                        sample_every=1,
                        temperature_K=10.0,
                        max_force_threshold_eV_per_A=1.0e6,
                    ),
                )
            ]
        ),
        resources=ResourcePlan(pools={"mlip": {}}),
    )

    result = run_sampling(config, tmp_path / "md-run")
    saved = json.loads((tmp_path / "md-run" / "sampling-results.json").read_text())

    assert result["n_candidates"] > 0
    assert (tmp_path / "md-run" / "candidates.extxyz").is_file()
    assert saved["candidates"][0]["sampling_method"] == "md"
