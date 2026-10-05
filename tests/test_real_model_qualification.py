"""Login-node contracts for the bounded compute-node qualification driver."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write

from matsim_agents.active_learning.config import ALConfig

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qualify_real_models", REPO / "deployments/perlmutter/jobs/qualify_real_models.py"
)
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


@pytest.mark.parametrize("backend", ["qe", "vasp"])
def test_config_roundtrip_and_backend_isolation(tmp_path, backend):
    (tmp_path / "elemental.json").write_text("{}")
    cfg = qualification.config(REPO, tmp_path, backend)
    restored = ALConfig.model_validate_json(cfg.model_dump_json())
    assert restored == cfg
    assert cfg.dft.backend == backend
    assert cfg.loop.max_dft_calculations == 6
    assert cfg.acquisition.n_select == 2
    assert cfg.trainer.validation_fraction == 0.2
    assert cfg.trainer.train_script.is_file()
    block = getattr(cfg.dft, backend)
    assert block.nodes_per_job == 1
    assert block.timeout_sec == 1800
    assert block.ranks_per_node * block.threads_per_rank == 64
    assert getattr(cfg.dft, "vasp" if backend == "qe" else "qe") is None
    if backend == "vasp":
        assert block.vasp_wrapper.is_file()
        assert block.incar_template.is_file()


@pytest.mark.parametrize("value", [None, np.nan, np.inf])
def test_comparison_rejects_missing_or_nonfinite_metrics(value):
    metrics = {
        "energy_mae_eV_per_atom": 0.1,
        "formation_energy_mae_eV_per_atom": 0.1,
        "force_mae_eV_per_A": value,
    }
    with pytest.raises(RuntimeError, match="Invalid candidate_metrics.force_mae"):
        qualification.require_metrics({"candidate_metrics": metrics, "incumbent_metrics": metrics})


def test_comparison_accepts_finite_metrics_without_requiring_promotion():
    metrics = {
        "energy_mae_eV_per_atom": 1.0,
        "formation_energy_mae_eV_per_atom": 1.0,
        "force_mae_eV_per_A": 1.0,
    }
    qualification.require_metrics(
        {"approved": False, "candidate_metrics": metrics, "incumbent_metrics": metrics}
    )


def test_bootstrap_refuses_existing_run(tmp_path):
    (tmp_path / "al").mkdir()
    with pytest.raises(RuntimeError, match="cannot overwrite"):
        qualification.bootstrap(REPO, tmp_path)


def test_resume_uses_distinct_reproducible_batches_and_preserves_partitions(tmp_path, monkeypatch):
    (tmp_path / "elemental.json").write_text("{}")
    cfg = qualification.config(REPO, tmp_path)
    (tmp_path / "al-config.json").write_text(cfg.model_dump_json())
    al = tmp_path / "al"
    al.mkdir()
    frames = [bulk("Si", "diamond", a=5.43 + index * 0.01) for index in range(6)]
    write(al / "dataset.extxyz", frames[:1])
    write(al / "validation.extxyz", frames[1:2])
    state = {
        "status": "complete",
        "n_dft_converged": 2,
        "n_dft_failed": 0,
        "training_status": "deferred",
        "model_comparison": None,
    }
    iteration0 = al / "iteration_0000"
    iteration0.mkdir()
    original = json.dumps(state)
    (iteration0 / "state.json").write_text(original)
    metrics = {
        "energy_mae_eV_per_atom": 0.1,
        "formation_energy_mae_eV_per_atom": 0.1,
        "force_mae_eV_per_A": 0.1,
    }
    seeds = []

    def run(config):
        index = config.loop.n_iterations - 1
        seeds.append(config.md.random_seed)
        write(al / "dataset.extxyz", frames[2 * index : 2 * index + 2], append=True)
        complete = {
            **state,
            "training_status": "completed",
            "model_comparison": {
                "candidate_metrics": metrics,
                "incumbent_metrics": metrics,
                "approved": False,
            },
        }
        directory = al / f"iteration_{index:04d}"
        directory.mkdir()
        (directory / "state.json").write_text(json.dumps(complete))

    monkeypatch.setattr(qualification, "run_active_learning", run)
    result = qualification.resume(tmp_path)
    assert seeds == [43, 44]
    assert result["training_count"] == 5
    assert (iteration0 / "state.json").read_text() == original
