from __future__ import annotations

import json
import sys
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

from benchmarks.e4_mlip.analyze import consistency, consolidate, plot
from benchmarks.e4_mlip.prepare import freeze_inventory, pbs_templates
from benchmarks.e4_mlip.run import (
    Structure,
    geometry_hash,
    load_inventory,
    main,
    numerical_values,
    resource_plan,
    sha256,
    verify_model,
    worker_task,
)
from matsim_agents.backends.mlip.pool import MLIPWorkerPool, WorkerSlot
from matsim_agents.execution.allocation import Allocation


@pytest.fixture
def sources(tmp_path):
    # Synthetic CPU-only test fixtures, never benchmark performance inputs.
    records = []
    for index in range(120):
        atoms = Atoms(
            "Nb128",
            positions=np.indices((4, 4, 8)).reshape(3, -1).T * 2,
            cell=[10, 10, 18],
            pbc=True,
        )
        atoms.positions[0, 0] += index * 0.0001
        path = tmp_path / f"test-{index}.xyz"
        write(path, atoms, format="extxyz")
        records.append(
            {
                "structure_id": f"test-{index}",
                "family": "bcc_refractory",
                "path": str(path),
                "source": "synthetic test fixture only",
            }
        )
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(records))
    return path


def test_freezes_exact_inventory_and_detects_tampering(sources, tmp_path):
    manifest = tmp_path / "structures.json"
    rows = freeze_inventory(sources, manifest)
    assert len(rows) == 120
    assert load_inventory(manifest) == rows
    with rows[0].path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_inventory(manifest)


@pytest.mark.parametrize("problem", ["count", "duplicate", "wrong-size"])
def test_invalid_inventory_is_not_published(sources, tmp_path, problem):
    rows = json.loads(sources.read_text())
    if problem == "count":
        rows.pop()
    elif problem == "duplicate":
        rows[-1] = rows[0]
    else:
        write(rows[0]["path"], Atoms("Nb", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True))
    sources.write_text(json.dumps(rows))
    output = tmp_path / "manifest.json"
    with pytest.raises(ValueError):
        freeze_inventory(sources, output)
    assert not output.exists()


@pytest.mark.parametrize("devices", [1, 2, 4, 8, 12])
def test_single_node_disjoint_placement(devices):
    placement = resource_plan(devices).resolve(Allocation("preview", ("test-host",)))
    pool = MLIPWorkerPool.from_placement(placement)
    assert len(pool.slots) == devices
    assert len({slot.env["ZE_AFFINITY_MASK"] for slot in pool.slots}) == devices
    assert len(placement.pool("mlip").slices) == 1


def test_unsupported_device_count_rejected():
    with pytest.raises(ValueError, match="Supported"):
        resource_plan(3)


def _probe(value):
    import os

    return {"mask": os.environ["E4B_TEST_MASK"], "value": value}


def test_warmup_runs_exactly_once_on_every_worker():
    with MLIPWorkerPool([WorkerSlot(i, {"E4B_TEST_MASK": str(i)}) for i in range(2)]) as pool:
        results = pool.on_each_worker(_probe, "probe")
        assert all(item.ok for item in results)
        assert [item.value["mask"] for item in results] == ["0", "1"]


def test_pbs_templates_are_single_node_and_default_dry_run(tmp_path):
    pbs_templates(tmp_path)
    files = list(tmp_path.glob("*.pbs"))
    assert len(files) == 5
    for file in files:
        text = file.read_text()
        assert "#PBS -l select=1" in text
        assert "${E4B_EXECUTE:-0}" in text
        assert "qsub" not in text and "mpiexec" not in text


@pytest.mark.parametrize(
    "energy,forces", [(float("nan"), [[0, 0, 0]]), (1, [[1, 2]]), (1, [[float("inf"), 0, 0]])]
)
def test_invalid_numerical_outputs_fail(energy, forces):
    with pytest.raises(ValueError, match="Invalid"):
        numerical_values(energy, forces, 1)


def test_model_species_and_operations_are_gated(tmp_path):
    path = tmp_path / "model"
    path.write_text("metadata fixture")
    file = {"path": str(path), "sha256": sha256(path)}
    metadata = {
        "checkpoint": file,
        "branch_checkpoint": file,
        "config": file,
        "backend": "hydragnn",
        "supported_species": [],
        "prediction_targets": [],
    }
    row = SimpleNamespace(species=["Nb"])
    blockers = verify_model(metadata, [row])
    assert any("species" in item.lower() for item in blockers)
    assert any("Energy/force" in item for item in blockers)
    metadata.update(
        supported_species=["Nb"],
        species_verified=True,
        species_evidence="fixture",
        prediction_targets=["energy", "forces"],
        numerical_tolerances_approved=True,
    )
    assert verify_model(metadata, [row]) == []


def test_dry_run_never_calls_gpu_or_worker(sources, tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    freeze_inventory(sources, manifest)
    checkpoint = tmp_path / "model"
    checkpoint.write_text("test only")
    file = {"path": str(checkpoint), "sha256": sha256(checkpoint)}
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "checkpoint": file,
                "branch_checkpoint": file,
                "config": file,
                "backend": "hydragnn",
                "supported_species": [],
                "prediction_targets": ["energy", "forces"],
            }
        )
    )
    monkeypatch.setattr(
        "benchmarks.e4_mlip.run._device", lambda: pytest.fail("Dry-run initialized a GPU")
    )
    monkeypatch.setattr(
        MLIPWorkerPool, "map", lambda *a: pytest.fail("Dry-run launched numerical tasks")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--manifest",
            str(manifest),
            "--metadata",
            str(metadata),
            "--devices",
            "4",
            "--mode",
            "inference",
            "--repetition",
            "1",
            "--output",
            str(tmp_path / "preflight"),
        ],
    )
    assert main() == 2
    report = json.loads((tmp_path / "preflight/preflight.json").read_text())
    assert report["gpu_initialized"] is False
    assert report["status"] == "NO-GO"
    assert len(report["structure_ids"]) == 120


def test_worker_times_cached_calculator_and_synchronizes(tmp_path, monkeypatch):
    class ToyCalculator(Calculator):
        implemented_properties = ["energy", "forces"]
        model = SimpleNamespace(
            parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type="xpu"))]
        )

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": 1.0, "forces": np.zeros((len(atoms), 3))}

    calls = []
    device = SimpleNamespace(
        synchronize=lambda: calls.append("sync"),
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 123,
        device_count=lambda: 1,
    )
    Entry = namedtuple("Entry", ["calculator", "num_branches", "active_branches", "uq_note"])
    cache = SimpleNamespace(loads=0)
    entry = Entry(ToyCalculator(), 0, 0, None)

    def get(args):
        if cache.loads == 0:
            cache.loads += 1
        entry.calculator.reset()
        return entry

    cache.get = get
    monkeypatch.setattr("benchmarks.e4_mlip.run._device", lambda: device)
    monkeypatch.setattr("benchmarks.e4_mlip.run._worker_calculators", lambda: cache)
    atoms = Atoms("Nb128", positions=np.zeros((128, 3)), cell=[10, 10, 10], pbc=True)
    path = tmp_path / "input.xyz"
    write(path, atoms)
    row = Structure(
        structure_id="test",
        family="bcc_refractory",
        path=path,
        sha256=sha256(path),
        geometry_sha256=geometry_hash(atoms),
        atom_count=128,
        species=["Nb"],
        source="test fixture",
    )
    payload = {
        "structure": row.model_dump(mode="json"),
        "mode": "inference",
        "result_path": str(tmp_path / "result.json"),
        "args": {"structure_path": str(path), "mlip_backend": "mace", "mlp_device": "xpu"},
        "submitted_at": 0,
    }
    first, second = worker_task(payload), worker_task(payload)
    assert first["cold_model"] is True and second["cold_model"] is False
    assert second["cache_loads"] == 1
    assert second["force_evaluations"] == 1
    assert len(calls) >= 10 and second["synchronized"] is True
    assert second["positions_A"] == atoms.positions.tolist()


def record(devices=1, seconds=10):
    row = {
        "structure_id": "test",
        "status": "ok",
        "converged": True,
        "energy_eV": 1.0,
        "forces_eV_A": [[0, 0, 0]],
        "max_force_eV_A": 0,
        "positions_A": [[0, 0, 0]],
        "cell_A": np.eye(3).tolist(),
        "steps": 1,
        "synchronized": True,
        "peak_allocated_bytes": 10,
        "force_evaluations": 2,
    }
    return {
        "backend": "hydragnn",
        "mode": "inference",
        "devices": devices,
        "repetition": 1,
        "measurement_valid": True,
        "attempted": 120,
        "results": [{**row, "structure_id": f"test-{index}"} for index in range(120)],
        "steady_seconds": seconds,
        "manifest_sha256": "a",
        "model_metadata_sha256": "b",
        "precision": "fp64",
        "fmax": 0.02,
        "max_steps": 200,
        "optimizer": "FIRE",
        "relax_cell": False,
        "total_seconds": seconds + 1,
        "cold_phase_seconds": 1,
    }


TOLERANCES = {"energy_eV": 1e-6, "force_eV_A": 1e-5, "position_A": 1e-4}


def test_speedup_efficiency_and_plots(tmp_path):
    rows = consolidate([record(), record(2, 5)], TOLERANCES)
    assert rows[1]["speedup"] == 2 and rows[1]["efficiency"] == 1
    plot(rows, tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 3


@pytest.mark.parametrize("problem", ["energy", "geometry", "convergence", "failed"])
def test_consistency_rejects_changed_outputs(problem):
    baseline, changed = record(), record(2, 5)
    row = changed["results"][0]
    if problem == "energy":
        row["energy_eV"] = 2
    elif problem == "geometry":
        row["positions_A"] = [[0.1, 0, 0]]
    elif problem == "convergence":
        row["converged"] = False
    else:
        row["status"] = "failed"
    assert consistency(baseline, changed, TOLERANCES)
    with pytest.raises(ValueError):
        consolidate([baseline, changed], TOLERANCES)


def test_missing_baseline_and_changed_workload_fail():
    with pytest.raises(ValueError, match="baseline"):
        consolidate([record(2)], TOLERANCES)
    changed = record(2)
    changed["manifest_sha256"] = "changed"
    with pytest.raises(ValueError, match="setting"):
        consolidate([record(), changed], TOLERANCES)
