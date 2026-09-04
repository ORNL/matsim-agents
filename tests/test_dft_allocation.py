from pathlib import Path
from threading import Lock
from time import sleep

import pytest
from ase import Atoms

from matsim_agents.active_learning.dft_backend import DFTJobSpec, DFTResult
from matsim_agents.active_learning.dft_runner import run_dft_batch
from matsim_agents.execution.allocation import (
    Allocation,
    discover_allocation,
    validate_dft_resources,
)


def test_pbs_allocation_deduplicates_nodefile(monkeypatch, tmp_path: Path):
    nodefile = tmp_path / "nodes"
    nodefile.write_text("aurora1\naurora1\naurora2\naurora2\n")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NUM_NODES", raising=False)
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    monkeypatch.setenv("MATSIM_GPUS_PER_NODE", "12")
    allocation = discover_allocation()
    assert allocation.scheduler == "pbs"
    assert allocation.nodes == ("aurora1", "aurora2")
    assert allocation.gpus_per_node == 12


def test_allocation_groups_are_disjoint_and_drop_no_full_group():
    allocation = Allocation("slurm", ("n0", "n1", "n2", "n3"), 8)
    assert allocation.groups(2) == [("n0", "n1"), ("n2", "n3")]


def test_resource_validation_requires_full_gpu_utilization():
    allocation = Allocation("slurm", ("n0", "n1"), 8)
    with pytest.raises(ValueError, match="exactly the allocated"):
        validate_dft_resources(allocation, nodes_per_job=1, ranks_per_node=4)
    validate_dft_resources(allocation, nodes_per_job=1, ranks_per_node=8)


def test_resource_validation_rejects_oversized_job():
    allocation = Allocation("pbs", ("n0",), 12)
    with pytest.raises(ValueError, match="allocation has 1"):
        validate_dft_resources(allocation, nodes_per_job=2, ranks_per_node=12)


def test_dispatcher_never_overlaps_jobs_on_the_same_partition(monkeypatch, tmp_path):
    nodefile = tmp_path / "nodes"
    nodefile.write_text("n0\nn1\n")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NUM_NODES", raising=False)
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    monkeypatch.setenv("MATSIM_GPUS_PER_NODE", "1")

    class Backend:
        name = "stub"
        nodes_per_job = 1
        ranks_per_node = 1
        threads_per_rank = 1
        timeout_sec = 10

        def __init__(self):
            self.lock = Lock()
            self.active: set[str] = set()
            self.overlap = False

        def run_one(self, spec):
            node = spec.assigned_nodes[0]
            with self.lock:
                if node in self.active:
                    self.overlap = True
                self.active.add(node)
            sleep(0.01)
            with self.lock:
                self.active.remove(node)
            return DFTResult("stub", spec.work_dir, 0, True, 0.0, None, None, 1, 0.01, spec.atoms)

    backend = Backend()
    specs = [
        DFTJobSpec(str(index), Atoms("H"), str(tmp_path / f"job-{index}")) for index in range(6)
    ]
    results = run_dft_batch(specs, backend)
    assert len(results) == 6
    assert backend.overlap is False
    assert {spec.assigned_nodes for spec in specs} == {("n0",), ("n1",)}
