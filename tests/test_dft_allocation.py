from pathlib import Path
from threading import Lock
from time import sleep

import pytest
from ase import Atoms

from matsim_agents.backends.dft.base import DFTJobSpec, DFTResult
from matsim_agents.backends.dft.runner import run_dft_batch
from matsim_agents.execution.allocation import Allocation, discover_allocation
from matsim_agents.execution.placement import ResourcePlan


def test_pbs_allocation_deduplicates_nodefile(monkeypatch, tmp_path: Path):
    nodefile = tmp_path / "nodes"
    nodefile.write_text("aurora1\naurora1\naurora2\naurora2\n")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NUM_NODES", raising=False)
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    allocation = discover_allocation()
    assert allocation.scheduler == "pbs"
    assert allocation.nodes == ("aurora1", "aurora2")
    assert allocation.gpus_per_node is None


def test_slurm_allocation_reports_gpu_count(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n0,n1")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "gpu:a100:4")
    allocation = discover_allocation()
    assert allocation.nodes == ("n0", "n1")
    assert allocation.gpus_per_node == 4


def _dft_pool(nodes: int, **dft):
    plan = ResourcePlan.model_validate(
        {"preset": "aurora", "pools": {"dft": {"nodes": "all", "devices": "all", **dft}}}
    )
    allocation = Allocation("pbs", tuple(f"n{index}" for index in range(nodes)))
    return plan.resolve(allocation).pool("dft")


class _Backend:
    name = "stub"
    timeout_sec = 10

    def __init__(self):
        self.lock = Lock()
        self.active: set[tuple[str, ...]] = set()
        self.overlap = False
        self.peak = 0

    def run_one(self, spec):
        hosts = spec.require_launch().hosts
        with self.lock:
            if hosts in self.active:
                self.overlap = True
            self.active.add(hosts)
            self.peak = max(self.peak, len(self.active))
        sleep(0.01)
        with self.lock:
            self.active.remove(hosts)
        return DFTResult("stub", spec.work_dir, 0, True, 0.0, None, None, 1, 0.01, spec.atoms)


def _specs(tmp_path, count):
    return [
        DFTJobSpec(str(index), Atoms("H"), str(tmp_path / f"job-{index}")) for index in range(count)
    ]


def test_dispatcher_never_overlaps_jobs_on_the_same_group(tmp_path):
    backend = _Backend()
    specs = _specs(tmp_path, 6)
    results = run_dft_batch(specs, backend, _dft_pool(4, nodes_per_job=2))
    assert len(results) == 6
    assert backend.overlap is False
    assert {spec.launch.hosts for spec in specs} == {("n0", "n1"), ("n2", "n3")}
    launch = specs[0].launch
    assert launch.wrapper_args() == ["2", "12", "8"]
    assert launch.environment({})["ZE_AFFINITY_MASK"] == ",".join(map(str, range(12)))


def test_dispatcher_respects_pool_concurrency_cap(tmp_path):
    backend = _Backend()
    run_dft_batch(_specs(tmp_path, 6), backend, _dft_pool(3, max_concurrent=1))
    assert backend.peak == 1


def test_dispatcher_requires_dft_launches(tmp_path):
    plan = ResourcePlan.model_validate({"preset": "aurora", "pools": {"mlip": {}}})
    pool = plan.resolve(Allocation("pbs", ("n0",))).pool("mlip")
    with pytest.raises(ValueError, match="no DFT launches"):
        run_dft_batch(_specs(tmp_path, 1), _Backend(), pool)


def test_backend_requires_a_launch(tmp_path):
    spec = DFTJobSpec("x", Atoms("H"), str(tmp_path))
    with pytest.raises(ValueError, match="dft pool of a resources: plan"):
        spec.require_launch()
