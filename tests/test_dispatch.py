"""Non-blocking candidate dispatch with a fake scheduler (M16). No real Slurm job is submitted."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from ase.build import bulk

from matsim_agents.discovery.candidates import CandidateManifest, CandidateRecord
from matsim_agents.discovery.dispatch import (
    FACILITIES,
    FacilityProfile,
    batches_of,
    dispatch,
    dispatchable,
    ingest_result,
    node_hours,
    reconcile,
)
from matsim_agents.execution import slurm_jobs as sj

PROFILE = FACILITIES["frontier"]


class FakeScheduler:
    """Records calls and never waits. ``states`` maps job id to sacct state."""

    def __init__(self):
        self.ids = itertools.count(1000)
        self.argvs: list[list[str]] = []
        self.states: dict[str, tuple[str, str, str]] = {}

    def __call__(self, argv):
        self.argvs.append(list(argv))
        if argv[0] == "sbatch":
            job = str(next(self.ids))
            self.states[job] = ("PENDING", "0:0", "00:00:00")
            return sj.CommandResult(0, f"{job}\n")
        wanted = argv[argv.index("-j") + 1].split(",")
        rows = "\n".join(
            f"{j}|{self.states[j][0]}|{self.states[j][1]}|{self.states[j][2]}" for j in wanted
        )
        return sj.CommandResult(0, rows + "\n")

    @property
    def submissions(self):
        return [a for a in self.argvs if a[0] == "sbatch"]


def _records(n, parents=()):
    out = []
    for k in range(n):
        atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)
        atoms.positions[0] += 0.001 * (k + 1)
        atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
        out.append(
            CandidateRecord.from_atoms(
                atoms, generation_method="t", recipe={"k": k}, parent_ids=parents
            )
        )
    return out


def _manifest(tmp_path, n=6):
    manifest = CandidateManifest(tmp_path / "cand.jsonl")
    records = _records(n)
    for r in records:
        manifest.add(r)
    return manifest, records


def _cmd(ids):
    return ("python", "run_batch.py", *ids)


def test_dispatch_batches_submit_once_and_survive_repeat(tmp_path):
    manifest, _ = _manifest(tmp_path, 6)
    sched = FakeScheduler()
    slurm = tmp_path / "jobs.json"
    report = dispatch(manifest, slurm, PROFILE, _cmd, batch_size=3, runner=sched)
    assert len(report.submitted) == 2 and not report.deferred and report.stop_reason is None
    assert all(s == "queued" for s in manifest.status.values())
    again = dispatch(manifest, slurm, PROFILE, _cmd, batch_size=3, runner=sched)
    assert again.submitted == () and len(sched.submissions) == 2
    reopened = CandidateManifest(tmp_path / "cand.jsonl")
    assert dispatchable(reopened) == []


def test_sbatch_arguments_use_profile_and_caps(tmp_path):
    manifest, _ = _manifest(tmp_path, 2)
    sched = FakeScheduler()
    dispatch(manifest, tmp_path / "j.json", PROFILE, _cmd, batch_size=2, gpus=1, runner=sched)
    argv = sched.submissions[0]
    assert argv[argv.index("-A") + 1] == "LRN110" and "-q" in argv and "debug" in argv
    assert "--gpus-per-node=1" in argv and argv[argv.index("-N") + 1] == "1"


def test_gpu_cap_defers_without_error_and_resumes_later(tmp_path):
    manifest, _ = _manifest(tmp_path, 8)
    sched = FakeScheduler()
    slurm = tmp_path / "j.json"
    report = dispatch(manifest, slurm, PROFILE, _cmd, batch_size=2, gpus=1, runner=sched)
    assert len(report.submitted) == sj.MAX_OUTSTANDING_GPU_JOBS == 2
    assert len(report.deferred) == 2 and "outstanding" in report.stop_reason
    assert len(sched.submissions) == 2
    for job in sched.states:
        sched.states[job] = ("COMPLETED", "0:0", "00:05:00")
    sj.refresh_manifest(slurm, runner=sched)
    more = dispatch(manifest, slurm, PROFILE, _cmd, batch_size=2, gpus=1, runner=sched)
    assert len(more.submitted) == 2


def test_max_submissions_limits_a_single_call(tmp_path):
    manifest, _ = _manifest(tmp_path, 6)
    report = dispatch(
        manifest, tmp_path / "j.json", PROFILE, _cmd, batch_size=2, max_submissions=1,
        runner=FakeScheduler(),
    )  # fmt: skip
    assert len(report.submitted) == 1 and len(report.deferred) == 2
    assert report.stop_reason == "submission limit reached"


def test_unverified_facility_refuses_to_dispatch(tmp_path):
    manifest, _ = _manifest(tmp_path, 2)
    sched = FakeScheduler()
    with pytest.raises(sj.SlurmJobError, match="not verified"):
        dispatch(
            manifest, tmp_path / "j.json", FACILITIES["aurora"], _cmd, batch_size=1, runner=sched
        )
    assert sched.argvs == []
    custom = FacilityProfile("x", "A", "debug", "batch", verified=False)
    with pytest.raises(sj.SlurmJobError):
        dispatch(manifest, tmp_path / "j.json", custom, _cmd, batch_size=1, runner=sched)


def test_stage_dependencies_hold_children_until_parents_are_done(tmp_path):
    manifest = CandidateManifest(tmp_path / "c.jsonl")
    (parent,) = _records(1)
    manifest.add(parent)
    child_atoms = bulk("Fe", "bcc", a=2.9, cubic=True)
    child = CandidateRecord.from_atoms(
        child_atoms, generation_method="t", parent_ids=[parent.candidate_id]
    )
    manifest.add(child)
    assert [r.candidate_id for r in dispatchable(manifest)] == [parent.candidate_id]
    ingest_result(manifest, parent.candidate_id, {"energy_eV": -1.0})
    assert [r.candidate_id for r in dispatchable(manifest)] == [child.candidate_id]
    with pytest.raises(ValueError):
        ingest_result(manifest, child.candidate_id, {})


def test_reconcile_never_treats_leaving_the_queue_as_success(tmp_path):
    manifest, records = _manifest(tmp_path, 4)
    sched = FakeScheduler()
    slurm = tmp_path / "j.json"
    dispatch(manifest, slurm, PROFILE, _cmd, batch_size=2, runner=sched)
    first, second = sorted(sched.states)
    sched.states[first] = ("COMPLETED", "0:0", "00:05:00")
    sched.states[second] = ("TIMEOUT", "0:0", "00:15:00")
    entries = sj.refresh_manifest(slurm, runner=sched)
    outcome = reconcile(manifest, entries)
    assert len(outcome["awaiting_result"]) == 2 and len(outcome["failed"]) == 2
    assert not any(s == "done" for s in manifest.status.values())
    failed = outcome["failed"][0]
    assert manifest.status[failed] == "failed"
    assert manifest.details[failed]["job_state"] == "TIMEOUT"
    assert manifest.details[failed]["diagnosis"]
    assert len(records) == 4


def test_measured_node_hours_come_from_scheduler_elapsed_time(tmp_path):
    manifest, _ = _manifest(tmp_path, 2)
    sched = FakeScheduler()
    slurm = tmp_path / "j.json"
    dispatch(manifest, slurm, PROFILE, _cmd, batch_size=2, runner=sched)
    assert node_hours(sj.load_manifest(slurm)) == {}  # nothing measured yet
    (job,) = sched.states
    sched.states[job] = ("COMPLETED", "0:0", "00:30:00")
    entries = sj.refresh_manifest(slurm, runner=sched)
    assert list(node_hours(entries).values()) == [0.5]


def test_batches_are_deterministic_and_validated():
    records = _records(5)
    assert batches_of(records, 2) == batches_of(list(reversed(records)), 2)
    assert [len(b) for b in batches_of(records, 2)] == [2, 2, 1]
    with pytest.raises(ValueError):
        batches_of(records, 0)
