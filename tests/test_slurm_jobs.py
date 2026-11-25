"""Non-blocking Slurm integration. A fake runner stands in for the scheduler."""

import pytest

from matsim_agents.execution.slurm_jobs import (
    CommandResult,
    JobSpec,
    SlurmJobError,
    build_sbatch_argv,
    load_manifest,
    parse_time_limit,
    refresh_manifest,
    retry_job,
    submit_job,
    summarize,
)


class FakeScheduler:
    def __init__(self, states=None):
        self.calls: list[list[str]] = []
        self.next_id = 1000
        self.states = states or {}

    def __call__(self, argv):
        self.calls.append(list(argv))
        if argv[0] == "sbatch":
            self.next_id += 1
            return CommandResult(0, f"{self.next_id};frontier\n")
        if argv[0] == "sacct":
            ids = argv[argv.index("-j") + 1].split(",")
            rows = [f"{i}|{self.states.get(i, 'RUNNING')}|0:0|00:01:00" for i in ids]
            return CommandResult(0, "\n".join(rows) + "\n")
        raise AssertionError(argv)

    def count(self, name):
        return sum(c[0] == name for c in self.calls)


def _spec(name="smoke", gpus=1, **kw):
    return JobSpec(
        name=name,
        command=("python", "-c", "print('hi there')"),
        gpus=gpus,
        **{"account": "ACCOUNT", **kw},
    )


def test_manifest_is_not_created_before_a_submission(tmp_path):
    path = tmp_path / "out" / "jobs.json"
    assert load_manifest(path) == [] and refresh_manifest(path, runner=FakeScheduler()) == []
    assert not path.parent.exists()


def test_submit_builds_a_safe_non_blocking_command(tmp_path):
    sched = FakeScheduler()
    entry, submitted = submit_job(_spec(), tmp_path / "jobs.json", runner=sched)
    argv = sched.calls[0]
    assert submitted and entry.job_id == "1001" and entry.state == "SUBMITTED"
    assert argv[:2] == ["sbatch", "--parsable"]
    assert argv[argv.index("-A") + 1] == "ACCOUNT" and argv[argv.index("-q") + 1] == "debug"
    assert argv[argv.index("-t") + 1] == "00:15:00"
    assert "--gpus-per-node=1" in argv and "--gpu-bind=closest" in argv
    assert argv[-2] == "--wrap" and argv[-1] == "python -c 'print('\"'\"'hi there'\"'\"')'"
    assert sched.count("sacct") == 0 and sched.count("squeue") == 0
    assert [e.job_id for e in load_manifest(tmp_path / "jobs.json")] == ["1001"]


def test_cpu_job_has_no_gpu_flags():
    argv = build_sbatch_argv(_spec(gpus=0))
    assert not any(a.startswith("--gpus") or a.startswith("--gpu-bind") for a in argv)


def test_duplicate_spec_is_not_submitted_twice(tmp_path):
    sched = FakeScheduler()
    path = tmp_path / "jobs.json"
    first, _ = submit_job(_spec(), path, runner=sched)
    again, submitted = submit_job(_spec(), path, runner=sched)
    assert not submitted and again.job_id == first.job_id and sched.count("sbatch") == 1


def test_gpu_cap_refuses_a_third_outstanding_job_but_not_cpu_jobs(tmp_path):
    sched = FakeScheduler()
    path = tmp_path / "jobs.json"
    submit_job(_spec("a"), path, runner=sched)
    submit_job(_spec("b"), path, runner=sched)
    with pytest.raises(SlurmJobError, match="2 GPU jobs are outstanding"):
        submit_job(_spec("c"), path, runner=sched)
    assert sched.count("sbatch") == 2
    submit_job(_spec("cpu", gpus=0), path, runner=sched)
    assert sched.count("sbatch") == 3


def test_cap_frees_after_refresh_marks_jobs_finished(tmp_path):
    sched = FakeScheduler({"1001": "COMPLETED", "1002": "COMPLETED"})
    path = tmp_path / "jobs.json"
    submit_job(_spec("a"), path, runner=sched)
    submit_job(_spec("b"), path, runner=sched)
    refresh_manifest(path, runner=sched)
    assert sched.count("sacct") == 1
    submit_job(_spec("c"), path, runner=sched)
    assert summarize(load_manifest(path)) == {
        "submitted": 3,
        "completed": 2,
        "failed": 0,
        "outstanding": 1,
        "outstanding_gpu": 1,
    }


def test_refresh_makes_one_call_and_skips_finished_jobs(tmp_path):
    sched = FakeScheduler({"1001": "COMPLETED"})
    path = tmp_path / "jobs.json"
    submit_job(_spec("a"), path, runner=sched)
    refresh_manifest(path, runner=sched)
    refresh_manifest(path, runner=sched)
    assert sched.count("sacct") == 1


def test_failed_job_is_diagnosed_and_retried_exactly_once(tmp_path):
    sched = FakeScheduler({"1001": "TIMEOUT", "1002": "TIMEOUT"})
    path = tmp_path / "jobs.json"
    first, _ = submit_job(_spec("a"), path, runner=sched)
    with pytest.raises(SlurmJobError, match="only failed jobs"):
        retry_job(first.spec_hash, path, diagnosis="x", runner=sched)
    refresh_manifest(path, runner=sched)
    failed = load_manifest(path)[0]
    assert failed.diagnosis == "the job reached its time limit"
    with pytest.raises(SlurmJobError, match="needs a diagnosis"):
        retry_job(first.spec_hash, path, diagnosis=" ", runner=sched)
    longer = _spec("a", time_limit="00:30:00")
    retry = retry_job(
        first.spec_hash, path, diagnosis="time limit too short", spec=longer, runner=sched
    )
    assert retry.retry_of == first.spec_hash and retry.job_id == "1002"
    refresh_manifest(path, runner=sched)
    with pytest.raises(SlurmJobError, match="already retried"):
        retry_job(first.spec_hash, path, diagnosis="again", runner=sched)
    with pytest.raises(SlurmJobError, match="already retried or is itself a retry"):
        retry_job(retry.spec_hash, path, diagnosis="again", runner=sched)
    assert sched.count("sbatch") == 2


def test_retry_with_identical_spec_keeps_hashes_unique(tmp_path):
    sched = FakeScheduler({"1001": "NODE_FAIL"})
    path = tmp_path / "jobs.json"
    first, _ = submit_job(_spec("a"), path, runner=sched)
    refresh_manifest(path, runner=sched)
    retry = retry_job(first.spec_hash, path, diagnosis="node failure", runner=sched)
    hashes = [e.spec_hash for e in load_manifest(path)]
    assert hashes == [first.spec_hash] and retry.retry_of == first.spec_hash


def test_scheduler_failure_leaves_no_manifest(tmp_path):
    path = tmp_path / "out" / "jobs.json"

    def broken(argv):
        return CommandResult(1, "", "invalid account")

    with pytest.raises(SlurmJobError, match="invalid account"):
        submit_job(_spec(), path, runner=broken)
    assert not path.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nodes": 2},
        {"gpus": 2},
        {"time_limit": "UNLIMITED"},
        {"time_limit": "00:00:00"},
        {"time_limit": "03:00:00"},
        {"time_limit": "00:75:00"},
        {"name": "bad name; rm"},
    ],
)
def test_invalid_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _spec(**kwargs)


def test_empty_command_is_rejected():
    with pytest.raises(ValueError):
        JobSpec(name="x", command=(), account="ACCOUNT")


def test_time_parsing():
    assert parse_time_limit("00:15:00") == 900 and parse_time_limit("1-00:00:00") == 86400


def test_spec_hash_depends_on_every_field():
    assert _spec().spec_hash == _spec().spec_hash
    assert _spec().spec_hash != _spec(time_limit="00:10:00").spec_hash
    assert _spec().spec_hash != _spec(gpus=0).spec_hash


def test_manifest_with_wrong_schema_is_refused(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text('{"schema": "other", "jobs": []}')
    with pytest.raises(SlurmJobError, match="not a"):
        load_manifest(path)
