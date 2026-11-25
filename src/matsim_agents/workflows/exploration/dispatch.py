"""Non-blocking batch dispatch of candidates to a scheduler.

Builds on ``CandidateManifest`` (identity, resume) and ``execution.slurm_jobs`` (duplicate
prevention, the outstanding-GPU-job cap, one-retry rule). Nothing here waits or polls.

Typical cycle:

1. ``submit_pending_batches`` submits what the caps allow and returns.
2. ``slurm_jobs.refresh_manifest`` then ``apply_scheduler_states`` read one ``sacct``.
3. ``record_result`` marks a candidate done once its result payload is parsed.

A job leaving the queue is not success; candidates stay ``queued`` until a result is recorded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from matsim_agents.execution import slurm_jobs as sj
from matsim_agents.workflows.exploration.candidates import CandidateManifest, CandidateRecord


@dataclass(frozen=True)
class FacilityProfile:
    """Scheduler settings for one facility, supplied by the deployment or the application.

    Unverified profiles refuse to dispatch.
    """

    name: str
    account: str
    qos: str
    partition: str
    verified: bool
    note: str = ""


def batches_of(records: Sequence[CandidateRecord], size: int) -> list[tuple[str, ...]]:
    """Deterministic batches by sorted candidate id."""
    if size < 1:
        raise ValueError("batch size must be >= 1")
    ids = sorted(r.candidate_id for r in records)
    return [tuple(ids[i : i + size]) for i in range(0, len(ids), size)]


def batch_name(prefix: str, ids: Sequence[str]) -> str:
    digest = hashlib.sha256("|".join(ids).encode()).hexdigest()[:12]
    return f"{prefix}-{digest}"


def ready_for_dispatch(manifest: CandidateManifest) -> list[CandidateRecord]:
    """Not yet submitted, with every parent candidate already done (stage dependencies)."""
    return [
        r
        for r in manifest.pending()
        if manifest.status[r.candidate_id] in ("generated", "queued")
        and not manifest.details.get(r.candidate_id, {}).get("job_id")
        and all(manifest.status.get(p) == "done" for p in r.parent_ids)
    ]


@dataclass(frozen=True)
class DispatchReport:
    submitted: tuple[tuple[str, str], ...]  # (job name, job id)
    reused: tuple[str, ...]  # job names already in the scheduler manifest
    deferred: tuple[tuple[str, ...], ...]  # batches held back
    stop_reason: str | None


CommandFor = Callable[[Sequence[str]], tuple[str, ...]]


def submit_pending_batches(
    manifest: CandidateManifest,
    slurm_manifest: str | Path,
    profile: FacilityProfile,
    command_for: CommandFor,
    *,
    batch_size: int,
    gpus: int = 0,
    time_limit: str = "00:15:00",
    prefix: str = "matsim-magnetic",
    max_submissions: int | None = None,
    runner: sj.Runner = sj.subprocess_runner,
    cwd: str | None = None,
) -> DispatchReport:
    """Submit pending candidate batches without waiting. Stops at the first scheduler refusal
    (for example the GPU cap) and reports the held-back batches."""
    if not profile.verified:
        raise sj.SlurmJobError(
            f"facility profile {profile.name!r} is not verified: {profile.note or 'no details'}"
        )
    submitted: list[tuple[str, str]] = []
    reused: list[str] = []
    deferred: list[tuple[str, ...]] = []
    stop: str | None = None
    for ids in batches_of(ready_for_dispatch(manifest), batch_size):
        if stop is not None or (max_submissions is not None and len(submitted) >= max_submissions):
            deferred.append(ids)
            stop = stop or "submission limit reached"
            continue
        spec = sj.JobSpec(
            name=batch_name(prefix, ids),
            command=command_for(ids),
            time_limit=time_limit,
            gpus=gpus,
            account=profile.account,
            qos=profile.qos,
            partition=profile.partition,
            cwd=cwd,
        )
        try:
            entry, was_submitted = sj.submit_job(spec, slurm_manifest, runner=runner)
        except sj.SlurmJobError as error:
            deferred.append(ids)
            stop = str(error)
            continue
        if was_submitted:
            submitted.append((spec.name, entry.job_id))
        else:
            reused.append(spec.name)
        for cid in ids:
            manifest.set_status(
                cid,
                "queued",
                {"job_id": entry.job_id, "job_name": spec.name, "spec_hash": entry.spec_hash},
            )
    return DispatchReport(tuple(submitted), tuple(reused), tuple(deferred), stop)


def elapsed_seconds(text: str | None) -> int | None:
    if not text:
        return None
    try:
        return sj.parse_time_limit(text)
    except ValueError:
        return None


def apply_scheduler_states(
    manifest: CandidateManifest, entries: Sequence[sj.JobEntry]
) -> dict[str, list[str]]:
    """Apply scheduler states to candidates after ``refresh_manifest``.

    Failed jobs fail their candidates (terminal, with the diagnosis). Completed jobs only mark
    the candidates ``awaiting_result``; they become ``done`` through ``record_result``.
    """
    by_job = {e.job_id: e for e in entries}
    out: dict[str, list[str]] = {"failed": [], "awaiting_result": [], "running": []}
    for cid, status in list(manifest.status.items()):
        job_id = manifest.details.get(cid, {}).get("job_id")
        entry = by_job.get(job_id) if job_id else None
        if entry is None or status != "queued":
            continue
        if entry.state in sj.TERMINAL_FAILED:
            manifest.set_status(
                cid,
                "failed",
                {
                    "job_state": entry.state,
                    "diagnosis": entry.diagnosis,
                    "elapsed": entry.elapsed,
                },
            )
            out["failed"].append(cid)
        elif entry.state in sj.TERMINAL_OK:
            manifest.set_status(cid, "queued", {"awaiting_result": True, "elapsed": entry.elapsed})
            out["awaiting_result"].append(cid)
        elif entry.state == "RUNNING":
            out["running"].append(cid)
    return out


def record_result(manifest: CandidateManifest, candidate_id: str, result: dict) -> None:
    """Mark a candidate done only with an explicit result payload."""
    if not result:
        raise ValueError("an empty result cannot mark a candidate done")
    manifest.set_status(candidate_id, "done", {"result": result})


def node_hours(entries: Sequence[sj.JobEntry]) -> dict[str, float]:
    """Measured node-hours per job name; jobs without elapsed time are omitted."""
    out: dict[str, float] = {}
    for e in entries:
        seconds = elapsed_seconds(e.elapsed)
        if seconds is not None:
            out[e.spec.name] = e.spec.nodes * seconds / 3600.0
    return out
