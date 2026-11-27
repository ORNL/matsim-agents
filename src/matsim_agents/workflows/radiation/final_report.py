"""Evidence-grounded campaign report built from the matrix, job manifest and ledger.

Every number comes from an input object; nothing is estimated here. The report keeps
*software completeness* (what exists and is tested) apart from *scientific qualification*
(what measured, non-synthetic evidence supports). The overall qualification is derived:

* ``qualified`` only if every matrix row is qualified;
* ``partially_qualified`` if some rows are qualified;
* ``not_qualified`` otherwise, including an empty matrix.

A scheduler job that reached ``COMPLETED`` is *not* evidence: its output has not been read
and hashed into the matrix, so it is listed under ``unverified``. Ledger artifacts whose
hashes no longer match are listed too.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from matsim_agents.core.fileio import atomic_write_text
from matsim_agents.execution.ledger import EvidenceLedger
from matsim_agents.execution.slurm_jobs import JobEntry, summarize
from matsim_agents.workflows.qualification_matrix import (
    QualificationMatrix,
    QualificationStatus,
)
from matsim_agents.workflows.radiation.materials import MaterialReadiness

FINAL_REPORT_SCHEMA = "matsim-radiation-final-report/1"
STANDING_STATEMENTS = (
    "software completeness is not scientific qualification",
    "synthetic evidence never counts toward qualification",
    "a completed scheduler job is not evidence until its output is read and recorded",
)


def overall_qualification(matrix: QualificationMatrix) -> str:
    """Derived from the rows; never set by hand."""
    statuses = [r.status for r in matrix.rows]
    qualified = sum(s is QualificationStatus.QUALIFIED for s in statuses)
    if statuses and qualified == len(statuses):
        return "qualified"
    return "partially_qualified" if qualified else "not_qualified"


def _matrix_section(matrix: QualificationMatrix) -> dict[str, Any]:
    return {
        "status_counts": matrix.status_counts(),
        "rows": [
            {
                "row_id": r.row_id,
                "title": r.title,
                "status": r.status.value,
                "criteria": len(r.criteria),
                "met": sorted(r.met()),
                "missing": r.missing(),
                "synthetic_evidence": sum(e.synthetic for e in r.evidence),
                "notes": list(r.notes),
            }
            for r in matrix.rows
        ],
    }


def _jobs_section(jobs: Sequence[JobEntry]) -> dict[str, Any]:
    return {
        **summarize(jobs),
        "jobs": [
            {
                "name": j.spec.name,
                "job_id": j.job_id,
                "state": j.state,
                "gpus": j.spec.gpus,
                "exit_code": j.exit_code,
                "diagnosis": j.diagnosis,
            }
            for j in jobs
        ],
    }


def _ledger_section(ledger: EvidenceLedger | None) -> dict[str, Any] | None:
    if ledger is None:
        return None
    summary = ledger.summary()
    return {
        "records": summary["records"],
        "evidence": summary["evidence"],
        "decisions": summary["decisions"],
        "hypotheses": {
            h: {"status": v["status"], "revisions": v["revisions"], "claim": v["claim"]}
            for h, v in summary["hypotheses"].items()
        },
        "artifact_problems": ledger.verify_artifacts(),
    }


def _unverified(
    matrix: QualificationMatrix, jobs: Sequence[JobEntry], ledger_problems: Sequence[str]
) -> list[str]:
    items = [
        f"job {j.job_id} ({j.spec.name}) is {j.state}; output not recorded in the matrix"
        for j in jobs
    ]
    for row in matrix.rows:
        items.extend(
            f"{row.row_id}/{e.criterion_id}: synthetic evidence, not counted"
            for e in row.evidence
            if e.synthetic
        )
        items.extend(
            f"{row.row_id}/{e.criterion_id}: evidence not marked measured"
            for e in row.evidence
            if not e.measured
        )
    items.extend(f"ledger: {p}" for p in ledger_problems)
    return items


def build_final_report(
    matrix: QualificationMatrix,
    *,
    jobs: Sequence[JobEntry] = (),
    ledger: EvidenceLedger | None = None,
    materials: Sequence[MaterialReadiness] = (),
    software: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A JSON-serialisable report. ``software`` is passed through under its own key."""
    ledger_section = _ledger_section(ledger)
    problems = ledger_section["artifact_problems"] if ledger_section else []
    overall = overall_qualification(matrix)
    return {
        "schema": FINAL_REPORT_SCHEMA,
        "scientific_qualification": {"overall": overall, "matrix": _matrix_section(matrix)},
        "software_completeness": dict(software) if software is not None else None,
        "jobs": _jobs_section(jobs),
        "ledger": ledger_section,
        "materials": [
            {
                "row_id": m.row_id,
                "name": m.name,
                "composition_fixed": m.composition_fixed,
                "matrix_status": m.matrix_status,
                "qualified": m.qualified,
                "missing_criteria": list(m.missing_criteria),
                "blockers": list(m.blockers),
            }
            for m in materials
        ],
        "unverified": _unverified(matrix, jobs, problems),
        "statements": list(STANDING_STATEMENTS),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """A short human-readable view of ``build_final_report`` output."""
    sci = report["scientific_qualification"]
    jobs = report["jobs"]
    lines = [
        "# Radiation-damage campaign report",
        "",
        f"Scientific qualification: **{sci['overall']}**",
        "",
        "| Row | Status | Met | Missing |",
        "| --- | --- | --- | --- |",
    ]
    for row in sci["matrix"]["rows"]:
        lines.append(
            f"| {row['row_id']} | {row['status']} | {len(row['met'])}/{row['criteria']} "
            f"| {', '.join(row['missing']) or '-'} |"
        )
    lines += [
        "",
        f"Jobs: {jobs['submitted']} submitted, {jobs['completed']} completed, "
        f"{jobs['failed']} failed, {jobs['outstanding']} outstanding.",
        "",
        "## Unverified",
        *(f"- {item}" for item in report["unverified"]),
        *([] if report["unverified"] else ["- none"]),
        "",
        "## Statements",
        *(f"- {s}" for s in report["statements"]),
        "",
    ]
    return "\n".join(lines)


def write_final_report(
    report: Mapping[str, Any], directory: str | os.PathLike
) -> tuple[Path, Path]:
    """Atomically write ``report.json`` and ``report.md`` into ``directory``."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    targets = (root / "final-report.json", root / "final-report.md")
    payloads = (json.dumps(report, indent=2, allow_nan=False) + "\n", render_markdown(report))
    for target, text in zip(targets, payloads, strict=True):
        atomic_write_text(target, text)
    return targets


__all__ = [
    "FINAL_REPORT_SCHEMA",
    "STANDING_STATEMENTS",
    "build_final_report",
    "overall_qualification",
    "render_markdown",
    "write_final_report",
]
