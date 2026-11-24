"""E7: final report from the matrix, job manifest and ledger (synthetic evidence only)."""

import json

from matsim_agents.execution.ledger import EvidenceLedger
from matsim_agents.execution.slurm_jobs import JobEntry, JobSpec
from matsim_agents.workflows.radiation_damage import run_radiation_damage
from matsim_agents.workflows.radiation_final_report import (
    FINAL_REPORT_SCHEMA,
    build_final_report,
    overall_qualification,
    render_markdown,
    write_final_report,
)
from matsim_agents.workflows.radiation_materials import assess_all
from matsim_agents.workflows.radiation_qualification import (
    QualificationMatrix,
    QualificationStatus,
    promote,
    record_evidence,
)
from tests.test_neb_workflow import _fake
from tests.test_radiation_damage_workflow import _config


def _job(state="SUBMITTED", gpus=1, job_id="1"):
    spec = JobSpec(name="smoke", command=("python", "-c", "print('hi there')"), gpus=gpus)
    return JobEntry(
        spec_hash=spec.spec_hash,
        spec=spec,
        job_id=job_id,
        state=state,
        submitted_at="2026-01-01T00:00:00+00:00",
    )


def _fill(matrix, row_id, artifact, *, synthetic=False):
    for crit in matrix.row(row_id).criteria:
        matrix = record_evidence(
            matrix,
            row_id,
            crit.criterion_id,
            artifact,
            summary="fixture",
            measured=True,
            synthetic=synthetic,
        )
    return matrix


def test_empty_matrix_is_not_qualified_and_report_says_so():
    report = build_final_report(QualificationMatrix())
    assert report["schema"] == FINAL_REPORT_SCHEMA
    assert report["scientific_qualification"]["overall"] == "not_qualified"
    assert report["scientific_qualification"]["matrix"]["status_counts"]["qualified"] == 0
    assert report["software_completeness"] is None and report["ledger"] is None
    assert any("software completeness is not scientific" in s for s in report["statements"])


def test_overall_is_derived_from_rows(tmp_path):
    artifact = tmp_path / "a.json"
    artifact.write_text("{}")
    matrix = _fill(QualificationMatrix(), "scalability", artifact)
    matrix = promote(matrix, "scalability", QualificationStatus.QUALIFIED)
    assert overall_qualification(matrix) == "partially_qualified"
    for row in matrix.rows:
        if row.status is not QualificationStatus.QUALIFIED:
            matrix = _fill(matrix, row.row_id, artifact)
            matrix = promote(matrix, row.row_id, QualificationStatus.QUALIFIED)
    assert overall_qualification(matrix) == "qualified"


def test_synthetic_evidence_is_listed_as_unverified_and_not_counted(tmp_path):
    artifact = tmp_path / "s.json"
    artifact.write_text("{}")
    matrix = _fill(QualificationMatrix(), "dft-neb", artifact, synthetic=True)
    report = build_final_report(matrix)
    row = next(
        r for r in report["scientific_qualification"]["matrix"]["rows"] if r["row_id"] == "dft-neb"
    )
    assert row["met"] == [] and row["synthetic_evidence"] == 4 and row["status"] == "not_assessed"
    assert sum("synthetic evidence" in u for u in report["unverified"]) == 4


def test_jobs_are_counted_from_the_manifest_and_not_treated_as_evidence():
    jobs = [
        _job("COMPLETED", job_id="1"),
        _job("SUBMITTED", job_id="2"),
        _job("FAILED", job_id="3"),
    ]
    report = build_final_report(QualificationMatrix(), jobs=jobs)
    assert report["jobs"]["submitted"] == 3 and report["jobs"]["completed"] == 1
    assert report["jobs"]["failed"] == 1 and report["jobs"]["outstanding"] == 1
    assert sum("output not recorded" in u for u in report["unverified"]) == 3
    assert report["scientific_qualification"]["overall"] == "not_qualified"


def test_ledger_summary_and_changed_artifact_are_reported(tmp_path):
    ledger = EvidenceLedger(tmp_path / "l.jsonl")
    run_radiation_damage(
        _config(tmp_path),
        tmp_path / "run",
        dispatch=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(),
        ledger=ledger,
    )
    report = build_final_report(QualificationMatrix(), ledger=ledger)
    section = report["ledger"]
    assert section["evidence"] >= 1 and section["records"] >= section["evidence"]
    assert section["artifact_problems"] == []


def test_materials_and_software_sections_pass_through():
    matrix = QualificationMatrix()
    report = build_final_report(
        matrix, materials=assess_all(matrix), software={"tests_passed": 3, "note": "toy"}
    )
    assert report["software_completeness"] == {"tests_passed": 3, "note": "toy"}
    assert len(report["materials"]) == 5 and not any(m["qualified"] for m in report["materials"])


def test_markdown_and_files_are_written_and_parse(tmp_path):
    report = build_final_report(QualificationMatrix(), jobs=[_job()])
    text = render_markdown(report)
    assert "Scientific qualification: **not_qualified**" in text and "1 submitted" in text
    json_path, md_path = write_final_report(report, tmp_path / "out")
    assert json.loads(json_path.read_text())["schema"] == FINAL_REPORT_SCHEMA
    assert md_path.read_text() == text
    assert not list((tmp_path / "out").glob(".rd-final-*"))
