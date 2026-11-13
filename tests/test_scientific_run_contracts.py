import json
from datetime import UTC, datetime

import pytest

from matsim_agents.execution.contracts import (
    EvidenceLevel,
    ProvenanceRecord,
    WorkflowResult,
    WorkflowStatus,
)
from matsim_agents.execution.run_directory import (
    ScientificRunDirectory,
    make_run_id,
    safe_component,
)


def _provenance() -> ProvenanceRecord:
    return ProvenanceRecord(workflow="relaxation", evidence_level=EvidenceLevel.MLIP_RELAXATION)


def test_run_directory_creates_canonical_restartable_layout(tmp_path):
    run = ScientificRunDirectory.create(
        tmp_path,
        workflow="relaxation",
        request={"structure": "Si.vasp"},
        resolved_config={"mode": "mlip"},
        provenance=_provenance(),
    )
    assert {path.name for path in run.path.iterdir()} >= {
        "structures",
        "calculations",
        "datasets",
        "models",
        "request.json",
        "resolved_config.json",
        "provenance.json",
        "events.jsonl",
    }
    assert ScientificRunDirectory.open(run.path).run_id == run.run_id
    assert json.loads((run.path / "request.json").read_text()) == {"structure": "Si.vasp"}


def test_run_ids_are_timestamped_and_collision_resistant():
    now = datetime(2026, 9, 1, 12, 30, tzinfo=UTC)
    first, second = make_run_id(now), make_run_id(now)
    assert first.startswith("2026-09-01T12-30-00Z_")
    assert first != second


def test_safe_component_blocks_path_traversal():
    assert "/" not in safe_component("../../outside/result")
    assert ".." not in safe_component("../../outside/result")


def test_failed_workflow_result_requires_reason():
    with pytest.raises(ValueError, match="failure_reason"):
        WorkflowResult(
            run_id="run",
            workflow="relaxation",
            status=WorkflowStatus.FAILED,
            evidence_level=EvidenceLevel.MLIP_RELAXATION,
            provenance=_provenance(),
        )


def test_text_and_json_writes_are_atomic_and_confined(tmp_path):
    run = ScientificRunDirectory.create(
        tmp_path,
        workflow="relaxation",
        request={},
        resolved_config={},
        provenance=_provenance(),
    )
    run.write_text("report.md", "first")
    run.write_text("report.md", "second")
    run.write_json("summary.json", {"complete": True})
    assert (run.path / "report.md").read_text() == "second"
    assert json.loads((run.path / "summary.json").read_text()) == {"complete": True}
    confined = run.write_text("../outside.txt", "confined to run directory")
    assert confined.parent == run.path
    assert not (run.path.parent / "outside.txt").exists()
    assert not [path for path in run.path.iterdir() if path.name.startswith(".report.md.")]


def test_failed_text_replace_preserves_previous_report(tmp_path, monkeypatch):
    run = ScientificRunDirectory.create(
        tmp_path,
        workflow="relaxation",
        request={},
        resolved_config={},
        provenance=_provenance(),
    )
    run.write_text("report.md", "committed report")

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr("matsim_agents.execution.run_directory.os.replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        run.write_text("report.md", "uncommitted report")
    assert (run.path / "report.md").read_text() == "committed report"
    assert not [path for path in run.path.iterdir() if path.name.startswith(".report.md.")]
