import csv
import json

from benchmarks.e3 import hypothesis_audit


def _h(hid, status, claim="Claim", evidence=("e-a",)):
    return {
        "id": hid,
        "claim": claim,
        "prediction": "Prediction",
        "status": status,
        "reason": "Reason",
        "evidence_ids": list(evidence),
    }


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _campaign(root):
    stream = root / "mat" / "llm_guided_adaptation" / "seed-1"
    rows = [{"evidence_id": "e-a", "formula": "A2B"}, {"evidence_id": "e-b", "formula": "AB2"}]
    _write(
        stream / "round-01/round-summary.json",
        {"outcomes": {"converged": 4}, "action_outcomes": rows},
    )
    _write(
        stream / "round-02/round-summary.json",
        {"outcomes": {"converged": 0, "nonconverged": 8}, "action_outcomes": []},
    )
    h1, h2 = _h("H1", "proposed"), _h("H2", "proposed")
    _write(
        stream / "round-01/acquisition/proposal-001.json",
        {
            "status": "accepted",
            "model_id": "m",
            "prompt_version": "p",
            "prompt": "x",
            "raw_response": "{}",
            "previous_hypotheses": [],
            "scientific_hypotheses": [h1, h2],
        },
    )
    r1 = [
        _h("H1", "challenged_by_mlip", evidence=("e-a", "e-b")),
        _h("H2", "proposed", claim="New claim"),
        _h("H3", "supported_by_mlip"),
    ]
    _write(
        stream / "round-01/hypothesis-assessment.json",
        {
            "status": "accepted",
            "model_id": "m",
            "version": "a",
            "prompt": "x",
            "previous_hypotheses": [h1, h2],
            "scientific_hypotheses": r1,
        },
    )
    _write(
        stream / "round-02/hypothesis-assessment.json",
        {
            "status": "accepted",
            "model_id": "m",
            "version": "a",
            "prompt": "x",
            "previous_hypotheses": r1,
            "scientific_hypotheses": [
                r1[0],
                _h("H2", "supported_by_mlip", claim="New claim"),
                r1[2],
            ],
        },
    )
    _write(
        stream / "round-02/hypothesis-assessment.timed-out.json",
        {"status": "request_started", "model_id": "m", "version": "a"},
    )
    _write(stream / "round-02/tasks/t0/result.json", {})
    return stream


def _flags(rows):
    return {(r["round"], r["hypothesis_id"], r["flag"]) for r in rows}


def test_audit_classifies_revisions_and_flags_review_cases(tmp_path):
    _campaign(tmp_path)
    tables = hypothesis_audit.audit(tmp_path)

    revisions = {
        (r["round"], r["record_kind"], r["hypothesis_id"]): r["revision_type"]
        for r in tables["evolution"]
    }
    assert revisions[(1, "proposal", "H1")] == "new"
    assert revisions[(1, "assessment", "H1")] == "status_only"
    assert revisions[(1, "assessment", "H2")] == "claim_changed"
    assert revisions[(2, "assessment", "H1")] == "unchanged"

    assert _flags(tables["flags"]) == {
        (1, "H1", "resolved_citing_multiple_compositions"),
        (1, "H2", "text_changed"),
        (1, "H3", "resolved_on_introduction"),
        (2, "H2", "status_change_without_new_converged_data"),
        (2, "", "round_no_converged_structures"),
        (2, "", "record_not_accepted"),
    }
    assert len(tables["interactions"]) == 4


def test_failed_round_assessment_uses_its_partial_outcomes(tmp_path):
    stream = tmp_path / "mat" / "llm_guided_adaptation" / "seed-1"
    before = _h("H1", "proposed")
    after = _h("H1", "supported_by_mlip", evidence=("e-a", "e-b"))
    _write(
        stream / "round-01/hypothesis-assessment-round_failed.json",
        {
            "status": "accepted",
            "prompt": "x",
            "additional_history": [
                {"evidence_id": "e-a", "formula": "A2B", "converged": 2},
                {"evidence_id": "e-b", "formula": "AB2", "converged": 1},
                {"evidence_id": "round-01-round_failed", "category": "round_failed"},
            ],
            "previous_hypotheses": [before],
            "scientific_hypotheses": [after],
        },
    )
    tables = hypothesis_audit.audit(tmp_path)

    (row,) = tables["evolution"]
    assert row["round_converged"] == 3
    assert row["evidence_formulas"] == "A2B;AB2"
    assert _flags(tables["flags"]) == {(1, "H1", "resolved_citing_multiple_compositions")}


def test_write_outputs_and_manifest_skip_task_directories(tmp_path):
    root = tmp_path / "campaign"
    _campaign(root)
    out = root / "hypothesis-audit"

    counts = hypothesis_audit.write(root, out)

    assert counts["evolution"] == 8
    with (out / "hypothesis-evolution.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 8
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["outputs"]) == {
        "interaction-index.csv",
        "hypothesis-evolution.csv",
        "review-flags.csv",
    }
    assert not any("/tasks/" in p or "hypothesis-audit" in p for p in manifest["run_files"])
    assert counts["run_files"] == 6
