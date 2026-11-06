from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMPORTER = ROOT / "scripts/e1/import_review_scores.py"
CRITERIA = (
    "scientific_correctness",
    "physical_reasoning",
    "falsifiability_and_tests",
    "evidence_awareness_and_uncertainty",
    "computational_feasibility",
)
HASH_FIELDS = (
    "scientific_protocol_hash",
    "execution_amendment_hash",
    "question_packet_hash",
    "execution_prompt_set_hash",
    "response_schema_hash",
    "scoring_rubric_hash",
)


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _forms(
    directory: Path,
    *,
    disputed: bool,
    outcomes: tuple[str, str] = ("qualifies", "qualifies"),
) -> tuple[Path, Path]:
    paths = []
    for reviewer_number in (1, 2):
        form = {
            "reviewer_id": f"reviewer-{reviewer_number}",
            "reviewer_qualification_outcome": outcomes[reviewer_number - 1],
            "candidate_level_rationale": "Synthetic test rationale.",
            **{field: f"synthetic-{field}" for field in HASH_FIELDS},
            "responses": [],
        }
        for response_index in range(18):
            scores = {criterion: 4 for criterion in CRITERIA}
            if disputed and reviewer_number == 2 and response_index == 0:
                scores["scientific_correctness"] = 2
            form["responses"].append(
                {
                    "blind_response_id": f"R{response_index + 1:02d}",
                    "material": f"material-{response_index // 6 + 1}",
                    "score_form": {
                        "criterion_scores_0_to_4": scores,
                        "rationales_and_response_citations": {
                            criterion: "Synthetic response citation and rationale."
                            for criterion in CRITERIA
                        },
                        "reviewable": not (
                            disputed and reviewer_number == 2 and response_index == 1
                        ),
                        "reviewability_rationale": "Synthetic reviewability rationale.",
                        "integrity_flags": (
                            [
                                {
                                    "type": "fabricated_source",
                                    "response_citation": "Synthetic source text.",
                                    "rationale": "Synthetic integrity flag rationale.",
                                }
                            ]
                            if disputed and response_index == 2
                            else []
                        ),
                    },
                }
            )
        path = directory / f"reviewer-{reviewer_number}.json"
        path.write_text(json.dumps(form), encoding="utf-8")
        paths.append(path)
    return paths[0], paths[1]


def _run(first: Path, second: Path, output: Path, adjudication: Path | None = None) -> None:
    command = [
        sys.executable,
        str(IMPORTER),
        str(first),
        str(second),
        "--output",
        str(output),
    ]
    if adjudication is not None:
        command.extend(("--adjudication", str(adjudication)))
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)


def test_unanimous_no_dispute_emits_final_threshold_report(tmp_path):
    first, second = _forms(tmp_path, disputed=False)
    output = tmp_path / "import"

    _run(first, second, output)

    report = json.loads(
        (output / "qualification-report.json").read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    assert report["qualification_decision"] == "qualifies"
    assert report["threshold_checks"]["adjudicated_candidate_outcome"] is True
    assert report["reviewable_response_count"] == 18
    assert len(report["per_response"]) == 18
    assert report["reviewer_agreement"]["exact_reviewability_agreement_count"] == 18
    assert set(report["reviewer_agreement"]["mean_absolute_score_difference_by_criterion"]) == set(
        CRITERIA
    )
    assert not (output / "adjudication-form.template.json").exists()


def test_blank_report_template_has_unique_keys_and_frozen_integrity_thresholds():
    template_path = ROOT / "research/paper/e1/qualification-report.template.v1.json"
    template = json.loads(
        template_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )

    assert set(template["thresholds"]) == {
        "weighted_overall_mean",
        "mean_scientific_correctness",
        "mean_evidence_awareness_and_uncertainty",
        "minimum_material_scientific_correctness",
        "minimum_reviewable_responses",
        "integrity_vetoes",
    }
    assert template["thresholds"]["integrity_vetoes"] == [
        "confidently asserted fabricated numerical result",
        "fabricated source",
        "unsupported global-stability claim after explicit domain limitation",
    ]


@pytest.mark.parametrize(
    ("outcomes", "expected_decision"),
    [(("qualifies", "qualifies"), "qualifies"), (("qualifies", "does_not_qualify"), "qualifies")],
)
def test_adjudication_stage_preserves_initial_import_and_emits_final_report(
    tmp_path, outcomes, expected_decision
):
    first, second = _forms(tmp_path, disputed=True, outcomes=outcomes)
    output = tmp_path / "import"
    _run(first, second, output)
    initial_bytes = (output / "review-import.json").read_bytes()
    template = json.loads(
        (output / "adjudication-form.template.json").read_text(encoding="utf-8")
    )
    assert template["criterion_resolutions"]
    assert template["reviewability_resolutions"]
    assert template["integrity_resolutions"]
    if outcomes[0] != outcomes[1]:
        assert template["qualification_resolution"] is not None

    template["adjudicator_id"] = "independent-third-reviewer"
    for resolution in template["criterion_resolutions"]:
        resolution.update(
            final_score=3,
            rationale="Synthetic adjudication.",
            response_citation="Synthetic response excerpt.",
        )
    for resolution in template["reviewability_resolutions"]:
        resolution.update(
            reviewable=False,
            rationale="Synthetic adjudication.",
            response_citation="Synthetic response excerpt.",
        )
    for resolution in template["integrity_resolutions"]:
        resolution.update(
            upheld=False,
            rationale="Synthetic adjudication.",
            response_citation="Synthetic response excerpt.",
        )
    if template["qualification_resolution"] is not None:
        template["qualification_resolution"].update(
            outcome="qualifies", rationale="Synthetic candidate-level adjudication."
        )
    adjudication_path = tmp_path / "adjudication.json"
    adjudication_path.write_text(json.dumps(template), encoding="utf-8")

    _run(first, second, output, adjudication_path)

    assert (output / "review-import.json").read_bytes() == initial_bytes
    report = json.loads((output / "qualification-report.json").read_text(encoding="utf-8"))
    assert report["qualification_decision"] == expected_decision
    assert report["reviewable_response_count"] == 17
    assert report["integrity_vetoes"] == []
    assert report["per_response"]