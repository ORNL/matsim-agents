#!/usr/bin/env python3
"""Validate two blinded E1 reviewer forms and emit paired-score/adjudication records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CRITERIA = (
    "scientific_correctness",
    "physical_reasoning",
    "falsifiability_and_tests",
    "evidence_awareness_and_uncertainty",
    "computational_feasibility",
)
VALID_OUTCOMES = {"qualifies", "does_not_qualify"}
VALID_INTEGRITY_FLAGS = {
    "fabricated_numerical_result",
    "fabricated_source",
    "unsupported_global_stability_claim",
}
WEIGHTS = {
    "scientific_correctness": 0.30,
    "physical_reasoning": 0.20,
    "falsifiability_and_tests": 0.20,
    "evidence_awareness_and_uncertainty": 0.20,
    "computational_feasibility": 0.10,
}
THRESHOLDS = {
    "weighted_overall_mean": 3.0,
    "mean_scientific_correctness": 3.0,
    "mean_evidence_awareness_and_uncertainty": 3.0,
    "minimum_material_scientific_correctness": 2.5,
    "minimum_reviewable_responses": 16,
    "integrity_vetoes": [
        "fabricated_numerical_result",
        "fabricated_source",
        "unsupported_global_stability_claim",
    ],
}


def _requires_adjudication(items: list[dict[str, Any]]) -> bool:
    return any(
        item.get("criteria_requiring_adjudication")
        or item.get("reviewability_adjudication_required")
        or bool(item.get("reviewer_flags"))
        or item.get("candidate_level_qualification_adjudication_required")
        for item in items
    )


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _adjudication_template(
    queue: list[dict[str, Any]], qualification_required: bool
) -> dict[str, Any]:
    criteria = []
    reviewability = []
    integrity = []
    for item in queue:
        identifier = item.get("blind_response_id")
        for criterion in item.get("criteria_requiring_adjudication", []):
            criteria.append(
                {
                    "blind_response_id": identifier,
                    "criterion": criterion,
                    "final_score": None,
                    "rationale": "",
                    "response_citation": "",
                }
            )
        for response_id in item.get("reviewability_adjudication_required", []):
            reviewability.append({
                "blind_response_id": response_id,
                "reviewable": None,
                "rationale": "",
                "response_citation": "",
            })
        if item.get("reviewer_flags"):
            for reviewer_flags in item["reviewer_flags"].values():
                for flag in reviewer_flags:
                    key = (identifier, flag["type"])
                    if key not in {
                        (entry["blind_response_id"], entry["flag_type"])
                        for entry in integrity
                    }:
                        integrity.append(
                            {
                                "blind_response_id": identifier,
                                "flag_type": flag["type"],
                                "upheld": None,
                                "rationale": "",
                                "response_citation": "",
                            }
                        )
    return {
        "adjudicator_id": "",
        "criterion_resolutions": criteria,
        "reviewability_resolutions": reviewability,
        "integrity_resolutions": integrity,
        "qualification_resolution": (
            {"outcome": None, "rationale": ""} if qualification_required else None
        ),
    }


def _validate_adjudication(
    path: Path, template: dict[str, Any], reviewer_ids: set[str]
) -> dict[str, Any]:
    form = json.loads(path.read_text(encoding="utf-8"))
    adjudicator = _require_text(form.get("adjudicator_id"), "adjudicator_id")
    if adjudicator in reviewer_ids:
        raise ValueError("The adjudicator must be independent of both reviewers")

    for field in ("criterion_resolutions", "reviewability_resolutions", "integrity_resolutions"):
        submitted = form.get(field)
        expected = template[field]
        if not isinstance(submitted, list):
            raise ValueError(f"{field} must be a list")
        key_fields = {
            "criterion_resolutions": ("blind_response_id", "criterion"),
            "reviewability_resolutions": ("blind_response_id",),
            "integrity_resolutions": ("blind_response_id", "flag_type"),
        }[field]
        def key(
            item: dict[str, Any], fields: tuple[str, ...] = key_fields
        ) -> tuple[Any, ...]:
            return tuple(item.get(name) for name in fields)

        if len(submitted) != len(expected) or {key(item) for item in submitted} != {
            key(item) for item in expected
        }:
            raise ValueError(f"{field} must resolve exactly the items in the adjudication template")
        for item in submitted:
            _require_text(item.get("rationale"), f"{field} rationale")
            if field == "criterion_resolutions":
                score = item.get("final_score")
                if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
                    raise ValueError("adjudicator final scores must be integers from 0 through 4")
                _require_text(item.get("response_citation"), "adjudicator response citation")
            elif field == "reviewability_resolutions":
                if not isinstance(item.get("reviewable"), bool):
                    raise ValueError("adjudicated reviewability must be true or false")
                _require_text(item.get("response_citation"), "adjudicated response citation")
            elif not isinstance(item.get("upheld"), bool):
                raise ValueError("integrity flag disposition upheld must be true or false")
            else:
                _require_text(item.get("response_citation"), "adjudicated response citation")

    expected_qualification = template["qualification_resolution"]
    submitted_qualification = form.get("qualification_resolution")
    if expected_qualification is None:
        if submitted_qualification is not None:
            raise ValueError("qualification_resolution must be null when reviewers agree")
    else:
        if not isinstance(submitted_qualification, dict):
            raise ValueError("candidate-level disagreement requires qualification_resolution")
        if submitted_qualification.get("outcome") not in VALID_OUTCOMES:
            raise ValueError(
                "adjudicator outcome must be qualifies or does_not_qualify"
            )
        _require_text(submitted_qualification.get("rationale"), "qualification rationale")
    return form


def _final_report(
    first: dict[str, Any],
    second: dict[str, Any],
    first_items: dict[str, dict[str, Any]],
    second_items: dict[str, dict[str, Any]],
    adjudication: list[dict[str, Any]],
    adjudicator_form: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if _requires_adjudication(adjudication) and adjudicator_form is None:
        return None
    adjudicator_form = adjudicator_form or _adjudication_template([], False)
    score_resolutions = {
        (item["blind_response_id"], item["criterion"]): item
        for item in adjudicator_form["criterion_resolutions"]
    }
    reviewability_resolutions = {
        item["blind_response_id"]: item
        for item in adjudicator_form["reviewability_resolutions"]
    }
    integrity_resolutions = {
        (item["blind_response_id"], item["flag_type"]): item
        for item in adjudicator_form["integrity_resolutions"]
    }
    per_response = {}
    criterion_values = {criterion: [] for criterion in CRITERIA}
    criterion_differences = {criterion: [] for criterion in CRITERIA}
    material_values: dict[str, dict[str, list[float]]] = {}
    integrity_vetoes = []
    reviewable_count = 0
    for identifier in sorted(first_items):
        first_item, second_item = first_items[identifier], second_items[identifier]
        material = first_item["material"]
        final_scores = {}
        for criterion in CRITERIA:
            first_score = first_item["score_form"]["criterion_scores_0_to_4"][criterion]
            second_score = second_item["score_form"]["criterion_scores_0_to_4"][criterion]
            criterion_differences[criterion].append(abs(first_score - second_score))
            resolution = score_resolutions.get((identifier, criterion))
            score = (
                resolution["final_score"]
                if resolution is not None
                else (first_score + second_score) / 2
            )
            final_scores[criterion] = score
            criterion_values[criterion].append(score)
            material_values.setdefault(material, {key: [] for key in CRITERIA})
            material_values[material][criterion].append(score)
        first_reviewable = first_item["score_form"]["reviewable"]
        second_reviewable = second_item["score_form"]["reviewable"]
        resolution = reviewability_resolutions.get(identifier)
        is_reviewable = (
            resolution["reviewable"]
            if resolution is not None
            else first_reviewable and second_reviewable
        )
        reviewable_count += int(is_reviewable)
        seen_flags = set()
        for reviewer_form in (first_item["score_form"], second_item["score_form"]):
            for flag in reviewer_form["integrity_flags"]:
                flag_key = (identifier, flag["type"])
                if flag_key in seen_flags:
                    continue
                seen_flags.add(flag_key)
                flag_resolution = integrity_resolutions[flag_key]
                if flag_resolution["upheld"]:
                    integrity_vetoes.append(
                        {
                            "blind_response_id": identifier,
                            "flag_type": flag["type"],
                            "reviewer_flags": [
                                {
                                    "reviewer_id": reviewer_id,
                                    **reviewer_flag,
                                }
                                for reviewer_id, reviewer_form in (
                                    (first["reviewer_id"], first_item["score_form"]),
                                    (second["reviewer_id"], second_item["score_form"]),
                                )
                                for reviewer_flag in reviewer_form["integrity_flags"]
                                if reviewer_flag["type"] == flag["type"]
                            ],
                            "adjudicator_response_citation": flag_resolution[
                                "response_citation"
                            ],
                            "adjudicator_rationale": flag_resolution["rationale"],
                        }
                    )
        per_response[identifier] = {
            "material": material,
            "reviewable": is_reviewable,
            "reviewer_scores": {
                first["reviewer_id"]: first_item["score_form"]["criterion_scores_0_to_4"],
                second["reviewer_id"]: second_item["score_form"]["criterion_scores_0_to_4"],
            },
            "criterion_score_differences": {
                criterion: abs(
                    first_item["score_form"]["criterion_scores_0_to_4"][criterion]
                    - second_item["score_form"]["criterion_scores_0_to_4"][criterion]
                )
                for criterion in CRITERIA
            },
            "reviewer_rationales_and_citations": {
                first["reviewer_id"]: first_item["score_form"]["rationales_and_response_citations"],
                second["reviewer_id"]: second_item["score_form"][
                    "rationales_and_response_citations"
                ],
            },
            "adjudicated_criterion_resolutions": {
                criterion: score_resolutions[(identifier, criterion)]
                for criterion in CRITERIA
                if (identifier, criterion) in score_resolutions
            },
            "reviewability_rationales_and_citations": {
                first["reviewer_id"]: {
                    "reviewable": first_reviewable,
                    "rationale": first_item["score_form"]["reviewability_rationale"],
                },
                second["reviewer_id"]: {
                    "reviewable": second_reviewable,
                    "rationale": second_item["score_form"]["reviewability_rationale"],
                },
                "adjudication": reviewability_resolutions.get(identifier),
            },
            "integrity_flags": {
                first["reviewer_id"]: first_item["score_form"]["integrity_flags"],
                second["reviewer_id"]: second_item["score_form"]["integrity_flags"],
                "adjudications": {
                    flag_type: resolution
                    for (response_id, flag_type), resolution in integrity_resolutions.items()
                    if response_id == identifier
                },
            },
            "final_adjudicated_scores": final_scores,
        }
    means = {criterion: sum(scores) / len(scores) for criterion, scores in criterion_values.items()}
    reviewer_agreement = {
        "exact_score_agreement_count_by_criterion": {
            criterion: sum(value == 0 for value in values)
            for criterion, values in criterion_differences.items()
        },
        "mean_absolute_score_difference_by_criterion": {
            criterion: sum(values) / len(values)
            for criterion, values in criterion_differences.items()
        },
        "exact_reviewability_agreement_count": sum(
            first_items[identifier]["score_form"]["reviewable"]
            == second_items[identifier]["score_form"]["reviewable"]
            for identifier in first_items
        ),
        "candidate_outcome_agreement": (
            first["reviewer_qualification_outcome"]
            == second["reviewer_qualification_outcome"]
        ),
    }
    material_means = {
        material: {criterion: sum(scores) / len(scores) for criterion, scores in criteria.items()}
        for material, criteria in material_values.items()
    }
    weighted_mean = sum(means[criterion] * weight for criterion, weight in WEIGHTS.items())
    checks = {
        "weighted_overall_mean": weighted_mean >= THRESHOLDS["weighted_overall_mean"],
        "mean_scientific_correctness": (
            means["scientific_correctness"] >= THRESHOLDS["mean_scientific_correctness"]
        ),
        "mean_evidence_awareness_and_uncertainty": (
            means["evidence_awareness_and_uncertainty"]
            >= THRESHOLDS["mean_evidence_awareness_and_uncertainty"]
        ),
        "material_level_scientific_correctness": all(
            values["scientific_correctness"]
            >= THRESHOLDS["minimum_material_scientific_correctness"]
            for values in material_means.values()
        ),
        "minimum_reviewable_responses": (
            reviewable_count >= THRESHOLDS["minimum_reviewable_responses"]
        ),
        "no_integrity_veto": not integrity_vetoes,
    }
    reviewer_outcome_disagreement = (
        first["reviewer_qualification_outcome"] != second["reviewer_qualification_outcome"]
    )
    if (
        first["reviewer_qualification_outcome"] == "qualifies"
        and second["reviewer_qualification_outcome"] == "qualifies"
    ):
        adjudicated_candidate_outcome = "qualifies"
    elif reviewer_outcome_disagreement:
        adjudicated_candidate_outcome = adjudicator_form["qualification_resolution"]["outcome"]
    else:
        adjudicated_candidate_outcome = "does_not_qualify"
    checks["adjudicated_candidate_outcome"] = adjudicated_candidate_outcome == "qualifies"
    qualifies = all(checks.values())
    return {
        "report_version": 1,
        "status": "final_qualification_decision",
        "reviewers": [first["reviewer_id"], second["reviewer_id"]],
        "adjudicator_id": adjudicator_form.get("adjudicator_id") or None,
        "reviewer_qualification_outcomes": {
            first["reviewer_id"]: first["reviewer_qualification_outcome"],
            second["reviewer_id"]: second["reviewer_qualification_outcome"],
        },
        "reviewer_candidate_level_rationales": {
            first["reviewer_id"]: first["candidate_level_rationale"],
            second["reviewer_id"]: second["candidate_level_rationale"],
        },
        "adjudicated_candidate_outcome": adjudicated_candidate_outcome,
        "adjudicated_candidate_rationale": (
            adjudicator_form["qualification_resolution"]["rationale"]
            if reviewer_outcome_disagreement
            else None
        ),
        "response_count": len(per_response),
        "reviewable_response_count": reviewable_count,
        "criterion_means_0_to_4": means,
        "reviewer_agreement": reviewer_agreement,
        "material_criterion_means_0_to_4": material_means,
        "weighted_overall_mean_0_to_4": weighted_mean,
        "thresholds": THRESHOLDS,
        "threshold_checks": checks,
        "integrity_vetoes": integrity_vetoes,
        "qualification_decision": "qualifies" if qualifies else "does_not_qualify",
        "decision_rationale": (
            "All numeric thresholds and the adjudicated candidate outcome are met."
            if qualifies
            else "A numeric threshold failed or the adjudicated outcome is does_not_qualify."
        ),
        "per_response": per_response,
    }


def load_form(path: Path) -> dict[str, Any]:
    form = json.loads(path.read_text(encoding="utf-8"))
    if form.get("reviewer_id") not in {"reviewer-1", "reviewer-2"}:
        raise ValueError(f"{path}: unexpected reviewer_id")
    if form.get("reviewer_qualification_outcome") not in VALID_OUTCOMES:
        raise ValueError(f"{path}: qualification outcome must be one of {sorted(VALID_OUTCOMES)}")
    items = form.get("responses")
    if not isinstance(items, list) or len(items) != 18:
        raise ValueError(f"{path}: expected exactly 18 response forms")
    identifiers = [item.get("blind_response_id") for item in items]
    if None in identifiers or len(set(identifiers)) != 18:
        raise ValueError(f"{path}: response IDs must be present and unique")
    for item in items:
        score_form = item.get("score_form") or {}
        scores = score_form.get("criterion_scores_0_to_4") or {}
        rationales = score_form.get("rationales_and_response_citations") or {}
        if set(scores) != set(CRITERIA) or set(rationales) != set(CRITERIA):
            raise ValueError(f"{path}: {item['blind_response_id']} must score all five criteria")
        for criterion in CRITERIA:
            value = scores[criterion]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4:
                raise ValueError(
                    f"{path}: {item['blind_response_id']} {criterion} must be an integer 0..4"
                )
            if not isinstance(rationales[criterion], str) or not rationales[criterion].strip():
                raise ValueError(
                    f"{path}: {item['blind_response_id']} {criterion} needs a rationale/citation"
                )
        flags = score_form.get("integrity_flags")
        if not isinstance(flags, list):
            raise ValueError(f"{path}: integrity_flags must be a list")
        for flag in flags:
            if flag.get("type") not in VALID_INTEGRITY_FLAGS:
                raise ValueError(f"{path}: unknown integrity flag {flag.get('type')!r}")
            if not flag.get("response_citation") or not flag.get("rationale"):
                raise ValueError(f"{path}: integrity flags require response citation and rationale")
        if not isinstance(score_form.get("reviewable"), bool):
            raise ValueError(
                f"{path}: {item['blind_response_id']} needs a reviewable true/false decision"
            )
        if (
            not isinstance(score_form.get("reviewability_rationale"), str)
            or not score_form["reviewability_rationale"].strip()
        ):
            raise ValueError(f"{path}: {item['blind_response_id']} needs a reviewability rationale")
    if (
        not isinstance(form.get("candidate_level_rationale"), str)
        or not form["candidate_level_rationale"].strip()
    ):
        raise ValueError(f"{path}: candidate-level qualification needs a rationale")
    if form.get("reviewer_qualification_outcome") not in VALID_OUTCOMES:
        raise ValueError(f"{path}: qualification outcome must be one of {sorted(VALID_OUTCOMES)}")
    return form


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewer_one", type=Path)
    parser.add_argument("reviewer_two", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adjudication", type=Path)
    args = parser.parse_args()
    first, second = load_form(args.reviewer_one), load_form(args.reviewer_two)
    if first["reviewer_id"] == second["reviewer_id"]:
        raise ValueError("The two score forms must come from different reviewers")
    for field in (
        "scientific_protocol_hash",
        "execution_amendment_hash",
        "question_packet_hash",
        "execution_prompt_set_hash",
        "response_schema_hash",
        "scoring_rubric_hash",
    ):
        if first.get(field) != second.get(field):
            raise ValueError(f"Reviewer forms disagree on {field}")
    first_items = {item["blind_response_id"]: item for item in first["responses"]}
    second_items = {item["blind_response_id"]: item for item in second["responses"]}
    if set(first_items) != set(second_items):
        raise ValueError("Reviewer forms do not cover the same 18 anonymous responses")

    adjudication = []
    comparisons = []
    integrity_adjudication = []
    for identifier in sorted(first_items):
        first_form = first_items[identifier]["score_form"]
        second_form = second_items[identifier]["score_form"]
        first_scores = first_form["criterion_scores_0_to_4"]
        second_scores = second_form["criterion_scores_0_to_4"]
        differences = {
            criterion: abs(first_scores[criterion] - second_scores[criterion])
            for criterion in CRITERIA
        }
        disputed = [criterion for criterion, difference in differences.items() if difference >= 2]
        if disputed:
            adjudication.append(
                {
                    "blind_response_id": identifier,
                    "criteria_requiring_adjudication": disputed,
                    "reviewer_scores": {
                        first["reviewer_id"]: first_scores,
                        second["reviewer_id"]: second_scores,
                    },
                    "adjudicator_final_scores": {criterion: None for criterion in disputed},
                    "adjudicator_rationales_and_citations": {
                        criterion: "" for criterion in disputed
                    },
                }
            )
        first_flags = first_form["integrity_flags"]
        second_flags = second_form["integrity_flags"]
        if first_flags or second_flags:
            integrity_adjudication.append(
                {
                    "blind_response_id": identifier,
                    "reviewer_flags": {
                        first["reviewer_id"]: first_flags,
                        second["reviewer_id"]: second_flags,
                    },
                    "adjudicator_disposition": None,
                    "adjudicator_rationale": "",
                }
            )
        comparisons.append(
            {
                "blind_response_id": identifier,
                "criterion_differences": differences,
                "adjudication_required": bool(disputed),
            }
        )

    outcome_disagreement = (
        first["reviewer_qualification_outcome"] != second["reviewer_qualification_outcome"]
    )
    first_reviewability = {
        item["blind_response_id"]: item["score_form"]["reviewable"] for item in first["responses"]
    }
    second_reviewability = {
        item["blind_response_id"]: item["score_form"]["reviewable"] for item in second["responses"]
    }
    reviewable_count = sum(first_reviewability.values())
    reviewability_disagreements = sorted(
        identifier
        for identifier in first_reviewability
        if first_reviewability[identifier] != second_reviewability[identifier]
    )
    if reviewable_count < 16:
        adjudication.append(
            {
                "operational_completeness_failure": True,
                "reviewable_response_count": reviewable_count,
                "minimum_required": 16,
                "note": "Candidate cannot qualify below the protocol completeness threshold.",
            }
        )
    if reviewability_disagreements:
        adjudication.append(
            {
                "reviewability_adjudication_required": reviewability_disagreements,
                "reviewer_decisions": {
                    first["reviewer_id"]: first_reviewability,
                    second["reviewer_id"]: second_reviewability,
                },
            }
        )
    adjudication.extend(integrity_adjudication)
    if outcome_disagreement:
        adjudication.append(
            {
                "candidate_level_qualification_adjudication_required": True,
                "reviewer_outcomes": {
                    first["reviewer_id"]: first["reviewer_qualification_outcome"],
                    second["reviewer_id"]: second["reviewer_qualification_outcome"],
                },
                "adjudicator_final_outcome": None,
                "adjudicator_rationale": "",
            }
        )

    adjudication_needed = _requires_adjudication(adjudication)
    adjudication_template = _adjudication_template(adjudication, outcome_disagreement)
    template_path = args.output / "adjudication-form.template.json"
    adjudicator_form = None
    if args.adjudication:
        if not adjudication_needed:
            raise ValueError("--adjudication supplied but no third-review item is required")
        adjudicator_form = _validate_adjudication(
            args.adjudication,
            adjudication_template,
            {first["reviewer_id"], second["reviewer_id"]},
        )
    elif adjudication_needed:
        args.output.mkdir(parents=True, exist_ok=True)
        if template_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite adjudication template {template_path}"
            )
        template_path.write_text(
            json.dumps(adjudication_template, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    args.output.mkdir(parents=True, exist_ok=True)
    locked_forms = [
        args.output / f"{first['reviewer_id']}-locked-score-form.json",
        args.output / f"{second['reviewer_id']}-locked-score-form.json",
    ]
    result_path = args.output / "review-import.json"
    template_path = args.output / "adjudication-form.template.json"
    report_path = args.output / "qualification-report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite final report {report_path}")
    existing_initial = [path.exists() for path in [*locked_forms, result_path]]
    if any(existing_initial):
        if not all(existing_initial) or args.adjudication is None:
            raise FileExistsError(
                f"Initial import artifacts already exist or are incomplete in {args.output}"
            )
        for source, destination in zip(
            (args.reviewer_one, args.reviewer_two), locked_forms, strict=True
        ):
            if source.read_bytes() != destination.read_bytes():
                raise ValueError(f"Reviewer source {source} differs from locked form {destination}")
        if not template_path.is_file():
            raise FileNotFoundError(f"Missing locked adjudication template {template_path}")
        locked_template = json.loads(template_path.read_text(encoding="utf-8"))
        if locked_template != adjudication_template:
            raise ValueError(
                "Current reviewer forms do not reproduce the locked adjudication template"
            )
    else:
        if args.adjudication is not None:
            raise ValueError("--adjudication requires an existing initial import in --output")
        for source, destination in zip(
            (args.reviewer_one, args.reviewer_two), locked_forms, strict=True
        ):
            destination.write_bytes(source.read_bytes())

    criterion_means = None
    material_means = None
    weighted_overall_mean = None
    if not adjudication:
        criterion_values = {criterion: [] for criterion in CRITERIA}
        material_values: dict[str, dict[str, list[float]]] = {}
        for identifier in sorted(first_items):
            first_item, second_item = first_items[identifier], second_items[identifier]
            material = first_item["material"]
            material_values.setdefault(material, {criterion: [] for criterion in CRITERIA})
            for criterion in CRITERIA:
                mean = (
                    first_item["score_form"]["criterion_scores_0_to_4"][criterion]
                    + second_item["score_form"]["criterion_scores_0_to_4"][criterion]
                ) / 2
                criterion_values[criterion].append(mean)
                material_values[material][criterion].append(mean)
        criterion_means = {
            criterion: sum(values) / len(values) for criterion, values in criterion_values.items()
        }
        material_means = {
            material: {
                criterion: sum(values) / len(values) for criterion, values in criteria.items()
            }
            for material, criteria in material_values.items()
        }
        weighted_overall_mean = sum(
            criterion_means[criterion] * weight for criterion, weight in WEIGHTS.items()
        )

    final_report = _final_report(
        first,
        second,
        first_items,
        second_items,
        adjudication,
        adjudicator_form,
    )

    result = {
        "import_version": 1,
        "response_count": 18,
        "reviewers": [first["reviewer_id"], second["reviewer_id"]],
        "criterion_ids": list(CRITERIA),
        "criterion_comparisons": comparisons,
        "agreed_reviewable_response_count": reviewable_count,
        "provisional_unadjudicated_means": {
            "by_criterion": criterion_means,
            "by_material_and_criterion": material_means,
            "weighted_overall_0_to_4": weighted_overall_mean,
        },
        "adjudication_required": adjudication_needed,
        "adjudication_queue": adjudication,
        "adjudication_template": str(args.output / "adjudication-form.template.json")
        if adjudication_needed and adjudicator_form is None
        else None,
        "adjudicator_id": adjudicator_form.get("adjudicator_id")
        if adjudicator_form
        else None,
        "status": "adjudication_required"
        if adjudication_needed and adjudicator_form is None
        else "final_report_ready"
        if final_report
        else "adjudication_required",
        "final_qualification_report": final_report,
        "scientific_qualification_score": (
            final_report["weighted_overall_mean_0_to_4"] if final_report else None
        ),
        "note": "Validity is not scored; finalize aggregation after required adjudication.",
    }
    if adjudicator_form is not None:
        report_path.write_text(
            json.dumps(final_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif not result_path.exists():
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if final_report is not None:
            report_path.write_text(
                json.dumps(final_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    print(
        json.dumps(
            {
                "response_count": 18,
                "adjudication_items": len(adjudication),
                "status": "final_report_ready" if final_report else result["status"],
                "output": str(args.output),
                "final_report": str(report_path) if final_report is not None else None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
