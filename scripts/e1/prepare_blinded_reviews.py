#!/usr/bin/env python3
"""Build blinded E1 reviewer forms from immutable original and recovery ledgers."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from random import SystemRandom
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/e1"
ORIGINAL = RUNS / "e1-r-mag-gpt-6-luna-2026-10-29"
RECOVERY = RUNS / "e1-r-mag-gpt-6-luna-recovery-2026-10-29"
SCORING = RUNS / "e1-r-mag-gpt-6-luna-scoring-2026-10-29"
ADMIN = RUNS / "e1-scoring-admin-2026-10-29"
ADMIN_CROSSWALK = ADMIN / "review-crosswalk.json"
REVIEWER_KEY = ADMIN / "reviewer-key.json"

CRITERIA = [
    {
        "id": "scientific_correctness",
        "name": "Scientific correctness",
        "weight": 0.30,
        "anchors": [
            "Central claims contradict the prompt/evidence or basic constraints.",
            "Major unsupported or incorrect claims.",
            "Mixed correctness; a material error or scope overclaim remains.",
            "Claims are sound; only minor noncentral errors.",
            "All material claims are correct, scoped, and tied to the supplied evidence.",
        ],
    },
    {
        "id": "physical_reasoning",
        "name": "Physical reasoning",
        "weight": 0.20,
        "anchors": [
            "No usable mechanism or causal logic.",
            "Mechanism labels without a coherent causal chain.",
            "Partly coherent chain with important assumptions omitted.",
            "Coherent mechanism and assumptions, with minor gaps.",
            "Distinguishes competing mechanisms and explains causal links and limits.",
        ],
    },
    {
        "id": "falsifiability_and_tests",
        "name": "Falsifiability and discriminating tests",
        "weight": 0.20,
        "anchors": [
            "No discriminating test.",
            "Test cannot separate the stated alternatives.",
            "Plausible test but outcomes or controls are incomplete.",
            "Feasible discriminating test and interpretable outcomes.",
            "Matched, feasible tests distinguish alternatives and state clear falsifiers.",
        ],
    },
    {
        "id": "evidence_awareness_and_uncertainty",
        "name": "Evidence awareness and uncertainty",
        "weight": 0.20,
        "anchors": [
            "Treats absent evidence as fact or asserts unsupported certainty.",
            "Misses major gaps/conflicts or scope limits.",
            "Identifies some gaps but overstates at least one conclusion.",
            "Calibrated uncertainty and key evidence gaps; minor omission.",
            "Separates observation, inference, and hypothesis; all material gaps/"
            "scope limits are explicit.",
        ],
    },
    {
        "id": "computational_feasibility",
        "name": "Computational feasibility",
        "weight": 0.10,
        "anchors": [
            "Proposed work cannot answer the question or is infeasible as stated.",
            "Major feasibility/control flaw.",
            "Potentially useful but underspecified or costly without justification.",
            "Feasible and adequately controlled, with minor omissions.",
            "Efficient, matched, reproducible plan with appropriate MLIP/DFT/experimental scope.",
        ],
    },
]

THRESHOLDS = {
    "mean_adjudicated_scientific_correctness": "at least 3.0/4 across D",
    "material_level_scientific_correctness": "no material-level mean below 2.5/4",
    "mean_adjudicated_evidence_awareness_and_uncertainty": "at least 3.0/4 across D",
    "weighted_overall_mean": "at least 3.0/4 on a 0-4 scale",
    "reviewable_responses": "at least 16 of 18",
    "integrity_vetoes": [
        "confidently asserted fabricated numerical result",
        "fabricated source",
        "unsupported global-stability claim after explicit domain limitation",
    ],
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    SCORING.mkdir(parents=True, exist_ok=True)
    ADMIN.mkdir(parents=True, exist_ok=True)
    question_file = ROOT / "research/paper/e1/d-questions.v1.json"
    schema_file = ROOT / "research/paper/e1/response-schema.v1.json"
    protocol_file = ROOT / "research/paper/e1/development-protocol.v1.md"
    amendment_file = ROOT / "research/paper/e1/development-execution-amendment.v1.md"
    questions_doc = read_json(question_file)
    questions = {item["question_id"]: item for item in questions_doc["questions"]}
    original_manifest = read_json(ORIGINAL / "manifest.json")
    recovery_manifest = read_json(RECOVERY / "manifest.json")
    manifest_questions = {item["question_id"]: item for item in original_manifest["questions"]}
    recovery_questions = {item["question_id"]: item for item in recovery_manifest["questions"]}
    prompt_fields = ("question_id", "material", "category", "prompt")
    assert set(manifest_questions) == set(questions)
    original = read_jsonl(ORIGINAL / "responses.jsonl")
    recovery = read_jsonl(RECOVERY / "responses.jsonl")
    assert len(original) == 18 and len(recovery) == 5
    assert original_manifest["decoding"]["max_tokens"] == 2048
    assert recovery_manifest["decoding"]["max_tokens"] == 32768

    valid_original = {
        item["question_id"]: item
        for item in original
        if item.get("status") == "complete" and item.get("parsed_response") is not None
    }
    failed_original = {
        item["question_id"]: item for item in original if item.get("status") == "error"
    }
    assert set(recovery_questions) == set(failed_original)
    assert all(
        tuple(manifest_questions[key][field] for field in prompt_fields)
        == tuple(recovery_questions[key][field] for field in prompt_fields)
        for key in recovery_questions
    )
    valid_recovery = {
        item["question_id"]: item
        for item in recovery
        if item.get("status") == "complete"
        and item.get("parsed_response") is not None
        and not item["response"].get("truncated")
    }
    assert len(valid_original) == 13 and len(failed_original) == 5
    assert len(valid_recovery) == 5
    assert set(valid_recovery) == set(failed_original)
    assert len(valid_original.keys() | valid_recovery.keys()) == 18
    assert set(valid_original.keys() | valid_recovery.keys()) == set(questions)

    old_key = read_json(REVIEWER_KEY) if REVIEWER_KEY.exists() else {"reviewers": {}}
    crosswalk = (
        read_json(ADMIN_CROSSWALK)
        if ADMIN_CROSSWALK.exists()
        else {
            "model_identity": {},
            "responses": {},
        }
    )
    stable_ids: dict[str, str] = {}
    prior_response_keys: dict[str, str] = {}
    for _reviewer, mappings in old_key.get("reviewers", {}).items():
        for mapping in mappings:
            stable_ids[mapping["question_id"]] = mapping["blind_response_id"]
            if mapping.get("original_response_key"):
                prior_response_keys[mapping["question_id"]] = mapping["original_response_key"]
    for question_id in set(valid_original) | set(valid_recovery):
        if question_id not in stable_ids:
            stable_ids[question_id] = "E1-" + secrets.token_hex(6).upper()

    response_sources: dict[str, tuple[str, dict[str, Any]]] = {}
    response_sources.update({key: ("original", value) for key, value in valid_original.items()})
    response_sources.update({key: ("recovery", value) for key, value in valid_recovery.items()})
    crosswalk["model_identity"] = {
        "catalog_name": original_manifest["candidate"]["catalog_name"],
        "model_id": original_manifest["candidate"]["model_id"],
        "revision": original_manifest["candidate"]["revision"],
        "provider": original_manifest["candidate"]["provider"],
        "upstream_provider": original_manifest["candidate"].get("mag_provider"),
    }
    crosswalk["responses"] = {
        stable_ids[question_id]: {
            "question_id": question_id,
            "response_key": record["response_key"],
            "original_response_key": prior_response_keys.get(question_id, record["response_key"]),
            "campaign": campaign,
            "campaign_directory": str(
                ORIGINAL.relative_to(ROOT) if campaign == "original" else RECOVERY.relative_to(ROOT)
            ),
            "attempt": record["attempt"],
            "original_first_attempt_status": (
                "complete" if question_id in valid_original else "truncated_technical_failure"
            ),
            "recovery_provenance": campaign == "recovery",
            "max_output_tokens": (
                original_manifest["decoding"]["max_tokens"]
                if campaign == "original"
                else recovery_manifest["decoding"]["max_tokens"]
            ),
        }
        for question_id, (campaign, record) in response_sources.items()
    }
    crosswalk["execution_summary"] = {
        "original_first_attempt_complete": 13,
        "original_requests": 18,
        "original_truncated_technical_failures": 5,
        "recovery_requests": 5,
        "recovery_complete": 5,
        "recovery_assisted_complete_responses": 18,
        "total_requests": 23,
    }

    by_question: dict[str, dict[str, Any]] = {}
    for question_id, (_campaign, record) in response_sources.items():
        question = manifest_questions[question_id]
        by_question[question_id] = {
            "blind_response_id": stable_ids[question_id],
            "material": question["material"],
            "task_type": question["category"],
            "question": question["prompt"],
            "evidence_and_assumptions": (
                "The question text is the complete supplied evidence and assumptions packet; "
                "no separate evidence file was supplied."
            ),
            "response": record["parsed_response"],
            "score_form": {
                "criterion_scores_0_to_4": {criterion["id"]: None for criterion in CRITERIA},
                "rationales_and_response_citations": {
                    criterion["id"]: "" for criterion in CRITERIA
                },
                "reviewable": None,
                "reviewability_rationale": "",
                "integrity_flags": [],
                "reviewer_notes": "",
            },
        }

    protocol_hash = sha256(protocol_file)
    amendment_hash = sha256(amendment_file)
    question_hash = sha256(question_file)
    execution_prompt_hash = hashlib.sha256(
        json.dumps(
            [manifest_questions[key] for key in sorted(manifest_questions)],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    schema_hash = sha256(schema_file)
    rubric_hash = hashlib.sha256(
        json.dumps({"criteria": CRITERIA, "thresholds": THRESHOLDS}, sort_keys=True).encode()
    ).hexdigest()

    for reviewer in ("reviewer-1", "reviewer-2"):
        items = list(by_question.values())
        SystemRandom().shuffle(items)
        packet = {
            "packet_version": 2,
            "reviewer_id": reviewer,
            "instructions_file": "research/paper/e1/reviewer-scoring-instructions.md",
            "scientific_protocol_hash": protocol_hash,
            "execution_amendment_hash": amendment_hash,
            "question_packet_hash": question_hash,
            "execution_prompt_set_hash": execution_prompt_hash,
            "response_schema_hash": schema_hash,
            "scoring_rubric_hash": rubric_hash,
            "reviewer_guidance": (
                "Score independently. This packet contains 18 complete responses. "
                "Do not infer model identity, provider, or execution condition."
            ),
            "criteria": CRITERIA,
            "qualification_thresholds": THRESHOLDS,
            "adjudication_rule": (
                "Third blinded adjudication is required for any criterion difference >=2 points "
                "or different reviewer qualification outcomes; otherwise use the two-reviewer mean."
            ),
            "reviewer_qualification_outcome": None,
            "candidate_level_rationale": "",
            "responses": items,
        }
        atomic_json(SCORING / f"{reviewer}-packet.json", packet)

    atomic_json(ADMIN_CROSSWALK, crosswalk)
    os.chmod(ADMIN_CROSSWALK, 0o600)
    new_key = {"reviewers": {}}
    for reviewer in ("reviewer-1", "reviewer-2"):
        new_key["reviewers"][reviewer] = [
            {
                "blind_response_id": stable_ids[qid],
                "original_response_key": crosswalk["responses"][stable_ids[qid]][
                    "original_response_key"
                ],
                "question_id": qid,
            }
            for qid in stable_ids
        ]
    atomic_json(REVIEWER_KEY, new_key)
    os.chmod(REVIEWER_KEY, 0o600)

    evaluation_record = {
        "evaluation": "GPT-6 Luna E1 development scientific review set",
        "created_utc_date": "2026-10-29",
        "original_campaign": {
            "directory": str(ORIGINAL.relative_to(ROOT)),
            "requests": 18,
            "first_attempt_complete": 13,
            "truncated_technical_failures": 5,
            "max_output_tokens": 2048,
            "input_tokens_reported": 9380,
            "output_tokens_reported": 25868,
            "provider_cost": "unavailable",
            "reasoning_tokens": "unavailable",
        },
        "separately_authorized_recovery": {
            "directory": str(RECOVERY.relative_to(ROOT)),
            "requests": 5,
            "complete": 5,
            "max_output_tokens": 32768,
            "input_tokens_reported": 2630,
            "output_tokens_reported": 11838,
            "total_latency_seconds": 130.882,
            "provider_cost": "unavailable",
            "reasoning_tokens": "unavailable",
        },
        "recovery_assisted_reviewable_responses": 18,
        "total_requests": 23,
        "interpretation": (
            "Original completion and recovery-assisted coverage are reported separately. "
            "The recovery does not convert original truncations into first-attempt successes. "
            "Scientific competence remains unscored pending blinded human review."
        ),
        "hashes": {
            "original_scientific_protocol_v1": protocol_hash,
            "execution_configuration_amendment_v1": amendment_hash,
            "development_question_artifact": question_hash,
            "executed_prompt_set": execution_prompt_hash,
            "response_schema": schema_hash,
            "scoring_rubric": rubric_hash,
        },
        "reviewer_packets": [
            "reviewer-1-packet.json",
            "reviewer-2-packet.json",
        ],
        "protected_provenance_crosswalk": str(ADMIN_CROSSWALK.relative_to(ROOT)),
        "review_status": "pending independent blinded reviewer scoring",
    }
    atomic_json(SCORING / "evaluation-record.json", evaluation_record)
    packets = [
        read_json(SCORING / f"{reviewer}-packet.json") for reviewer in ("reviewer-1", "reviewer-2")
    ]
    packet_ids = [{item["blind_response_id"] for item in packet["responses"]} for packet in packets]
    assert all(len(packet["responses"]) == 18 for packet in packets)
    assert packet_ids[0] == packet_ids[1]
    forbidden_keys = {
        "provider",
        "model_id",
        "catalog_name",
        "endpoint",
        "campaign",
        "response_key",
    }

    def assert_blinded(value: Any) -> None:
        if isinstance(value, dict):
            assert not forbidden_keys.intersection(value)
            for child in value.values():
                assert_blinded(child)
        elif isinstance(value, list):
            for child in value:
                assert_blinded(child)

    for packet in packets:
        assert_blinded(packet)
    print(
        json.dumps(
            {
                "questions": len(by_question),
                "original_complete": len(valid_original),
                "recovery_complete": len(valid_recovery),
                "total_requests": 23,
                "protocol_hash": protocol_hash,
                "amendment_hash": amendment_hash,
                "question_hash": question_hash,
                "execution_prompt_hash": execution_prompt_hash,
                "schema_hash": schema_hash,
                "scoring_dir": str(SCORING.relative_to(ROOT)),
                "reviewer_packets_share_ids": packet_ids[0] == packet_ids[1],
                "reviewer_packets_blinding_keys_checked": True,
                "crosswalk_permissions": oct(ADMIN_CROSSWALK.stat().st_mode & 0o777),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
