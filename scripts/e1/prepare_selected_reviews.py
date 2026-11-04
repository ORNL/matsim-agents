#!/usr/bin/env python3
"""Prepare private, blinded, unscored review packets from completed E1 run ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path
from random import SystemRandom
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_blinded_reviews import CRITERIA, THRESHOLDS

from matsim_agents.workflows.e1 import assess_semantic_validity

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = ROOT / "research/paper/e1/d-questions.v1.json"
SCHEMA_PATH = ROOT / "research/paper/e1/response-schema.v1.json"
PROTOCOL_PATH = ROOT / "research/paper/e1/development-protocol.v1.md"
AMENDMENT_PATH = ROOT / "research/paper/e1/development-execution-amendment.v1.md"
PROMPT_AMENDMENT_PATH = ROOT / "research/paper/e1/development-prompt-amendment.v2.md"
REVIEWER_INSTRUCTIONS = "research/paper/e1/reviewer-scoring-instructions.md"
FORBIDDEN_KEYS = {
    "provider",
    "model_id",
    "catalog_name",
    "endpoint",
    "campaign",
    "response_key",
    "response_status",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _assert_blinded(value: Any) -> None:
    if isinstance(value, dict):
        if FORBIDDEN_KEYS.intersection(value):
            raise ValueError("review packet contains a protected identity/provenance field")
        for child in value.values():
            _assert_blinded(child)
    elif isinstance(value, list):
        for child in value:
            _assert_blinded(child)


def _latest_responses(ledger: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(ledger):
        question_id = row.get("question_id")
        if not question_id:
            continue
        if question_id not in latest or row.get("attempt", 0) >= latest[question_id].get("attempt", 0):
            latest[question_id] = row
    return latest


def _review_response(row: dict[str, Any]) -> Any:
    parsed = row.get("parsed_response")
    if isinstance(parsed, dict):
        return parsed
    raw = row.get("response")
    if isinstance(raw, dict):
        return raw.get("text") or ""
    return raw if isinstance(raw, str) else ""


def _build_packet(
    reviewer_id: str,
    by_question: dict[str, dict[str, Any]],
    hashes: dict[str, str],
) -> dict[str, Any]:
    items = list(by_question.values())
    SystemRandom().shuffle(items)
    return {
        "packet_version": 2,
        "reviewer_id": reviewer_id,
        "instructions_file": REVIEWER_INSTRUCTIONS,
        **hashes,
        "reviewer_guidance": (
            "Score independently. This packet contains the 18 supplied development questions. "
            "Some answer text may be partial or malformed; judge scientific quality and "
            "reviewability separately. Do not infer model identity or execution condition."
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing review output: {output_root}")

    old_umask = os.umask(0o077)
    try:
        output_root.mkdir(parents=True, mode=0o700)
        (output_root / "reviewer-1").mkdir(mode=0o700)
        (output_root / "reviewer-2").mkdir(mode=0o700)
        admin_root = output_root / "admin"
        admin_root.mkdir(mode=0o700)

        question_doc = _read_json(QUESTIONS_PATH)
        canonical_questions = {
            item["question_id"]: item for item in question_doc["questions"]
        }
        if len(canonical_questions) != 18:
            raise ValueError("canonical E1 D packet must contain exactly 18 unique questions")
        protocol_hash = _sha256_file(PROTOCOL_PATH)
        amendment_hash = _sha256_file(AMENDMENT_PATH)
        prompt_amendment_hash = _sha256_file(PROMPT_AMENDMENT_PATH)
        question_hash = _sha256_file(QUESTIONS_PATH)
        schema_hash = _sha256_file(SCHEMA_PATH)
        rubric_hash = _sha256_bytes(
            json.dumps({"criteria": CRITERIA, "thresholds": THRESHOLDS}, sort_keys=True).encode()
        )

        crosswalk: dict[str, Any] = {}
        for run_dir in args.run_dir:
            run_dir = run_dir.resolve()
            manifest_path = run_dir / "manifest.json"
            run_path = run_dir / "run.json"
            ledger_path = run_dir / "responses.jsonl"
            if not all(path.is_file() for path in (manifest_path, run_path, ledger_path)):
                raise FileNotFoundError(f"run directory is missing manifest, run summary, or ledger: {run_dir}")
            manifest = _read_json(manifest_path)
            run = _read_json(run_path)
            questions = {item["question_id"]: item for item in manifest.get("questions", [])}
            if set(questions) != set(canonical_questions):
                raise ValueError(f"{run_dir}: manifest does not contain the frozen 18-question D set")
            latest = _latest_responses(ledger_path)
            if set(latest) != set(canonical_questions):
                raise ValueError(f"{run_dir}: expected one recorded latest outcome for each of 18 questions")
            placeholder_ids = sorted(
                question_id
                for question_id, row in latest.items()
                if assess_semantic_validity(row.get("parsed_response"))["semantic_validity"]
                in {"placeholder_only", "invalid"}
            )
            if placeholder_ids:
                raise ValueError(
                    f"{run_dir}: schema-placeholder responses are not scientifically reviewable: "
                    + ", ".join(placeholder_ids)
                )

            blind_candidate_id = "E1-CAND-" + secrets.token_hex(6).upper()
            response_map: dict[str, Any] = {}
            by_question: dict[str, dict[str, Any]] = {}
            for question_id, question in questions.items():
                row = latest[question_id]
                blind_response_id = "E1-RSP-" + secrets.token_hex(6).upper()
                response_map[blind_response_id] = {
                    "question_id": question_id,
                    "response_key": row.get("response_key"),
                    "status": row.get("status"),
                    "attempt": row.get("attempt"),
                }
                by_question[question_id] = {
                    "blind_response_id": blind_response_id,
                    "material": question["material"],
                    "task_type": question["category"],
                    "question": question["prompt"],
                    "evidence_and_assumptions": (
                        "The question text is the complete supplied evidence and assumptions packet; "
                        "no separate evidence file was supplied."
                    ),
                    "response": _review_response(row),
                    "score_form": {
                        "criterion_scores_0_to_4": {
                            criterion["id"]: None for criterion in CRITERIA
                        },
                        "rationales_and_response_citations": {
                            criterion["id"]: "" for criterion in CRITERIA
                        },
                        "reviewable": None,
                        "reviewability_rationale": "",
                        "integrity_flags": [],
                        "reviewer_notes": "",
                    },
                }

            execution_prompt_hash = _sha256_bytes(
                json.dumps(
                    [questions[key] for key in sorted(questions)],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
            response_packet_hash = _sha256_bytes(
                json.dumps(
                    [
                        {
                            "blind_response_id": by_question[qid]["blind_response_id"],
                            "response": by_question[qid]["response"],
                        }
                        for qid in sorted(by_question)
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
            hashes = {
                "scientific_protocol_hash": protocol_hash,
                "execution_amendment_hash": amendment_hash,
                "prompt_amendment_hash": prompt_amendment_hash,
                "question_packet_hash": question_hash,
                "execution_prompt_set_hash": execution_prompt_hash,
                "response_schema_hash": schema_hash,
                "scoring_rubric_hash": rubric_hash,
                "response_packet_hash": response_packet_hash,
            }
            packets = {
                reviewer: _build_packet(reviewer, by_question, hashes)
                for reviewer in ("reviewer-1", "reviewer-2")
            }
            for packet in packets.values():
                _assert_blinded(packet)
                identifiers = [item["blind_response_id"] for item in packet["responses"]]
                if len(identifiers) != 18 or len(set(identifiers)) != 18:
                    raise ValueError("review packet response IDs are not unique and complete")
            id_sets = [
                {item["blind_response_id"] for item in packet["responses"]}
                for packet in packets.values()
            ]
            if id_sets[0] != id_sets[1]:
                raise ValueError("reviewer packets do not share the same anonymous response IDs")

            (output_root / "reviewer-1" / blind_candidate_id).mkdir(mode=0o700)
            (output_root / "reviewer-2" / blind_candidate_id).mkdir(mode=0o700)
            _atomic_json(
                output_root / "reviewer-1" / blind_candidate_id / "packet.json",
                packets["reviewer-1"],
            )
            _atomic_json(
                output_root / "reviewer-2" / blind_candidate_id / "packet.json",
                packets["reviewer-2"],
            )
            candidate = manifest["candidate"]
            crosswalk[blind_candidate_id] = {
                "model_id": candidate["model_id"],
                "provider": candidate["provider"],
                "mag_provider": candidate.get("mag_provider"),
                "run_directory": str(run_dir.relative_to(ROOT)),
                "run_state": run.get("state"),
                "prompt_variant": run.get("prompt_variant", "v1"),
                "system_prompt_hash": run.get("system_prompt_hash"),
                "manifest_hash": run.get("manifest_hash"),
                "response_packet_hash": response_packet_hash,
                "responses": response_map,
            }

        _atomic_json(admin_root / "provenance-crosswalk.json", crosswalk)
        _atomic_json(
            admin_root / "preparation-record.json",
            {
                "status": "unscored_pending_independent_review",
                "candidate_count": len(crosswalk),
                "reviewer_packet_count": 2 * len(crosswalk),
                "created_at_utc": __import__("datetime").datetime.now(
                    __import__("datetime").UTC
                ).isoformat(),
                "note": (
                    "Packets are unscored and must be assigned only after reviewer qualification, "
                    "conflict checks, and rubric/protocol approval. Keep admin provenance files "
                    "separate from reviewer delivery."
                ),
            },
        )
        print(
            json.dumps(
                {
                    "candidate_packets": len(crosswalk),
                    "reviewer_packets": 2 * len(crosswalk),
                    "output_root": str(output_root.relative_to(ROOT)),
                    "admin_permissions": oct(admin_root.stat().st_mode & 0o777),
                },
                sort_keys=True,
            )
        )
    finally:
        os.umask(old_umask)


if __name__ == "__main__":
    main()