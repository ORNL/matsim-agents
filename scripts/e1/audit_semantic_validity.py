#!/usr/bin/env python3
"""Write a semantic-validity audit sidecar for an archived E1 run without modifying it.

The archived ``responses.jsonl`` and ``run.json`` are read-only inputs; their SHA-256
digests are recorded so the audit can be tied to the exact bytes it assessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from matsim_agents.workflows.e1 import assess_semantic_validity

AUDIT_VERSION = "e1-semantic-audit-v1"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_run(run_dir: Path) -> dict:
    ledger = run_dir / "responses.jsonl"
    run_summary = run_dir / "run.json"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
    responses = []
    for row in rows:
        semantic = assess_semantic_validity(row.get("parsed_response"))
        recorded = row.get("scientific_reviewability")
        if row.get("status") != "complete":
            audited = recorded
        elif semantic["semantic_validity"] == "valid":
            audited = "reviewable"
        else:
            audited = "nonreviewable_semantic_output"
        responses.append(
            {
                "question_id": row.get("question_id"),
                "response_key": row.get("response_key"),
                "attempt": row.get("attempt"),
                "recorded_status": row.get("status"),
                "recorded_scientific_reviewability": recorded,
                "technical_completion": (
                    "truncated" if (row.get("response") or {}).get("truncated") else "completed"
                ),
                "schema_compliance": "compliant" if row.get("schema_error") is None else (
                    "noncompliant"
                ),
                **semantic,
                "audited_scientific_reviewability": audited,
            }
        )
    return {
        "audit_version": AUDIT_VERSION,
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "run_directory": str(run_dir),
        "responses_jsonl_sha256": _sha256_file(ledger),
        "run_json_sha256": _sha256_file(run_summary) if run_summary.is_file() else None,
        "summary": {
            "records": len(responses),
            "semantic_validity": dict(Counter(item["semantic_validity"] for item in responses)),
            "recorded_reviewable": sum(
                item["recorded_scientific_reviewability"] == "reviewable" for item in responses
            ),
            "audited_reviewable": sum(
                item["audited_scientific_reviewability"] == "reviewable" for item in responses
            ),
        },
        "note": (
            "Sidecar audit only. The archived ledger and run summary are unmodified; recorded "
            "fields are preserved verbatim. No scientific-quality score is implied."
        ),
        "responses": responses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="E1 output directories")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        before = _sha256_file(run_dir / "responses.jsonl")
        audit = audit_run(run_dir)
        destination = run_dir / "semantic-audit.json"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing audit: {destination}")
        destination.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if _sha256_file(run_dir / "responses.jsonl") != before:
            raise RuntimeError(f"archived ledger changed during audit: {run_dir}")
        print(json.dumps({"run_dir": str(run_dir), **audit["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
