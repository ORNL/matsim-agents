"""Run one bounded E1-R request per eligible MAG catalog model ID."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from matsim_agents.workflows.e1 import (
    CURRENT_PROMPT_VARIANT,
    E1DevelopmentManifest,
    run_e1_development,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
SWEEP_ROOT = RUNS_ROOT / "e1" / "remote-2026-11-02" / "catalog-sweep"
CATALOG_PATH = SWEEP_ROOT / "catalog.json"
FOLLOWUP_ROOT = RUNS_ROOT / "e1" / "remote-2026-11-02" / "catalog-sweep-allocated-followup"
SOURCE_MANIFEST = (
    RUNS_ROOT
    / "e1"
    / "remote-2026-11-02"
    / "manifests"
    / "gemini-2-5-flash-dlfp-a-canary-4k.json"
)
GOOGLE_OUTPUT_TOKENS = 8192
CBORG_OUTPUT_TOKENS = 4096
MAX_INPUT_BYTES = 16384


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _fetch_catalog() -> list[str]:
    token = os.environ.get("AMSC_TOKEN")
    if not token:
        raise RuntimeError("AMSC_TOKEN is not set in the current environment")
    base_url = os.environ.get(
        "MATSIM_MAG_BASE_URL", "https://mag.genesis.americansciencecloud.org/v1"
    ).rstrip("/")
    request = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        entries = json.load(response).get("data", [])
    return sorted({str(entry["id"]) for entry in entries if entry.get("id")})


def _catalog_ids(refresh: bool) -> list[str]:
    if CATALOG_PATH.is_file() and not refresh:
        payload = json.loads(CATALOG_PATH.read_text())
        return list(payload["model_ids"])
    model_ids = _fetch_catalog()
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(
        json.dumps(
            {"captured_at_utc": _timestamp(), "model_ids": model_ids},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return model_ids


def _completed_mag_models() -> set[str]:
    completed: set[str] = set()
    for ledger_path in RUNS_ROOT.rglob("responses.jsonl"):
        manifest_path = ledger_path.parent / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            candidate = manifest.get("candidate") or {}
            if manifest.get("cohort") != "E1-R" or candidate.get("provider") != "mag":
                continue
            question_ids = {
                question.get("question_id")
                for question in manifest.get("questions", [])
                if question.get("question_id")
            }
            latest: dict[str, dict[str, Any]] = {}
            for line in ledger_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                question_id = row.get("question_id")
                if question_id and (
                    question_id not in latest
                    or row.get("attempt", 0) >= latest[question_id].get("attempt", 0)
                ):
                    latest[question_id] = row
            if question_ids and question_ids.issubset(latest) and all(
                latest[question_id].get("status") == "complete"
                and isinstance(latest[question_id].get("parsed_response"), dict)
                for question_id in question_ids
            ):
                completed.add(candidate["model_id"])
        except (KeyError, OSError, json.JSONDecodeError):
            continue
    return completed


def _attempted_mag_models() -> set[str]:
    attempted: set[str] = set()
    for marker_path in RUNS_ROOT.rglob("request-started.json"):
        try:
            model_id = json.loads(marker_path.read_text()).get("model_id")
            if model_id:
                attempted.add(model_id)
        except (OSError, json.JSONDecodeError):
            continue

    for ledger_path in RUNS_ROOT.rglob("responses.jsonl"):
        if not ledger_path.is_file() or ledger_path.stat().st_size == 0:
            continue
        manifest_path = ledger_path.parent / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            candidate = manifest.get("candidate") or {}
            if manifest.get("cohort") == "E1-R" and candidate.get("provider") == "mag":
                attempted.add(candidate["model_id"])
        except (KeyError, OSError, json.JSONDecodeError):
            continue
    return attempted


def _slug(model_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-.")[:72]
    digest = hashlib.sha256(model_id.encode()).hexdigest()[:12]
    return f"{readable or 'model'}-{digest}"


def _route(model_id: str) -> str | None:
    if "@" not in model_id:
        return None
    return model_id.rsplit("@", 1)[1]


def _manifest_for(
    model_id: str,
    question: dict[str, Any],
    protocol_version: str,
    base_url: str,
    suite_size: int,
    max_output_tokens: int,
    approval_reference: str,
) -> E1DevelopmentManifest:
    route = _route(model_id)
    allocation_reference = (
        "User-authorized allocated-route MAG sweep; one request per candidate; "
        f"{max_output_tokens} output-token ceiling; {suite_size} requests"
    )
    candidate = {
        "catalog_name": model_id,
        "model_id": model_id,
        "revision": model_id,
        "max_output_tokens_supported": None,
        "revision_semantics": "mutable_alias",
        "provider": "mag",
        "mag_provider": route,
    }
    approval = {
        "approval_reference": approval_reference,
        "approved_by": os.environ.get("USER", "workspace user"),
        "provider": "mag",
        "mag_provider": route,
        "model_id": model_id,
        "revision": model_id,
        "revision_semantics": "mutable_alias",
        "mutable_alias_accepted": True,
        "endpoint": base_url,
        "project_authorization_reference": "User authorization in chat on 2026-11-02",
        "allocation_limit_reference": allocation_reference,
        "allocation_charge_description": (
            f"At most one MAG request for this model with up to {max_output_tokens} "
            f"generated tokens plus provider-defined input processing; full sweep "
            f"ceiling {suite_size} requests and "
            f"per-request limits are recorded in each manifest; billed under account rates."
        ),
        "data_use_conditions": (
            "Only the supplied D-LFP-A scientific prompt is sent to MAG and the "
            "catalog-listed upstream route; no proprietary or personal data is included."
        ),
        "retention_conditions": (
            "Subject to the applicable MAG and upstream account retention terms; "
            "no additional retention guarantee is assumed."
        ),
        "max_input_message_bytes_per_question": MAX_INPUT_BYTES,
        "max_input_message_bytes_campaign": MAX_INPUT_BYTES,
        "max_total_requests": 1,
        "max_attempts_per_question": 1,
        "max_output_tokens_per_response": max_output_tokens,
        "max_output_tokens_campaign": max_output_tokens,
    }
    return E1DevelopmentManifest.model_validate(
        {
            "schema_version": 1,
            "partition": "development",
            "cohort": "E1-R",
            "candidate": candidate,
            "decoding": {
                "temperature": 0.0,
                "max_tokens": max_output_tokens,
                "top_p": None,
                "seed": None,
            },
            "remote_approval": approval,
            "protocol_version": protocol_version,
            "prompt_variant": CURRENT_PROMPT_VARIANT,
            "questions": [question],
        }
    )


def _was_attempted(output_dir: Path) -> bool:
    marker = output_dir / "request-started.json"
    ledger = output_dir / "responses.jsonl"
    return marker.exists() or (ledger.is_file() and ledger.stat().st_size > 0)


def _record_result(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="inventory candidates without inference")
    parser.add_argument("--refresh-catalog", action="store_true", help="fetch and freeze a fresh MAG catalog")
    parser.add_argument(
        "--routes",
        nargs="+",
        choices=("google", "cborg"),
        default=("google", "cborg"),
        help="allocated MAG upstream routes to include",
    )
    args = parser.parse_args()

    model_ids = _catalog_ids(args.refresh_catalog)
    embedding_ids = {model_id for model_id in model_ids if "embedding" in model_id.lower()}
    completed = _completed_mag_models()
    allocated_routes = set(args.routes)
    eligible = [
        model_id
        for model_id in model_ids
        if model_id not in embedding_ids
        and model_id not in completed
        and _route(model_id) in allocated_routes
    ]
    outputs_root = FOLLOWUP_ROOT / "outputs"
    attempted = _attempted_mag_models()
    pending = [
        model_id
        for model_id in eligible
        if model_id not in attempted
        and not _was_attempted(outputs_root / _slug(model_id))
    ]
    requests = [
        (
            model_id,
            GOOGLE_OUTPUT_TOKENS if _route(model_id) == "google" else CBORG_OUTPUT_TOKENS,
        )
        for model_id in pending
    ]
    print(
        f"catalog_ids={len(model_ids)} embeddings_excluded={len(embedding_ids)} "
        f"properly_collected_excluded={len(completed & set(model_ids))} "
        f"previously_attempted_excluded={len(attempted & set(model_ids))} "
        f"allocated_candidates={len(eligible)} pending_ids={len(pending)} "
        f"requests={len(requests)} "
        f"output_token_ceiling={sum(max_tokens for _, max_tokens in requests)}",
        flush=True,
    )
    if args.dry_run:
        for model_id, max_tokens in requests:
            print(f"pending: {model_id} max_tokens={max_tokens}")
        return 0

    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(f"source E1 manifest missing: {SOURCE_MANIFEST}")
    source = json.loads(SOURCE_MANIFEST.read_text())
    question = source["questions"][0]
    protocol_version = source["protocol_version"]
    base_url = os.environ.get(
        "MATSIM_MAG_BASE_URL", "https://mag.genesis.americansciencecloud.org/v1"
    ).rstrip("/")
    if not base_url.endswith("/v1"):
        raise ValueError("MATSIM_MAG_BASE_URL must end in /v1")

    outputs_root.mkdir(parents=True, exist_ok=True)
    summary_path = FOLLOWUP_ROOT / "summary.jsonl"
    approval_reference = "chat-authorization-2026-11-02-allocated-followup"
    for index, (model_id, max_tokens) in enumerate(requests, start=1):
        output_dir = outputs_root / _slug(model_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        marker = output_dir / "request-started.json"
        marker.write_text(
            json.dumps(
                {"model_id": model_id, "started_at_utc": _timestamp()},
                sort_keys=True,
            )
            + "\n"
        )
        try:
            manifest = _manifest_for(
                model_id,
                question,
                protocol_version,
                base_url,
                len(requests),
                max_tokens,
                approval_reference,
            )
            result = run_e1_development(
                manifest,
                output_dir,
                max_new_requests=1,
            )
            state = result.get("state", "unknown")
            completed_count = result.get("completed_responses", 0)
            failed_count = result.get("failed_responses", 0)
            record = {
                "model_id": model_id,
                "max_output_tokens": max_tokens,
                "state": state,
                "completed_responses": completed_count,
                "failed_responses": failed_count,
                "finished_at_utc": _timestamp(),
            }
            print(
                f"[{index}/{len(requests)}] {model_id}: {state} "
                f"completed={completed_count} failed={failed_count}",
                flush=True,
            )
        except Exception as error:
            record = {
                "model_id": model_id,
                "max_output_tokens": max_tokens,
                "state": "runner_exception",
                "error_type": type(error).__name__,
                "finished_at_utc": _timestamp(),
            }
            print(
                f"[{index}/{len(requests)}] {model_id}: runner_exception "
                f"{type(error).__name__}",
                flush=True,
            )
        _record_result(summary_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
