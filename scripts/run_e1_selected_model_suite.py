"""Run the 18-question E1 suite on a bounded, explicitly selected MAG model set."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from matsim_agents.workflows.e1 import (
    CURRENT_PROMPT_VARIANT,
    E1DevelopmentManifest,
    _e1_system_prompt,
    _preflight_remote_inputs,
    run_e1_development,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs" / "e1"
SOURCE_MANIFEST = RUNS_ROOT / "e1-r-mag-gpt-6-luna-2026-10-29" / "manifest.json"
CATALOG_PATH = RUNS_ROOT / "remote-2026-11-02" / "catalog-sweep" / "catalog.json"
OUTPUT_ROOT = RUNS_ROOT / "remote-2026-11-02" / "selected-confirmed-claude-prompt-schema-18q"
MAX_INPUT_BYTES = 16384

MODELS = (
    {"model_id": "claude-haiku-5-5-low@google", "max_tokens": 8192},
    {"model_id": "claude-haiku-5-5-low-compact@google", "max_tokens": 8192},
)
CLAUDE_MODELS = {model["model_id"] for model in MODELS if model["model_id"].startswith("claude-")}


def _route(model_id: str) -> str:
    return model_id.rsplit("@", 1)[1]


def _slug(model_id: str) -> str:
    return model_id.replace("@", "-")


def _manifest_for(
    model: dict[str, Any], questions: list[dict[str, Any]], protocol_version: str, endpoint: str
) -> E1DevelopmentManifest:
    model_id = model["model_id"]
    route = _route(model_id)
    max_tokens = model["max_tokens"]
    request_limit = len(questions)
    return E1DevelopmentManifest.model_validate(
        {
            "schema_version": 1,
            "partition": "development",
            "cohort": "E1-R",
            "candidate": {
                "catalog_name": model_id,
                "model_id": model_id,
                "revision": model_id,
                "max_output_tokens_supported": None,
                "revision_semantics": "mutable_alias",
                "provider": "mag",
                "mag_provider": route,
            },
            "decoding": {"temperature": 0.0, "max_tokens": max_tokens},
            "remote_approval": {
                "approval_reference": "chat-authorization-2026-11-02-selected-e1-suite",
                "approved_by": os.environ.get("USER", "workspace user"),
                "provider": "mag",
                "mag_provider": route,
                "model_id": model_id,
                "revision": model_id,
                "revision_semantics": "mutable_alias",
                "mutable_alias_accepted": True,
                "endpoint": endpoint,
                "project_authorization_reference": (
                    "User authorization in chat on 2026-11-02 for the selected six-model E1 suite"
                ),
                "allocation_limit_reference": (
                    f"At most {request_limit} requests, one per E1 question, "
                    f"with a {max_tokens}-token per-response ceiling; Claude canary is part of the suite."
                ),
                "allocation_charge_description": (
                    f"At most {request_limit} MAG requests for this model with up to "
                    f"{max_tokens} generated tokens each, plus provider-defined input processing."
                ),
                "data_use_conditions": (
                    "Only the 18 supplied E1 development prompts are sent to MAG and the "
                    "catalog-listed upstream route; no proprietary or personal data is included."
                ),
                "retention_conditions": (
                    "Subject to the applicable MAG and upstream account retention terms; "
                    "no additional retention guarantee is assumed."
                ),
                "max_input_message_bytes_per_question": MAX_INPUT_BYTES,
                "max_input_message_bytes_campaign": MAX_INPUT_BYTES * request_limit,
                "max_total_requests": request_limit,
                "max_attempts_per_question": 1,
                "max_output_tokens_per_response": max_tokens,
                "max_output_tokens_campaign": request_limit * max_tokens,
            },
            "protocol_version": protocol_version,
            "prompt_variant": CURRENT_PROMPT_VARIANT,
            "questions": questions,
        }
    )


def _canary_is_valid(output_dir: Path, question_id: str) -> bool:
    ledger_path = output_dir / "responses.jsonl"
    if not ledger_path.is_file():
        return False
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [row for row in rows if row.get("question_id") == question_id]
    if not matches:
        return False
    latest = max(matches, key=lambda row: row.get("attempt", 0))
    parsed = latest.get("parsed_response")
    return (
        latest.get("status") == "complete"
        and isinstance(parsed, dict)
        and len(parsed) == 10
        and not latest.get("schema_error")
        and latest.get("semantic_validity", "valid") == "valid"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="send the approved MAG requests")
    args = parser.parse_args()

    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(f"18-question source manifest missing: {SOURCE_MANIFEST}")
    if not CATALOG_PATH.is_file():
        raise FileNotFoundError(f"cached MAG catalog missing: {CATALOG_PATH}")
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    questions = source["questions"]
    if len(questions) != 18:
        raise ValueError(f"expected 18 E1 questions, found {len(questions)}")
    catalog_ids = set(json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["model_ids"])
    missing = [model["model_id"] for model in MODELS if model["model_id"] not in catalog_ids]
    if missing:
        raise ValueError(f"selected model IDs are absent from the cached MAG catalog: {missing}")

    base_url = os.environ.get(
        "MATSIM_MAG_BASE_URL", "https://mag.genesis.americansciencecloud.org/v1"
    ).rstrip("/")
    if not base_url.endswith("/v1"):
        raise ValueError("MATSIM_MAG_BASE_URL must end in /v1")
    total_requests = len(MODELS) * len(questions)
    total_output_tokens = sum(model["max_tokens"] * len(questions) for model in MODELS)
    manifests = {
        model["model_id"]: _manifest_for(model, questions, source["protocol_version"], base_url)
        for model in MODELS
    }
    input_sizes = {
        model_id: sum(
            item["count"]
            for item in _preflight_remote_inputs(
                manifest, _e1_system_prompt(manifest.prompt_variant)
            ).values()
        )
        for model_id, manifest in manifests.items()
    }
    print(
        f"models={len(MODELS)} questions={len(questions)} max_requests={total_requests} "
        f"max_output_tokens={total_output_tokens} max_input_bytes={max(input_sizes.values())} "
        f"endpoint={base_url}",
        flush=True,
    )
    for model in MODELS:
        print(f"selected: {model['model_id']} max_tokens={model['max_tokens']}", flush=True)
    if not args.execute:
        print("dry run only; pass --execute to send requests", flush=True)
        return 0
    if not os.environ.get("AMSC_TOKEN"):
        raise RuntimeError("AMSC_TOKEN is not set in the current environment")

    for index, model in enumerate(MODELS, start=1):
        model_id = model["model_id"]
        output_dir = OUTPUT_ROOT / _slug(model_id)
        manifest = manifests[model_id]
        try:
            if model_id in CLAUDE_MODELS and not _canary_is_valid(output_dir, questions[0]["question_id"]):
                canary_result = run_e1_development(manifest, output_dir, max_new_requests=1)
                if not _canary_is_valid(output_dir, questions[0]["question_id"]):
                    print(
                        f"[{index}/{len(MODELS)}] {model_id}: canary failed; remaining questions not sent "
                        f"(state={canary_result.get('state')})",
                        flush=True,
                    )
                    continue
            result = run_e1_development(
                manifest,
                output_dir,
                max_new_requests=len(questions),
            )
            print(
                f"[{index}/{len(MODELS)}] {model_id}: {result.get('state')} "
                f"completed={result.get('completed_responses', 0)} "
                f"failed={result.get('failed_responses', 0)}",
                flush=True,
            )
        except Exception as error:
            print(
                f"[{index}/{len(MODELS)}] {model_id}: runner_exception {type(error).__name__}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())