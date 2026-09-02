#!/usr/bin/env python3
"""Qualify every first-class LLM through a multi-round scientific debate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from matsim_agents.backends.llm.provider import get_chat_model
from matsim_agents.workflows.debate import (
    DebateParticipant,
    ScientificDebateConfig,
    ScientificDebateResult,
    run_scientific_debate,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "deployments" / "common" / "open-model-catalog.json"

THERMOELECTRIC_HYPOTHESIS = (
    "What candidate material provides an optimal thermoelectric functional property—"
    "specifically a high dimensionless figure of merit ZT near 800 K—while remaining "
    "chemically stable and composed of reasonably abundant elements?"
)


def load_catalog(path: Path = CATALOG) -> list[dict[str, Any]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("the first-class model catalog must contain at least two models")
    return entries


def catalog_participants(
    entries: list[dict[str, Any]],
    *,
    environment: dict[str, str] | None = None,
) -> list[DebateParticipant]:
    """Resolve every catalog model to its facility endpoint or fail closed."""

    env = os.environ if environment is None else environment
    shared_url = env.get("MATSIM_VLLM_BASE_URL")
    participants = []
    missing = []
    for entry in entries:
        endpoint = env.get(str(entry["base_url_env"])) or shared_url
        if not endpoint:
            missing.append(str(entry["base_url_env"]))
            continue
        participants.append(
            DebateParticipant(
                name=str(entry["name"]),
                provider="vllm",
                model=str(entry["model"]),
                base_url=endpoint,
                role=(
                    "independent materials scientist assessing thermoelectric transport, "
                    "stability, abundance, and experimental falsifiability"
                ),
            )
        )
    if missing:
        raise ValueError("missing endpoint variables for first-class models: " + ", ".join(missing))
    return participants


def validate_enclave_result(
    result: ScientificDebateResult,
    participants: list[DebateParticipant],
    rounds: int,
) -> list[str]:
    errors = []
    expected = {participant.name for participant in participants}
    contribution_ids = [turn.turn_id for turn in result.turns] + [result.synthesis_turn_id]
    if len(contribution_ids) != len(set(contribution_ids)):
        errors.append("model contribution IDs are not unique")
    if any(not contribution_id.strip() for contribution_id in contribution_ids):
        errors.append("one or more model contributions have no ID")
    for round_index in range(1, rounds + 1):
        turns = [turn for turn in result.turns if turn.round == round_index]
        observed = {turn.participant for turn in turns}
        if observed != expected:
            errors.append(
                f"round {round_index} model set differs: missing={sorted(expected - observed)} "
                f"unexpected={sorted(observed - expected)}"
            )
        empty = [turn.participant for turn in turns if not turn.response.strip()]
        if empty:
            errors.append(f"round {round_index} has empty responses from {sorted(empty)}")
    if not result.synthesis.strip():
        errors.append("final synthesis is empty")
    return errors


def execute_llm_enclave(
    *,
    output: Path,
    rounds: int,
    catalog: Path = CATALOG,
    environment: dict[str, str] | None = None,
    model_factory=get_chat_model,
) -> dict[str, Any]:
    if rounds < 2:
        raise ValueError("LLM portability debate requires at least two rounds")
    entries = load_catalog(catalog)
    participants = catalog_participants(entries, environment=environment)
    output.mkdir(parents=True, exist_ok=True)
    config = ScientificDebateConfig(
        hypothesis=THERMOELECTRIC_HYPOTHESIS,
        participants=participants,
        rounds=rounds,
        output_root=str(output / "runs"),
    )
    try:
        debate = run_scientific_debate(config, model_factory=model_factory)
        errors = validate_enclave_result(debate, participants, rounds)
        payload = {
            "schema_version": 1,
            "benchmark": "first-class-llm-scientific-enclave",
            "status": "passed" if not errors else "failed",
            "hypothesis": THERMOELECTRIC_HYPOTHESIS,
            "required_rounds": rounds,
            "required_models": [participant.model for participant in participants],
            "required_participants": [participant.name for participant in participants],
            "turn_count": len(debate.turns),
            "debate_run_directory": debate.run_directory,
            "transcript_path": debate.transcript_path,
            "dialogue_path": debate.dialogue_path,
            "errors": errors,
        }
    except Exception as error:
        payload = {
            "schema_version": 1,
            "benchmark": "first-class-llm-scientific-enclave",
            "status": "failed",
            "hypothesis": THERMOELECTRIC_HYPOTHESIS,
            "required_rounds": rounds,
            "required_models": [participant.model for participant in participants],
            "required_participants": [participant.name for participant in participants],
            "errors": [f"{type(error).__name__}: {error}"],
        }
    (output / "llm_enclave_result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("--rounds must be at least 2")
    result = execute_llm_enclave(
        output=args.output,
        rounds=args.rounds,
        catalog=args.catalog,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
