from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from matsim_agents.workflows.e1 import (
    E1DevelopmentManifest,
    run_e1_development,
)


def _scientific_response() -> str:
    return json.dumps(
        {
            "conclusion": "The evidence is insufficient to distinguish the hypotheses.",
            "hypotheses": ["H1", "H2"],
            "reasoning_summary": "The prompt supplies no comparative measurements.",
            "assumptions": [],
            "supporting_evidence": [],
            "missing_or_conflicting_evidence": ["Comparative measurements"],
            "falsifiable_predictions": [],
            "proposed_tests": ["Measure both samples under matched conditions."],
            "outcome_interpretation": [],
            "confidence": "low",
        }
    )


def _manifest() -> E1DevelopmentManifest:
    return E1DevelopmentManifest.model_validate(
        {
            "protocol_version": "e1-dev-v1",
            "cohort": "E1-L",
            "candidate": {
                "catalog_name": "test-model",
                "model_id": "org/test-model",
                "revision": "abc123",
                "provider": "vllm",
            },
            "questions": [
                {
                    "question_id": "D-LFP-A",
                    "material": "LiFePO4",
                    "category": "A",
                    "prompt": "Compare two stated hypotheses.",
                },
                {
                    "question_id": "D-PHOS-B",
                    "material": "phosphorene",
                    "category": "B",
                    "prompt": "Audit the supplied hypothetical report.",
                },
            ],
        }
    )


def _remote_manifest(**approval_overrides) -> E1DevelopmentManifest:
    manifest_data = _manifest().model_dump(mode="json")
    manifest_data["cohort"] = "E1-R"
    manifest_data["candidate"].update(
        {"provider": "openai", "model_id": "test-remote-model"}
    )
    manifest_data["remote_approval"] = {
        "approval_reference": "test-only",
        "approved_by": "test",
        "provider": "openai",
        "mag_provider": None,
        "model_id": "test-remote-model",
        "revision": manifest_data["candidate"]["revision"],
        "revision_semantics": "immutable",
        "mutable_alias_accepted": False,
        "endpoint": "https://api.openai.com/v1",
        "project_authorization_reference": "project-authorization-test-fixture",
        "allocation_limit_reference": "allocation-approval-test-fixture",
        "allocation_charge_description": "Test fixture allocation usage, not a real charge.",
        "data_use_conditions": "Test fixture: synthetic prompts only.",
        "retention_conditions": "Test fixture: no real submission.",
        "max_input_message_bytes_per_question": 10000,
        "max_input_message_bytes_campaign": 20000,
        "max_total_requests": 2,
        "max_attempts_per_question": 1,
        "max_output_tokens_per_response": 32768,
        "max_output_tokens_campaign": 65536,
        **approval_overrides,
    }
    manifest_data["decoding"]["max_tokens"] = 32768
    return E1DevelopmentManifest.model_validate(manifest_data)


def test_development_runner_is_restart_safe_and_excludes_held_out(tmp_path):
    calls = []

    class FakeModel:
        def invoke(self, messages):
            calls.append(messages[-1].content)
            return AIMessage(content=f"response {len(calls)}")

    factory_calls = []

    def model_factory(**_kwargs):
        factory_calls.append(True)
        return FakeModel()

    manifest = _manifest()
    assert manifest.decoding.max_tokens == 32768
    first = run_e1_development(manifest, tmp_path, model_factory=model_factory)
    second = run_e1_development(manifest, tmp_path, model_factory=model_factory)

    records = [
        json.loads(line)
        for line in (tmp_path / "responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert first["state"] == second["state"] == "complete"
    assert first["completed_responses"] == 2
    assert second["completed_responses"] == 2
    assert len(records) == 2
    assert len(factory_calls) == 1
    assert calls == [question.prompt for question in manifest.questions]
    assert all(record["question_id"].startswith("D-") for record in records)
    run_metadata = json.loads((tmp_path / "run.json").read_text())
    assert run_metadata["partition"] == "development"
    assert run_metadata["protocol_artifact_hashes"]["development_questions"] == hashlib.sha256(
        Path("research/paper/e1/d-questions.v1.json").read_bytes()
    ).hexdigest()

    with pytest.raises(ValidationError):
        E1DevelopmentManifest.model_validate(
            {
                **manifest.model_dump(mode="json"),
                "questions": [
                    {
                        "question_id": "H-LFP-A",
                        "material": "LiFePO4",
                        "category": "A",
                        "prompt": "Held-out prompt",
                    }
                ],
            }
        )
    with pytest.raises(ValidationError, match="must be unique"):
        E1DevelopmentManifest.model_validate(
            {
                **manifest.model_dump(mode="json"),
                "questions": [
                    manifest.questions[0].model_dump(mode="json"),
                    manifest.questions[0].model_dump(mode="json"),
                ],
            }
        )


def test_development_runner_rejects_manifest_mismatch_and_recovers_partial_tail(tmp_path):
    class FakeModel:
        def invoke(self, _messages):
            return AIMessage(content="response")

    manifest = _manifest()
    run_e1_development(manifest, tmp_path, model_factory=lambda **_kwargs: FakeModel())
    ledger_path = tmp_path / "responses.jsonl"
    complete_records = ledger_path.read_text(encoding="utf-8")
    ledger_path.write_text(complete_records + '{"status":"complete"', encoding="utf-8")

    calls = []

    def unexpected_factory(**_kwargs):
        calls.append(True)
        return FakeModel()

    run_e1_development(manifest, tmp_path, model_factory=unexpected_factory)
    assert calls == []
    assert ledger_path.read_text(encoding="utf-8") == complete_records

    ledger_path.write_text(complete_records.rstrip("\n"), encoding="utf-8")
    run_e1_development(manifest, tmp_path, model_factory=unexpected_factory)
    assert calls == []
    assert ledger_path.read_text(encoding="utf-8") == complete_records

    changed_manifest = manifest.model_copy(update={"protocol_version": "e1-dev-v2"})
    with pytest.raises(ValueError, match="different manifest"):
        run_e1_development(changed_manifest, tmp_path, model_factory=unexpected_factory)
    assert calls == []


def test_mag_candidate_route_is_part_of_manifest_and_factory_args(tmp_path):
    manifest_data = _manifest().model_dump(mode="json")
    manifest_data["cohort"] = "E1-R"
    manifest_data["candidate"].update(
        {
            "provider": "mag",
            "mag_provider": "openai",
            "model_id": "gpt-6-luna@openai",
            "revision": "gpt-6-luna@openai",
            "revision_semantics": "mutable_alias",
        }
    )
    manifest_data["remote_approval"] = {
        "approval_reference": "test-only",
        "approved_by": "test",
        "provider": "mag",
        "mag_provider": "openai",
        "model_id": "gpt-6-luna@openai",
        "revision": manifest_data["candidate"]["revision"],
        "revision_semantics": "mutable_alias",
        "mutable_alias_accepted": True,
        "endpoint": "https://mag.genesis.americansciencecloud.org/v1",
        "project_authorization_reference": "project-authorization-test-fixture",
        "allocation_limit_reference": "allocation-approval-test-fixture",
        "allocation_charge_description": "Test fixture allocation usage, not a real charge.",
        "data_use_conditions": "Test fixture: synthetic prompts only.",
        "retention_conditions": "Test fixture: no real submission.",
        "max_input_message_bytes_per_question": 10000,
        "max_input_message_bytes_campaign": 20000,
        "max_total_requests": 2,
        "max_attempts_per_question": 1,
        "max_output_tokens_per_response": 32768,
        "max_output_tokens_campaign": 65536,
    }
    manifest_data["decoding"]["max_tokens"] = 32768
    manifest = E1DevelopmentManifest.model_validate(manifest_data)
    factory_calls = []

    class FakeModel:
        def invoke(self, _messages):
            return AIMessage(content="response")

    def model_factory(**kwargs):
        factory_calls.append(kwargs)
        return FakeModel()

    run_e1_development(manifest, tmp_path, model_factory=model_factory)

    assert factory_calls[0]["provider"] == "mag"
    assert factory_calls[0]["mag_provider"] == "openai"
    assert factory_calls[0]["max_tokens"] == 32768
    candidate = json.loads((tmp_path / "manifest.json").read_text())["candidate"]
    assert candidate["mag_provider"] == "openai"
    assert candidate["revision_semantics"] == "mutable_alias"


def test_remote_manifest_requires_allocation_contract_and_binds_approval():
    manifest_data = _remote_manifest().model_dump(mode="json")
    manifest_data["remote_approval"].pop("allocation_limit_reference")
    with pytest.raises(ValidationError, match="allocation_limit_reference"):
        E1DevelopmentManifest.model_validate(manifest_data)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _remote_manifest(spending_ceiling_usd=0.001)

    mag_data = _manifest().model_dump(mode="json")
    mag_data["cohort"] = "E1-R"
    mag_data["candidate"].update(
        {
            "provider": "mag",
            "mag_provider": "openai",
            "model_id": "account-model",
            "revision": "gpt-6-luna@openai",
            "revision_semantics": "mutable_alias",
        }
    )
    mag_data["remote_approval"] = {
        **_remote_manifest().remote_approval.model_dump(mode="json"),
        "provider": "mag",
        "mag_provider": "openai",
        "model_id": "account-model",
        "revision": "gpt-6-luna@openai",
        "revision_semantics": "mutable_alias",
        "mutable_alias_accepted": False,
        "endpoint": "https://mag.genesis.americansciencecloud.org/v1",
    }
    with pytest.raises(ValidationError, match="mutable_alias_accepted"):
        E1DevelopmentManifest.model_validate(mag_data)

    manifest_data = _remote_manifest().model_dump(mode="json")
    manifest_data["remote_approval"]["model_id"] = "different-model"
    with pytest.raises(ValidationError, match="approval must match"):
        E1DevelopmentManifest.model_validate(manifest_data)


def test_remote_retry_budget_cannot_exceed_total_request_limit(tmp_path):
    class AlwaysFails:
        def invoke(self, _messages):
            raise RuntimeError("transport failure")

    manifest = _remote_manifest(
        max_attempts_per_question=2,
        max_total_requests=2,
        max_input_message_bytes_campaign=40000,
        max_output_tokens_campaign=65536,
    )
    run_e1_development(
        manifest,
        tmp_path,
        model_factory=lambda **_kwargs: AlwaysFails(),
    )

    with pytest.raises(ValueError, match="approved total request limit"):
        run_e1_development(
            manifest,
            tmp_path,
            model_factory=lambda **_kwargs: AlwaysFails(),
            retry_errors=True,
        )


def test_remote_input_preflight_blocks_before_model_or_output_creation(tmp_path):
    manifest = _remote_manifest(
        allocation_limit_reference="allocation-approval-pending-test-fixture",
        max_input_message_bytes_per_question=100,
        max_input_message_bytes_campaign=200,
    )
    factory_calls = []

    def model_factory(**_kwargs):
        factory_calls.append(True)
        raise AssertionError("preflight must fail before model construction")

    with pytest.raises(ValueError, match="input preflight exceeds utf8_bytes limit"):
        run_e1_development(manifest, tmp_path / "not-created", model_factory=model_factory)

    assert factory_calls == []
    assert not (tmp_path / "not-created").exists()


def test_remote_runner_preserves_responses_and_writes_allocation_ledger(tmp_path):
    captured_messages = []

    class FakeModel:
        def invoke(self, messages):
            captured_messages.append(messages)
            return AIMessage(
                content=_scientific_response(),
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    "output_token_details": {"reasoning": 7},
                },
            )

    manifest = _remote_manifest()
    result = run_e1_development(
        manifest,
        tmp_path,
        model_factory=lambda **_kwargs: FakeModel(),
    )

    response_records = [
        json.loads(line)
        for line in (tmp_path / "responses.jsonl").read_text().splitlines()
    ]
    allocation_records = [
        json.loads(line)
        for line in (tmp_path / "allocation-usage.jsonl").read_text().splitlines()
    ]
    assert result["state"] == "complete"
    assert all(record["raw_response"]["content"] == _scientific_response() for record in response_records)
    assert all(record["parsed_response"]["confidence"] == "low" for record in response_records)
    assert all(record["schema_error"] is None for record in response_records)
    assert len(allocation_records) == len(manifest.questions)
    assert allocation_records[0]["billing_basis"] == "project_allocation"
    assert allocation_records[0]["prompt_tokens_reported"] == 10
    assert allocation_records[0]["completion_tokens_reported"] == 20
    assert allocation_records[0]["reasoning_tokens_reported"] == 7
    assert allocation_records[0]["latency_s"] is not None
    assert all(record["input_preflight"]["unit"] == "utf8_bytes" for record in response_records)
    assert all(record["cohort"] == "E1-R" for record in allocation_records)
    assert all(record["endpoint"] == "https://api.openai.com/v1" for record in response_records)
    assert all(record["requested_model_id"] == "test-remote-model" for record in response_records)
    assert '"required"' in captured_messages[0][0].content
    assert "E1ScientificResponse" in captured_messages[0][0].content


def test_remote_truncated_reply_is_recorded_as_failed_attempt(tmp_path):
    class TruncatedModel:
        def invoke(self, _messages):
            return AIMessage(
                content="",
                response_metadata={"finish_reason": "length"},
                usage_metadata={
                    "input_tokens": 536,
                    "output_tokens": 2048,
                    "total_tokens": 2584,
                },
            )

    manifest = _remote_manifest()
    result = run_e1_development(
        manifest,
        tmp_path,
        model_factory=lambda **_kwargs: TruncatedModel(),
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "responses.jsonl").read_text().splitlines()
    ]

    assert result["state"] == "complete_with_errors"
    assert all(record["status"] == "error" for record in records)
    assert all(record["parsed_response"] is None for record in records)
    assert all(record["response"]["truncated"] for record in records)


def test_remote_runner_records_errors_and_retries_only_when_requested(tmp_path):
    class FlakyModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary transport error")
            return AIMessage(content=_scientific_response())

    model = FlakyModel()
    manifest = _remote_manifest(
        max_attempts_per_question=2,
        max_total_requests=4,
        max_input_message_bytes_campaign=40000,
        max_output_tokens_campaign=131072,
    )
    first = run_e1_development(
        manifest,
        tmp_path,
        model_factory=lambda **_kwargs: model,
    )
    records = [
        json.loads(line) for line in (tmp_path / "responses.jsonl").read_text().splitlines()
    ]
    assert first["state"] == "complete_with_errors"
    assert records[0]["status"] == "error"
    assert records[0]["attempt"] == 1
    assert "temporary transport error" in records[0]["error_message"]
    allocation_records = [
        json.loads(line)
        for line in (tmp_path / "allocation-usage.jsonl").read_text().splitlines()
    ]
    assert allocation_records[0]["status"] == "error"
    assert allocation_records[0]["billing_basis"] == "project_allocation"

    calls_before_restart = model.calls
    run_e1_development(
        manifest,
        tmp_path,
        model_factory=lambda **_kwargs: model,
    )
    assert model.calls == calls_before_restart

    retried = run_e1_development(
        manifest,
        tmp_path,
        model_factory=lambda **_kwargs: model,
        retry_errors=True,
    )
    all_records = [
        json.loads(line) for line in (tmp_path / "responses.jsonl").read_text().splitlines()
    ]
    assert retried["state"] == "complete"
    assert all_records[-1]["status"] == "complete"
    assert all_records[-1]["attempt"] == 2