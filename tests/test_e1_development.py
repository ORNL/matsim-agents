from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from matsim_agents.workflows.e1 import (
    E1DevelopmentManifest,
    run_e1_development,
)


def _manifest() -> E1DevelopmentManifest:
    return E1DevelopmentManifest.model_validate(
        {
            "protocol_version": "e1-dev-v1",
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
    assert json.loads((tmp_path / "run.json").read_text())["partition"] == "development"

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