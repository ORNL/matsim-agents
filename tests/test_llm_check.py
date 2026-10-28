from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage

from matsim_agents.workflows.llm_check import LLMCheckConfig, run_llm_check


class _FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, _messages):
        return AIMessage(content=next(self.responses))


class _RoutingFakeModel:
    def invoke(self, messages):
        prompt = str(messages[-1].content)
        if "PORTABILITY_OK" in prompt:
            return AIMessage(content="PORTABILITY_OK")
        if "matching this schema" in prompt:
            return AIMessage(
                content=json.dumps(
                    {
                        "objective": "Stable silicon",
                        "hypothesis": "Diamond silicon is stable.",
                        "proposed_compositions": ["Si"],
                        "scientific_rationale": "Compare phase energies.",
                    }
                )
            )
        return AIMessage(content="Nonempty scientific discussion response.")


def test_llm_check_persists_all_required_stages(tmp_path):
    hypothesis = {
        "objective": "Stable silicon",
        "hypothesis": "Diamond silicon is stable.",
        "proposed_compositions": ["Si"],
        "scientific_rationale": "Compare phase energies.",
    }
    responses = [
        "PORTABILITY_OK",
        json.dumps(hypothesis),
        "Diamond Si is the proposal.",
        "The proposal needs a competing-phase comparison.",
        "Revised after the critique: compare diamond against competing phases.",
    ]
    cfg = LLMCheckConfig(provider="vllm", output_root=str(tmp_path))
    result = run_llm_check(
        cfg,
        model_factory=lambda **_kwargs: _FakeModel(responses),
        readiness_probe=lambda _cfg: {"served_models": [cfg.model]},
    )
    assert result.status == "complete"
    assert all(result.stages.values())
    run = Path(result.run_directory)
    assert json.loads((run / "result.json").read_text())["model"] == cfg.model
    assert (run / "structured_hypothesis.json").is_file()
    records = [json.loads(line) for line in (run / "contributions.jsonl").read_text().splitlines()]
    assert [record["id"] for record in records] == [
        "generation",
        "structured",
        "proposal",
        "critique",
        "revision",
    ]
    assert records[3]["text"] == "The proposal needs a competing-phase comparison."
    assert (run / "prompts" / "revision.json").is_file()
    assert (run / "transcript.md").is_file()
    assert result.incomplete_replies == []
    assert "api_key" not in (run / "resolved_config.json").read_text()


def test_llm_check_persists_readiness_failure(tmp_path):
    cfg = LLMCheckConfig(provider="vllm", output_root=str(tmp_path))

    def fail(_cfg):
        raise RuntimeError("model is not served")

    result = run_llm_check(cfg, readiness_probe=fail)
    assert result.status == "failed"
    assert result.stages["readiness"] is False
    assert "not served" in result.failure_reason
    assert (Path(result.run_directory) / "result.json").is_file()


def test_llm_check_exercises_configured_concurrency(tmp_path):
    cfg = LLMCheckConfig(
        provider="vllm",
        output_root=str(tmp_path),
        concurrent_requests=3,
    )
    result = run_llm_check(
        cfg,
        model_factory=lambda **_kwargs: _RoutingFakeModel(),
        readiness_probe=lambda _cfg: {"served_models": [cfg.model]},
    )
    records = [
        json.loads(line)
        for line in (Path(result.run_directory) / "contributions.jsonl").read_text().splitlines()
    ]
    concurrent = [record for record in records if record["kind"] == "concurrent_generation"]
    assert result.status == "complete"
    assert sorted(record["id"] for record in concurrent) == [
        "concurrent-001",
        "concurrent-002",
        "concurrent-003",
    ]
    assert all(record["text"] == "PORTABILITY_OK" for record in concurrent)


def test_llm_check_long_context_output_is_bounded_and_recorded(tmp_path, monkeypatch):
    import matsim_agents.workflows.llm_check as llm_check

    monkeypatch.setattr(llm_check, "_long_context_prompt", lambda _path, _tokens: ("stress", 16384))
    invoke_model = llm_check.invoke_model

    def measured_invoke(model, messages):
        reply = invoke_model(model, messages)
        if str(messages[-1].content) == "stress":
            return reply.model_copy(update={"latency_s": 10.0})
        return reply

    monkeypatch.setattr(llm_check, "invoke_model", measured_invoke)
    calls = []

    def model_factory(**kwargs):
        calls.append(kwargs)
        if "max_tokens" in kwargs:
            return _StressModel()
        return _RoutingFakeModel()

    cfg = LLMCheckConfig(
        provider="vllm",
        model="local-model",
        output_root=str(tmp_path),
        context_length=32768,
        long_context_output_tokens=16384,
        long_context_prompt_tokens=16384,
        long_context_tokenizer_path="/local/model",
    )
    result = run_llm_check(
        cfg,
        model_factory=model_factory,
        readiness_probe=lambda _cfg: {"served_models": [cfg.model]},
    )

    assert result.status == "complete"
    assert result.long_context_output_passed is True
    assert calls[-1]["max_tokens"] == 16384
    run = Path(result.run_directory)
    stress = json.loads((run / "long-context-output.json").read_text())
    assert stress["requested_prompt_tokens"] == 16384
    assert stress["requested_max_output_tokens"] == 16384
    assert stress["completion_tokens"] == 15565
    assert stress["tokens_per_second"] > 0
    assert any(
        record["id"] == "long-context-output"
        for record in map(json.loads, (run / "contributions.jsonl").read_text().splitlines())
    )


def test_llm_check_long_context_limits_must_fit_model_context():
    import pytest

    with pytest.raises(ValueError, match="exceed context_length"):
        LLMCheckConfig(
            provider="vllm",
            model="local-model",
            context_length=32768,
            long_context_output_tokens=16384,
            long_context_prompt_tokens=16385,
            long_context_tokenizer_path="/local/model",
        )


def test_llm_check_long_context_requires_server_reported_prompt_tokens(tmp_path, monkeypatch):
    import matsim_agents.workflows.llm_check as llm_check

    monkeypatch.setattr(llm_check, "_long_context_prompt", lambda _path, _tokens: ("stress", 16384))

    def model_factory(**kwargs):
        return _StressModel(prompt_tokens=16000) if "max_tokens" in kwargs else _RoutingFakeModel()

    cfg = LLMCheckConfig(
        provider="vllm",
        model="local-model",
        output_root=str(tmp_path),
        context_length=32768,
        long_context_output_tokens=16384,
        long_context_prompt_tokens=16384,
        long_context_tokenizer_path="/local/model",
    )
    result = run_llm_check(
        cfg,
        model_factory=model_factory,
        readiness_probe=lambda _cfg: {"served_models": [cfg.model]},
    )

    assert result.status == "failed"
    assert "differ from the tokenizer target" in result.failure_reason
    stress = json.loads((Path(result.run_directory) / "long-context-output.json").read_text())
    assert stress["prompt_tokens"] == 16000


class _StressModel:
    def __init__(self, prompt_tokens=16384):
        self.prompt_tokens = prompt_tokens

    def invoke(self, _messages):
        return AIMessage(
            content="QUAL_OK " * 10,
            response_metadata={"finish_reason": "length"},
            usage_metadata={
                "input_tokens": self.prompt_tokens,
                "output_tokens": 15565,
                "total_tokens": self.prompt_tokens + 15565,
            },
        )
