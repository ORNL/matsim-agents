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
    assert (run / "discussion.json").is_file()
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
    generation = json.loads(
        (Path(result.run_directory) / "generation.json").read_text(encoding="utf-8")
    )
    assert result.status == "complete"
    assert generation["concurrent_responses"] == ["PORTABILITY_OK"] * 3
