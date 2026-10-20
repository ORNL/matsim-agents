from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from matsim_agents.workflows.debate import (
    DebateParticipant,
    ScientificDebateConfig,
    run_scientific_debate,
)


class _Model:
    def __init__(self, name: str):
        self.name = name
        self.prompts = []
        self.system_prompts = []

    def invoke(self, messages):
        self.system_prompts.append(messages[0].content)
        self.prompts.append(messages[-1].content)
        return SimpleNamespace(content=f"argument from {self.name} call {len(self.prompts)}")


def test_single_call_mode_invokes_one_model_exactly_once(tmp_path):
    model = _Model("baseline")
    result = run_scientific_debate(
        ScientificDebateConfig(
            hypothesis="Propose candidate Nb-Ta-O formulas.",
            participants=[DebateParticipant(name="baseline", provider="vllm", model="baseline")],
            rounds=1,
            single_call=True,
            output_root=str(tmp_path),
        ),
        model_factory=lambda **kwargs: model,
    )

    assert len(model.prompts) == 1
    assert len(result.turns) == 1
    assert len(result.verdicts) == 1
    assert result.verdicts[0].response == result.turns[0].response


def test_single_call_mode_includes_final_response_instruction(tmp_path):
    model = _Model("baseline")
    run_scientific_debate(
        ScientificDebateConfig(
            hypothesis="Propose candidate Nb-Ta-O formulas.",
            participants=[DebateParticipant(name="baseline", provider="vllm", model="baseline")],
            rounds=1,
            single_call=True,
            final_response_instruction='Return JSON with a "formulas" array.',
            output_root=str(tmp_path),
        ),
        model_factory=lambda **kwargs: model,
    )

    assert len(model.prompts) == 1
    assert 'Return JSON with a "formulas" array.' in model.system_prompts[0]


def test_single_call_mode_rejects_multiple_models_or_rounds():
    participant = DebateParticipant(name="baseline", provider="vllm", model="baseline")
    with pytest.raises(ValueError, match="exactly one participant and one round"):
        ScientificDebateConfig(
            hypothesis="h",
            participants=[participant],
            rounds=2,
            single_call=True,
        )


def test_models_debate_each_other_for_user_selected_rounds(tmp_path):
    models = {}

    def factory(*, provider, model, base_url, max_tokens):
        models[model] = _Model(model)
        return models[model]

    cfg = ScientificDebateConfig(
        hypothesis="A metastable Si phase is stabilized by pressure.",
        rounds=3,
        output_root=str(tmp_path),
        participants=[
            DebateParticipant(name="theorist", provider="vllm", model="model-a"),
            DebateParticipant(name="skeptic", provider="ollama", model="model-b"),
            DebateParticipant(name="experimentalist", provider="openai", model="model-c"),
        ],
    )
    result = run_scientific_debate(cfg, model_factory=factory)
    assert result.rounds_completed == 3
    assert len(result.turns) == 9
    assert len(result.verdicts) == 3
    assert [turn.participant for turn in result.turns[:6]] == [
        "theorist",
        "skeptic",
        "experimentalist",
        "skeptic",
        "experimentalist",
        "theorist",
    ]
    assert "argument from model-a" in models["model-b"].prompts[-1]
    run = tmp_path / result.run_id
    assert result.run_id.startswith("debate-")
    # One canonical record per contribution; no duplicate transcripts or empty folders.
    assert {path.name for path in run.iterdir()} == {
        "contributions.jsonl",
        "events.jsonl",
        "prompts",
        "provenance.json",
        "request.json",
        "resolved_config.json",
        "result.json",
        "transcript.md",
    }
    records = [json.loads(line) for line in (run / "contributions.jsonl").read_text().splitlines()]
    identifiers = [record["id"] for record in records]
    assert identifiers == [turn.turn_id for turn in result.turns] + [
        verdict.contribution_id for verdict in result.verdicts
    ]
    assert len(identifiers) == len(set(identifiers)) == 12
    assert [record["kind"] for record in records] == ["argument"] * 9 + ["verdict"] * 3
    assert sorted(path.stem for path in (run / "prompts").iterdir()) == sorted(identifiers)
    assert len({models[name].system_prompts[0] for name in models}) == 1
    summary = json.loads((run / "result.json").read_text())
    assert "turns" not in summary and "synthesis" not in summary
    assert summary["turn_ids"] == identifiers[:9]
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert [
        event["payload"]["id"] for event in events if event["event"] == "contribution_completed"
    ] == identifiers
    markdown = (run / "transcript.md").read_text()
    assert markdown.startswith("# Scientific debate")
    assert markdown.count("\n## ") == 12


def test_designated_synthesis_remains_available_for_role_based_debate(tmp_path):
    models = {}

    def factory(*, provider, model, base_url, max_tokens):
        models[model] = _Model(model)
        return models[model]

    result = run_scientific_debate(
        ScientificDebateConfig(
            hypothesis="Candidate X has high ZT.",
            output_root=str(tmp_path),
            debate_mode="role_based",
            synthesis_method="designated_model",
            synthesis_participant="theorist",
            participants=[
                DebateParticipant(name="theorist", provider="vllm", model="a", role="theorist"),
                DebateParticipant(
                    name="experimentalist",
                    provider="vllm",
                    model="b",
                    role="experimentalist",
                ),
            ],
        ),
        model_factory=factory,
    )
    assert len(result.verdicts) == 1
    assert result.verdicts[0].participant == "theorist"
    assert result.synthesis_turn_id == result.verdicts[0].contribution_id


def test_debate_rejects_duplicate_names_and_unknown_synthesizer():
    participant = DebateParticipant(name="same", provider="vllm", model="a")
    with pytest.raises(ValueError, match="unique"):
        ScientificDebateConfig(hypothesis="h", participants=[participant, participant])
    with pytest.raises(ValueError, match="synthesis_participant"):
        ScientificDebateConfig(
            hypothesis="h",
            participants=[
                participant,
                DebateParticipant(name="other", provider="vllm", model="b"),
            ],
            synthesis_participant="missing",
        )


@pytest.mark.parametrize(
    "name,mode,method,verdict_count",
    [
        ("equal-independent.yaml", "equal", "independent_verdicts", 2),
        ("equal-designated.yaml", "equal", "designated_model", 1),
        ("role-based-independent.yaml", "role_based", "independent_verdicts", 2),
        ("role-based-designated.yaml", "role_based", "designated_model", 1),
    ],
)
def test_documented_debate_modalities_match_the_config_contract(
    name, mode, method, verdict_count, tmp_path
):
    raw = yaml.safe_load((Path("examples/debate") / name).read_text())
    raw["output_root"] = str(tmp_path)
    config = ScientificDebateConfig.model_validate(raw)
    result = run_scientific_debate(
        config,
        model_factory=lambda *, provider, model, base_url, max_tokens: _Model(model),
    )
    assert config.debate_mode == mode
    assert config.synthesis_method == method
    assert len(result.verdicts) == verdict_count


class _ReasoningModel:
    """Replies like a reasoning model served without a reasoning parser."""

    def __init__(self, name: str, *, cut_off: bool = False):
        self.name = name
        self.cut_off = cut_off
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        if self.cut_off:
            return SimpleNamespace(
                content=f"<think>{self.name} still thinking",
                response_metadata={"finish_reason": "length"},
                usage_metadata={"input_tokens": 10, "output_tokens": 64},
            )
        return SimpleNamespace(
            content=f"{self.name} private reasoning</think>{self.name} answer",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 10, "output_tokens": 20},
        )


def test_peers_see_answers_only_and_incomplete_replies_are_flagged(tmp_path):
    models = {"a": _ReasoningModel("a"), "b": _ReasoningModel("b", cut_off=True)}
    result = run_scientific_debate(
        ScientificDebateConfig(
            hypothesis="h",
            rounds=1,
            output_root=str(tmp_path),
            participants=[
                DebateParticipant(name="a", provider="vllm", model="a"),
                DebateParticipant(name="b", provider="vllm", model="b"),
            ],
        ),
        model_factory=lambda **kwargs: models[kwargs["model"]],
    )
    assert result.turns[0].response == "a answer"
    assert "private reasoning" not in models["b"].prompts[0]
    assert '<contribution id="round-001-turn-0001" speaker="a">' in models["b"].prompts[0]
    assert result.turns[1].response == "" and result.turns[1].incomplete
    assert "[No answer: the token limit ended" in models["a"].prompts[-1]
    assert result.incomplete_contributions == [
        result.turns[1].turn_id,
        result.verdicts[1].contribution_id,
    ]
    records = [
        json.loads(line) for line in Path(result.contributions_path).read_text().splitlines()
    ]
    assert records[0]["reasoning"] == "a private reasoning"
    assert records[0]["completion_tokens"] == 20 and records[0]["finish_reason"] == "stop"
    assert records[1]["truncated"] and records[1]["reasoning"] == "b still thinking"
    markdown = Path(result.transcript_path).read_text()
    assert "<summary>Reasoning (19 characters)</summary>" in markdown
    assert "> **Incomplete:** the token limit ended this reply" in markdown


def test_reply_token_limit_defaults_and_participant_override(tmp_path):
    limits = {}

    def factory(*, provider, model, base_url, max_tokens):
        limits[model] = max_tokens
        return _Model(model)

    run_scientific_debate(
        ScientificDebateConfig(
            hypothesis="h",
            rounds=1,
            output_root=str(tmp_path),
            participants=[
                DebateParticipant(name="a", provider="vllm", model="a"),
                DebateParticipant(
                    name="b", provider="vllm", model="b", max_completion_tokens=16384
                ),
            ],
        ),
        model_factory=factory,
    )
    assert limits == {"a": 8192, "b": 16384}


def test_long_debates_drop_whole_oldest_contributions_and_name_them(tmp_path):
    class Wordy(_Model):
        def invoke(self, messages):
            super().invoke(messages)
            return SimpleNamespace(content=f"{self.name}-{len(self.prompts)} " + "x" * 600)

    models = {"a": Wordy("a"), "b": Wordy("b")}
    result = run_scientific_debate(
        ScientificDebateConfig(
            hypothesis="h",
            rounds=3,
            output_root=str(tmp_path),
            max_transcript_chars=1_500,
            participants=[
                DebateParticipant(name="a", provider="vllm", model="a"),
                DebateParticipant(name="b", provider="vllm", model="b"),
            ],
        ),
        model_factory=lambda **kwargs: models[kwargs["model"]],
    )
    last_prompt = models["b"].prompts[-2]  # turn 6; the last prompt is b's verdict
    # Two whole contributions fit; the others are named, not cut mid-text.
    assert last_prompt.count("<contribution ") == 2
    assert last_prompt.count("x" * 600) == 2
    assert "[3 earlier contributions are not shown" in last_prompt
    records = [
        json.loads(line) for line in Path(result.contributions_path).read_text().splitlines()
    ]
    assert records[5]["omitted_context"] == [turn.turn_id for turn in result.turns[:3]]
    assert records[5]["context"] == [turn.turn_id for turn in result.turns[3:5]]
