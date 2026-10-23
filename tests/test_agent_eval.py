from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from matsim_agents.campaign.agent_eval import (
    AgentAnswer,
    AgentEvalConfig,
    aggregate_answers,
    case_prompt,
    grade_answer,
    load_suite,
    parse_answer,
    run_agent_eval,
)
from matsim_agents.campaign.benchmark import BenchmarkArm
from matsim_agents.execution.ledger import EvidenceLedger
from matsim_agents.workflows.debate import DebateParticipant

SUITE = Path(__file__).resolve().parents[1] / "benchmarks" / "agent_eval"


class _Model:
    """Chat model stub whose reply is computed from the prompt."""

    def __init__(self, reply, usage=(100, 20)):
        self.reply = reply
        self.usage = usage

    def invoke(self, messages):
        return AIMessage(
            content=self.reply(messages[-1].content),
            usage_metadata={
                "input_tokens": self.usage[0],
                "output_tokens": self.usage[1],
                "total_tokens": sum(self.usage),
            },
        )


def _oracle(prompt: str) -> str:
    """Answer each bundled case as its grader expects."""
    case = next(c for c in load_suite(SUITE).cases if c.claim in prompt)
    ids = re.findall(r"ev-[0-9a-f]{24}", prompt)
    decisive = case.expected_status != "unresolved"
    return json.dumps(
        {
            "status": case.expected_status,
            "evidence_ids": sorted(set(ids)) if decisive else [],
            "request_dft": case.dft_request_justified,
            "rationale": "rules",
        }
    )


def _participants(n=2):
    return [DebateParticipant(name=f"m{i}", provider="fake", model=f"model-{i}") for i in range(n)]


def _case(case_id):
    suite = load_suite(SUITE)
    case = next(c for c in suite.cases if c.case_id == case_id)
    return suite, case


def _ledger(tmp_path, case_id):
    suite, case = _case(case_id)
    ledger = EvidenceLedger(tmp_path / f"{case_id}.jsonl")
    records = {
        label: ledger.append(spec.record(suite.suite_id, case.case_id, label))
        for label, spec in case.evidence.items()
    }
    return case, records, ledger


def test_bundled_suite_covers_every_category_and_enters_a_ledger(tmp_path):
    suite = load_suite(SUITE)
    assert {c.expected_status for c in suite.cases} == {"supported", "contradicted", "unresolved"}
    assert {
        "missing_evidence",
        "ambiguous",
        "contradictory_evidence",
        "inadmissible_evidence",
    } <= {c.category for c in suite.cases}
    for case in suite.cases:
        _ledger(tmp_path, case.case_id)


def test_an_mlip_only_stability_claim_is_wrong_however_argued(tmp_path):
    case, records, ledger = _ledger(tmp_path, "sic-2h-mlip-only")
    surrogate = records["surrogate-hull"].evidence_id
    grade = grade_answer(
        case, records, ledger, AgentAnswer(status="supported", evidence_ids=[surrogate])
    )
    assert not grade.correct
    assert any("converged_dft" in p for p in grade.problems)
    grade = grade_answer(case, records, ledger, AgentAnswer(status="unresolved", request_dft=True))
    assert grade.correct and grade.correctly_inconclusive and grade.dft_request_correct


def test_failed_evidence_and_invented_ids_are_not_attribution(tmp_path):
    case, records, ledger = _ledger(tmp_path, "sic-4h-failed-dft")
    crashed = records["crashed"].evidence_id
    grade = grade_answer(
        case, records, ledger, AgentAnswer(status="contradicted", evidence_ids=[crashed])
    )
    assert not grade.correct and grade.correctly_inconclusive is False
    assert any("not evidence for or against" in p for p in grade.problems)
    case, records, ledger = _ledger(tmp_path, "sic-3c-on-hull")
    grade = grade_answer(
        case, records, ledger, AgentAnswer(status="supported", evidence_ids=["ev-" + "0" * 24])
    )
    assert not grade.correct and not grade.evidence_attribution
    assert grade.correctly_inconclusive is None


def test_a_decisive_answer_must_cite_the_deciding_evidence(tmp_path):
    case, records, ledger = _ledger(tmp_path, "sic-rocksalt-on-hull")
    hull = records["hull"].evidence_id
    assert grade_answer(
        case, records, ledger, AgentAnswer(status="contradicted", evidence_ids=[hull])
    ).correct
    grade = grade_answer(case, records, ledger, AgentAnswer(status="contradicted"))
    assert not grade.correct and not grade.evidence_attribution
    assert not grade_answer(case, records, ledger, None).correct


def test_conflicting_admissible_evidence_is_unresolved(tmp_path):
    case, records, ledger = _ledger(tmp_path, "sic-zincblende-conflicting-dft")
    soft = records["hull-soft"].evidence_id
    # The ledger admits the claim; the grader still requires unresolved.
    grade = grade_answer(
        case, records, ledger, AgentAnswer(status="supported", evidence_ids=[soft])
    )
    assert grade.problems == ["answered supported; expected unresolved"]


def test_panel_majority_and_ties():
    yes = AgentAnswer(status="supported", evidence_ids=["ev-a"], request_dft=False)
    no = AgentAnswer(status="contradicted", evidence_ids=["ev-b"], request_dft=True)
    assert aggregate_answers([yes, yes, no]).status == "supported"
    assert aggregate_answers([yes, yes, no]).evidence_ids == ["ev-a"]
    tie = aggregate_answers([yes, no])
    assert tie.status == "unresolved" and tie.evidence_ids == []
    assert aggregate_answers([]) is None


def test_parse_answer_takes_the_last_valid_object():
    text = 'thinking {"status": "maybe"} then {"status": "unresolved", "request_dft": true}'
    assert parse_answer(text) == AgentAnswer(status="unresolved", request_dft=True)
    with pytest.raises(ValueError):
        parse_answer("no json")


def test_arms_share_one_completion_budget():
    config = AgentEvalConfig(participants=_participants(2), completion_tokens_per_case=6000)
    assert [config.calls(arm) for arm in config.arms] == [0, 1, 2, 6]
    with pytest.raises(ValueError, match="at least two participants"):
        AgentEvalConfig(participants=_participants(1))
    with pytest.raises(ValueError, match="under 256 tokens"):
        AgentEvalConfig(participants=_participants(2), completion_tokens_per_case=1000)
    with pytest.raises(ValueError, match="must be among"):
        AgentEvalConfig(participants=_participants(2), arms=[BenchmarkArm.RANDOM])


def test_example_config_is_valid():
    import yaml

    with open(SUITE / "example-config.yaml", encoding="utf-8") as stream:
        config = AgentEvalConfig.model_validate(yaml.safe_load(stream))
    assert len(config.arms) == 4


def test_case_prompt_lists_ids_and_rules(tmp_path):
    case, records, _ = _ledger(tmp_path, "sic-6h-incomplete-references")
    prompt = case_prompt(case, records)
    assert records["partial-hull"].evidence_id in prompt
    assert "complete reference phases" in prompt and case.claim in prompt


def test_run_compares_arms_at_equal_budget_and_keeps_an_audit_trail(tmp_path):
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return _Model(_oracle)

    config = AgentEvalConfig(
        participants=_participants(2),
        completion_tokens_per_case=6000,
        decoding={"temperature": 0.3},
    )
    result = run_agent_eval(SUITE, config, output_dir=tmp_path, model_factory=factory)
    summary = {s.arm: s for s in result.summaries}
    n_cases = len(load_suite(SUITE).cases)
    n_unresolved = sum(c.expected_status == "unresolved" for c in load_suite(SUITE).cases)
    assert summary[BenchmarkArm.NO_LLM].metrics["correct"] == pytest.approx(n_unresolved / n_cases)
    assert summary[BenchmarkArm.NO_LLM].llm_cost["calls"] == 0
    for arm in config.arms[1:]:
        assert summary[arm].metrics["correct"] == 1.0, arm
        assert summary[arm].metrics["evidence_attribution"] == 1.0
        assert summary[arm].llm_cost["calls"] == config.calls(arm) * n_cases
        assert summary[arm].llm_cost["completion_tokens"] == 20 * config.calls(arm) * n_cases
        assert summary[arm].numerical_cost["dft_calculations"] == 0
    # Every model call got the arm's share of the budget and the decoding settings.
    assert {c["max_tokens"] for c in calls} == {6000, 3000, 1000}
    assert all(c["temperature"] == 0.3 for c in calls)
    comparison = next(
        c
        for c in result.comparisons
        if c.metric == "correct" and c.treatment == BenchmarkArm.SINGLE_LLM
    )
    assert comparison.n_pairs == n_cases and comparison.mean_difference > 0
    run_dir = Path(result.run_dir)
    answers = [json.loads(line) for line in (run_dir / "answers.jsonl").read_text().splitlines()]
    assert len(answers) == n_cases * len(config.arms)
    single = next(a for a in answers if a["arm"] == "single_llm")
    assert single["models"] == ["fake:model-0"]
    debate_dir = Path(single["run_directories"][0])
    assert list((debate_dir / "prompts").glob("*.json"))
    assert (run_dir / "ledgers" / f"{single['case_id']}.jsonl").is_file()
    saved = json.loads((run_dir / "summary.json").read_text())
    assert saved["protocol_digest"] == result.protocol_digest
    assert saved["config"]["decoding"] == {"temperature": 0.3}


def test_unparseable_replies_are_graded_wrong(tmp_path):
    config = AgentEvalConfig(
        participants=_participants(1),
        arms=[BenchmarkArm.NO_LLM, BenchmarkArm.SINGLE_LLM],
        completion_tokens_per_case=1024,
    )
    result = run_agent_eval(
        SUITE,
        config,
        output_dir=tmp_path,
        model_factory=lambda **_: _Model(lambda _: "I cannot say."),
        cases=["sic-3c-on-hull", "si2c-no-evidence"],
    )
    single = next(s for s in result.summaries if s.arm == BenchmarkArm.SINGLE_LLM)
    assert single.metrics["correct"] == 0.0
    assert single.llm_cost["parse_failures"] == 2
    with pytest.raises(ValueError, match="unknown agent-eval cases"):
        run_agent_eval(SUITE, config, output_dir=tmp_path, cases=["nope"])


def test_cli_reports_arms_and_comparisons(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import matsim_agents.campaign.agent_eval as agent_eval
    from matsim_agents.cli import app

    monkeypatch.setattr(
        agent_eval.run_agent_eval,
        "__kwdefaults__",
        {"model_factory": lambda **_: _Model(_oracle), "cases": None},
    )
    config = tmp_path / "eval.yaml"
    config.write_text(
        "participants:\n  - {name: a, provider: fake, model: m}\n"
        "arms: [no_llm, single_llm]\ncompletion_tokens_per_case: 1024\n"
    )
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "agent-eval",
            str(config),
            "-o",
            str(tmp_path / "out"),
            "--case",
            "sic-3c-on-hull",
            "--case",
            "si2c-no-evidence",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "single_llm: correct 1.00" in result.output
    assert "single_llm - no_llm correct: +0.500" in result.output
