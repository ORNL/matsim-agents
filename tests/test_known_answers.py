"""Known-answer benchmark: replay tier, grader, ledger verdicts, replayer, and CLI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from matsim_agents.campaign.known_answers import (
    REPLAYER,
    KnownAnswerSuite,
    load_suite,
    run_known_answers,
)
from matsim_agents.execution.ledger import EvidenceLedger

SUITE = Path(__file__).resolve().parents[1] / "benchmarks" / "known_answers"


@pytest.fixture(scope="module")
def replay_run(tmp_path_factory):
    output = tmp_path_factory.mktemp("known-answers")
    ledger = EvidenceLedger(output / "ledger.jsonl")
    return run_known_answers(SUITE, tier="replay", output_dir=output, ledger=ledger), ledger


def _copy_suite(tmp_path: Path) -> Path:
    suite = tmp_path / "suite"
    shutil.copytree(SUITE, suite, ignore=shutil.ignore_patterns("record.py", "__pycache__"))
    return suite


def test_suite_lists_five_cases_with_contracts_and_recordings():
    suite = load_suite(SUITE)
    assert len(suite.cases) == 5
    for case in suite.cases:
        assert (SUITE / case.contract).is_file()
        assert list((SUITE / "replay" / case.case_id).glob("relax-*/pw.out"))


def test_replay_tier_passes_every_case(replay_run):
    result, _ = replay_run
    assert result.tier == "replay"
    assert [v.case_id for v in result.verdicts] == [c.case_id for c in load_suite(SUITE).cases]
    assert result.passed, {v.case_id: v.reasons for v in result.verdicts if not v.passed}


def test_step_limited_case_is_graded_inconclusive(replay_run):
    result, _ = replay_run
    verdict = next(v for v in result.verdicts if v.case_id == "si-step-limited")
    assert verdict.status.value == "inconclusive"
    assert verdict.passed


def test_verdicts_are_ledger_decisions_citing_evidence(replay_run):
    result, ledger = replay_run
    decisions = {d.subject: d for d in ledger.decisions() if d.action.startswith("known_answer_")}
    assert set(decisions) == {v.case_id for v in result.verdicts}
    for verdict in result.verdicts:
        decision = decisions[verdict.case_id]
        assert decision.decision_id == verdict.decision_id
        assert decision.action == "known_answer_pass"
        assert decision.evidence_ids and set(decision.evidence_ids) == set(verdict.evidence_ids)
        assert all(ledger.get(eid) is not None for eid in decision.evidence_ids)
        assert not ledger.problems(decision)


def test_grader_fails_a_wrong_answer(tmp_path):
    suite = _copy_suite(tmp_path)
    data = yaml.safe_load((suite / "cases.yaml").read_text())
    case = next(c for c in data["cases"] if c["case_id"] == "si-diamond-vs-beta-tin")
    case["expected_ground_state"] = "Si-beta-tin.vasp"
    (suite / "cases.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    result = run_known_answers(
        suite, tier="replay", output_dir=tmp_path / "out", cases=["si-diamond-vs-beta-tin"]
    )
    (verdict,) = result.verdicts
    assert not result.passed
    assert any("ground state Si-diamond.vasp" in reason for reason in verdict.reasons)
    ledger = EvidenceLedger(tmp_path / "out" / "known-answer-ledger.jsonl")
    assert [d.action for d in ledger.decisions() if d.subject == verdict.case_id] == [
        "known_answer_fail"
    ]


def test_out_of_window_observable_fails(tmp_path):
    suite = _copy_suite(tmp_path)
    data = yaml.safe_load((suite / "cases.yaml").read_text())
    case = next(c for c in data["cases"] if c["case_id"] == "si-lattice-constant")
    lattice = next(o for o in case["observables"] if o.get("lattice_constant"))
    lattice["minimum"], lattice["maximum"] = 5.0, 5.1
    (suite / "cases.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    result = run_known_answers(
        suite, tier="replay", output_dir=tmp_path / "out", cases=["si-lattice-constant"]
    )
    (verdict,) = result.verdicts
    assert not verdict.passed
    assert any(reason.startswith("lattice_constant(") for reason in verdict.reasons)


def test_unknown_case_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown known-answer cases"):
        run_known_answers(SUITE, tier="replay", output_dir=tmp_path, cases=["nope"])


def test_suite_rejects_duplicate_case_ids():
    case = load_suite(SUITE).cases[0].model_dump()
    with pytest.raises(ValueError, match="unique"):
        KnownAnswerSuite(suite_id="dup", cases=[case, case])


def test_replayer_rejects_a_changed_input(tmp_path):
    recorded = next((SUITE / "replay" / "si-lattice-constant").glob("relax-*"))
    work = tmp_path / recorded.name / "calculations" / "qe"
    work.mkdir(parents=True)
    changed = work / "pw.in"
    changed.write_text(recorded.joinpath("pw.in").read_text().replace("45", "46", 1))
    replay_dir = str(recorded.parent)

    proc = subprocess.run(
        [sys.executable, str(REPLAYER), replay_dir, str(work), str(changed)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0 and "differs from the recorded" in proc.stderr

    changed.write_text(recorded.joinpath("pw.in").read_text())
    proc = subprocess.run(
        [sys.executable, str(REPLAYER), replay_dir, str(work), str(changed)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout == recorded.joinpath("pw.out").read_text()


def test_cli_exit_codes(tmp_path):
    from typer.testing import CliRunner

    from matsim_agents.cli import app

    ok = CliRunner().invoke(
        app,
        ["benchmark", "known-answers", "--case", "si-step-limited", "-o", str(tmp_path / "a")],
    )
    assert ok.exit_code == 0, ok.output
    assert "PASS si-step-limited" in ok.output

    suite = _copy_suite(tmp_path)
    data = yaml.safe_load((suite / "cases.yaml").read_text())
    data["cases"][-1]["expected_status"] = "sufficiently_characterized"
    (suite / "cases.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    bad = CliRunner().invoke(
        app,
        [
            "benchmark",
            "known-answers",
            str(suite),
            "--case",
            data["cases"][-1]["case_id"],
            "-o",
            str(tmp_path / "b"),
        ],
    )
    assert bad.exit_code == 1
    assert "FAIL" in bad.output

    tier = CliRunner().invoke(app, ["benchmark", "known-answers", "--tier", "cloud"])
    assert tier.exit_code != 0
