"""Known-answer benchmark: characterization contracts whose correct answers are known.

A suite directory holds ``cases.yaml`` (the cases and their expected answers),
one contract per case, and, for the offline tier, recorded ``pw.x`` runs under
``replay/<case_id>/<step_id>/`` (``pw.in`` and ``pw.out``). Each case runs the
real characterization pipeline; on the ``replay`` tier the DFT launcher is
replaced by a replayer that checks the written input against the recorded one
and returns the recorded output, on the ``live`` tier ``pw.x`` runs.

The grader is deterministic and is the benchmark's NO_LLM arm: it checks the
report's status, observables, conclusion, and missing evidence against the
case's expectations. Every verdict that rests on ledger evidence is written to
the ledger as a decision citing that evidence.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from matsim_agents.campaign.characterization import load_contract, run_characterization
from matsim_agents.execution.contracts import CharacterizationReport, CharacterizationStatus
from matsim_agents.execution.ledger import DecisionRecord, EvidenceLedger

CASES_FILE = "cases.yaml"
ACTOR = "benchmark:known-answer"
REPLAYER = Path(__file__).with_name("_replay_pw.py")


class ObservableExpectation(BaseModel):
    """An observable that must lie in ``[minimum, maximum]``.

    ``structure`` selects the structure by input file name; ``lattice_constant``
    grades ``relaxed_structure`` by the cubic lattice constant of its relaxed
    primitive cell (fcc primitive vectors: a = sqrt(2) |a_i|).
    """

    name: str
    structure: str | None = None
    minimum: float = -math.inf
    maximum: float = math.inf
    lattice_constant: bool = False

    @model_validator(mode="after")
    def _bounded(self) -> ObservableExpectation:
        if not self.minimum <= self.maximum:
            raise ValueError(f"{self.name}: minimum exceeds maximum")
        if self.lattice_constant and self.name != "relaxed_structure":
            raise ValueError("lattice_constant grades relaxed_structure only")
        return self


class KnownAnswerCase(BaseModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    question: str
    contract: str = Field(description="Contract path, relative to the suite directory.")
    expected_status: CharacterizationStatus
    expected_ground_state: str | None = Field(
        None, description="Input file name of the structure the conclusion must name."
    )
    observables: list[ObservableExpectation] = Field(default_factory=list)
    missing_contains: list[str] = Field(
        default_factory=list, description="Substrings some missing-evidence entry must contain."
    )


class KnownAnswerSuite(BaseModel):
    suite_id: str
    cases: list[KnownAnswerCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> KnownAnswerSuite:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("known-answer case IDs must be unique")
        return self


class KnownAnswerCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class KnownAnswerVerdict(BaseModel):
    case_id: str
    passed: bool
    status: CharacterizationStatus
    checks: list[KnownAnswerCheck]
    evidence_ids: list[str] = Field(default_factory=list)
    decision_id: str | None = None

    @property
    def reasons(self) -> list[str]:
        return [check.detail for check in self.checks if not check.passed]


class KnownAnswerResult(BaseModel):
    suite_id: str
    tier: Literal["replay", "live"]
    verdicts: list[KnownAnswerVerdict]

    @property
    def passed(self) -> bool:
        return all(verdict.passed for verdict in self.verdicts)


def load_suite(suite_dir: str | Path) -> KnownAnswerSuite:
    import yaml

    data = yaml.safe_load((Path(suite_dir) / CASES_FILE).read_text())
    return KnownAnswerSuite.model_validate(data)


def _cubic_lattice_constant(path: str) -> float:
    from ase.io import read

    lengths = read(path).cell.lengths()
    return float(math.sqrt(2.0) * sum(lengths) / 3.0)


def grade_case(case: KnownAnswerCase, report: CharacterizationReport) -> KnownAnswerVerdict:
    """Compare a report with a case's known answer (deterministic, no model)."""
    checks = [
        KnownAnswerCheck(
            name="status",
            passed=report.status == case.expected_status,
            detail=f"status {report.status.value}, expected {case.expected_status.value}",
        )
    ]
    if case.expected_ground_state is not None:
        # The ranked structure at zero relative energy, or the structure a hull
        # assessment reports on; the conclusion must name it too.
        conclusion = report.conclusion or ""
        ground = next(
            (
                Path(value.structure).name
                for value in report.observables
                if value.structure
                and (
                    (value.name == "relative_energy_per_atom" and float(value.value) == 0.0)
                    or value.name == "energy_above_hull"
                )
            ),
            None,
        )
        checks.append(
            KnownAnswerCheck(
                name="ground_state",
                passed=ground == case.expected_ground_state
                and case.expected_ground_state in conclusion,
                detail=f"ground state {ground}, expected {case.expected_ground_state}",
            )
        )
    for expectation in case.observables:
        matches = [
            value
            for value in report.observables
            if value.name == expectation.name
            and (
                expectation.structure is None
                or (value.structure and Path(value.structure).name == expectation.structure)
            )
        ]
        label = expectation.name + (f"[{expectation.structure}]" if expectation.structure else "")
        if len(matches) != 1:
            checks.append(
                KnownAnswerCheck(
                    name=label, passed=False, detail=f"{label}: {len(matches)} values reported"
                )
            )
            continue
        value = matches[0].value
        if expectation.lattice_constant:
            value = _cubic_lattice_constant(str(value))
            label = f"lattice_constant({label})"
        number = float(value)
        checks.append(
            KnownAnswerCheck(
                name=label,
                passed=expectation.minimum <= number <= expectation.maximum,
                detail=f"{label} = {number:.6g}, expected in "
                f"[{expectation.minimum:g}, {expectation.maximum:g}]",
            )
        )
    for text in case.missing_contains:
        checks.append(
            KnownAnswerCheck(
                name=f"missing:{text}",
                passed=any(text in entry for entry in report.missing),
                detail=f"missing evidence mentions {text!r}: {report.missing}",
            )
        )
    return KnownAnswerVerdict(
        case_id=case.case_id,
        passed=all(check.passed for check in checks),
        status=report.status,
        checks=checks,
        evidence_ids=[item.evidence_id for item in report.evidence],
    )


def run_known_answers(
    suite_dir: str | Path,
    *,
    tier: Literal["replay", "live"] = "replay",
    output_dir: str | Path,
    ledger: EvidenceLedger | None = None,
    cases: list[str] | None = None,
) -> KnownAnswerResult:
    """Run and grade the suite's cases; record each evidence-backed verdict in ``ledger``."""
    suite_dir = Path(suite_dir).resolve()
    suite = load_suite(suite_dir)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if ledger is None:
        ledger = EvidenceLedger(output_dir / "known-answer-ledger.jsonl")
    unknown = sorted(set(cases or []) - {case.case_id for case in suite.cases})
    if unknown:
        raise ValueError(f"unknown known-answer cases {unknown}")
    verdicts = []
    for case in suite.cases:
        if cases and case.case_id not in cases:
            continue
        contract = load_contract(suite_dir / case.contract)
        update: dict = {"output_root": output_dir / case.case_id}
        if tier == "replay":
            assert contract.methods.dft is not None, f"{case.case_id}: replay needs a DFT method"
            replay = str(suite_dir / "replay" / case.case_id)
            dft = contract.methods.dft.model_copy(
                update={
                    "launcher": [sys.executable, str(REPLAYER), replay, "{work_dir}", "{input}"]
                }
            )
            update["methods"] = contract.methods.model_copy(update={"dft": dft})
        contract = contract.model_copy(update=update)
        report = run_characterization(contract, ledger=ledger)
        verdict = grade_case(case, report)
        if verdict.evidence_ids:
            decision = ledger.append(
                DecisionRecord(
                    action="known_answer_pass" if verdict.passed else "known_answer_fail",
                    subject=case.case_id,
                    evidence_ids=verdict.evidence_ids,
                    rationale="; ".join(check.detail for check in verdict.checks),
                    actor=f"{ACTOR}:{suite.suite_id}:{tier}",
                )
            )
            verdict = verdict.model_copy(update={"decision_id": decision.decision_id})
        verdicts.append(verdict)
    return KnownAnswerResult(suite_id=suite.suite_id, tier=tier, verdicts=verdicts)


__all__ = [
    "KnownAnswerCase",
    "KnownAnswerResult",
    "KnownAnswerSuite",
    "KnownAnswerVerdict",
    "ObservableExpectation",
    "grade_case",
    "load_suite",
    "run_known_answers",
]
