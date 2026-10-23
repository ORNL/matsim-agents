"""Evaluate LLM agents on claims whose evidence, and correct verdict, are fixed.

Each case gives a claim and evidence records; an agent answers whether the
evidence supports, contradicts, or leaves the claim unresolved, which evidence
it rests on, and whether a DFT calculation is needed. A rule grader scores the
answer with the evidence ledger's claim rules, so a stability claim resting on
MLIP evidence is wrong however it is argued. Arms (no LLM, one LLM, several
LLMs without and with cross-critique) share one completion-token budget per
case and are compared case by case. LLM cost (tokens, latency) is reported
apart from numerical cost (DFT calculations requested).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from matsim_agents.backends.llm.provider import get_chat_model
from matsim_agents.campaign.benchmark import (
    BenchmarkArm,
    BenchmarkMetric,
    BenchmarkObservation,
    PairedComparison,
    compare_paired_metric,
)
from matsim_agents.execution.contracts import EvidenceLevel
from matsim_agents.execution.ledger import (
    ClaimType,
    EvidenceKind,
    EvidenceLedger,
    EvidenceOutcome,
    EvidenceRecord,
    HypothesisRecord,
    HypothesisStatus,
    LedgerQuantity,
    MethodRecord,
)
from matsim_agents.workflows.debate import (
    DebateParticipant,
    ScientificDebateConfig,
    run_scientific_debate,
)

Verdict = Literal["supported", "contradicted", "unresolved"]
AGENT_ARMS = (
    BenchmarkArm.NO_LLM,
    BenchmarkArm.SINGLE_LLM,
    BenchmarkArm.MULTI_LLM_NO_CRITIQUE,
    BenchmarkArm.MULTI_LLM_CROSS_CRITIQUE,
)
MIN_TOKENS_PER_CALL = 256
METRICS = (
    BenchmarkMetric(name="correct", direction="maximize"),
    BenchmarkMetric(name="evidence_attribution", direction="maximize"),
    BenchmarkMetric(name="correctly_inconclusive", direction="maximize"),
    BenchmarkMetric(name="dft_request_correct", direction="maximize"),
    BenchmarkMetric(name="completion_tokens", direction="minimize", unit="tokens"),
)
_DEFAULT_COMPARISONS = (
    (BenchmarkArm.SINGLE_LLM, BenchmarkArm.NO_LLM),
    (BenchmarkArm.MULTI_LLM_NO_CRITIQUE, BenchmarkArm.SINGLE_LLM),
    (BenchmarkArm.MULTI_LLM_CROSS_CRITIQUE, BenchmarkArm.MULTI_LLM_NO_CRITIQUE),
)


class CaseEvidence(BaseModel):
    """An evidence record of a case; its evidence_id is derived from the content."""

    kind: EvidenceKind
    subject: str
    method: MethodRecord
    evidence_level: EvidenceLevel
    outcome: EvidenceOutcome
    values: list[LedgerQuantity] = Field(default_factory=list)
    reference_complete: bool | None = None
    failure_reason: str | None = None

    def record(self, suite_id: str, case_id: str, label: str) -> EvidenceRecord:
        return EvidenceRecord(
            **self.model_dump(), campaign_id=suite_id, task_id=f"{case_id}:{label}"
        )


class AgentEvalCase(BaseModel):
    case_id: str
    category: str
    subject: str
    claim: str
    claim_type: ClaimType
    expected_status: Verdict
    decisive_evidence: list[str] = Field(
        default_factory=list, description="Labels of evidence a decisive answer must cite."
    )
    dft_request_justified: bool
    evidence: dict[str, CaseEvidence] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _decisive_evidence_exists(self) -> AgentEvalCase:
        if unknown := sorted(set(self.decisive_evidence) - set(self.evidence)):
            raise ValueError(f"{self.case_id}: decisive_evidence names unknown labels {unknown}")
        if self.expected_status != "unresolved" and not self.decisive_evidence:
            raise ValueError(f"{self.case_id}: a decisive expected status needs decisive_evidence")
        return self


class AgentEvalSuite(BaseModel):
    suite_id: str
    cases: list[AgentEvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_cases(self) -> AgentEvalSuite:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("agent-eval case_ids must be unique")
        return self


def load_suite(suite_dir: str | Path) -> AgentEvalSuite:
    with open(Path(suite_dir) / "cases.yaml", encoding="utf-8") as stream:
        return AgentEvalSuite.model_validate(yaml.safe_load(stream))


class AgentEvalConfig(BaseModel):
    """The arms, models, decoding, and budget of one evaluation."""

    participants: list[DebateParticipant] = Field(min_length=1)
    arms: list[BenchmarkArm] = Field(default_factory=lambda: list(AGENT_ARMS))
    rounds: int = Field(2, ge=1, description="Critique rounds of the cross-critique arm.")
    completion_tokens_per_case: int = Field(
        16_384, gt=0, description="Shared by every arm: split evenly across an arm's model calls."
    )
    decoding: dict[str, float | int | str | bool] = Field(
        default_factory=lambda: {"temperature": 0.0},
        description="Passed to the model factory for every call.",
    )
    random_seeds: list[int] = Field(default_factory=lambda: [0], min_length=1)
    comparisons: list[tuple[BenchmarkArm, BenchmarkArm]] = Field(
        default_factory=lambda: list(_DEFAULT_COMPARISONS),
        description="(treatment, control) pairs; pairs with an unconfigured arm are skipped.",
    )
    confidence_level: float = Field(0.95, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validate(self) -> AgentEvalConfig:
        if unsupported := [arm.value for arm in self.arms if arm not in AGENT_ARMS]:
            raise ValueError(
                f"agent-eval arms must be among {[a.value for a in AGENT_ARMS]}, not {unsupported}"
            )
        if len(self.arms) != len(set(self.arms)):
            raise ValueError("agent-eval arms must be unique")
        if len(self.random_seeds) != len(set(self.random_seeds)):
            raise ValueError("agent-eval random_seeds must be unique")
        names = [participant.name for participant in self.participants]
        if len(names) != len(set(names)):
            raise ValueError("agent-eval participant names must be unique")
        multi = {BenchmarkArm.MULTI_LLM_NO_CRITIQUE, BenchmarkArm.MULTI_LLM_CROSS_CRITIQUE}
        if multi & set(self.arms) and len(self.participants) < 2:
            raise ValueError("multi-LLM arms need at least two participants")
        for arm in self.arms:
            if (calls := self.calls(arm)) and self.completion_tokens_per_case // calls < (
                MIN_TOKENS_PER_CALL
            ):
                raise ValueError(
                    f"completion_tokens_per_case leaves {arm.value} under "
                    f"{MIN_TOKENS_PER_CALL} tokens for each of its {calls} calls"
                )
        return self

    def calls(self, arm: BenchmarkArm) -> int:
        """Model calls the arm makes for one case."""
        n = len(self.participants)
        return {
            BenchmarkArm.NO_LLM: 0,
            BenchmarkArm.SINGLE_LLM: 1,
            BenchmarkArm.MULTI_LLM_NO_CRITIQUE: n,
            BenchmarkArm.MULTI_LLM_CROSS_CRITIQUE: n * self.rounds + n,
        }[arm]


class AgentAnswer(BaseModel):
    status: Verdict
    evidence_ids: list[str] = Field(default_factory=list)
    request_dft: bool = False
    rationale: str = ""


class AgentEvalGrade(BaseModel):
    correct: bool
    evidence_attribution: bool
    correctly_inconclusive: bool | None = Field(
        None, description="Only for cases whose expected status is unresolved."
    )
    dft_request_correct: bool
    problems: list[str] = Field(default_factory=list)


class ArmAnswer(BaseModel):
    """What one arm answered for one case, and what it cost."""

    answer: AgentAnswer | None
    responses: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    run_directories: list[str] = Field(default_factory=list)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    parse_failures: int = 0


class AgentEvalRecord(ArmAnswer):
    arm: BenchmarkArm
    random_seed: int
    case_id: str
    evidence_ids: dict[str, str]
    grade: AgentEvalGrade


class ArmSummary(BaseModel):
    arm: BenchmarkArm
    answers: int
    metrics: dict[str, float]
    llm_cost: dict[str, float]
    numerical_cost: dict[str, int]


class AgentEvalResult(BaseModel):
    suite_id: str
    protocol_digest: str
    run_dir: str
    summaries: list[ArmSummary]
    comparisons: list[PairedComparison]


ANSWER_FORMAT = """Return JSON only, with this exact structure:
{"status": "supported|contradicted|unresolved", "evidence_ids": ["ev-..."],
 "request_dft": true|false, "rationale": "..."}
Cite evidence only by the evidence_id values given. "supported" or "contradicted" must cite the
evidence that decides the claim. Do not wrap the JSON in Markdown."""

_RULES = """Rules for judging the claim:
- A failed_to_compute or not_converged record is not evidence for or against a material.
- A thermodynamic_stability claim is decided only by converged DFT stability (hull) evidence with
  complete reference phases; MLIP evidence supports at most an energy_ranking claim.
- An energy_ranking claim needs at least MLIP relaxation evidence of kind energy_ranking or
  stability.
- When admissible records disagree, the claim is unresolved.
- Answer unresolved when the evidence does not decide the claim, and request DFT only when a DFT
  calculation could decide it."""


def case_prompt(case: AgentEvalCase, records: dict[str, EvidenceRecord]) -> str:
    """The question put to every arm: the claim, its evidence, and the rules."""
    evidence = [
        record.model_dump(
            mode="json",
            include={
                "evidence_id",
                "kind",
                "subject",
                "method",
                "evidence_level",
                "outcome",
                "values",
                "reference_complete",
                "failure_reason",
            },
        )
        for record in records.values()
    ]
    return (
        f"Claim ({case.claim_type}) about {case.subject}: {case.claim}\n\n"
        f"Evidence records:\n{json.dumps(evidence, indent=1) if evidence else 'none'}\n\n"
        f"{_RULES}\n\nDoes the evidence support, contradict, or leave the claim unresolved?"
    )


def parse_answer(text: str) -> AgentAnswer:
    """The last valid answer object in a reply."""
    decoder = json.JSONDecoder()
    answer = None
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text, offset)
            if isinstance(candidate, dict) and "status" in candidate:
                answer = AgentAnswer.model_validate(candidate)
        except ValueError:
            continue
    if answer is None:
        raise ValueError("reply contains no valid answer")
    return answer


def grade_answer(
    case: AgentEvalCase,
    records: dict[str, EvidenceRecord],
    ledger: EvidenceLedger,
    answer: AgentAnswer | None,
) -> AgentEvalGrade:
    """Score one answer; a decisive answer the ledger would reject is wrong."""
    unresolved = case.expected_status == "unresolved"
    if answer is None:
        return AgentEvalGrade(
            correct=False,
            evidence_attribution=False,
            correctly_inconclusive=False if unresolved else None,
            dft_request_correct=False,
            problems=["no parseable answer"],
        )
    known = {record.evidence_id for record in records.values()}
    decisive = {records[label].evidence_id for label in case.decisive_evidence}
    problems = []
    if invented := [i for i in answer.evidence_ids if i not in known]:
        problems.append(f"cites evidence that is not in the case: {invented}")
    if answer.status != "unresolved":
        problems += ledger.problems(
            HypothesisRecord(
                hypothesis_id=f"agent-eval:{case.case_id}",
                claim_type=case.claim_type,
                scientific_claim=case.claim,
                status=HypothesisStatus(answer.status),
                evidence_ids=answer.evidence_ids,
            )
        )
        if decisive and not decisive & set(answer.evidence_ids):
            problems.append("does not cite the evidence that decides the claim")
    if answer.status != case.expected_status:
        problems.append(f"answered {answer.status}; expected {case.expected_status}")
    attribution = not invented and (
        answer.status == "unresolved"
        or (
            bool(answer.evidence_ids)
            and (not decisive or bool(decisive & set(answer.evidence_ids)))
        )
    )
    return AgentEvalGrade(
        correct=not problems,
        evidence_attribution=attribution,
        correctly_inconclusive=(answer.status == "unresolved") if unresolved else None,
        dft_request_correct=answer.request_dft == case.dft_request_justified,
        problems=problems,
    )


def aggregate_answers(answers: list[AgentAnswer]) -> AgentAnswer | None:
    """Majority verdict; without a strict majority the panel is unresolved."""
    if not answers:
        return None
    status, votes = Counter(a.status for a in answers).most_common(1)[0]
    if votes * 2 <= len(answers):
        status = "unresolved"
    majority = [a for a in answers if a.status == status]
    return AgentAnswer(
        status=status,
        evidence_ids=list(dict.fromkeys(i for a in majority for i in a.evidence_ids)),
        request_dft=sum(a.request_dft for a in answers) * 2 > len(answers),
        rationale="\n\n".join(a.rationale for a in majority),
    )


def _debate_answers(
    cfg: ScientificDebateConfig, model_factory: Callable[..., Any]
) -> tuple[list[AgentAnswer], ArmAnswer]:
    result = run_scientific_debate(cfg, model_factory=model_factory)
    answers, failures = [], 0
    for verdict in result.verdicts:
        try:
            answers.append(parse_answer(verdict.response))
        except ValueError:
            failures += 1
    with open(result.contributions_path, encoding="utf-8") as stream:
        contributions = [json.loads(line) for line in stream if line.strip()]
    return answers, ArmAnswer(
        answer=None,
        responses=[verdict.response for verdict in result.verdicts],
        models=[f"{p.provider}:{p.model}" for p in cfg.participants],
        run_directories=[result.run_directory],
        calls=len(contributions),
        prompt_tokens=sum(c.get("prompt_tokens") or 0 for c in contributions),
        completion_tokens=sum(c.get("completion_tokens") or 0 for c in contributions),
        latency_s=round(sum(c.get("latency_s") or 0.0 for c in contributions), 3),
        parse_failures=failures,
    )


def _merge(parts: list[ArmAnswer], answer: AgentAnswer | None) -> ArmAnswer:
    return ArmAnswer(
        answer=answer,
        responses=[r for p in parts for r in p.responses],
        models=[m for p in parts for m in p.models],
        run_directories=[d for p in parts for d in p.run_directories],
        calls=sum(p.calls for p in parts),
        prompt_tokens=sum(p.prompt_tokens for p in parts),
        completion_tokens=sum(p.completion_tokens for p in parts),
        latency_s=round(sum(p.latency_s for p in parts), 3),
        parse_failures=sum(p.parse_failures for p in parts),
    )


def run_arm(
    arm: BenchmarkArm,
    prompt: str,
    config: AgentEvalConfig,
    *,
    output_root: Path,
    model_factory: Callable[..., Any],
) -> ArmAnswer:
    """Ask one arm about one case."""
    if arm == BenchmarkArm.NO_LLM:
        # The baseline every LLM arm must beat: never decide, always ask for DFT.
        return ArmAnswer(answer=AgentAnswer(status="unresolved", request_dft=True))
    tokens = config.completion_tokens_per_case // config.calls(arm)
    participants = [
        p.model_copy(update={"max_completion_tokens": None}) for p in config.participants
    ]
    factory = partial(model_factory, **config.decoding)

    def single(participant: DebateParticipant) -> ScientificDebateConfig:
        return ScientificDebateConfig(
            hypothesis=prompt,
            participants=[participant],
            rounds=1,
            single_call=True,
            max_completion_tokens=tokens,
            final_response_instruction=ANSWER_FORMAT,
            output_root=str(output_root),
        )

    if arm == BenchmarkArm.SINGLE_LLM:
        answers, part = _debate_answers(single(participants[0]), factory)
        return _merge([part], answers[0] if answers else None)
    if arm == BenchmarkArm.MULTI_LLM_NO_CRITIQUE:
        runs = [_debate_answers(single(p), factory) for p in participants]
        answers = [a for found, _ in runs for a in found]
        return _merge([part for _, part in runs], aggregate_answers(answers))
    answers, part = _debate_answers(
        ScientificDebateConfig(
            hypothesis=prompt,
            participants=participants,
            rounds=config.rounds,
            synthesis_method="independent_verdicts",
            max_completion_tokens=tokens,
            final_response_instruction=ANSWER_FORMAT,
            output_root=str(output_root),
        ),
        factory,
    )
    return _merge([part], aggregate_answers(answers))


def _digest(suite: AgentEvalSuite, config: AgentEvalConfig) -> str:
    payload = {"suite": suite.model_dump(mode="json"), "config": config.model_dump(mode="json")}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _summaries(records: list[AgentEvalRecord], arms: list[BenchmarkArm]) -> list[ArmSummary]:
    summaries = []
    for arm in arms:
        mine = [r for r in records if r.arm == arm]
        inconclusive = [
            r.grade.correctly_inconclusive
            for r in mine
            if r.grade.correctly_inconclusive is not None
        ]
        n = max(len(mine), 1)
        summaries.append(
            ArmSummary(
                arm=arm,
                answers=len(mine),
                metrics={
                    "correct": sum(r.grade.correct for r in mine) / n,
                    "evidence_attribution": sum(r.grade.evidence_attribution for r in mine) / n,
                    "correctly_inconclusive": (
                        sum(inconclusive) / len(inconclusive) if inconclusive else 0.0
                    ),
                    "dft_request_correct": sum(r.grade.dft_request_correct for r in mine) / n,
                },
                llm_cost={
                    "calls": sum(r.calls for r in mine),
                    "prompt_tokens": sum(r.prompt_tokens for r in mine),
                    "completion_tokens": sum(r.completion_tokens for r in mine),
                    "latency_s": round(sum(r.latency_s for r in mine), 3),
                    "parse_failures": sum(r.parse_failures for r in mine),
                },
                numerical_cost={
                    "dft_calculations": 0,
                    "dft_requests": sum(bool(r.answer and r.answer.request_dft) for r in mine),
                },
            )
        )
    return summaries


def _observations(
    records: list[AgentEvalRecord], digest: str, case_ids: list[str], seeds: list[int]
) -> list[BenchmarkObservation]:
    observations = []
    for record in records:
        metrics = {
            "correct": float(record.grade.correct),
            "evidence_attribution": float(record.grade.evidence_attribution),
            "dft_request_correct": float(record.grade.dft_request_correct),
            "completion_tokens": float(record.completion_tokens),
        }
        if record.grade.correctly_inconclusive is not None:
            metrics["correctly_inconclusive"] = float(record.grade.correctly_inconclusive)
        observations.append(
            BenchmarkObservation(
                protocol_digest=digest,
                arm=record.arm,
                # Arms are paired case by case: one pair per (seed, case).
                random_seed=seeds.index(record.random_seed) * len(case_ids)
                + case_ids.index(record.case_id),
                metrics=metrics,
                dft_calculations_attempted=0,
                node_hours_consumed=0.0,
                completed=True,
            )
        )
    return observations


def run_agent_eval(
    suite_dir: str | Path,
    config: AgentEvalConfig,
    *,
    output_dir: str | Path,
    model_factory: Callable[..., Any] = get_chat_model,
    cases: list[str] | None = None,
) -> AgentEvalResult:
    """Ask every arm about every case for every seed, grade, and compare the arms.

    Each call writes a new directory under ``output_dir`` holding the case
    ledgers, every model call's prompt and reply (one debate run each),
    ``answers.jsonl`` with the grades, and ``summary.json``.
    """
    suite = load_suite(suite_dir)
    if unknown := sorted(set(cases or []) - {case.case_id for case in suite.cases}):
        raise ValueError(f"unknown agent-eval cases {unknown}")
    selected = [case for case in suite.cases if not cases or case.case_id in cases]
    suite = suite.model_copy(update={"cases": selected})
    digest = _digest(suite, config)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="agent-eval-", dir=output_dir))
    case_records: dict[str, dict[str, EvidenceRecord]] = {}
    ledgers: dict[str, EvidenceLedger] = {}
    for case in selected:
        ledger = EvidenceLedger(run_dir / "ledgers" / f"{case.case_id}.jsonl")
        case_records[case.case_id] = {
            label: ledger.append(spec.record(suite.suite_id, case.case_id, label))
            for label, spec in case.evidence.items()
        }
        ledgers[case.case_id] = ledger
    records: list[AgentEvalRecord] = []
    with open(run_dir / "answers.jsonl", "w", encoding="utf-8") as answers_file:
        for seed in config.random_seeds:
            for case in selected:
                found = case_records[case.case_id]
                prompt = case_prompt(case, found)
                for arm in config.arms:
                    answer = run_arm(
                        arm,
                        prompt,
                        config,
                        output_root=run_dir / "arms" / arm.value / f"seed-{seed}" / case.case_id,
                        model_factory=model_factory,
                    )
                    record = AgentEvalRecord(
                        **answer.model_dump(),
                        arm=arm,
                        random_seed=seed,
                        case_id=case.case_id,
                        evidence_ids={label: r.evidence_id for label, r in found.items()},
                        grade=grade_answer(case, found, ledgers[case.case_id], answer.answer),
                    )
                    answers_file.write(record.model_dump_json() + "\n")
                    answers_file.flush()
                    records.append(record)
    observations = _observations(
        records, digest, [case.case_id for case in selected], config.random_seeds
    )
    comparisons = []
    for treatment, control in config.comparisons:
        if treatment not in config.arms or control not in config.arms:
            continue
        for metric in METRICS:
            try:
                comparisons.append(
                    compare_paired_metric(
                        observations,
                        treatment=treatment,
                        control=control,
                        metric=metric,
                        confidence_level=config.confidence_level,
                    )
                )
            except ValueError:
                continue  # fewer than two pairs carry this metric
    result = AgentEvalResult(
        suite_id=suite.suite_id,
        protocol_digest=digest,
        run_dir=str(run_dir),
        summaries=_summaries(records, config.arms),
        comparisons=comparisons,
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {"config": config.model_dump(mode="json"), **result.model_dump(mode="json")}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return result
