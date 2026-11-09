"""E3 hypothesis reassessment after DFT ingestion.

Two paths write a revision of the same per-material hypothesis into a strategy's
own ledger:

* ``deterministic_reassessment`` applies the frozen protocol rules.  With fewer
  qualified validation/test groups than the protocol requires, the transfer
  hypothesis cannot be tested and stays UNRESOLVED.
* ``llm_reassessment`` asks the strategy's LLM for a revision, validates it
  (status enum, citations from this ledger only, failed evidence never cited for
  SUPPORTED/CONTRADICTED, no numbers absent from the prompt) and rejects rather
  than repairs an invalid reply.  Token cost is recorded either way.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from benchmarks.e3.controller import CAMPAIGN, Context
from benchmarks.e3.run import _evidence_summary, unsupported_numbers
from matsim_agents.execution.ledger import (
    CostRecord,
    DecisionRecord,
    EvidenceLedger,
    EvidenceOutcome,
    EvidenceRecord,
    HypothesisRecord,
    HypothesisStatus,
)

PROMPT_TEMPLATE = "e3-reassessment-v1"
REJECT_ACTION = "reject_reassessment"
RESOLVED = {HypothesisStatus.SUPPORTED, HypothesisStatus.CONTRADICTED}


def hypothesis_id(material: str) -> str:
    return f"{CAMPAIGN}:{material}:transfer"


@dataclass
class Reassessment:
    accepted: bool
    status: str | None
    revision_id: str | None
    decision_id: str | None
    problems: list[str]
    cost: CostRecord


def _head(ledger: EvidenceLedger, hid: str) -> str | None:
    current = ledger.current_hypotheses().get(hid)
    return current.revision_id if current else None


def _material_evidence(ledger: EvidenceLedger, ctx: Context) -> list[EvidenceRecord]:
    acquired = {d.subject for d in ledger.decisions() if d.action == "freeze_dft_request"}
    return [
        r
        for r in ledger.evidence()
        if r.subject == ctx.material or r.subject.startswith(f"{ctx.material}:") or r.subject in acquired
    ]


def deterministic_reassessment(ledger: EvidenceLedger, ctx: Context, actor: str) -> Reassessment:
    splits = ctx.protocol["splits"]
    n_val, n_test = len(ctx.validation_groups), len(ctx.test_groups)
    evidence = _material_evidence(ledger, ctx)
    measured_dft = [e for e in evidence if e.kind == "relaxation" and e.outcome == EvidenceOutcome.MEASURED]
    insufficient = n_val < splits["min_validation_groups"] or n_test < splits["min_test_groups"]
    if insufficient:
        rationale = (
            f"Insufficient validation evidence: {n_val} validation and {n_test} test reference groups "
            f"qualified (protocol requires {splits['min_validation_groups']} and "
            f"{splits['min_test_groups']}). {len(measured_dft)} acquired DFT labels are training "
            "candidates, not validation data, so they cannot decide transfer."
        )
    else:
        rationale = (
            "Validation partition is sufficient; the transfer decision is made by the validation "
            "metrics against the frozen accuracy criteria after any authorized adaptation."
        )
    hid = hypothesis_id(ctx.material)
    record = ledger.append(
        HypothesisRecord(
            hypothesis_id=hid,
            prior_revision_id=_head(ledger, hid),
            campaign_id=CAMPAIGN,
            claim_type="other",
            scientific_claim=ctx.material_protocol["hypothesis"].strip(),
            falsifiable_prediction=json.dumps(
                {k: v for k, v in ctx.material_protocol["accuracy_criteria"].items() if isinstance(v, (int, float))},
                sort_keys=True,
            ),
            status=HypothesisStatus.UNRESOLVED,
            evidence_ids=[e.evidence_id for e in evidence],
            scientific_rationale=rationale,
            proposed_tests=["qualify >= 2 independent validation groups and >= 1 test group under the frozen signature"]
            if insufficient
            else [],
            agent_model_id=actor,
        )
    )
    return Reassessment(True, record.status.value, record.revision_id, None, [], CostRecord())


def reassessment_prompt(ledger: EvidenceLedger, ctx: Context) -> str:
    evidence = [_evidence_summary(e) for e in _material_evidence(ledger, ctx)]
    criteria = {k: v for k, v in ctx.material_protocol["accuracy_criteria"].items() if isinstance(v, (int, float))}
    current = ledger.current_hypotheses().get(hypothesis_id(ctx.material))
    return (
        f"You reassess the E3 hypothesis for {ctx.material} after new DFT evidence.\n"
        f"Hypothesis: {ctx.material_protocol['hypothesis'].strip()}\n"
        f"Current status: {current.status.value if current else 'none'}\n"
        f"Frozen accuracy criteria: {json.dumps(criteria, sort_keys=True)}\n"
        f"Qualified reference groups: validation={len(ctx.validation_groups)} test={len(ctx.test_groups)} "
        f"(required validation>={ctx.protocol['splits']['min_validation_groups']} "
        f"test>={ctx.protocol['splits']['min_test_groups']})\n"
        f"Evidence: {json.dumps(evidence)}\n"
        "Acquired DFT labels are training candidates, not validation data.\n"
        "Reply with JSON only: {\"status\": one of proposed|supported|contradicted|unresolved, "
        "\"scientific_claim\": str, \"rationale\": str, \"cited_evidence_ids\": [ids from Evidence], "
        "\"proposed_tests\": [str]}. Use only numbers that appear above.\n"
    )


def _parse(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    return data


def validate_reply(data: dict[str, Any], ledger: EvidenceLedger, prompt: str) -> list[str]:
    problems = []
    try:
        status = HypothesisStatus(data.get("status"))
    except ValueError:
        return [f"invalid status {data.get('status')!r}"]
    if status == HypothesisStatus.SUPERSEDED:
        problems.append("superseded is not a reassessment outcome")
    for key in ("scientific_claim", "rationale"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            problems.append(f"missing {key}")
    cited = data.get("cited_evidence_ids")
    if not isinstance(cited, list) or not all(isinstance(i, str) for i in cited):
        return problems + ["cited_evidence_ids must be a list of IDs"]
    unknown = [i for i in cited if not isinstance(ledger.get(i), EvidenceRecord)]
    if unknown:
        problems.append(f"cites evidence not in this strategy's ledger: {unknown}")
    if status in RESOLVED:
        if not cited:
            problems.append(f"a {status.value} claim must cite evidence")
        failed = [i for i in cited if isinstance(r := ledger.get(i), EvidenceRecord) and r.outcome != EvidenceOutcome.MEASURED]
        if failed:
            problems.append(f"{status.value} claim cites failed or unconverged evidence {failed}")
    text = " ".join(str(data.get(k, "")) for k in ("scientific_claim", "rationale")) + " " + " ".join(
        map(str, data.get("proposed_tests") or [])
    )
    bad = unsupported_numbers(text, prompt)
    if bad:
        problems.append(f"numbers not in the prompt: {bad}")
    return problems


def llm_reassessment(ledger: EvidenceLedger, ctx: Context, chat: Any, actor: str, iteration: int) -> Reassessment:
    prompt = reassessment_prompt(ledger, ctx)
    started = time.perf_counter()
    response = chat.invoke(prompt)
    seconds = time.perf_counter() - started
    usage = getattr(response, "usage_metadata", None) or {}
    cost = CostRecord(
        llm_prompt_tokens=int(usage.get("input_tokens", 0)),
        llm_completion_tokens=int(usage.get("output_tokens", 0)),
        llm_seconds=seconds,
    )
    text = getattr(response, "content", response)
    try:
        data = _parse(text if isinstance(text, str) else json.dumps(text))
        problems = validate_reply(data, ledger, prompt)
    except (ValueError, json.JSONDecodeError) as exc:
        data, problems = {}, [f"unparseable reply: {exc}"]
    hid = hypothesis_id(ctx.material)
    cited = [e.evidence_id for e in _material_evidence(ledger, ctx)]
    if problems:
        decision = ledger.append(
            DecisionRecord(
                campaign_id=CAMPAIGN,
                iteration=iteration,
                action=REJECT_ACTION,
                subject=hid,
                evidence_ids=cited,
                rationale="; ".join(problems),
                actor=actor,
                policy_version=PROMPT_TEMPLATE,
                observed_cost=cost,
            )
        )
        return Reassessment(False, None, None, decision.decision_id, problems, cost)
    record = ledger.append(
        HypothesisRecord(
            hypothesis_id=hid,
            prior_revision_id=_head(ledger, hid),
            campaign_id=CAMPAIGN,
            claim_type="other",
            scientific_claim=data["scientific_claim"].strip(),
            status=HypothesisStatus(data["status"]),
            evidence_ids=list(dict.fromkeys(data["cited_evidence_ids"])),
            scientific_rationale=data["rationale"].strip(),
            proposed_tests=[str(t) for t in data.get("proposed_tests") or []],
            agent_model_id=actor,
        )
    )
    decision = ledger.append(
        DecisionRecord(
            campaign_id=CAMPAIGN,
            iteration=iteration,
            action="reassess_hypothesis",
            subject=hid,
            evidence_ids=record.evidence_ids or cited,
            rationale=f"accepted revision {record.revision_id}",
            actor=actor,
            policy_version=PROMPT_TEMPLATE,
            observed_cost=cost,
        )
    )
    return Reassessment(True, record.status.value, record.revision_id, decision.decision_id, [], cost)


def reassess_round(state: Any, ctx: Context, ledger_dir: Any, chat: Any = None, llm_actor: str | None = None) -> dict[str, Reassessment]:
    """Reassess each strategy whose round is fully ingested; the rule arm is deterministic.

    The LLM arm uses ``chat`` when given, otherwise the deterministic rule as well.
    """
    from benchmarks.e3.controller import ledger_path
    from benchmarks.e3.run import LLM
    from benchmarks.e3.state import E3State

    record = state.round(state.current_round)
    results = {}
    for strategy, st in record.strategies.items():
        if st.state != E3State.EVIDENCE_INGESTED:
            continue
        ledger = EvidenceLedger(ledger_path(ledger_dir, strategy))
        if strategy == LLM and chat is not None:
            result = llm_reassessment(ledger, ctx, chat, llm_actor or f"{LLM}:llm", state.current_round)
        else:
            result = deterministic_reassessment(ledger, ctx, f"{strategy}:{PROMPT_TEMPLATE}-deterministic")
        results[strategy] = result
        if result.accepted:
            if result.revision_id not in st.hypothesis_revision_ids:
                st.hypothesis_revision_ids.append(result.revision_id)
            st.reassessment_status = result.status
            st.state = (
                E3State.SCIENTIFICALLY_UNRESOLVED
                if result.status == HypothesisStatus.UNRESOLVED.value
                else E3State.REASSESSED
            )
        elif result.decision_id not in st.rejected_reassessments:
            st.rejected_reassessments.append(result.decision_id)
        cost = ledger.cost_summary()
        st.llm_prompt_tokens = int(cost["llm_prompt_tokens"])
        st.llm_completion_tokens = int(cost["llm_completion_tokens"])
        st.llm_seconds = float(cost["llm_seconds"])
    return results
