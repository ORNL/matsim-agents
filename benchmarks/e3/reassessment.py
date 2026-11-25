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
from dataclasses import dataclass, field
from typing import Any

from benchmarks.e3.controller import CAMPAIGN, Context
from benchmarks.e3.run import _evidence_summary, unsupported_numbers
from matsim_agents.core.contracts import EvidenceLevel
from matsim_agents.execution.ledger import (
    CostRecord,
    DecisionRecord,
    EvidenceLedger,
    EvidenceOutcome,
    EvidenceRecord,
    HypothesisRecord,
    HypothesisStatus,
)

PROMPT_TEMPLATE = "e3-reassessment-v2"
REJECT_ACTION = "reject_reassessment"
RESOLVED = {HypothesisStatus.SUPPORTED, HypothesisStatus.CONTRADICTED}


def hypothesis_id(material: str) -> str:
    return f"{CAMPAIGN}:{material}:transfer"


def _transfer_claim(ctx: Context) -> str:
    if ctx.material_protocol.get("hypothesis"):
        return ctx.material_protocol["hypothesis"].strip()
    return (
        f"Pretrained interatomic potentials transfer to {ctx.material_protocol['name']} "
        "within the frozen accuracy criteria and declared method/reference convention."
    )


@dataclass
class Reassessment:
    accepted: bool
    status: str | None
    revision_id: str | None
    decision_id: str | None
    problems: list[str]
    cost: CostRecord
    revision_ids: list[str] = field(default_factory=list)


def _head(ledger: EvidenceLedger, hid: str) -> str | None:
    current = ledger.current_hypotheses().get(hid)
    return current.revision_id if current else None


def _require_scientific_state(ledger: EvidenceLedger, ctx: Context) -> None:
    history = ledger.history(hypothesis_id(ctx.material))
    if not history or history[0].status != HypothesisStatus.PROPOSED:
        raise ValueError(
            "missing initial scientific hypothesis state; legacy resume is not allowed"
        )
    for hid, current in ledger.current_hypotheses().items():
        if not hid.startswith(f"{CAMPAIGN}:{ctx.material}:"):
            continue
        if (
            not current.falsifiable_prediction
            or not current.falsifiable_prediction.strip()
            or not current.scientific_rationale.strip()
            or not current.evidence_ids
        ):
            raise ValueError(f"incomplete scientific hypothesis state for {hid}")


def _material_evidence(ledger: EvidenceLedger, ctx: Context) -> list[EvidenceRecord]:
    acquired = {d.subject for d in ledger.decisions() if d.action == "freeze_dft_request"}
    return [
        r
        for r in ledger.evidence()
        if r.subject == ctx.material
        or r.subject.startswith(f"{ctx.material}:")
        or r.subject in acquired
    ]


def deterministic_reassessment(
    ledger: EvidenceLedger,
    ctx: Context,
    actor: str,
    phase: str = "round",
    iteration: int | None = None,
) -> Reassessment:
    if phase != "initial" and not _head(ledger, hypothesis_id(ctx.material)):
        raise ValueError(
            "missing initial scientific hypothesis state; legacy resume is not allowed"
        )
    if phase != "initial":
        _require_scientific_state(ledger, ctx)
    if phase == "initial" and _head(ledger, hypothesis_id(ctx.material)):
        raise ValueError("initial scientific hypothesis state already exists")
    splits = ctx.protocol["splits"]
    n_val, n_test = len(ctx.validation_groups), len(ctx.test_groups)
    evidence = _material_evidence(ledger, ctx)
    measured_dft = [
        e for e in evidence if e.kind == "relaxation" and e.outcome == EvidenceOutcome.MEASURED
    ]
    insufficient = n_val < splits["min_validation_groups"] or n_test < splits["min_test_groups"]
    if insufficient:
        rationale = (
            f"Insufficient validation evidence: {n_val} validation and {n_test} "
            "test reference groups "
            f"qualified (protocol requires {splits['min_validation_groups']} and "
            f"{splits['min_test_groups']}). {len(measured_dft)} acquired DFT labels are training "
            "candidates, not validation data, so they cannot decide transfer."
        )
    else:
        rationale = (
            "Validation partition is sufficient; the transfer decision is made by the "
            "validation "
            "metrics against the frozen accuracy criteria after any authorized adaptation."
        )
    hid = hypothesis_id(ctx.material)
    previous = [
        r
        for h, r in ledger.current_hypotheses().items()
        if h.startswith(f"{CAMPAIGN}:{ctx.material}:") and h != hid
    ]
    record = ledger.append(
        HypothesisRecord(
            hypothesis_id=hid,
            prior_revision_id=_head(ledger, hid),
            campaign_id=CAMPAIGN,
            claim_type="other",
            scientific_claim=_transfer_claim(ctx),
            falsifiable_prediction=json.dumps(
                {
                    k: v
                    for k, v in ctx.material_protocol["accuracy_criteria"].items()
                    if isinstance(v, (int, float))
                },
                sort_keys=True,
            ),
            status=HypothesisStatus.PROPOSED if phase == "initial" else HypothesisStatus.UNRESOLVED,
            evidence_ids=[e.evidence_id for e in evidence],
            scientific_rationale=rationale,
            proposed_tests=[
                "qualify >= 2 independent validation groups and >= 1 test group "
                "under the frozen signature"
            ]
            if insufficient
            else [],
            agent_model_id=actor,
        )
    )
    ledger.append(
        DecisionRecord(
            campaign_id=CAMPAIGN,
            iteration=iteration,
            action=f"reassess_hypothesis:{phase}",
            subject=hid,
            evidence_ids=record.evidence_ids,
            actor=actor,
            policy_version=PROMPT_TEMPLATE,
            rationale=json.dumps({"revision_ids": [record.revision_id], "phase": phase}),
        )
    )
    revisions = [record.revision_id]
    for prior in previous:
        revised = ledger.append(
            HypothesisRecord(
                hypothesis_id=prior.hypothesis_id,
                prior_revision_id=prior.revision_id,
                campaign_id=CAMPAIGN,
                claim_type=prior.claim_type,
                scientific_claim=prior.scientific_claim,
                falsifiable_prediction=prior.falsifiable_prediction,
                status=HypothesisStatus.UNRESOLVED,
                evidence_ids=record.evidence_ids,
                scientific_rationale=(
                    "Frozen transfer policy cannot resolve this additional material claim. "
                    "Claim and prediction retained; no numerical scientific support inferred. "
                    + rationale
                ),
                proposed_tests=prior.proposed_tests,
                agent_model_id=actor,
            )
        )
        revisions.append(revised.revision_id)
    return Reassessment(
        True,
        record.status.value,
        record.revision_id,
        None,
        [],
        CostRecord(),
        revisions,
    )


def reassessment_prompt(ledger: EvidenceLedger, ctx: Context, phase: str = "round") -> str:
    evidence = [_evidence_summary(e) for e in _material_evidence(ledger, ctx)]
    criteria = {
        k: v
        for k, v in ctx.material_protocol["accuracy_criteria"].items()
        if isinstance(v, (int, float))
    }
    previous = [
        r.model_dump(mode="json")
        for r in ledger.current_hypotheses().values()
        if r.hypothesis_id.startswith(f"{CAMPAIGN}:{ctx.material}:")
    ]
    return (
        f"You reassess the E3 hypothesis for {ctx.material} after new DFT evidence.\n"
        f"Hypothesis: {_transfer_claim(ctx)}\n"
        f"Frozen scientific domain: {json.dumps(ctx.material_protocol, sort_keys=True)}\n"
        f"Lifecycle phase: {phase}\n"
        f"Previous scientific state: {json.dumps(previous, sort_keys=True)}\n"
        f"Frozen accuracy criteria: {json.dumps(criteria, sort_keys=True)}\n"
        f"Qualified reference groups: validation={len(ctx.validation_groups)} "
        f"test={len(ctx.test_groups)} "
        f"(required validation>={ctx.protocol['splits']['min_validation_groups']} "
        f"test>={ctx.protocol['splits']['min_test_groups']})\n"
        f"Evidence: {json.dumps(evidence)}\n"
        "Acquired DFT labels are training candidates, not validation data.\n"
        "Revise every previous hypothesis using its stable hypothesis_id; do not omit any. "
        "Explain what changed or why it remains unchanged. Separate falsifiable material "
        "claims from optimization/convergence diagnostics. Initial claims are proposed. "
        "MLIP results do not establish DFT accuracy or thermodynamic stability. A transfer "
        "claim cannot be resolved without qualified independent validation/test groups.\n"
        'Reply with JSON only: {"hypotheses": [{"hypothesis_id": str, '
        '"status": one of proposed|supported|contradicted|unresolved, '
        '"scientific_claim": str, "falsifiable_prediction": str, "rationale": str, '
        '"cited_evidence_ids": [ids from Evidence], "proposed_tests": [str]}]}. '
        f"Use {hypothesis_id(ctx.material)} for the initial transfer hypothesis. "
        "Use only numbers that appear above.\n"
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
    required = {
        "hypothesis_id",
        "status",
        "scientific_claim",
        "falsifiable_prediction",
        "rationale",
        "cited_evidence_ids",
        "proposed_tests",
    }
    if set(data) != required:
        problems.append("hypothesis must contain exactly the required fields")
    try:
        status = HypothesisStatus(data.get("status"))
    except ValueError:
        return [f"invalid status {data.get('status')!r}"]
    if status == HypothesisStatus.SUPERSEDED:
        problems.append("superseded is not a reassessment outcome")
    for key in ("hypothesis_id", "scientific_claim", "falsifiable_prediction", "rationale"):
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
        failed = [
            i
            for i in cited
            if isinstance(r := ledger.get(i), EvidenceRecord)
            and r.outcome != EvidenceOutcome.MEASURED
        ]
        if failed:
            problems.append(f"{status.value} claim cites failed or unconverged evidence {failed}")
        non_dft = [
            i
            for i in cited
            if isinstance(r := ledger.get(i), EvidenceRecord)
            and (
                r.method.backend != "vasp"
                or r.kind != "characterization"
                or r.evidence_level
                not in {
                    EvidenceLevel.CONVERGED_DFT,
                    EvidenceLevel.HIGHER_ACCURACY_DFT,
                }
            )
        ]
        if non_dft:
            problems.append("resolved transfer requires qualified DFT validation characterization")
    if not cited:
        problems.append("hypothesis must cite supplied evidence")
    tests = data.get("proposed_tests")
    if not isinstance(tests, list) or not all(isinstance(t, str) and t.strip() for t in tests):
        problems.append("proposed_tests must be a list of non-empty strings")
    text = (
        " ".join(
            str(data.get(k, ""))
            for k in ("scientific_claim", "falsifiable_prediction", "rationale")
        )
        + " "
        + " ".join(map(str, data.get("proposed_tests") or []))
    )
    bad = unsupported_numbers(text, prompt)
    if bad:
        problems.append(f"numbers not in the prompt: {bad}")
    return problems


def _audit(ledger: EvidenceLedger, payload: dict[str, Any]) -> None:
    path = ledger.path.with_name("hypothesis-audit.jsonl")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, default=str, sort_keys=True) + "\n")


def _require_llm_audit(ledger: EvidenceLedger, ctx: Context) -> None:
    """Reject missing, interrupted or rejected LLM revision histories."""
    path = ledger.path.with_name("hypothesis-audit.jsonl")
    if not path.is_file():
        raise ValueError("missing initial LLM scientific audit; cannot resume")
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
    except (ValueError, OSError) as exc:
        raise ValueError("invalid scientific audit; cannot resume") from exc
    acquisition_rows = [row for row in rows if row.get("phase") == "acquisition"]
    if len(acquisition_rows) % 3 or any(
        [r.get("event") for r in acquisition_rows[offset : offset + 3]]
        != ["prompt", "response", "accepted"]
        for offset in range(0, len(acquisition_rows), 3)
    ):
        raise ValueError("rejected or incomplete LLM acquisition audit; cannot resume")
    revision_rows = [
        row for row in rows if row.get("phase") in {"initial", "pre_acquisition", "round", "final"}
    ]
    if len(revision_rows) % 3 or not revision_rows:
        raise ValueError("incomplete LLM scientific audit; cannot resume")
    for offset in range(0, len(revision_rows), 3):
        prompt, response, accepted = revision_rows[offset : offset + 3]
        if (
            [prompt.get("event"), response.get("event"), accepted.get("event")]
            != ["prompt", "response", "accepted"]
            or prompt.get("phase") != response.get("phase")
            or prompt.get("phase") != accepted.get("phase")
            or not accepted.get("revision_ids")
            or any(
                not isinstance(ledger.get(rid), HypothesisRecord)
                for rid in accepted["revision_ids"]
            )
        ):
            raise ValueError("rejected or incomplete LLM scientific audit; cannot resume")
    initial = revision_rows[2]
    initial_history = ledger.history(hypothesis_id(ctx.material))
    if (
        revision_rows[0].get("phase") != "initial"
        or not initial_history
        or initial_history[0].revision_id not in initial["revision_ids"]
    ):
        raise ValueError("initial hypothesis is not linked to the LLM scientific audit")


def llm_reassessment(
    ledger: EvidenceLedger,
    ctx: Context,
    chat: Any,
    actor: str,
    iteration: int,
    phase: str = "round",
) -> Reassessment:
    if chat is None:
        raise ValueError("LLM hypothesis revision requires a chat model; no deterministic fallback")
    previous = {
        hid: r
        for hid, r in ledger.current_hypotheses().items()
        if hid.startswith(f"{CAMPAIGN}:{ctx.material}:")
    }
    if phase != "initial" and not previous:
        raise ValueError(
            "missing initial scientific hypothesis state; legacy resume is not allowed"
        )
    if phase != "initial":
        _require_scientific_state(ledger, ctx)
        _require_llm_audit(ledger, ctx)
        audit_path = ledger.path.with_name("hypothesis-audit.jsonl")
        if any(
            row.get("event") == "accepted"
            and row.get("phase") == phase
            and row.get("iteration") == iteration
            for line in audit_path.read_text().splitlines()
            for row in [json.loads(line)]
        ):
            raise ValueError(f"{phase} revision for iteration {iteration} already exists")
    if phase == "initial" and previous:
        raise ValueError("initial scientific hypothesis state already exists")
    prompt = reassessment_prompt(ledger, ctx, phase)
    audit = {
        "phase": phase,
        "iteration": iteration,
        "actor": actor,
        "prompt": prompt,
        "previous": [r.model_dump(mode="json") for r in previous.values()],
        "evidence": [r.model_dump(mode="json") for r in _material_evidence(ledger, ctx)],
    }
    _audit(ledger, {**audit, "event": "prompt"})
    started = time.perf_counter()
    try:
        response = chat.invoke(prompt)
    except Exception as exc:
        _audit(ledger, {**audit, "event": "invoke_error", "error": str(exc)})
        raise
    seconds = time.perf_counter() - started
    usage = getattr(response, "usage_metadata", None) or {}
    cost = CostRecord(
        llm_prompt_tokens=int(usage.get("input_tokens", 0)),
        llm_completion_tokens=int(usage.get("output_tokens", 0)),
        llm_seconds=seconds,
    )
    text = getattr(response, "content", response)
    _audit(ledger, {**audit, "event": "response", "raw_response": text, "usage": usage})
    try:
        data = _parse(text if isinstance(text, str) else json.dumps(text))
        rows = data.get("hypotheses")
        if set(data) != {"hypotheses"} or not isinstance(rows, list) or not rows:
            raise ValueError("reply requires a non-empty hypotheses list")
        problems = []
        seen = set()
        allowed = {r.evidence_id for r in _material_evidence(ledger, ctx)}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("hypothesis must be an object")
            problems.extend(validate_reply(row, ledger, prompt))
            hid = row.get("hypothesis_id")
            if (
                not isinstance(hid, str)
                or not hid.startswith(f"{CAMPAIGN}:{ctx.material}:")
                or hid in seen
            ):
                problems.append("invalid or duplicate stable hypothesis_id")
            seen.add(hid)
            if not set(row.get("cited_evidence_ids") or []) <= allowed:
                problems.append("hypothesis cites evidence outside the supplied material context")
            if phase == "initial" and row.get("status") != "proposed":
                problems.append("initial scientific hypotheses must be proposed")
            if row.get("status") in {"supported", "contradicted"} and (
                len(ctx.validation_groups) < ctx.protocol["splits"]["min_validation_groups"]
                or len(ctx.test_groups) < ctx.protocol["splits"]["min_test_groups"]
            ):
                problems.append(
                    "insufficient independent validation/test groups to resolve transfer"
                )
        if not set(previous) <= seen or hypothesis_id(ctx.material) not in seen:
            problems.append("hypotheses omit previous stable IDs or the transfer hypothesis")
    except (ValueError, TypeError, KeyError) as exc:
        data, problems = {}, [f"unparseable reply: {exc}"]
    hid = hypothesis_id(ctx.material)
    cited = [e.evidence_id for e in _material_evidence(ledger, ctx)]
    if problems:
        _audit(ledger, {**audit, "event": "rejected", "raw_response": text, "problems": problems})
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
    records = []
    for row in rows:
        records.append(
            ledger.append(
                HypothesisRecord(
                    hypothesis_id=row["hypothesis_id"],
                    prior_revision_id=_head(ledger, row["hypothesis_id"]),
                    campaign_id=CAMPAIGN,
                    claim_type="other",
                    scientific_claim=row["scientific_claim"].strip(),
                    falsifiable_prediction=row["falsifiable_prediction"].strip(),
                    status=HypothesisStatus(row["status"]),
                    evidence_ids=list(dict.fromkeys(row["cited_evidence_ids"])),
                    scientific_rationale=row["rationale"].strip(),
                    proposed_tests=row["proposed_tests"],
                    agent_model_id=actor,
                )
            )
        )
    record = next(r for r in records if r.hypothesis_id == hypothesis_id(ctx.material))
    decision = ledger.append(
        DecisionRecord(
            campaign_id=CAMPAIGN,
            iteration=iteration,
            action=f"reassess_hypothesis:{phase}",
            subject=hid,
            evidence_ids=record.evidence_ids or cited,
            rationale=json.dumps(
                {"revision_ids": [r.revision_id for r in records], "phase": phase}
            ),
            actor=actor,
            policy_version=PROMPT_TEMPLATE,
            observed_cost=cost,
        )
    )
    _audit(ledger, {**audit, "event": "accepted", "revision_ids": [r.revision_id for r in records]})
    return Reassessment(
        True,
        record.status.value,
        record.revision_id,
        decision.decision_id,
        [],
        cost,
        [r.revision_id for r in records],
    )


def reassess_round(
    state: Any, ctx: Context, ledger_dir: Any, chat: Any = None, llm_actor: str | None = None
) -> dict[str, Reassessment]:
    """Reassess each strategy whose round is fully ingested; the rule arm is deterministic.

    The LLM arm requires a model. Missing initial/previous rounds are not backfilled.
    """
    from benchmarks.e3.controller import ledger_path
    from benchmarks.e3.run import LLM
    from benchmarks.e3.state import E3State

    record = next((r for r in state.rounds if r.round == state.current_round), None)
    if record is None:
        raise ValueError("missing current scientific round state")
    for strategy, st in record.strategies.items():
        if st.state in {E3State.REASSESSED, E3State.SCIENTIFICALLY_UNRESOLVED}:
            ledger = EvidenceLedger(ledger_path(ledger_dir, strategy))
            _require_scientific_state(ledger, ctx)
            if strategy == LLM:
                _require_llm_audit(ledger, ctx)
            if strategy == LLM:
                _require_llm_audit(ledger, ctx)
            if (
                not st.hypothesis_revision_ids
                or not st.reassessment_status
                or any(
                    not isinstance(ledger.get(rid), HypothesisRecord)
                    for rid in st.hypothesis_revision_ids
                )
            ):
                raise ValueError(f"{strategy}: incomplete scientific round state")
            heads = [
                r.revision_id
                for hid, r in ledger.current_hypotheses().items()
                if hid.startswith(f"{CAMPAIGN}:{ctx.material}:")
            ]
            if any(rid not in st.hypothesis_revision_ids for rid in heads):
                raise ValueError(f"{strategy}: current scientific state is not recorded")
            if strategy == LLM:
                path = ledger.path.with_name("hypothesis-audit.jsonl")
                accepted = [
                    row
                    for line in path.read_text().splitlines()
                    if (row := json.loads(line)).get("event") == "accepted"
                ]
                required = {"round"}
                if state.current_round == int(ctx.protocol["budget"]["rounds"]):
                    required.add("final")
                for phase in required:
                    if not any(
                        row.get("phase") == phase
                        and row.get("iteration") == state.current_round
                        and set(row.get("revision_ids", [])) <= set(st.hypothesis_revision_ids)
                        for row in accepted
                    ):
                        raise ValueError(f"{strategy}: missing accepted {phase} audit")
    if (
        any(
            st.state == E3State.EVIDENCE_INGESTED for s, st in record.strategies.items() if s == LLM
        )
        and chat is None
    ):
        raise ValueError("LLM reassessment is mandatory and requires a chat model")
    for strategy, st in record.strategies.items():
        if st.state != E3State.EVIDENCE_INGESTED:
            continue
        ledger = EvidenceLedger(ledger_path(ledger_dir, strategy))
        _require_scientific_state(ledger, ctx)
        for number in range(1, state.current_round):
            prior_round = next((r for r in state.rounds if r.round == number), None)
            prior = prior_round.strategies.get(strategy) if prior_round else None
            if (
                prior is None
                or not prior.hypothesis_revision_ids
                or not prior.reassessment_status
                or any(
                    not isinstance(ledger.get(rid), HypothesisRecord)
                    for rid in prior.hypothesis_revision_ids
                )
                or not any(
                    d.action == "reassess_hypothesis:round" and d.iteration == number
                    for d in ledger.decisions()
                )
            ):
                raise ValueError(f"{strategy}: missing scientific revision for round {number}")
    results = {}
    for strategy, st in record.strategies.items():
        if st.state != E3State.EVIDENCE_INGESTED:
            continue
        ledger = EvidenceLedger(ledger_path(ledger_dir, strategy))
        if strategy == LLM:
            result = llm_reassessment(
                ledger, ctx, chat, llm_actor or f"{LLM}:llm", state.current_round
            )
        else:
            result = deterministic_reassessment(
                ledger,
                ctx,
                f"{strategy}:{PROMPT_TEMPLATE}-deterministic",
                iteration=state.current_round,
            )
        results[strategy] = result
        if result.accepted:
            for revision_id in result.revision_ids:
                if revision_id not in st.hypothesis_revision_ids:
                    st.hypothesis_revision_ids.append(revision_id)
            st.reassessment_status = result.status
            st.state = (
                E3State.SCIENTIFICALLY_UNRESOLVED
                if result.status == HypothesisStatus.UNRESOLVED.value
                else E3State.REASSESSED
            )
            if state.current_round == int(ctx.protocol["budget"]["rounds"]):
                final = (
                    llm_reassessment(
                        ledger, ctx, chat, llm_actor or f"{LLM}:llm", state.current_round, "final"
                    )
                    if strategy == LLM
                    else deterministic_reassessment(
                        ledger,
                        ctx,
                        strategy,
                        "final",
                        iteration=state.current_round,
                    )
                )
                if not final.accepted:
                    st.state = E3State.EVIDENCE_INGESTED
                    st.reassessment_status = None
                    st.rejected_reassessments.append(final.decision_id)
                    results[strategy] = final
                else:
                    st.hypothesis_revision_ids.extend(final.revision_ids)
                    st.reassessment_status = final.status
                    st.state = (
                        E3State.SCIENTIFICALLY_UNRESOLVED
                        if final.status == HypothesisStatus.UNRESOLVED.value
                        else E3State.REASSESSED
                    )
                    results[strategy] = final
        elif result.decision_id not in st.rejected_reassessments:
            st.rejected_reassessments.append(result.decision_id)
        cost = ledger.cost_summary()
        st.llm_prompt_tokens = int(cost["llm_prompt_tokens"])
        st.llm_completion_tokens = int(cost["llm_completion_tokens"])
        st.llm_seconds = float(cost["llm_seconds"])
    return results
