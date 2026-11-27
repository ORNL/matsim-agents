"""Persistent hypothesis, evidence, and revision contracts for agentic studies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from matsim_agents.backends.llm.hypotheses import INSTRUCTIONS as HYPOTHESIS_INSTRUCTIONS
from matsim_agents.backends.llm.hypotheses import validate_hypotheses
from matsim_agents.backends.llm.provider import get_chat_model
from matsim_agents.backends.llm.replies import invoke_model
from matsim_agents.backends.llm.servers import api_key_from_env
from matsim_agents.backends.llm.transcript import Contribution, Transcript, prompt_messages
from matsim_agents.core.contracts import (
    ComputeBudget,
    EvidenceLevel,
    ProvenanceRecord,
    WorkflowStatus,
)
from matsim_agents.execution.run_directory import ScientificRunDirectory
from matsim_agents.workflows.debate import (
    DebateParticipant,
    ScientificDebateConfig,
    run_scientific_debate,
)
from matsim_agents.workflows.exploration.phase import (
    PhaseExplorationPolicy,
    PhaseExplorationWorkflowResult,
)


class PropertyTask(BaseModel):
    property_name: str
    method: str
    success_criterion: str
    required_fidelity: EvidenceLevel
    rationale: str


class ScientificHypothesis(BaseModel):
    objective: str
    hypothesis: str
    proposed_compositions: list[str]
    scientific_rationale: str
    assumptions: list[str] = Field(default_factory=list)
    property_tasks: list[PropertyTask] = Field(default_factory=list)
    estimated_cost: str | None = None
    parent_run_id: str | None = None
    discussion_mode: Literal["single_llm", "multi_llm_debate"] | None = None
    discussion_run_id: str | None = None
    source_contribution_ids: list[str] = Field(default_factory=list)
    scientific_hypotheses: list[dict[str, Any]] = Field(default_factory=list)


class HypothesisRevision(BaseModel):
    parent_run_id: str
    disposition: Literal["supported", "rejected", "partially_supported", "inconclusive"]
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    remaining_gaps: list[str] = Field(default_factory=list)
    revised_hypothesis: ScientificHypothesis


class SingleLLMDiscussionConfig(BaseModel):
    provider: str = "ollama"
    model: str | None = None
    base_url: str | None = None
    max_completion_tokens: int | None = Field(None, gt=0)
    api_key_env: str | None = Field(None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


class MultiLLMDebateDiscussionConfig(BaseModel):
    participants: list[DebateParticipant] = Field(min_length=2)
    rounds: int = Field(2, ge=1, le=100)
    debate_mode: Literal["equal", "role_based"] = "equal"
    synthesis_method: Literal["independent_verdicts", "designated_model"] = "independent_verdicts"
    synthesis_participant: str | None = None
    max_transcript_chars: int = Field(60_000, ge=1_000)


class HypothesisDiscussionConfig(BaseModel):
    mode: Literal["single_llm", "multi_llm_debate"] = "single_llm"
    single_llm: SingleLLMDiscussionConfig = Field(default_factory=SingleLLMDiscussionConfig)
    multi_llm_debate: MultiLLMDebateDiscussionConfig | None = None

    @model_validator(mode="after")
    def _required_mode_configuration(self) -> HypothesisDiscussionConfig:
        if self.mode == "multi_llm_debate" and self.multi_llm_debate is None:
            raise ValueError("mode=multi_llm_debate requires multi_llm_debate configuration")
        return self


class InvestigationConfig(BaseModel):
    objective: str
    output_root: str = "./runs"
    parent_run: str | None = None
    phase_policy: PhaseExplorationPolicy = Field(default_factory=PhaseExplorationPolicy)
    budget: ComputeBudget = Field(default_factory=ComputeBudget)
    hypothesis_discussion: HypothesisDiscussionConfig | None = None


class InvestigationResult(BaseModel):
    run_id: str
    run_directory: str
    status: WorkflowStatus
    hypothesis: ScientificHypothesis
    explorations: list[PhaseExplorationWorkflowResult] = Field(default_factory=list)
    report_path: str


_HYPOTHESIS_JSON_INSTRUCTION = """Return JSON only with these proposal fields, plus the required
scientific_hypotheses field:
{
  "hypothesis": "testable scientific claim",
  "proposed_compositions": ["valid chemical formula", "..."],
  "scientific_rationale": "reasoning and relevant mechanisms",
  "assumptions": ["assumption", "..."]
}
Do not wrap the JSON in Markdown. Propose at least one composition."""
_REVISION_INSTRUCTION = (
    f"{HYPOTHESIS_INSTRUCTIONS}\n"
    "Include scientific_hypotheses in the same JSON object as the investigation proposal. "
    "The stable evidence ID `investigation-objective` identifies the requested research "
    "objective and is not a numerical observation. Cite only supplied evidence IDs. "
    "Never describe peer opinions, MLIP predictions, or convergence as DFT evidence."
)


def _previous_hypotheses(previous: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not previous:
        return []
    direct = previous.get("scientific_hypotheses")
    if isinstance(direct, list):
        return direct
    hypothesis = previous.get("hypothesis")
    if isinstance(hypothesis, dict) and isinstance(hypothesis.get("scientific_hypotheses"), list):
        return hypothesis["scientific_hypotheses"]
    return []


def _previous_evidence(previous: dict[str, Any] | None) -> tuple[set[str], set[str], set[str]]:
    """Return only explicitly scoped evidence identifiers from a prior result."""
    valid: set[str] = set()
    mlip: set[str] = set()
    dft: set[str] = set()
    if not previous:
        return valid, mlip, dft
    records = previous.get("evidence_records", previous.get("evidence", []))
    if isinstance(records, list):
        for item in records:
            if not isinstance(item, dict):
                continue
            evidence_id = item.get("evidence_id") or item.get("id")
            if not isinstance(evidence_id, str):
                continue
            valid.add(evidence_id)
            if str(item.get("outcome", "")).lower() != "measured":
                continue
            scope = " ".join(
                str(item.get(key, "")) for key in ("evidence_level", "method", "backend")
            ).lower()
            if "dft" in scope:
                dft.add(evidence_id)
            elif "mlip" in scope or "mace" in scope:
                mlip.add(evidence_id)
    for reference in _previous_exploration_evidence(previous):
        evidence_id = reference["evidence_id"]
        valid.add(evidence_id)
        mlip.add(evidence_id)
    return valid, mlip, dft


def _previous_exploration_evidence(previous: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Assign stable MLIP-scoped citations to measured, converged prior relaxations."""
    if not previous or not isinstance(previous.get("explorations"), list):
        return []
    references = []
    for exploration in previous["explorations"]:
        if not isinstance(exploration, dict):
            continue
        composition = exploration.get("composition")
        if isinstance(composition, dict):
            composition = composition.get("formula")
        for stage in ("initial", "after_retraining"):
            result = exploration.get(stage)
            if not isinstance(result, dict):
                continue
            relaxations = result.get("relaxations", [])
            if not isinstance(relaxations, list):
                continue
            for relaxation in relaxations:
                if (
                    not isinstance(relaxation, dict)
                    or not relaxation.get("converged")
                    or relaxation.get("final_energy_eV") is None
                ):
                    continue
                identity = json.dumps(
                    [composition, stage, relaxation], sort_keys=True, separators=(",", ":")
                )
                digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
                references.append(
                    {
                        "evidence_id": f"prior-mlip-{digest}",
                        "scope": "measured converged MLIP relaxation; not DFT",
                        "composition": composition,
                        "stage": stage,
                        "relaxation": relaxation,
                    }
                )
    return references


def _evidence_context(previous: dict[str, Any] | None) -> tuple[set[str], set[str], set[str]]:
    valid, mlip, dft = _previous_evidence(previous)
    valid.add("investigation-objective")
    for row in _previous_hypotheses(previous):
        if isinstance(row, dict):
            ids = row.get("evidence_ids", [])
            if isinstance(ids, list):
                valid.update(item for item in ids if isinstance(item, str))
    return valid, mlip, dft


def _parse_hypothesis_proposal(text: str, objective: str) -> ScientificHypothesis:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError("LLM hypothesis proposal is not valid JSON") from error
    compositions = payload.get("proposed_compositions")
    if not isinstance(compositions, list) or not compositions:
        raise ValueError("LLM hypothesis proposal must contain proposed_compositions")
    return ScientificHypothesis(
        objective=objective,
        hypothesis=str(payload["hypothesis"]),
        proposed_compositions=[str(value) for value in compositions],
        scientific_rationale=str(payload["scientific_rationale"]),
        assumptions=[str(value) for value in payload.get("assumptions", [])],
    )


def build_hypothesis_from_discussion(
    objective: str,
    discussion: HypothesisDiscussionConfig,
    *,
    output_root: str,
    previous: dict[str, Any] | None = None,
    model_factory: Callable[..., Any] = get_chat_model,
) -> ScientificHypothesis:
    """Produce the investigation hypothesis using one LLM or a reusable debate."""

    previous_state = _previous_hypotheses(previous)
    valid_evidence, mlip_evidence, dft_evidence = _evidence_context(previous)
    context = (
        "\n\nPrevious investigation results and evidence:\n"
        f"{json.dumps(previous, sort_keys=True)}\n\n"
        "Previous scientific hypothesis state (retain every stable ID):\n"
        f"{json.dumps(previous_state, sort_keys=True)}\n\n"
        "Previous evidence references with stable IDs and method scope:\n"
        f"{json.dumps(_previous_exploration_evidence(previous), sort_keys=True)}"
        if previous
        else ""
    )
    if discussion.mode == "single_llm":
        spec = discussion.single_llm
        api_key = api_key_from_env(spec.api_key_env, owner="single-LLM hypothesis generation")
        model = model_factory(
            provider=spec.provider,
            model=spec.model,
            base_url=spec.base_url,
            **({"api_key": api_key} if api_key is not None else {}),
            **(
                {"max_tokens": spec.max_completion_tokens}
                if spec.max_completion_tokens is not None
                else {}
            ),
        )
        messages = [
            SystemMessage(
                content="Formulate a testable materials-science hypothesis. "
                + _HYPOTHESIS_JSON_INSTRUCTION
                + "\n\n"
                + _REVISION_INSTRUCTION
            ),
            HumanMessage(
                content=(
                    f"Scientific objective:\n{objective}{context}\n\n"
                    f"Available evidence IDs (MLIP: {sorted(mlip_evidence)}; "
                    f"DFT: {sorted(dft_evidence)}): {sorted(valid_evidence)}"
                )
            ),
        ]
        discussion_run = ScientificRunDirectory.create(
            output_root,
            workflow="single_llm_hypothesis_discussion",
            request={"objective": objective, "previous": previous},
            resolved_config=spec.model_dump(mode="json"),
            provenance=ProvenanceRecord(
                workflow="single_llm_hypothesis_discussion",
                evidence_level=EvidenceLevel.HYPOTHESIS,
            ),
            prefix="hypothesis",
            directories=(),
        )
        transcript = Transcript(
            discussion_run.path,
            title="Single-LLM hypothesis discussion",
            preamble=f"Run `{discussion_run.run_id}`.\n\n**Objective.** {objective}",
        )
        prompt_path = discussion_run.path / "prompts" / "proposal-001.json"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(
            json.dumps(prompt_messages(messages), indent=2) + "\n", encoding="utf-8"
        )
        try:
            reply = invoke_model(model, messages)
        except Exception as error:
            discussion_run.append_event("model_call_failed", {"error": str(error)})
            raise
        transcript.add(
            Contribution.from_reply(
                reply,
                id="proposal-001",
                kind="proposal",
                speaker="proposer",
                provider=spec.provider,
                model=spec.model,
                context=[],
            ),
            prompt=messages,
        )
        discussion_run.append_event("contribution_completed", {"id": "proposal-001"})
        if reply.truncated or not reply.text.strip():
            discussion_run.append_event(
                "hypothesis_validation_failed",
                {"id": "proposal-001", "error": "reply is incomplete"},
            )
            raise ValueError("incomplete scientific hypothesis proposal")
        try:
            payload = json.loads(reply.text)
            if not isinstance(payload, dict) or "scientific_hypotheses" not in payload:
                raise ValueError("LLM proposal must include scientific_hypotheses")
            state = validate_hypotheses(
                payload["scientific_hypotheses"],
                valid_evidence,
                previous_state,
                mlip_evidence=mlip_evidence,
                dft_evidence=dft_evidence,
            )
            hypothesis = _parse_hypothesis_proposal(reply.text, objective)
        except Exception as error:
            discussion_run.append_event(
                "hypothesis_validation_failed",
                {"id": "proposal-001", "error": str(error)},
            )
            raise
        hypothesis.discussion_mode = "single_llm"
        hypothesis.discussion_run_id = discussion_run.run_id
        hypothesis.scientific_hypotheses = state
        return hypothesis

    spec = discussion.multi_llm_debate
    if spec is None:
        raise ValueError("mode=multi_llm_debate requires multi_llm_debate configuration")
    debate = run_scientific_debate(
        ScientificDebateConfig(
            hypothesis=f"Scientific objective:\n{objective}{context}",
            participants=spec.participants,
            rounds=spec.rounds,
            output_root=str(output_root),
            debate_mode=spec.debate_mode,
            synthesis_method=spec.synthesis_method,
            synthesis_participant=spec.synthesis_participant,
            max_transcript_chars=spec.max_transcript_chars,
            final_response_instruction=_HYPOTHESIS_JSON_INSTRUCTION,
            evidence_ids=sorted(valid_evidence),
            mlip_evidence_ids=sorted(mlip_evidence),
            dft_evidence_ids=sorted(dft_evidence),
            initial_hypotheses=previous_state,
        ),
        model_factory=model_factory,
    )
    proposals = [
        _parse_hypothesis_proposal(verdict.response, objective) for verdict in debate.verdicts
    ]
    compositions = list(
        dict.fromkeys(
            composition for proposal in proposals for composition in proposal.proposed_compositions
        )
    )
    assumptions = list(
        dict.fromkeys(assumption for proposal in proposals for assumption in proposal.assumptions)
    )
    hypothesis = ScientificHypothesis(
        objective=objective,
        hypothesis="\n\n".join(
            f"[{verdict.participant}] {proposal.hypothesis}"
            for verdict, proposal in zip(debate.verdicts, proposals, strict=True)
        ),
        proposed_compositions=compositions,
        scientific_rationale="\n\n".join(
            f"[{verdict.participant}] {proposal.scientific_rationale}"
            for verdict, proposal in zip(debate.verdicts, proposals, strict=True)
        ),
        assumptions=assumptions,
        discussion_mode="multi_llm_debate",
        discussion_run_id=debate.run_id,
        source_contribution_ids=[verdict.contribution_id for verdict in debate.verdicts],
        scientific_hypotheses=debate.scientific_hypotheses,
    )
    return hypothesis


def run_investigation(
    cfg: InvestigationConfig,
    *,
    phase_runner: Callable[[str, PhaseExplorationPolicy, str], PhaseExplorationWorkflowResult],
    hypothesis_builder: Callable[[str, dict[str, Any] | None], ScientificHypothesis] | None = None,
    model_factory: Callable[..., Any] = get_chat_model,
) -> InvestigationResult:
    """Compose hypothesis generation with the reusable phase workflow."""

    previous = None
    parent_id = None
    if cfg.parent_run:
        parent = ScientificRunDirectory.open(cfg.parent_run)
        parent_id = parent.run_id
        results_path = parent.path / "results.json"
        previous = (
            __import__("json").loads(results_path.read_text()) if results_path.exists() else None
        )
    if hypothesis_builder is not None:
        hypothesis = hypothesis_builder(cfg.objective, previous)
    elif cfg.hypothesis_discussion is not None:
        hypothesis = build_hypothesis_from_discussion(
            cfg.objective,
            cfg.hypothesis_discussion,
            output_root=str(Path(cfg.output_root) / "discussions"),
            previous=previous,
            model_factory=model_factory,
        )
    else:
        raise ValueError("configure hypothesis_discussion or provide hypothesis_builder")
    hypothesis.parent_run_id = parent_id
    provenance = ProvenanceRecord(
        workflow="agentic_investigation",
        evidence_level=EvidenceLevel.HYPOTHESIS,
        parent_run_id=parent_id,
    )
    run = ScientificRunDirectory.create(
        cfg.output_root,
        workflow="agentic_investigation",
        request={"objective": cfg.objective, "parent_run": cfg.parent_run},
        resolved_config=cfg.model_dump(mode="json"),
        provenance=provenance,
    )
    compositions = hypothesis.proposed_compositions
    if cfg.budget.max_candidates is not None:
        compositions = compositions[: cfg.budget.max_candidates]
    explorations = [
        phase_runner(comp, cfg.phase_policy, str(run.path / "calculations" / comp))
        for comp in compositions
    ]
    report = run.path / "report.md"
    lines = [
        f"# Investigation: {cfg.objective}",
        "",
        hypothesis.hypothesis,
        "",
        "## Candidates",
        "",
    ]
    for result in explorations:
        lines.append(
            f"- {result.composition}: {len(result.initial.phase_candidates)} phase candidate(s)"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = InvestigationResult(
        run_id=run.run_id,
        run_directory=str(run.path),
        status=WorkflowStatus.COMPLETE,
        hypothesis=hypothesis,
        explorations=explorations,
        report_path=str(report),
    )
    run.write_json("hypothesis.json", hypothesis)
    run.write_json("results.json", result)
    return result


__all__ = [
    "HypothesisDiscussionConfig",
    "HypothesisRevision",
    "InvestigationConfig",
    "InvestigationResult",
    "MultiLLMDebateDiscussionConfig",
    "PropertyTask",
    "ScientificHypothesis",
    "SingleLLMDiscussionConfig",
    "build_hypothesis_from_discussion",
    "run_investigation",
]
