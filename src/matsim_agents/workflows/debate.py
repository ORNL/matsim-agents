"""Persistent multi-model scientific hypothesis debate workflow."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from matsim_agents.backends.llm.provider import get_chat_model
from matsim_agents.backends.llm.replies import invoke_model
from matsim_agents.backends.llm.servers import api_key_from_env, llm_endpoints
from matsim_agents.backends.llm.transcript import Contribution, Transcript
from matsim_agents.execution.contracts import EvidenceLevel, ProvenanceRecord, WorkflowStatus
from matsim_agents.execution.placement import ResourcePlan
from matsim_agents.execution.run_directory import ScientificRunDirectory

log = logging.getLogger(__name__)


class DebateParticipant(BaseModel):
    """One independently configured model and its scientific perspective."""

    name: str = Field(min_length=1)
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    server: str | None = Field(
        None,
        min_length=1,
        description=(
            "Name of a server in the llm pool of the debate's resources: plan (or of "
            "the plan served by `matsim-agents resources serve`); replaces provider, "
            "model, and base_url."
        ),
    )
    role: str = "independent scientific reviewer"
    max_completion_tokens: int | None = Field(
        None, gt=0, description="Overrides the debate's max_completion_tokens for this model."
    )
    api_key_env: str | None = Field(
        None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Environment variable holding this participant's API key; never recorded.",
    )

    @model_validator(mode="after")
    def _model_source(self) -> DebateParticipant:
        if self.server is not None:
            if set_fields := [f for f in ("provider", "model", "base_url") if getattr(self, f)]:
                raise ValueError(
                    f"participant {self.name!r} names server {self.server!r}; remove "
                    f"{', '.join(set_fields)}"
                )
        elif not (self.provider and self.model):
            raise ValueError(f"participant {self.name!r} needs provider and model, or server")
        return self


class ScientificDebateConfig(BaseModel):
    hypothesis: str = Field(min_length=1)
    participants: list[DebateParticipant] = Field(min_length=1)
    rounds: int = Field(2, ge=1, le=100)
    output_root: str = "./runs"
    debate_mode: Literal["equal", "role_based"] = "equal"
    synthesis_method: Literal["independent_verdicts", "designated_model"] = "independent_verdicts"
    synthesis_participant: str | None = None
    max_transcript_chars: int = Field(
        60_000,
        ge=1_000,
        description=(
            "Most characters of earlier answers shown in one prompt; the oldest whole "
            "contributions beyond it are left out and named."
        ),
    )
    max_completion_tokens: int | None = Field(
        8192,
        gt=0,
        description=(
            "Most tokens one reply may generate, including a reasoning model's reasoning; "
            "None uses the provider's default."
        ),
    )
    final_response_instruction: str | None = None
    single_call: bool = False
    resources: ResourcePlan | None = Field(
        None,
        description="Plan whose llm pool serves the participants that name a server.",
    )

    @model_validator(mode="after")
    def _validate_participants(self) -> ScientificDebateConfig:
        names = [participant.name for participant in self.participants]
        if len(names) != len(set(names)):
            raise ValueError("debate participant names must be unique")
        if self.synthesis_participant is not None and self.synthesis_participant not in names:
            raise ValueError("synthesis_participant must name a configured participant")
        if self.synthesis_method == "independent_verdicts" and self.synthesis_participant:
            raise ValueError(
                "synthesis_participant is only valid with synthesis_method=designated_model"
            )
        if self.single_call and (len(self.participants) != 1 or self.rounds != 1):
            raise ValueError("single_call requires exactly one participant and one round")
        if not self.single_call and len(self.participants) < 2:
            raise ValueError("scientific debate requires at least two participants")
        if self.resources is not None and not any(p.server is not None for p in self.participants):
            raise ValueError("resources: is set, but no participant names a server")
        return self


class DebateTurn(BaseModel):
    turn_id: str
    round: int
    participant: str
    provider: str
    model: str
    response: str = Field(description="The answer, without the model's reasoning.")
    incomplete: bool = Field(
        False, description="The token limit cut the reply off, or it has no answer."
    )


class DebateVerdict(BaseModel):
    contribution_id: str
    participant: str
    provider: str
    model: str
    response: str = Field(description="The answer, without the model's reasoning.")
    incomplete: bool = Field(
        False, description="The token limit cut the reply off, or it has no answer."
    )


class ScientificDebateResult(BaseModel):
    run_id: str
    run_directory: str
    status: WorkflowStatus
    hypothesis: str
    rounds_completed: int
    turns: list[DebateTurn]
    verdicts: list[DebateVerdict]
    synthesis: str
    synthesis_turn_id: str | None = None
    incomplete_contributions: list[str] = Field(default_factory=list)
    transcript_path: str
    contributions_path: str


_NO_ANSWER = "[No answer: the token limit ended this reply during the model's reasoning.]"
_CUT_OFF = "[The token limit cut this reply off here.]"


def _contribution_block(contribution_id: str, speaker: str, text: str, incomplete: bool) -> str:
    body = text.strip() or _NO_ANSWER
    if incomplete and text.strip():
        body += f"\n{_CUT_OFF}"
    return f'<contribution id="{contribution_id}" speaker="{speaker}">\n{body}\n</contribution>'


def _peer_context(turns: list[DebateTurn], limit: int) -> tuple[str, list[str], list[str]]:
    """Earlier answers a speaker is shown, as ``(text, shown ids, omitted ids)``.

    Contributions are kept whole, newest first, until ``limit`` characters; the
    oldest ones that do not fit are left out and named. A newest contribution
    longer than the limit is shortened, with a marker.
    """

    blocks: list[tuple[str, str]] = []
    used = 0
    for turn in reversed(turns):
        block = _contribution_block(turn.turn_id, turn.participant, turn.response, turn.incomplete)
        if blocks and used + len(block) > limit:
            break
        if len(block) > limit:
            block = block[: limit - 60] + "\n[... shortened to fit the prompt ...]\n</contribution>"
        blocks.append((turn.turn_id, block))
        used += len(block)
    blocks.reverse()
    shown = [turn_id for turn_id, _ in blocks]
    omitted = [turn.turn_id for turn in turns[: len(turns) - len(blocks)]]
    text = "\n\n".join(block for _, block in blocks)
    if omitted:
        text = (
            f"[{len(omitted)} earlier contributions are not shown, to keep this prompt within "
            f"its budget: {', '.join(omitted)}]\n\n{text}"
        )
    return text, shown, omitted


def run_scientific_debate(
    cfg: ScientificDebateConfig,
    *,
    model_factory: Callable[..., Any] = get_chat_model,
) -> ScientificDebateResult:
    """Run the debate; participants that name a server use its endpoint.

    Servers come from ``cfg.resources`` (started for the debate and stopped
    afterwards) or from the plan served by ``matsim-agents resources serve``.
    """

    if not any(participant.server is not None for participant in cfg.participants):
        return _run_scientific_debate(cfg, model_factory=model_factory)
    with llm_endpoints(cfg.resources, output_dir=cfg.output_root) as services:
        participants = []
        for participant in cfg.participants:
            if participant.server is None:
                participants.append(participant)
                continue
            endpoint = services.endpoint(participant.server)
            participants.append(
                participant.model_copy(
                    update={
                        "provider": endpoint.provider,
                        "model": endpoint.model,
                        "base_url": endpoint.base_url,
                        "api_key_env": participant.api_key_env or endpoint.api_key_env,
                        # Saved configs record the resolved endpoint, which reloads on its own.
                        "server": None,
                    }
                )
            )
        resolved = cfg.model_copy(update={"participants": participants, "resources": None})
        return _run_scientific_debate(resolved, model_factory=model_factory)


_EQUAL_INSTRUCTION = (
    "You are one equal member of a scientific debate panel. No participant has greater "
    "authority or a privileged role. Independently evaluate the hypothesis, explicitly "
    "identify peer claims you support or dispute, expose assumptions, propose falsification "
    "tests, and update your position when warranted. Do not seek superficial consensus. "
    "Distinguish evidence from speculation."
)
_VERDICT_INSTRUCTION = (
    "Give your independent final assessment of this scientific debate. Report supported "
    "points, unresolved disputes, strongest evidence, assumptions, decisive tests, and a "
    "calibrated verdict. Do not claim to speak for the other panel members or manufacture "
    "consensus."
)
_OWN_WORDS = (
    "Peer contributions are quoted between <contribution> tags. Write only your own "
    "statement: refer to peers by name, and do not copy or continue their text."
)


def _turn_instruction(cfg: ScientificDebateConfig, participant: DebateParticipant) -> str:
    if cfg.debate_mode == "equal":
        instruction = _EQUAL_INSTRUCTION
    else:
        instruction = (
            f"You are {participant.name}, acting as {participant.role}. Debate the "
            "scientific hypothesis independently. Explicitly identify which peer claims "
            "you support or dispute, expose assumptions, propose falsification tests, "
            "and update your position when warranted. Do not seek superficial consensus. "
            "Distinguish evidence from speculation."
        )
    if cfg.single_call and cfg.final_response_instruction:
        instruction += f"\n\nRequired response format:\n{cfg.final_response_instruction}"
    return instruction


class _Debate:
    """One debate run: its run directory, transcript, and participant models."""

    def __init__(self, cfg: ScientificDebateConfig, model_factory: Callable[..., Any]) -> None:
        self.cfg = cfg
        provenance = ProvenanceRecord(
            workflow="scientific_hypothesis_debate",
            evidence_level=EvidenceLevel.HYPOTHESIS,
            numerical_settings={
                "rounds": cfg.rounds,
                "participants": [
                    participant.model_dump(mode="json") for participant in cfg.participants
                ],
            },
        )
        self.run = ScientificRunDirectory.create(
            cfg.output_root,
            workflow="scientific_hypothesis_debate",
            request={"hypothesis": cfg.hypothesis},
            resolved_config=cfg.model_dump(mode="json"),
            provenance=provenance,
            prefix="debate",
            directories=(),
        )
        self.transcript = Transcript(
            self.run.path,
            title="Scientific debate",
            preamble=f"Run `{self.run.run_id}`.\n\n**Hypothesis.** {cfg.hypothesis}",
        )
        self.models = {}
        for participant in cfg.participants:
            key = api_key_from_env(
                participant.api_key_env, owner=f"participant {participant.name!r}"
            )
            tokens = participant.max_completion_tokens or cfg.max_completion_tokens
            self.models[participant.name] = model_factory(
                provider=participant.provider,
                model=participant.model,
                base_url=participant.base_url,
                **({"api_key": key} if key is not None else {}),
                **({"max_tokens": tokens} if tokens is not None else {}),
            )

    def speak(
        self,
        participant: DebateParticipant,
        contribution_id: str,
        kind: str,
        messages: list[Any],
        *,
        round_number: int | None,
        context: tuple[str, list[str], list[str]],
    ) -> Contribution:
        """Ask one participant, record the contribution at once, and return it."""

        _, shown, omitted = context
        reply = invoke_model(self.models[participant.name], messages)
        contribution = self.transcript.add(
            Contribution.from_reply(
                reply,
                id=contribution_id,
                kind=kind,
                speaker=participant.name,
                provider=participant.provider,
                model=participant.model,
                round=round_number,
                context=shown,
                omitted_context=omitted,
            ),
            prompt=messages,
        )
        self.run.append_event("contribution_completed", {"id": contribution_id})
        if contribution.incomplete:
            log.warning(
                "Debate %s: %s's %s is incomplete (finish reason %s, %d answer characters)",
                self.run.run_id,
                participant.name,
                contribution_id,
                contribution.finish_reason,
                len(contribution.text),
            )
        return contribution

    def argue(self) -> list[DebateTurn]:
        cfg = self.cfg
        turns: list[DebateTurn] = []
        for round_index in range(cfg.rounds):
            offset = round_index % len(cfg.participants)
            for participant in cfg.participants[offset:] + cfg.participants[:offset]:
                context = _peer_context(turns, cfg.max_transcript_chars)
                contribution = self.speak(
                    participant,
                    f"round-{round_index + 1:03d}-turn-{len(turns) + 1:04d}",
                    "argument",
                    [
                        SystemMessage(
                            content=(
                                f"{_turn_instruction(cfg, participant)}\n\n{_OWN_WORDS}\n\n"
                                f"This is round {round_index + 1} of {cfg.rounds}: give your "
                                "current argument and direct challenges to the other models."
                            )
                        ),
                        HumanMessage(
                            content=(
                                f"Hypothesis:\n{cfg.hypothesis}\n\n"
                                "Panel contributions so far:\n"
                                f"{context[0] or 'No peer has spoken yet.'}"
                            )
                        ),
                    ],
                    round_number=round_index + 1,
                    context=context,
                )
                turns.append(
                    DebateTurn(
                        turn_id=contribution.id,
                        round=round_index + 1,
                        participant=participant.name,
                        provider=participant.provider,
                        model=participant.model,
                        response=contribution.text,
                        incomplete=contribution.incomplete,
                    )
                )
        return turns

    def judge(self, turns: list[DebateTurn]) -> list[DebateVerdict]:
        cfg = self.cfg
        if cfg.single_call:
            # The single answer is the verdict.
            turn = turns[0]
            return [
                DebateVerdict(
                    contribution_id=turn.turn_id,
                    participant=turn.participant,
                    provider=turn.provider,
                    model=turn.model,
                    response=turn.response,
                    incomplete=turn.incomplete,
                )
            ]
        if cfg.synthesis_method == "independent_verdicts":
            judges = cfg.participants
        else:
            name = cfg.synthesis_participant or cfg.participants[0].name
            judges = [next(p for p in cfg.participants if p.name == name)]
        instruction = f"{_VERDICT_INSTRUCTION}\n\n{_OWN_WORDS}"
        if cfg.final_response_instruction:
            instruction += f"\n\nRequired response format:\n{cfg.final_response_instruction}"
        context = _peer_context(turns, cfg.max_transcript_chars)
        verdicts = []
        for index, participant in enumerate(judges, start=1):
            contribution = self.speak(
                participant,
                f"verdict-{index:03d}-turn-{len(turns) + index:04d}",
                "verdict" if cfg.synthesis_method == "independent_verdicts" else "synthesis",
                [
                    SystemMessage(content=instruction),
                    HumanMessage(
                        content=(
                            f"Hypothesis:\n{cfg.hypothesis}\n\nDebate contributions:\n{context[0]}"
                        )
                    ),
                ],
                round_number=None,
                context=context,
            )
            verdicts.append(
                DebateVerdict(
                    contribution_id=contribution.id,
                    participant=participant.name,
                    provider=participant.provider,
                    model=participant.model,
                    response=contribution.text,
                    incomplete=contribution.incomplete,
                )
            )
        return verdicts

    def finish(
        self, turns: list[DebateTurn], verdicts: list[DebateVerdict]
    ) -> ScientificDebateResult:
        cfg = self.cfg
        incomplete = list(
            dict.fromkeys(
                [turn.turn_id for turn in turns if turn.incomplete]
                + [verdict.contribution_id for verdict in verdicts if verdict.incomplete]
            )
        )
        result = ScientificDebateResult(
            run_id=self.run.run_id,
            run_directory=str(self.run.path),
            status=WorkflowStatus.COMPLETE,
            hypothesis=cfg.hypothesis,
            rounds_completed=cfg.rounds,
            turns=turns,
            verdicts=verdicts,
            synthesis="\n\n".join(
                f"[{verdict.participant} independent verdict]\n{verdict.response}"
                for verdict in verdicts
            ),
            synthesis_turn_id=(
                verdicts[0].contribution_id if cfg.synthesis_method == "designated_model" else None
            ),
            incomplete_contributions=incomplete,
            transcript_path=str(self.transcript.markdown_path),
            contributions_path=str(self.transcript.contributions_path),
        )
        # The texts live in contributions.jsonl; the result names them by id.
        self.run.write_json(
            "result.json",
            result.model_dump(mode="json", exclude={"turns", "verdicts", "synthesis"})
            | {
                "turn_ids": [turn.turn_id for turn in turns],
                "verdict_ids": [verdict.contribution_id for verdict in verdicts],
            },
        )
        return result


def _run_scientific_debate(
    cfg: ScientificDebateConfig,
    *,
    model_factory: Callable[..., Any],
) -> ScientificDebateResult:
    """Run round-robin critique rounds and record every contribution as it is made.

    Participants speak sequentially within a round. Therefore every response
    can question all earlier rounds and the peers who have already spoken in
    the current round. The order rotates each round to avoid permanently giving
    one model the informational advantage of speaking last. Peers see each
    other's answers only, never their reasoning.
    """

    debate = _Debate(cfg, model_factory)
    turns = debate.argue()
    return debate.finish(turns, debate.judge(turns))


__all__ = [
    "DebateParticipant",
    "DebateTurn",
    "DebateVerdict",
    "ScientificDebateConfig",
    "ScientificDebateResult",
    "run_scientific_debate",
]
