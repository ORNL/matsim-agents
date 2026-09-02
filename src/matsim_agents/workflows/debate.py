"""Persistent multi-model scientific hypothesis debate workflow."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from matsim_agents.backends.llm.provider import get_chat_model
from matsim_agents.execution.contracts import EvidenceLevel, ProvenanceRecord, WorkflowStatus
from matsim_agents.execution.run_directory import ScientificRunDirectory


class DebateParticipant(BaseModel):
    """One independently configured model and its scientific perspective."""

    name: str = Field(min_length=1)
    provider: str
    model: str
    base_url: str | None = None
    role: str = "independent scientific reviewer"


class ScientificDebateConfig(BaseModel):
    hypothesis: str = Field(min_length=1)
    participants: list[DebateParticipant] = Field(min_length=2)
    rounds: int = Field(2, ge=1, le=100)
    output_root: str = "./runs"
    synthesis_participant: str | None = None
    max_transcript_chars: int = Field(60_000, ge=1_000)

    @model_validator(mode="after")
    def _validate_participants(self) -> ScientificDebateConfig:
        names = [participant.name for participant in self.participants]
        if len(names) != len(set(names)):
            raise ValueError("debate participant names must be unique")
        if self.synthesis_participant is not None and self.synthesis_participant not in names:
            raise ValueError("synthesis_participant must name a configured participant")
        return self


class DebateTurn(BaseModel):
    turn_id: str
    round: int
    participant: str
    provider: str
    model: str
    response: str


class ScientificDebateResult(BaseModel):
    run_id: str
    run_directory: str
    status: WorkflowStatus
    hypothesis: str
    rounds_completed: int
    turns: list[DebateTurn]
    synthesis: str
    synthesis_turn_id: str
    transcript_path: str
    dialogue_path: str


def _content(response: Any) -> str:
    value = getattr(response, "content", response)
    return value if isinstance(value, str) else str(value)


def _transcript(turns: list[DebateTurn], limit: int) -> str:
    text = "\n\n".join(
        f"[round {turn.round}; {turn.participant}]\n{turn.response}" for turn in turns
    )
    return text[-limit:]


def run_scientific_debate(
    cfg: ScientificDebateConfig,
    *,
    model_factory: Callable[..., Any] = get_chat_model,
) -> ScientificDebateResult:
    """Run round-robin critique rounds and persist the complete enclave transcript.

    Participants speak sequentially within a round. Therefore every response
    can question all earlier rounds and the peers who have already spoken in
    the current round. The order rotates each round to avoid permanently giving
    one model the informational advantage of speaking last.
    """

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
    run = ScientificRunDirectory.create(
        cfg.output_root,
        workflow="scientific_hypothesis_debate",
        request={"hypothesis": cfg.hypothesis},
        resolved_config=cfg.model_dump(mode="json"),
        provenance=provenance,
    )
    models = {
        participant.name: model_factory(
            provider=participant.provider,
            model=participant.model,
            base_url=participant.base_url,
        )
        for participant in cfg.participants
    }
    turns: list[DebateTurn] = []
    for round_index in range(cfg.rounds):
        offset = round_index % len(cfg.participants)
        order = cfg.participants[offset:] + cfg.participants[:offset]
        for participant in order:
            history = _transcript(turns, cfg.max_transcript_chars) or "No peer has spoken yet."
            response = models[participant.name].invoke(
                [
                    SystemMessage(
                        content=(
                            f"You are {participant.name}, acting as {participant.role}. "
                            "Debate the scientific hypothesis independently. Explicitly identify "
                            "which peer claims you support or dispute, expose assumptions, propose "
                            "falsification tests, and update your position when warranted. Do not "
                            "seek superficial consensus. Distinguish evidence from speculation."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Hypothesis:\n{cfg.hypothesis}\n\n"
                            f"Debate transcript so far:\n{history}\n\n"
                            f"This is round {round_index + 1} of {cfg.rounds}. Respond with your "
                            "current argument and direct challenges to the other models."
                        )
                    ),
                ]
            )
            turns.append(
                DebateTurn(
                    turn_id=f"round-{round_index + 1:03d}-turn-{len(turns) + 1:04d}",
                    round=round_index + 1,
                    participant=participant.name,
                    provider=participant.provider,
                    model=participant.model,
                    response=_content(response),
                )
            )

    synthesizer_name = cfg.synthesis_participant or cfg.participants[0].name
    synthesis = _content(
        models[synthesizer_name].invoke(
            [
                SystemMessage(
                    content=(
                        "Synthesize a scientific debate without erasing disagreement. Report: "
                        "consensus, unresolved disputes, strongest evidence, assumptions, decisive "
                        "experiments or calculations, and a calibrated verdict on the hypothesis."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Hypothesis:\n{cfg.hypothesis}\n\nTranscript:\n"
                        f"{_transcript(turns, cfg.max_transcript_chars)}"
                    ),
                ),
            ]
        )
    )
    synthesis_turn_id = f"synthesis-turn-{len(turns) + 1:04d}"
    dialogue = [
        {
            "contribution_id": "hypothesis-0000",
            "kind": "assigned_hypothesis",
            "speaker": "user",
            "text": cfg.hypothesis,
        },
        *[
            {
                "contribution_id": turn.turn_id,
                "kind": "model_argument",
                "round": turn.round,
                "speaker": turn.participant,
                "provider": turn.provider,
                "model": turn.model,
                "text": turn.response,
            }
            for turn in turns
        ],
        {
            "contribution_id": synthesis_turn_id,
            "kind": "model_synthesis",
            "speaker": synthesizer_name,
            "provider": next(
                participant.provider
                for participant in cfg.participants
                if participant.name == synthesizer_name
            ),
            "model": next(
                participant.model
                for participant in cfg.participants
                if participant.name == synthesizer_name
            ),
            "text": synthesis,
        },
    ]
    dialogue_path = run.write_json("dialogue.json", dialogue)
    transcript_path = run.write_json(
        "debate_transcript.json",
        {
            "hypothesis": cfg.hypothesis,
            "turns": [turn.model_dump(mode="json") for turn in turns],
            "synthesis": synthesis,
        },
    )
    result = ScientificDebateResult(
        run_id=run.run_id,
        run_directory=str(run.path),
        status=WorkflowStatus.COMPLETE,
        hypothesis=cfg.hypothesis,
        rounds_completed=cfg.rounds,
        turns=turns,
        synthesis=synthesis,
        synthesis_turn_id=synthesis_turn_id,
        transcript_path=str(transcript_path),
        dialogue_path=str(dialogue_path),
    )
    run.write_json("results.json", result)
    return result


__all__ = [
    "DebateParticipant",
    "DebateTurn",
    "ScientificDebateConfig",
    "ScientificDebateResult",
    "run_scientific_debate",
]
