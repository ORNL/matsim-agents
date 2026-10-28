"""Append-only transcripts of conversations between people, tools, and models.

A transcript directory holds:

- ``contributions.jsonl``: one record per contribution, appended as it is made,
  with the answer, the model's reasoning, how the reply ended, token counts,
  latency, and which earlier contributions the model was shown;
- ``prompts/<id>.json``: the exact messages sent for each model contribution;
- ``transcript.md``: the same conversation for reading, with reasoning folded.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from matsim_agents.backends.llm.replies import ModelReply

_ROLES = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}


class Contribution(BaseModel):
    """One message of a transcript: a model reply, a person's turn, or a tool result."""

    id: str
    kind: str
    speaker: str
    provider: str | None = None
    model: str | None = None
    round: int | None = None
    text: str
    reasoning: str | None = None
    finish_reason: str | None = None
    truncated: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None
    provider_metadata: dict[str, str] | None = None
    context: list[str] = Field(
        default_factory=list, description="Earlier contributions the model was shown."
    )
    omitted_context: list[str] = Field(
        default_factory=list,
        description="Earlier contributions left out to keep the prompt within its budget.",
    )
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def from_reply(cls, reply: ModelReply, **fields: Any) -> Contribution:
        return cls(
            text=reply.text,
            reasoning=reply.reasoning,
            finish_reason=reply.finish_reason,
            truncated=reply.truncated,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            latency_s=reply.latency_s,
            provider_metadata=reply.provider_metadata,
            **fields,
        )

    @property
    def incomplete(self) -> bool:
        """True when the token limit cut the reply off or it has no answer."""
        return self.truncated or not self.text.strip()


def prompt_messages(messages: Sequence[Any]) -> list[dict[str, str]]:
    """Messages as ``{"role", "content"}`` records."""

    return [
        {
            "role": _ROLES.get(getattr(message, "type", ""), str(getattr(message, "type", ""))),
            "content": str(getattr(message, "content", message)),
        }
        for message in messages
    ]


def _markdown(contribution: Contribution) -> str:
    heading = f"{contribution.kind.replace('_', ' ').capitalize()} · {contribution.speaker}"
    if contribution.round is not None:
        heading += f" · round {contribution.round}"
    details = [f"`{contribution.id}`"]
    if contribution.model:
        details.append(contribution.model)
    if contribution.completion_tokens is not None:
        details.append(f"{contribution.completion_tokens} completion tokens")
    if contribution.latency_s is not None:
        details.append(f"{contribution.latency_s:.1f} s")
    if contribution.omitted_context:
        details.append(f"{len(contribution.omitted_context)} earlier contributions not shown")
    lines = [f"## {heading}", "", " · ".join(details), ""]
    if contribution.truncated:
        lines += [
            f"> **Incomplete:** the token limit ended this reply "
            f"(finish reason `{contribution.finish_reason}`).",
            "",
        ]
    lines += [contribution.text.strip() or "_No answer._", ""]
    if contribution.reasoning:
        lines += [
            "<details>",
            f"<summary>Reasoning ({len(contribution.reasoning)} characters)</summary>",
            "",
            contribution.reasoning.strip(),
            "",
            "</details>",
            "",
        ]
    return "\n".join(lines) + "\n"


class Transcript:
    """Write a conversation to a directory as it happens (see the module docstring)."""

    def __init__(self, directory: str | Path, *, title: str, preamble: str | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.contributions_path = self.directory / "contributions.jsonl"
        self.markdown_path = self.directory / "transcript.md"
        self.count = len(self.contributions())
        # Concurrent requests (llm-check) append from several threads.
        self._lock = threading.Lock()
        if not self.markdown_path.exists():
            header = f"# {title}\n\n" + (f"{preamble.strip()}\n\n" if preamble else "")
            self.markdown_path.write_text(header, encoding="utf-8")

    def add(
        self, contribution: Contribution, *, prompt: Sequence[Any] | None = None
    ) -> Contribution:
        """Append ``contribution``; ``prompt`` is the messages a model was sent for it."""

        with self._lock:
            if prompt is not None:
                prompts = self.directory / "prompts"
                prompts.mkdir(exist_ok=True)
                (prompts / f"{contribution.id}.json").write_text(
                    json.dumps(prompt_messages(prompt), indent=2) + "\n", encoding="utf-8"
                )
            with self.contributions_path.open("a", encoding="utf-8") as stream:
                stream.write(contribution.model_dump_json() + "\n")
            with self.markdown_path.open("a", encoding="utf-8") as stream:
                stream.write(_markdown(contribution))
            self.count += 1
        return contribution

    def contributions(self) -> list[Contribution]:
        if not self.contributions_path.exists():
            return []
        return [
            Contribution.model_validate_json(line)
            for line in self.contributions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


__all__ = ["Contribution", "Transcript", "prompt_messages"]
