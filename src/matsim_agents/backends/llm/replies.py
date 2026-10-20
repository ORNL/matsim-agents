"""What a chat model returned for one request.

A reply separates the answer from the model's reasoning and records how the
generation ended, so callers parse and pass on only the answer and transcripts
can show when a reply was cut off at the token limit.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, computed_field

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
# finish/stop reasons that mean the token limit ended the reply (OpenAI and
# vLLM: "length"; Anthropic: "max_tokens").
_TRUNCATED = frozenset({"length", "max_tokens"})


def split_reasoning(text: str) -> tuple[str, str | None]:
    """Return ``(answer, reasoning)`` for a reply that may contain a think block.

    Reasoning models served without a reasoning parser write their reasoning
    before ``</think>``; the chat template usually opens ``<think>`` in the
    prompt, so often only the closing tag is generated. A ``<think>`` that is
    never closed is reasoning cut off at the token limit: there is no answer.
    """

    head, closed, tail = text.rpartition(_THINK_CLOSE)
    if not closed:
        stripped = text.lstrip()
        if stripped.startswith(_THINK_OPEN):
            return "", stripped[len(_THINK_OPEN) :].strip() or None
        return text, None
    reasoning = head.strip()
    if reasoning.startswith(_THINK_OPEN):
        reasoning = reasoning[len(_THINK_OPEN) :].strip()
    return tail.strip(), reasoning or None


class ModelReply(BaseModel):
    """The answer of one model request and how it was produced."""

    text: str
    reasoning: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None

    @computed_field
    @property
    def truncated(self) -> bool:
        """True when the token limit ended the reply."""
        return self.finish_reason in _TRUNCATED


def _content_parts(content: Any) -> tuple[str, str | None]:
    """Text and reasoning of a message content (a string or provider content blocks)."""

    if isinstance(content, str):
        return content, None
    if not isinstance(content, list):
        return str(content), None
    texts: list[str] = []
    thoughts: list[str] = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict) and block.get("type") == "thinking":
            thoughts.append(str(block.get("thinking", "")))
        elif isinstance(block, dict) and block.get("type") == "text":
            texts.append(str(block.get("text", "")))
    return "".join(texts), "\n\n".join(thoughts) or None


def reply_from_message(message: Any, *, latency_s: float | None = None) -> ModelReply:
    """Build a reply from a LangChain message (or any object with ``content``)."""

    text, reasoning = _content_parts(getattr(message, "content", message))
    extra = getattr(message, "additional_kwargs", None) or {}
    reasoning = reasoning or extra.get("reasoning_content") or extra.get("reasoning")
    if reasoning is None:
        text, reasoning = split_reasoning(text)
    metadata = getattr(message, "response_metadata", None) or {}
    finish = (
        metadata.get("finish_reason") or metadata.get("stop_reason") or metadata.get("done_reason")
    )
    usage = getattr(message, "usage_metadata", None) or {}
    return ModelReply(
        text=text,
        reasoning=reasoning,
        finish_reason=finish,
        prompt_tokens=usage.get("input_tokens"),
        completion_tokens=usage.get("output_tokens"),
        latency_s=latency_s,
    )


def invoke_model(model: Any, messages: Sequence[Any]) -> ModelReply:
    """Send ``messages`` to a chat model and return its reply."""

    start = time.perf_counter()
    message = model.invoke(list(messages))
    return reply_from_message(message, latency_s=round(time.perf_counter() - start, 3))


__all__ = ["ModelReply", "invoke_model", "reply_from_message", "split_reasoning"]
