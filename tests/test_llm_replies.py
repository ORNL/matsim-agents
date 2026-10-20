"""Model replies: answers separated from reasoning, and how each reply ended."""

from __future__ import annotations

import os
import stat
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from matsim_agents.backends.llm.provider import ChatVLLM, get_chat_model
from matsim_agents.backends.llm.replies import invoke_model, reply_from_message, split_reasoning
from matsim_agents.execution.contracts import EvidenceLevel, ProvenanceRecord
from matsim_agents.execution.run_directory import ScientificRunDirectory


@pytest.mark.parametrize(
    ("text", "answer", "reasoning"),
    [
        ("plain answer", "plain answer", None),
        ("<think>plan</think>\n\nanswer", "answer", "plan"),
        # The chat template opened <think> in the prompt.
        (" The user wants X.\n</think>\nanswer", "answer", "The user wants X."),
        # Reasoning cut off at the token limit: no answer.
        ("<think>still planning", "", "still planning"),
        ("</think>answer", "answer", None),
    ],
)
def test_split_reasoning(text, answer, reasoning):
    assert split_reasoning(text) == (answer, reasoning)


def _completion(content, *, reasoning=None, finish="stop", usage=(12, 34)):
    message = SimpleNamespace(content=content)
    if reasoning is not None:
        message.reasoning = reasoning
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]),
    )


@pytest.mark.parametrize(
    ("completion", "answer", "reasoning"),
    [
        (_completion("why</think>answer"), "answer", "why"),
        # vLLM with --reasoning-parser returns the reasoning separately.
        (_completion("answer", reasoning="why"), "answer", "why"),
        (_completion(None, reasoning="why", finish="length"), "", "why"),
    ],
)
def test_vllm_replies_carry_reasoning_finish_reason_and_usage(completion, answer, reasoning):
    client = MagicMock()
    client.chat.completions.create.return_value = completion
    with patch("openai.OpenAI", return_value=client):
        reply = invoke_model(ChatVLLM(model="m"), [HumanMessage(content="q")])
    assert (reply.text, reply.reasoning) == (answer, reasoning)
    assert reply.finish_reason == completion.choices[0].finish_reason
    assert reply.truncated is (reply.finish_reason == "length")
    assert (reply.prompt_tokens, reply.completion_tokens) == (12, 34)
    assert reply.latency_s is not None


def test_replies_from_other_providers():
    anthropic = AIMessage(
        content=[{"type": "thinking", "thinking": "why"}, {"type": "text", "text": "answer"}],
        response_metadata={"stop_reason": "max_tokens"},
        usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    )
    reply = reply_from_message(anthropic)
    assert (reply.text, reply.reasoning, reply.truncated) == ("answer", "why", True)
    ollama = AIMessage(
        content="<think>why</think>answer", response_metadata={"done_reason": "stop"}
    )
    reply = reply_from_message(ollama)
    assert (reply.text, reply.reasoning, reply.truncated) == ("answer", "why", False)
    assert reply_from_message(SimpleNamespace(content="plain")).text == "plain"


def test_reply_token_limit_maps_to_each_provider(monkeypatch):
    assert get_chat_model(provider="vllm", model="m", max_tokens=777).max_completion_tokens == 777
    ollama = MagicMock()
    with patch.dict("sys.modules", {"langchain_ollama": MagicMock(ChatOllama=ollama)}):
        get_chat_model(provider="ollama", model="m", max_tokens=777)
    assert ollama.call_args.kwargs["num_predict"] == 777
    openai = MagicMock()
    with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=openai)}):
        get_chat_model(provider="openai", model="m", max_tokens=777)
    assert openai.call_args.kwargs["max_tokens"] == 777


def _run(tmp_path, **kwargs):
    return ScientificRunDirectory.create(
        tmp_path,
        workflow="w",
        request={},
        resolved_config={},
        provenance=ProvenanceRecord(workflow="w", evidence_level=EvidenceLevel.HYPOTHESIS),
        **kwargs,
    )


def test_run_files_get_ordinary_file_permissions(tmp_path):
    # Not mkstemp's owner-only 0600: collaborators sharing the project read runs.
    umask = os.umask(0o022)
    os.umask(umask)
    path = _run(tmp_path).write_json("result.json", {"ok": True})
    assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~umask


def test_run_prefix_and_directories(tmp_path):
    run = _run(tmp_path, prefix="debate", directories=())
    assert run.run_id.startswith("debate-")
    assert not any(path.is_dir() for path in run.path.iterdir())
