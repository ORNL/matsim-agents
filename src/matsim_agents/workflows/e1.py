"""Development-only runner for the E1 scientific-reasoning feasibility study."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.backends.llm.provider import get_chat_model
from matsim_agents.backends.llm.replies import invoke_model


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class E1Question(_StrictModel):
    question_id: str = Field(pattern=r"^D-[A-Z0-9-]+$")
    material: str = Field(min_length=1)
    category: Literal["A", "B", "C"]
    prompt: str = Field(min_length=1)


class E1Candidate(_StrictModel):
    catalog_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    provider: str = Field(min_length=1)


class E1Decoding(_StrictModel):
    temperature: Literal[0.0] = 0.0
    max_tokens: int = Field(default=16384, gt=0, le=16384)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    seed: int | None = None


class E1DevelopmentManifest(_StrictModel):
    schema_version: Literal[1] = 1
    partition: Literal["development"] = "development"
    candidate: E1Candidate
    decoding: E1Decoding = E1Decoding()
    protocol_version: str = Field(min_length=1)
    questions: tuple[E1Question, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_question_ids(self) -> E1DevelopmentManifest:
        ids = [question.question_id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("E1 question IDs must be unique")
        return self


ModelFactory = Callable[..., Any]
SessionFactory = Callable[[E1Candidate, E1Decoding, ModelFactory], Any]


@contextmanager
def _model_session(
    candidate: E1Candidate, decoding: E1Decoding, model_factory: ModelFactory
) -> Iterator[Any]:
    kwargs: dict[str, Any] = {
        "provider": candidate.provider,
        "model": candidate.model_id,
        "temperature": decoding.temperature,
        "max_tokens": decoding.max_tokens,
    }
    if decoding.top_p is not None:
        kwargs["top_p"] = decoding.top_p
    if decoding.seed is not None:
        kwargs["seed"] = decoding.seed
    yield model_factory(**kwargs)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _write_run_metadata(path: Path, metadata: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _completed_response_keys(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        last_complete_record = content.rfind(b"\n") + 1
        trailing_record = content[last_complete_record:]
        with path.open("r+b") as ledger:
            try:
                json.loads(trailing_record)
            except json.JSONDecodeError:
                ledger.truncate(last_complete_record)
                content = content[:last_complete_record]
            else:
                ledger.seek(0, os.SEEK_END)
                ledger.write(b"\n")
                ledger.flush()
                os.fsync(ledger.fileno())
                content += b"\n"
    for line_number, line in enumerate(content.splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid E1 ledger JSON on line {line_number}") from error
        if record.get("status") == "complete":
            completed.add(record["response_key"])
    return completed


def _validate_output_identity(manifest_path: Path, ledger_path: Path, manifest_hash: str) -> None:
    if ledger_path.exists() and not manifest_path.exists():
        raise ValueError("E1 response ledger exists without its manifest; refusing to mix runs")
    if manifest_path.exists():
        existing_hash = _sha256(json.loads(manifest_path.read_text(encoding="utf-8")))
        if existing_hash != manifest_hash:
            raise ValueError("E1 output directory already belongs to a different manifest")


def _write_manifest(path: Path, manifest_data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run_e1_development(
    manifest: E1DevelopmentManifest,
    output_dir: str | Path,
    *,
    model_factory: ModelFactory = get_chat_model,
    session_factory: SessionFactory = _model_session,
) -> dict[str, Any]:
    """Run only manifest-supplied development questions for one candidate.

    The runner has no held-out source/path parameter. Successful response keys
    are skipped on restart; each newly completed reply is flushed to JSONL.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "responses.jsonl"
    manifest_path = output / "manifest.json"
    manifest_data = manifest.model_dump(mode="json")
    manifest_hash = _sha256(manifest_data)
    _validate_output_identity(manifest_path, ledger_path, manifest_hash)
    protocol_hash = _sha256(
        {"protocol_version": manifest.protocol_version, "decoding": manifest.decoding.model_dump()}
    )
    completed = _completed_response_keys(ledger_path)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "partition": "development",
        "candidate": manifest.candidate.model_dump(mode="json"),
        "manifest_hash": manifest_hash,
        "protocol_hash": protocol_hash,
        "question_ids": [question.question_id for question in manifest.questions],
        "state": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    metadata_path = output / "run.json"
    _write_manifest(manifest_path, manifest_data)
    _write_run_metadata(metadata_path, metadata)

    pending = []
    for question in manifest.questions:
        prompt_hash = _sha256(question.prompt)
        response_key = _sha256(
            {
                "question": question.model_dump(mode="json"),
                "candidate": manifest.candidate.model_dump(mode="json"),
                "prompt_hash": prompt_hash,
                "decoding": manifest.decoding.model_dump(mode="json"),
                "protocol_hash": protocol_hash,
            }
        )
        if response_key not in completed:
            pending.append((question, prompt_hash, response_key))

    try:
        if pending:
            with (
                session_factory(manifest.candidate, manifest.decoding, model_factory) as model,
                ledger_path.open("a", encoding="utf-8") as ledger,
            ):
                for question, prompt_hash, response_key in pending:
                    reply = invoke_model(model, [HumanMessage(content=question.prompt)])
                    record = {
                        "schema_version": 1,
                        "status": "complete",
                        "response_key": response_key,
                        "question_id": question.question_id,
                        "material": question.material,
                        "category": question.category,
                        "candidate": manifest.candidate.model_dump(mode="json"),
                        "manifest_hash": manifest_hash,
                        "protocol_hash": protocol_hash,
                        "prompt_hash": prompt_hash,
                        "response": reply.model_dump(mode="json"),
                        "completed_at_utc": datetime.now(UTC).isoformat(),
                    }
                    ledger.write(json.dumps(record, sort_keys=True) + "\n")
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    completed.add(response_key)
        metadata["state"] = "complete"
    except Exception as error:
        metadata["state"] = "failed"
        metadata["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata["completed_responses"] = len(completed)
        metadata["updated_at_utc"] = datetime.now(UTC).isoformat()
        _write_run_metadata(metadata_path, metadata)
    return metadata


__all__ = [
    "E1Candidate",
    "E1Decoding",
    "E1DevelopmentManifest",
    "E1Question",
    "run_e1_development",
]