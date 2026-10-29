"""Development-only runner for the E1 scientific-reasoning feasibility study."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.backends.llm.provider import get_chat_model
from matsim_agents.backends.llm.replies import invoke_model_with_raw


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class E1Question(_StrictModel):
    question_id: str = Field(pattern=r"^D-[A-Z0-9-]+$")
    material: str = Field(min_length=1)
    category: Literal["A", "B", "C"]
    prompt: str = Field(min_length=1)


class E1ScientificResponse(_StrictModel):
    conclusion: str
    hypotheses: list[str]
    reasoning_summary: str
    assumptions: list[str]
    supporting_evidence: list[str]
    missing_or_conflicting_evidence: list[str]
    falsifiable_predictions: list[str]
    proposed_tests: list[str]
    outcome_interpretation: list[str]
    confidence: str


class E1Candidate(_StrictModel):
    catalog_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    mag_provider: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_mag_provider(self) -> E1Candidate:
        if self.mag_provider is not None and self.provider.lower() != "mag":
            raise ValueError("mag_provider is only valid for MAG candidates")
        return self


class E1Decoding(_StrictModel):
    temperature: Literal[0.0] = 0.0
    max_tokens: int = Field(default=16384, gt=0, le=16384)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    seed: int | None = None


class E1RemoteApproval(_StrictModel):
    approval_reference: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    mag_provider: str | None = Field(default=None, min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    spending_ceiling_usd: float = Field(gt=0.0, allow_inf_nan=False)
    max_input_tokens_per_question: int = Field(gt=0)
    max_attempts_per_question: int = Field(default=1, gt=0)
    input_usd_per_million_tokens: float = Field(ge=0.0, allow_inf_nan=False)
    output_usd_per_million_tokens: float = Field(ge=0.0, allow_inf_nan=False)


class E1DevelopmentManifest(_StrictModel):
    schema_version: Literal[1] = 1
    partition: Literal["development"] = "development"
    cohort: Literal["E1-L", "E1-R"]
    candidate: E1Candidate
    decoding: E1Decoding = E1Decoding()
    remote_approval: E1RemoteApproval | None = None
    protocol_version: str = Field(min_length=1)
    questions: tuple[E1Question, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> E1DevelopmentManifest:
        ids = [question.question_id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("E1 question IDs must be unique")
        local_providers = {"vllm", "huggingface", "ollama"}
        remote_providers = {"openai", "anthropic", "mag"}
        provider = self.candidate.provider.lower()
        expected = local_providers if self.cohort == "E1-L" else remote_providers
        if provider not in expected:
            raise ValueError(f"provider {provider!r} is not valid for cohort {self.cohort}")
        if self.cohort == "E1-R":
            if self.remote_approval is None:
                raise ValueError("E1-R requires explicit route and spending approval")
            approved_route = (
                self.remote_approval.provider.lower(),
                self.remote_approval.mag_provider,
                self.remote_approval.model_id,
                self.remote_approval.revision,
            )
            candidate_route = (
                provider,
                self.candidate.mag_provider,
                self.candidate.model_id,
                self.candidate.revision,
            )
            if approved_route != candidate_route:
                raise ValueError("E1-R approval must match the candidate provider, model, and revision")
            worst_case_usd = len(self.questions) * self.remote_approval.max_attempts_per_question * (
                self.remote_approval.max_input_tokens_per_question
                * self.remote_approval.input_usd_per_million_tokens
                + self.decoding.max_tokens
                * self.remote_approval.output_usd_per_million_tokens
            ) / 1_000_000
            if worst_case_usd > self.remote_approval.spending_ceiling_usd:
                raise ValueError(
                    "E1-R worst-case token spend exceeds the approved spending ceiling "
                    f"(${worst_case_usd:.6f} > "
                    f"${self.remote_approval.spending_ceiling_usd:.6f})"
                )
        elif self.remote_approval is not None:
            raise ValueError("remote_approval is only valid for E1-R")
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
    if candidate.provider.lower() == "mag":
        kwargs["mag_provider"] = candidate.mag_provider
    yield model_factory(**kwargs)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _protocol_artifact_hashes() -> dict[str, str]:
    artifact_dir = Path(__file__).resolve().parents[3] / "research" / "paper" / "e1"
    artifacts = {
        "development_questions": artifact_dir / "d-questions.v1.json",
        "response_schema": artifact_dir / "response-schema.v1.json",
        "protocol_rubric_provenance": artifact_dir / "development-protocol.v1.md",
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in artifacts.items()
        if path.exists()
    }


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
        if record.get("status") in {"complete", "error"}:
            completed.add(record["response_key"])
    return completed


def _response_attempts(path: Path) -> dict[str, tuple[str, int]]:
    latest: dict[str, tuple[str, int]] = {}
    if not path.exists():
        return latest
    _completed_response_keys(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") in {"complete", "error"}:
            key = record["response_key"]
            latest[key] = (record["status"], max(record.get("attempt", 1), latest.get(key, ("", 0))[1]))
    return latest


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
    retry_errors: bool = False,
) -> dict[str, Any]:
    """Run only manifest-supplied development questions for one candidate.

    The runner has no held-out source/path parameter. Completed response keys
    are skipped on restart; failures are only retried when explicitly requested.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "responses.jsonl"
    manifest_path = output / "manifest.json"
    manifest_data = manifest.model_dump(mode="json")
    manifest_hash = _sha256(manifest_data)
    _validate_output_identity(manifest_path, ledger_path, manifest_hash)
    protocol_artifact_hashes = _protocol_artifact_hashes()
    protocol_hash = _sha256(
        {
            "protocol_version": manifest.protocol_version,
            "decoding": manifest.decoding.model_dump(),
            "response_schema": E1ScientificResponse.model_json_schema(),
            "artifact_hashes": protocol_artifact_hashes,
        }
    )
    attempts = _response_attempts(ledger_path)
    completed = {key for key, (status, _) in attempts.items() if status == "complete"}
    if manifest.cohort == "E1-R" and ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("status") in {"complete", "error"}:
                _append_cost_record(output / "costs.jsonl", manifest, record)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "partition": "development",
        "cohort": manifest.cohort,
        "candidate": manifest.candidate.model_dump(mode="json"),
        "manifest_hash": manifest_hash,
        "protocol_hash": protocol_hash,
        "protocol_artifact_hashes": protocol_artifact_hashes,
        "question_ids": [question.question_id for question in manifest.questions],
        "state": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    metadata_path = output / "run.json"

    pending = []
    system_prompt = _e1_system_prompt()
    for question in manifest.questions:
        prompt_hash = _sha256(
            {
                "system": system_prompt,
                "question": question.prompt,
                "response_schema": E1ScientificResponse.model_json_schema(),
            }
        )
        response_key = _sha256(
            {
                "question": question.model_dump(mode="json"),
                "candidate": manifest.candidate.model_dump(mode="json"),
                "prompt_hash": prompt_hash,
                "decoding": manifest.decoding.model_dump(mode="json"),
                "protocol_hash": protocol_hash,
            }
        )
        previous = attempts.get(response_key)
        if previous is None or (previous[0] == "error" and retry_errors):
            attempt = 1 if previous is None else previous[1] + 1
            approval = manifest.remote_approval
            if (
                manifest.cohort == "E1-R"
                and approval is not None
                and attempt > approval.max_attempts_per_question
            ):
                raise ValueError("E1-R retry exceeds the approved max_attempts_per_question")
            pending.append((question, prompt_hash, response_key, attempt))

    _write_manifest(manifest_path, manifest_data)
    _write_run_metadata(metadata_path, metadata)

    try:
        if pending:
            with (
                session_factory(manifest.candidate, manifest.decoding, model_factory) as model,
                ledger_path.open("a", encoding="utf-8") as ledger,
            ):
                for question, prompt_hash, response_key, attempt in pending:
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=question.prompt),
                    ]
                    try:
                        raw_message, reply = invoke_model_with_raw(model, messages)
                    except Exception as error:
                        record = {
                            "schema_version": 1,
                            "status": "error",
                            "attempt": attempt,
                            "response_key": response_key,
                            "question_id": question.question_id,
                            "candidate": manifest.candidate.model_dump(mode="json"),
                            "cohort": manifest.cohort,
                            "manifest_hash": manifest_hash,
                            "protocol_hash": protocol_hash,
                            "prompt_hash": prompt_hash,
                            "error_type": type(error).__name__,
                            "error_message": _safe_error_message(str(error)),
                            "completed_at_utc": datetime.now(UTC).isoformat(),
                        }
                        ledger.write(json.dumps(record, sort_keys=True) + "\n")
                        ledger.flush()
                        os.fsync(ledger.fileno())
                        attempts[response_key] = ("error", attempt)
                        if manifest.cohort == "E1-R":
                            _append_cost_record(output / "costs.jsonl", manifest, record)
                        continue
                    raw_response = (
                        raw_message.model_dump(mode="json")
                        if hasattr(raw_message, "model_dump")
                        else {"content": getattr(raw_message, "content", str(raw_message))}
                    )
                    try:
                        parsed_response = E1ScientificResponse.model_validate_json(reply.text)
                        parsed_data = parsed_response.model_dump(mode="json")
                        schema_error = None
                    except Exception as error:
                        parsed_data = None
                        schema_error = f"{type(error).__name__}: {_safe_error_message(str(error))}"
                    record = {
                        "schema_version": 1,
                        "status": "complete",
                        "attempt": attempt,
                        "response_key": response_key,
                        "question_id": question.question_id,
                        "material": question.material,
                        "category": question.category,
                        "candidate": manifest.candidate.model_dump(mode="json"),
                        "cohort": manifest.cohort,
                        "manifest_hash": manifest_hash,
                        "protocol_hash": protocol_hash,
                        "prompt_hash": prompt_hash,
                        "raw_response": raw_response,
                        "response": reply.model_dump(mode="json"),
                        "parsed_response": parsed_data,
                        "schema_error": schema_error,
                        "completed_at_utc": datetime.now(UTC).isoformat(),
                    }
                    ledger.write(json.dumps(record, sort_keys=True) + "\n")
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    completed.add(response_key)
                    attempts[response_key] = ("complete", attempt)
                    if manifest.cohort == "E1-R":
                        _append_cost_record(output / "costs.jsonl", manifest, record)
        error_count = sum(status == "error" for status, _ in attempts.values())
        metadata["state"] = "complete_with_errors" if error_count else "complete"
    except Exception as error:
        metadata["state"] = "failed"
        metadata["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata["completed_responses"] = len(completed)
        metadata["failed_responses"] = sum(
            status == "error" for status, _ in attempts.values()
        )
        metadata["updated_at_utc"] = datetime.now(UTC).isoformat()
        _write_run_metadata(metadata_path, metadata)
    return metadata


_E1_SYSTEM_PROMPT = """Answer the supplied materials-science question using only its stated facts and assumptions. Return one JSON object matching the required response schema, with no markdown fences. Keep reasoning_summary concise and provide a scientific rationale, not private chain-of-thought. Use empty arrays where a field does not apply. Distinguish supplied evidence, inference, and hypotheses; do not invent measurements, sources, or results."""


def _e1_system_prompt() -> str:
    return f"{_E1_SYSTEM_PROMPT}\n\nRequired JSON Schema:\n{json.dumps(E1ScientificResponse.model_json_schema(), sort_keys=True)}"


def _safe_error_message(message: str) -> str:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AMSC_TOKEN", "MATSIM_VLLM_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", message)
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", message)
    return message[:1000]


def _append_cost_record(
    path: Path, manifest: E1DevelopmentManifest, response: dict[str, Any]
) -> None:
    approval = manifest.remote_approval
    if approval is None:
        raise ValueError("E1-R cost ledger requires remote approval data")
    existing_keys = set()
    if path.exists():
        existing_keys = {
            (record["response_key"], record["attempt"])
            for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        }
    identity = (response["response_key"], response.get("attempt", 1))
    if identity in existing_keys:
        return
    reply = response.get("response") or {}
    prompt_tokens = reply.get("prompt_tokens")
    completion_tokens = reply.get("completion_tokens")
    estimated_cost = None
    if prompt_tokens is not None and completion_tokens is not None:
        estimated_cost = (
            prompt_tokens * approval.input_usd_per_million_tokens
            + completion_tokens * approval.output_usd_per_million_tokens
        ) / 1_000_000
    cost_record = {
        "response_key": response["response_key"],
        "question_id": response["question_id"],
        "candidate": response["candidate"],
        "cohort": "E1-R",
        "attempt": response["attempt"],
        "status": response["status"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": estimated_cost,
        "provider_reported_cost_usd": None,
        "pricing_approval_reference": approval.approval_reference,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
    }
    with path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(cost_record, sort_keys=True) + "\n")
        ledger.flush()
        os.fsync(ledger.fileno())


__all__ = [
    "E1Candidate",
    "E1Decoding",
    "E1DevelopmentManifest",
    "E1Question",
    "E1RemoteApproval",
    "E1ScientificResponse",
    "run_e1_development",
]