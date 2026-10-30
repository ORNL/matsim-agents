"""Development-only runner for the E1 scientific-reasoning feasibility study."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
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
    max_output_tokens_supported: int | None = Field(default=None, gt=0)
    revision_semantics: Literal["immutable", "mutable_alias"] = "immutable"
    provider: str = Field(min_length=1)
    mag_provider: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_mag_provider(self) -> E1Candidate:
        if self.mag_provider is not None and self.provider.lower() != "mag":
            raise ValueError("mag_provider is only valid for MAG candidates")
        return self


class E1Decoding(_StrictModel):
    temperature: Literal[0.0] = 0.0
    max_tokens: int = Field(default=32768, gt=0, le=32768)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    seed: int | None = None


class E1RemoteApproval(_StrictModel):
    approval_reference: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    mag_provider: str | None = Field(default=None, min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    revision_semantics: Literal["immutable", "mutable_alias"]
    mutable_alias_accepted: bool = False
    endpoint: str = Field(min_length=1)
    project_authorization_reference: str = Field(min_length=1)
    allocation_limit_reference: str = Field(min_length=1)
    allocation_charge_description: str = Field(min_length=1)
    data_use_conditions: str = Field(min_length=1)
    retention_conditions: str = Field(min_length=1)
    max_input_message_bytes_per_question: int = Field(gt=0)
    max_input_message_bytes_campaign: int = Field(gt=0)
    max_total_requests: int = Field(gt=0)
    max_attempts_per_question: int = Field(gt=0)
    max_output_tokens_per_response: int = Field(gt=0)
    max_output_tokens_campaign: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_alias_approval(self) -> E1RemoteApproval:
        if self.revision_semantics == "mutable_alias" and not self.mutable_alias_accepted:
            raise ValueError("mutable MAG alias requires explicit mutable_alias_accepted approval")
        return self


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
        if (
            self.candidate.max_output_tokens_supported is not None
            and self.decoding.max_tokens > self.candidate.max_output_tokens_supported
        ):
            raise ValueError("E1 decoding exceeds the candidate model output-token limit")
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
                self.remote_approval.revision_semantics,
            )
            candidate_route = (
                provider,
                self.candidate.mag_provider,
                self.candidate.model_id,
                self.candidate.revision,
                self.candidate.revision_semantics,
            )
            if approved_route != candidate_route:
                raise ValueError("E1-R approval must match the candidate provider, model, and revision")
            if self.remote_approval.max_total_requests < len(self.questions):
                raise ValueError("E1-R request limit must cover every development question")
            if self.remote_approval.max_total_requests > (
                len(self.questions) * self.remote_approval.max_attempts_per_question
            ):
                raise ValueError("E1-R request limit exceeds the approved per-question attempt limit")
            if (
                self.candidate.revision_semantics == "mutable_alias"
                and not self.remote_approval.mutable_alias_accepted
            ):
                raise ValueError("E1-R mutable alias use requires explicit owner approval")
            if self.decoding.max_tokens > self.remote_approval.max_output_tokens_per_response:
                raise ValueError("E1-R decoding exceeds the approved per-response output limit")
            if (
                self.remote_approval.max_total_requests * self.decoding.max_tokens
                > self.remote_approval.max_output_tokens_campaign
            ):
                raise ValueError("E1-R worst-case output exceeds the approved campaign output limit")
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


def _request_count(path: Path) -> int:
    if not path.exists():
        return 0
    _completed_response_keys(path)
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if (record := json.loads(line)).get("status") in {"complete", "error"}
    )


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
    max_new_requests: int | None = None,
) -> dict[str, Any]:
    """Run only manifest-supplied development questions for one candidate.

    The runner has no held-out source/path parameter. Completed response keys
    are skipped on restart; failures are only retried when explicitly requested.
    """

    system_prompt = _e1_system_prompt()
    input_preflight = _preflight_remote_inputs(manifest, system_prompt)
    endpoint = _provider_endpoint(manifest.candidate.provider)
    if manifest.cohort == "E1-R" and manifest.remote_approval.endpoint != endpoint:
        raise ValueError(
            "E1-R configured endpoint does not match the approved endpoint: "
            f"{endpoint!r} != {manifest.remote_approval.endpoint!r}"
        )
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
    previous_request_count = _request_count(ledger_path)
    if manifest.cohort == "E1-R" and ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("status") in {"complete", "error"}:
                _append_allocation_record(output / "allocation-usage.jsonl", manifest, record)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "partition": "development",
        "cohort": manifest.cohort,
        "candidate": manifest.candidate.model_dump(mode="json"),
        "endpoint": endpoint,
        "manifest_hash": manifest_hash,
        "protocol_hash": protocol_hash,
        "protocol_artifact_hashes": protocol_artifact_hashes,
        "question_ids": [question.question_id for question in manifest.questions],
        "input_preflight": input_preflight,
        "state": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    metadata_path = output / "run.json"

    pending = []
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

    if max_new_requests is not None:
        if max_new_requests <= 0:
            raise ValueError("max_new_requests must be positive")
        pending = pending[:max_new_requests]

    if manifest.cohort == "E1-R" and pending:
        approval = manifest.remote_approval
        request_count = previous_request_count + len(pending)
        if request_count > approval.max_total_requests:
            raise ValueError(
                "E1-R dispatch exceeds approved total request limit: "
                f"{request_count} > {approval.max_total_requests}"
            )

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
                    request_started = time.perf_counter()
                    request_started_at = datetime.now(UTC).isoformat()
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
                            "endpoint": endpoint,
                            "requested_model_id": manifest.candidate.model_id,
                            "request_started_at_utc": request_started_at,
                            "latency_s": round(time.perf_counter() - request_started, 3),
                            "provider_response_metadata": {},
                            "input_preflight": input_preflight.get(question.question_id),
                            "error_type": type(error).__name__,
                            "error_message": _safe_error_message(str(error)),
                            "completed_at_utc": datetime.now(UTC).isoformat(),
                        }
                        ledger.write(json.dumps(record, sort_keys=True) + "\n")
                        ledger.flush()
                        os.fsync(ledger.fileno())
                        attempts[response_key] = ("error", attempt)
                        if manifest.cohort == "E1-R":
                            _append_allocation_record(output / "allocation-usage.jsonl", manifest, record)
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
                    response_status = (
                        "complete"
                        if manifest.cohort != "E1-R"
                        or (schema_error is None and not reply.truncated)
                        else "error"
                    )
                    record = {
                        "schema_version": 1,
                        "status": response_status,
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
                        "endpoint": endpoint,
                        "requested_model_id": manifest.candidate.model_id,
                        "request_started_at_utc": request_started_at,
                        "latency_s": reply.latency_s,
                        "input_preflight": input_preflight.get(question.question_id),
                        "raw_response": raw_response,
                        "provider_response_metadata": raw_response.get("response_metadata", {}),
                        "provider_usage_metadata": raw_response.get("usage_metadata", {}),
                        "reasoning_tokens_reported": _reasoning_tokens_reported(raw_response),
                        "response": reply.model_dump(mode="json"),
                        "parsed_response": parsed_data,
                        "schema_error": schema_error,
                        "scientific_reviewability": (
                            "reviewable"
                            if response_status == "complete"
                            else "nonreviewable_technical_output"
                        ),
                        "nonreviewable_reason": (
                            "truncated_by_output_limit"
                            if reply.truncated
                            else "schema_validation_failed"
                            if schema_error is not None
                            else None
                        ),
                        "completed_at_utc": datetime.now(UTC).isoformat(),
                    }
                    ledger.write(json.dumps(record, sort_keys=True) + "\n")
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    if response_status == "complete":
                        completed.add(response_key)
                    attempts[response_key] = (response_status, attempt)
                    if manifest.cohort == "E1-R":
                        _append_allocation_record(output / "allocation-usage.jsonl", manifest, record)
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


def _reasoning_tokens_reported(raw_response: dict[str, Any]) -> int | None:
    usage = raw_response.get("usage_metadata") or {}
    details = usage.get("output_token_details") or usage.get("completion_tokens_details") or {}
    value = details.get("reasoning") or details.get("reasoning_tokens") or usage.get("reasoning_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _preflight_remote_inputs(
    manifest: E1DevelopmentManifest, system_prompt: str
) -> dict[str, dict[str, Any]]:
    if manifest.cohort != "E1-R":
        return {}
    approval = manifest.remote_approval
    if approval is None:
        raise ValueError("E1-R input preflight requires remote approval")

    counts: dict[str, dict[str, Any]] = {}
    campaign_total = 0
    for question in manifest.questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.prompt},
        ]
        count = len(
            json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        method = "serialized-messages-utf8-bytes"
        unit = "utf8_bytes"
        question_limit = approval.max_input_message_bytes_per_question
        campaign_limit = approval.max_input_message_bytes_campaign

        campaign_total += count
        if count > question_limit:
            raise ValueError(
                f"E1 input preflight exceeds {unit} limit for {question.question_id}: "
                f"{count} > {question_limit}"
            )
        if campaign_total > campaign_limit:
            raise ValueError(
                f"E1 input preflight exceeds campaign {unit} limit: "
                f"{campaign_total} > {campaign_limit}"
            )
        counts[question.question_id] = {"count": count, "method": method, "unit": unit}
    worst_case_campaign_bytes = sum(
        count["count"] * approval.max_attempts_per_question for count in counts.values()
    )
    if worst_case_campaign_bytes > approval.max_input_message_bytes_campaign:
        raise ValueError(
            "E1 input preflight exceeds worst-case campaign utf8_bytes limit: "
            f"{worst_case_campaign_bytes} > {approval.max_input_message_bytes_campaign}"
        )
    return counts


def _provider_endpoint(provider: str) -> str | None:
    provider = provider.lower()
    environment_defaults = {
        "mag": ("MATSIM_MAG_BASE_URL", "https://mag.genesis.americansciencecloud.org/v1"),
        "vllm": ("MATSIM_VLLM_BASE_URL", "http://localhost:8000/v1"),
        "ollama": ("MATSIM_OLLAMA_BASE_URL", "http://localhost:11434"),
        "openai": ("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "anthropic": ("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
    }
    environment_default = environment_defaults.get(provider)
    if environment_default is None:
        return None
    environment_name, default = environment_default
    return os.environ.get(environment_name, default)


def _append_allocation_record(
    path: Path, manifest: E1DevelopmentManifest, response: dict[str, Any]
) -> None:
    approval = manifest.remote_approval
    if approval is None:
        raise ValueError("E1-R allocation ledger requires remote approval data")
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
    cost_record = {
        "response_key": response["response_key"],
        "question_id": response["question_id"],
        "candidate": response["candidate"],
        "cohort": "E1-R",
        "attempt": response["attempt"],
        "status": response["status"],
        "prompt_tokens_reported": reply.get("prompt_tokens"),
        "completion_tokens_reported": reply.get("completion_tokens"),
        "reasoning_tokens_reported": response.get("reasoning_tokens_reported"),
        "finish_reason": reply.get("finish_reason"),
        "latency_s": reply.get("latency_s"),
        "billing_basis": "project_allocation",
        "approval_reference": approval.approval_reference,
        "allocation_limit_reference": approval.allocation_limit_reference,
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