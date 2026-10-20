"""Independent readiness and portability qualification for LLM backends."""

from __future__ import annotations

import functools
import json
import os
import platform
import resource
import sys
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from matsim_agents.backends.llm.provider import DEFAULT_MODELS, get_chat_model
from matsim_agents.backends.llm.servers import llm_endpoints
from matsim_agents.execution.contracts import EvidenceLevel, ProvenanceRecord, WorkflowStatus
from matsim_agents.execution.placement import ResourcePlan
from matsim_agents.execution.run_directory import ScientificRunDirectory
from matsim_agents.workflows.investigation import ScientificHypothesis

PROVIDERS = frozenset(DEFAULT_MODELS)


class LLMCheckConfig(BaseModel):
    provider: str = "ollama"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    output_root: str = "./runs/llm-check"
    temperature: float = 0.0
    timeout_sec: int = Field(120, ge=1)
    expected_accelerators: int | None = Field(None, ge=1)
    tensor_parallel_size: int | None = Field(None, ge=1)
    pipeline_parallel_size: int = Field(1, ge=1)
    expected_devices: list[int] | None = Field(
        None,
        description=(
            "In-process models: device ordinals the model may occupy. The distributed "
            "stage checks the loaded model's device map against them."
        ),
    )
    context_length: int | None = Field(None, ge=1)
    concurrent_requests: int = Field(1, ge=1)
    deterministic_prompt: str = "Return exactly the token PORTABILITY_OK and no additional text."
    scientific_objective: str = "Propose a testable hypothesis for stable elemental silicon."
    server: str | None = Field(
        None,
        description=(
            "Check this server of the plan's llm pool: from resources (default: its "
            "only server) or from the plan served by `matsim-agents resources serve`."
        ),
    )
    resources: ResourcePlan | None = None

    @property
    def uses_plan_server(self) -> bool:
        return self.server is not None or self.resources is not None

    @model_validator(mode="after")
    def _validate_provider(self) -> LLMCheckConfig:
        if self.uses_plan_server:
            if set_fields := sorted({"provider", "model", "base_url"} & self.model_fields_set):
                raise ValueError(
                    "llm-check takes its model from the plan's server; remove "
                    + ", ".join(set_fields)
                )
            return self
        self.provider = self.provider.lower()
        if self.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(PROVIDERS)}")
        self.model = self.model or DEFAULT_MODELS[self.provider]
        if self.provider == "vllm":
            self.base_url = self.base_url or "http://localhost:8000/v1"
            if not self.base_url.rstrip("/").endswith("/v1"):
                raise ValueError("vLLM base_url must identify the OpenAI-compatible /v1 root")
        return self


class LLMCheckResult(BaseModel):
    run_id: str
    run_directory: str
    status: WorkflowStatus
    provider: str
    model: str
    stages: dict[str, bool]
    failure_reason: str | None = None


def _get_json(url: str, timeout: int, api_key: str | None = None) -> Any:
    headers = {"Accept": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def probe_readiness(cfg: LLMCheckConfig) -> dict[str, Any]:
    """Check artifacts, credentials, or an already-running provider endpoint."""

    if cfg.provider == "vllm":
        key = cfg.api_key or os.environ.get("MATSIM_VLLM_API_KEY")
        payload = _get_json(f"{cfg.base_url.rstrip('/')}/models", cfg.timeout_sec, key)
        models = [entry.get("id") for entry in payload.get("data", [])]
        if cfg.model not in models:
            raise RuntimeError(f"configured model {cfg.model!r} not served; available={models}")
        return {"endpoint": cfg.base_url, "served_models": models}
    if cfg.provider == "ollama":
        base = cfg.base_url or os.environ.get("MATSIM_OLLAMA_BASE_URL", "http://localhost:11434")
        payload = _get_json(f"{base.rstrip('/')}/api/tags", cfg.timeout_sec)
        models = [entry.get("name") for entry in payload.get("models", [])]
        if cfg.model not in models:
            raise RuntimeError(
                f"configured Ollama model {cfg.model!r} not installed; available={models}"
            )
        return {"endpoint": base, "installed_models": models}
    if cfg.provider == "huggingface":
        location = Path(os.environ.get("MATSIM_HF_MODEL_PATH", cfg.model or ""))
        if location.exists():
            required = [location / "config.json"]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise RuntimeError(f"incomplete local Hugging Face model: missing {missing}")
            return {"model_path": str(location.resolve()), "local": True}
        return {"model_id": cfg.model, "local": False}
    credential = "OPENAI_API_KEY" if cfg.provider == "openai" else "ANTHROPIC_API_KEY"
    if not (cfg.api_key or os.environ.get(credential)):
        raise RuntimeError(f"{cfg.provider} readiness requires {credential}")
    return {"credential": credential, "configured": True}


def _content(response: Any) -> str:
    value = getattr(response, "content", response)
    return str(value).strip()


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
    start, stop = stripped.find("{"), stripped.rfind("}")
    if start < 0 or stop < start:
        raise ValueError("response does not contain a JSON object")
    value = json.loads(stripped[start : stop + 1])
    if not isinstance(value, dict):
        raise ValueError("structured response must be a JSON object")
    return value


def _accelerator_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "ZE_AFFINITY_MASK": os.environ.get("ZE_AFFINITY_MASK"),
    }
    try:
        import torch

        record.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "accelerator_count": torch.cuda.device_count(),
                "accelerator_names": [
                    torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
                ],
            }
        )
    except ImportError:
        record["torch_available"] = False
    return record


def _model_devices(model: Any) -> set[str]:
    """Devices a loaded in-process model occupies: ordinals, or names such as ``cpu``."""

    pipeline = getattr(getattr(model, "llm", None), "pipeline", None)
    loaded = getattr(pipeline, "model", None)
    device_map = getattr(loaded, "hf_device_map", None)
    if device_map:
        values = device_map.values()
    elif loaded is not None and hasattr(loaded, "device"):
        values = [loaded.device]
    else:
        return set()
    devices = set()
    for value in values:
        index = getattr(value, "index", value)
        devices.add(str(index) if isinstance(index, int) else str(value).split(":")[-1])
    return devices


def _model_identity(cfg: LLMCheckConfig, readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "temperature": cfg.temperature,
        "tensor_parallel_size": cfg.tensor_parallel_size,
        "pipeline_parallel_size": cfg.pipeline_parallel_size,
        "expected_devices": cfg.expected_devices,
        "context_length": cfg.context_length,
        "readiness": readiness,
    }


def _raise_setup_failure(reason: str, _cfg: LLMCheckConfig) -> dict[str, Any]:
    raise RuntimeError(reason)


def run_llm_check(
    cfg: LLMCheckConfig,
    *,
    model_factory: Callable[..., Any] = get_chat_model,
    readiness_probe: Callable[[LLMCheckConfig], dict[str, Any]] = probe_readiness,
) -> LLMCheckResult:
    """Run all LLM checks and persist a terminal result even on failure.

    With a plan server, the check runs against that server's endpoint, and its
    tensor-parallel size, device count, and context length default to the
    placement's.
    """

    if not cfg.uses_plan_server:
        return _run_llm_check(cfg, model_factory=model_factory, readiness_probe=readiness_probe)
    stack = ExitStack()
    try:
        services = stack.enter_context(llm_endpoints(cfg.resources, output_dir=cfg.output_root))
        endpoint = services.endpoint(cfg.server)
    except Exception as exc:  # noqa: BLE001 - a server that cannot start is a recorded failure
        stack.close()
        reason = f"plan server {cfg.server or '(only server)'} unavailable: {exc}"
        unresolved = cfg.model_copy(
            update={"provider": "plan-server", "model": cfg.server, "resources": None}
        )
        return _run_llm_check(
            unresolved,
            model_factory=model_factory,
            readiness_probe=functools.partial(_raise_setup_failure, reason),
        )
    with stack:
        resolved = cfg.model_copy(
            update={
                "provider": endpoint.provider,
                "model": endpoint.model,
                "base_url": endpoint.base_url,
                # None falls back to the provider default (MATSIM_VLLM_API_KEY or EMPTY).
                "api_key": cfg.api_key or endpoint.api_key,
                "tensor_parallel_size": cfg.tensor_parallel_size or endpoint.tensor_parallel_size,
                "pipeline_parallel_size": endpoint.pipeline_parallel_size
                or cfg.pipeline_parallel_size,
                # A vLLM server's devices are its workers; an in-process model is checked
                # against its assigned devices, not every device this process can see.
                "expected_accelerators": cfg.expected_accelerators
                or (endpoint.device_count if endpoint.provider == "vllm" else None),
                "expected_devices": cfg.expected_devices
                or (list(endpoint.devices) if endpoint.devices is not None else None),
                "context_length": cfg.context_length or endpoint.max_model_len,
                # Saved configs record the resolved endpoint, which reloads on its own.
                "server": None,
                "resources": None,
            }
        )
        return _run_llm_check(
            resolved, model_factory=model_factory, readiness_probe=readiness_probe
        )


def _run_llm_check(
    cfg: LLMCheckConfig,
    *,
    model_factory: Callable[..., Any],
    readiness_probe: Callable[[LLMCheckConfig], dict[str, Any]],
) -> LLMCheckResult:
    provenance = ProvenanceRecord(
        workflow="llm_readiness_check",
        evidence_level=EvidenceLevel.HYPOTHESIS,
        backend=cfg.provider,
        model_identifier=cfg.model,
        numerical_settings=cfg.model_dump(mode="json", exclude={"api_key"}),
    )
    run = ScientificRunDirectory.create(
        cfg.output_root,
        workflow="llm_readiness_check",
        request={"provider": cfg.provider, "model": cfg.model},
        resolved_config=cfg.model_dump(mode="json", exclude={"api_key"}),
        provenance=provenance,
    )
    stages = {
        name: False
        for name in ("readiness", "load", "generation", "structured", "discussion", "distributed")
    }
    failure = None
    accelerator = _accelerator_record()
    run.write_json(
        "environment.json",
        {"python": sys.version, "platform": platform.platform(), "accelerator": accelerator},
    )
    try:
        readiness = readiness_probe(cfg)
        stages["readiness"] = True
        run.write_json("health.json", readiness)

        before_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        load_start = time.perf_counter()
        model = model_factory(
            provider=cfg.provider,
            model=cfg.model,
            temperature=cfg.temperature,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
        )
        construction_seconds = time.perf_counter() - load_start
        stages["load"] = True
        identity = _model_identity(cfg, readiness)
        run.write_json("model_identity.json", identity)

        generation_start = time.perf_counter()
        generated = _content(model.invoke([HumanMessage(content=cfg.deterministic_prompt)]))
        generation_seconds = time.perf_counter() - generation_start
        if "PORTABILITY_OK" not in generated:
            raise RuntimeError("deterministic generation did not contain PORTABILITY_OK")
        stages["generation"] = True
        concurrency_start = time.perf_counter()
        if cfg.concurrent_requests > 1:
            with ThreadPoolExecutor(max_workers=cfg.concurrent_requests) as executor:
                concurrent_outputs = list(
                    executor.map(
                        lambda _index: _content(
                            model.invoke([HumanMessage(content=cfg.deterministic_prompt)])
                        ),
                        range(cfg.concurrent_requests),
                    )
                )
            if any("PORTABILITY_OK" not in output for output in concurrent_outputs):
                raise RuntimeError("one or more concurrent generations failed the invariant")
        else:
            concurrent_outputs = [generated]
        concurrent_seconds = time.perf_counter() - concurrency_start
        run.write_json(
            "generation.json",
            {
                "prompt": cfg.deterministic_prompt,
                "response": generated,
                "concurrent_responses": concurrent_outputs,
            },
        )

        schema = ScientificHypothesis.model_json_schema()
        structured_prompt = (
            f"{cfg.scientific_objective}\nReturn only JSON matching this schema:\n"
            f"{json.dumps(schema, sort_keys=True)}"
        )
        structured_text = _content(model.invoke([HumanMessage(content=structured_prompt)]))
        hypothesis = ScientificHypothesis.model_validate(_json_object(structured_text))
        stages["structured"] = True
        run.write_json("structured_hypothesis.json", hypothesis)

        proposal = _content(model.invoke([HumanMessage(content=cfg.scientific_objective)]))
        critique_prompt = f"Critique this hypothesis and identify a missing test:\n{proposal}"
        critique = _content(
            model.invoke(
                [
                    HumanMessage(content=cfg.scientific_objective),
                    AIMessage(content=proposal),
                    HumanMessage(content=critique_prompt),
                ]
            )
        )
        revision_prompt = (
            "Revise the proposal in response to the critique and mention the critique."
        )
        revision = _content(
            model.invoke(
                [
                    SystemMessage(
                        content="Revise scientific claims only when evidence supports it."
                    ),
                    HumanMessage(content=cfg.scientific_objective),
                    AIMessage(content=proposal),
                    HumanMessage(content=critique),
                    HumanMessage(content=revision_prompt),
                ]
            )
        )
        if not all((proposal, critique, revision)):
            raise RuntimeError("proposal, critique, and revision must all be nonempty")
        stages["discussion"] = True
        run.write_json(
            "discussion.json", {"proposal": proposal, "critique": critique, "revision": revision}
        )

        if cfg.expected_devices is not None:
            used = _model_devices(model)
            outside = sorted(set(used) - {str(device) for device in cfg.expected_devices})
            if not used or outside:
                raise RuntimeError(
                    f"in-process model occupies {sorted(used)}; expected only devices "
                    f"{cfg.expected_devices}"
                )
            identity["mapped_devices"] = sorted(used)
            run.write_json("model_identity.json", identity)
        if cfg.expected_accelerators is not None:
            if cfg.provider == "vllm":
                workers = (cfg.tensor_parallel_size or 0) * cfg.pipeline_parallel_size
                if workers != cfg.expected_accelerators:
                    raise RuntimeError(
                        "for remote vLLM, expected_accelerators must equal tensor_parallel_size "
                        "x pipeline_parallel_size; verify actual workers in the server launch log"
                    )
            elif accelerator.get("accelerator_count") != cfg.expected_accelerators:
                raise RuntimeError(
                    f"expected {cfg.expected_accelerators} accelerators, found "
                    f"{accelerator.get('accelerator_count', 0)}"
                )
        stages["distributed"] = True
        after_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        run.write_json(
            "performance.json",
            {
                "client_or_local_model_construction_seconds": construction_seconds,
                "generation_seconds": generation_seconds,
                "concurrent_generation_seconds": concurrent_seconds,
                "max_rss_before": before_memory,
                "max_rss_after": after_memory,
                "concurrent_requests_configured": cfg.concurrent_requests,
            },
        )
        status = WorkflowStatus.COMPLETE
    except Exception as exc:  # noqa: BLE001 - readiness failures are artifacts
        status = WorkflowStatus.FAILED
        failure = str(exc)
    result = LLMCheckResult(
        run_id=run.run_id,
        run_directory=str(run.path),
        status=status,
        provider=cfg.provider,
        model=cfg.model or "",
        stages=stages,
        failure_reason=failure,
    )
    run.write_json("result.json", result)
    run.append_event("run_finished", {"status": status, "failure_reason": failure})
    return result


__all__ = ["LLMCheckConfig", "LLMCheckResult", "PROVIDERS", "probe_readiness", "run_llm_check"]
