#!/usr/bin/env python3
"""Run one identical search prompt across six open-source LLMs.

This script is a practical benchmark harness for matsim-agents users who want
an apples-to-apples comparison of model behavior on the same prompt.

It records per-model latency, raw response text, extracted compositions,
keyword hits, and errors, then writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from matsim_agents.discovery.composition import extract_compositions
from matsim_agents.llm import get_chat_model


@dataclass
class ModelSpec:
    name: str
    provider: str
    model: str
    base_url_env: str | None = None
    fallback_base_url_env: str | None = None
    api_key_env: str | None = None


DEFAULT_SPECS = [
    ModelSpec(
        name="qwen2.5-72b-instruct",
        provider="vllm",
        model="Qwen/Qwen2.5-72B-Instruct",
        base_url_env="MATSIM_VLLM_QWEN72_BASE_URL",
        fallback_base_url_env="MATSIM_VLLM_BASE_URL",
        api_key_env="MATSIM_VLLM_API_KEY",
    ),
    ModelSpec(
        name="qwen2.5-14b-instruct",
        provider="ollama",
        model="qwen2.5:14b",
        base_url_env="MATSIM_OLLAMA_BASE_URL",
    ),
    ModelSpec(
        name="llama-3.1-70b-instruct",
        provider="vllm",
        model="meta-llama/Llama-3.1-70B-Instruct",
        base_url_env="MATSIM_VLLM_LLAMA70_BASE_URL",
        fallback_base_url_env="MATSIM_VLLM_BASE_URL",
        api_key_env="MATSIM_VLLM_API_KEY",
    ),
    ModelSpec(
        name="llama-3.1-8b-instruct",
        provider="ollama",
        model="llama3.1:8b",
        base_url_env="MATSIM_OLLAMA_BASE_URL",
    ),
    ModelSpec(
        name="mixtral-8x22b-instruct",
        provider="vllm",
        model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        base_url_env="MATSIM_VLLM_MIXTRAL_BASE_URL",
        fallback_base_url_env="MATSIM_VLLM_BASE_URL",
        api_key_env="MATSIM_VLLM_API_KEY",
    ),
    ModelSpec(
        name="deepseek-r1-distill-qwen-32b",
        provider="vllm",
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        base_url_env="MATSIM_VLLM_DEEPSEEK32_BASE_URL",
        fallback_base_url_env="MATSIM_VLLM_BASE_URL",
        api_key_env="MATSIM_VLLM_API_KEY",
    ),
]


def _resolve_env(primary: str | None, fallback: str | None = None) -> str | None:
    if primary:
        value = os.environ.get(primary)
        if value:
            return value
    if fallback:
        value = os.environ.get(fallback)
        if value:
            return value
    return None


def _load_specs(spec_file: Path | None) -> list[ModelSpec]:
    if spec_file is None:
        return DEFAULT_SPECS

    payload = json.loads(spec_file.read_text())
    if not isinstance(payload, list):
        raise ValueError("Spec file must be a JSON list.")

    specs: list[ModelSpec] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Spec entry {idx} must be a JSON object.")
        specs.append(
            ModelSpec(
                name=item["name"],
                provider=item["provider"],
                model=item["model"],
                base_url_env=item.get("base_url_env"),
                fallback_base_url_env=item.get("fallback_base_url_env"),
                api_key_env=item.get("api_key_env"),
            )
        )
    return specs


def _coerce_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content)


def _keyword_score(text: str, keywords: list[str]) -> dict[str, bool]:
    lower = text.lower()
    return {kw: (kw.lower() in lower) for kw in keywords}


def run_eval(prompt: str, specs: list[ModelSpec], keywords: list[str], temperature: float) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []

    for spec in specs:
        base_url = _resolve_env(spec.base_url_env, spec.fallback_base_url_env)
        api_key = _resolve_env(spec.api_key_env)

        row: dict[str, Any] = {
            "name": spec.name,
            "provider": spec.provider,
            "model": spec.model,
            "base_url": base_url,
            "status": "ok",
            "latency_sec": None,
            "response_text": "",
            "response_chars": 0,
            "compositions": [],
            "keyword_hits": {},
            "error": None,
        }

        t0 = time.perf_counter()
        try:
            chat_model = get_chat_model(
                provider=spec.provider,
                model=spec.model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
            )
            reply = chat_model.invoke([HumanMessage(content=prompt)])
            text = _coerce_text(reply.content)
            row["response_text"] = text
            row["response_chars"] = len(text)
            row["compositions"] = sorted({c.formula for c in extract_compositions(text)})
            row["keyword_hits"] = _keyword_score(text, keywords)
        except Exception as exc:  # pragma: no cover - depends on external services
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            row["latency_sec"] = round(time.perf_counter() - t0, 3)

        results.append(row)

    successful = [r for r in results if r["status"] == "ok"]
    mean_latency = (
        round(sum(float(r["latency_sec"]) for r in successful) / len(successful), 3)
        if successful
        else None
    )

    summary = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "keywords": keywords,
        "num_models": len(specs),
        "num_success": len(successful),
        "num_errors": len(specs) - len(successful),
        "mean_latency_sec_success": mean_latency,
    }

    return {
        "summary": summary,
        "models": [asdict(s) for s in specs],
        "results": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one identical search prompt across six open-source models and write JSON results."
        )
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Single prompt to run against every model.",
    )
    parser.add_argument(
        "--spec-file",
        type=Path,
        default=None,
        help="Optional JSON list of model specs. Defaults to the built-in six-model matrix.",
    )
    parser.add_argument(
        "--keywords",
        default="stability,band gap,formation energy,synthesis",
        help="Comma-separated keywords used for simple response scoring.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature used for all models.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to runs/model-eval-<timestamp>.json.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    specs = _load_specs(args.spec_file)
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]

    out_path = args.out
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = Path("runs") / f"model-eval-{ts}.json"

    report = run_eval(
        prompt=args.prompt,
        specs=specs,
        keywords=keywords,
        temperature=args.temperature,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print("Model prompt evaluation completed.")
    print(f"Output: {out_path}")
    print(
        "Summary: "
        f"success={report['summary']['num_success']}/{report['summary']['num_models']} "
        f"mean_latency_sec={report['summary']['mean_latency_sec_success']}"
    )

    for row in report["results"]:
        comp_count = len(row["compositions"])
        print(
            f"- {row['name']}: {row['status']} "
            f"latency={row['latency_sec']}s comps={comp_count} chars={row['response_chars']}"
        )
        if row["status"] == "error":
            print(f"  error: {row['error']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
