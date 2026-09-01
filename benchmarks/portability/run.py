#!/usr/bin/env python3
"""Plan and execute the fixed cross-facility portability benchmark.

Scientific choices live in ``config/science.yaml``.  Machine overlays may set
only scheduler/resource/launcher values, which prevents a site script from
silently changing the scientific problem being compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FACILITIES = {"frontier", "aurora", "perlmutter"}
FACILITY_KEYS = {
    "facility",
    "scheduler",
    "accelerator",
    "gpus_per_node",
    "ranks_per_node",
    "launchers",
}


def load_yaml(path: Path) -> dict[str, Any]:
    # Checked-in .yaml uses JSON syntax (a strict YAML subset), keeping the
    # smoke gate usable while the Python environment itself is being tested.
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def resolved_configuration(facility: str) -> dict[str, Any]:
    if facility not in FACILITIES:
        raise ValueError(f"unsupported facility {facility!r}")
    science = load_yaml(HERE / "config" / "science.yaml")
    overlay = load_yaml(HERE / "config" / f"{facility}.yaml")
    unexpected = set(overlay) - FACILITY_KEYS
    if unexpected:
        raise ValueError(f"facility overlay changes non-deployment keys: {sorted(unexpected)}")
    return {"science": science, "deployment": overlay}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    science = config["science"]
    structure = HERE / science["structure"]
    if not structure.is_file():
        errors.append(f"missing fixed structure: {structure}")
    deployment = config["deployment"]
    for backend, relative in deployment["launchers"].items():
        launcher = ROOT / relative
        if not launcher.is_file():
            errors.append(f"missing {backend} launcher: {launcher}")
    al = science["active_learning"]
    if al["retrain"] or al["promote_model"]:
        errors.append("portability benchmark must not retrain or promote a model")
    return errors


def environment_record(config: dict[str, Any]) -> dict[str, Any]:
    deployment = config["deployment"]
    structure = HERE / config["science"]["structure"]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "facility": deployment["facility"],
        "scheduler": deployment["scheduler"],
        "hostname": platform.node(),
        "python": sys.version,
        "platform": platform.platform(),
        "structure_sha256": _sha256(structure),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
        ).stdout.strip(),
        "allocation": {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_ID", "SLURM_JOB_NUM_NODES", "PBS_JOBID", "PBS_NODEFILE")
            if os.environ.get(key)
        },
        "executables": {name: shutil.which(name) for name in ("python", "pw.x", "vasp_std")},
    }


def build_plan(config: dict[str, Any], suite: str, backend: str) -> list[dict[str, Any]]:
    science = config["science"]
    if suite == "smoke":
        return [{"stage": "environment"}, {"stage": "launcher", "backend": backend}]
    if suite == "relaxation":
        return [
            {
                "stage": "relaxation",
                "adapter": "deterministic",
                "production_target": backend,
            }
        ]
    if suite == "active-learning":
        return [{"stage": "active-learning", **science["active_learning"], "backend": backend}]
    if suite == "llm-discussion":
        return [
            {
                "stage": "llm-discussion",
                "turns": ["proposal", "critique", "revision"],
                "phase_dispatch": True,
            }
        ]
    return [{"stage": "phase-exploration", **science["phase_exploration"], "backend": backend}]


def execute_smoke(config: dict[str, Any], backend: str) -> dict[str, Any]:
    if backend == "mlip":
        completed = subprocess.run(
            [sys.executable, "-c", "import ase, matsim_agents; print(ase.__version__)"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    else:
        relative = config["deployment"]["launchers"][backend]
        completed = subprocess.run(
            ["bash", "-n", str(ROOT / relative)], text=True, capture_output=True, check=False
        )
    return {
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def validate_llm_check(path: Path) -> dict[str, Any]:
    """Require a successful, fully qualified LLM-check directory."""

    result_path = path / "result.json"
    identity_path = path / "model_identity.json"
    if not result_path.is_file() or not identity_path.is_file():
        raise ValueError("LLM-check run must contain result.json and model_identity.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise ValueError("referenced LLM-check run did not complete successfully")
    required = {"readiness", "load", "generation", "structured", "discussion", "distributed"}
    if not all(result.get("stages", {}).get(stage) for stage in required):
        raise ValueError("referenced LLM-check run did not pass every required stage")
    return {"run_id": result.get("run_id"), "run_directory": str(path), "model": identity}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facility", required=True, choices=sorted(FACILITIES))
    parser.add_argument(
        "--suite",
        default="smoke",
        choices=[
            "smoke",
            "relaxation",
            "active-learning",
            "phase-exploration",
            "llm-discussion",
            "all",
        ],
    )
    parser.add_argument("--backend", default="qe", choices=["mlip", "qe", "vasp"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute supported stages; otherwise emit a deterministic plan.",
    )
    parser.add_argument(
        "--live-llm",
        action="store_true",
        help="Use the configured LLM for proposal and critique instead of deterministic responses.",
    )
    parser.add_argument(
        "--llm-check-run",
        type=Path,
        help="Successful matsim-agents llm-check directory required by --live-llm.",
    )
    args = parser.parse_args()

    llm_qualification = None
    if args.live_llm:
        if args.llm_check_run is None:
            parser.error("--live-llm requires --llm-check-run")
        llm_qualification = validate_llm_check(args.llm_check_run)
        identity = llm_qualification["model"]
        qualified_provider = str(identity["provider"])
        qualified_model = str(identity["model"])
        if os.environ.get("MATSIM_LLM_PROVIDER", qualified_provider) != qualified_provider:
            parser.error("MATSIM_LLM_PROVIDER differs from the qualified LLM-check provider")
        if os.environ.get("MATSIM_LLM_MODEL", qualified_model) != qualified_model:
            parser.error("MATSIM_LLM_MODEL differs from the qualified LLM-check model")
        os.environ.setdefault("MATSIM_LLM_PROVIDER", qualified_provider)
        os.environ.setdefault("MATSIM_LLM_MODEL", qualified_model)
        if identity.get("base_url"):
            os.environ.setdefault("MATSIM_VLLM_BASE_URL", str(identity["base_url"]))

    config = resolved_configuration(args.facility)
    errors = validate_inputs(config)
    if errors:
        raise SystemExit("\n".join(errors))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "environment.json").write_text(
        json.dumps(environment_record(config), indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "resolved_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    suites = (
        ["smoke", "relaxation", "active-learning", "phase-exploration", "llm-discussion"]
        if args.suite == "all"
        else [args.suite]
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "facility": args.facility,
        "suite": args.suite,
        "backend": args.backend,
        "status": "planned",
        "plan": [step for suite in suites for step in build_plan(config, suite, args.backend)],
        "llm_qualification": llm_qualification,
    }
    if args.execute:
        from benchmarks.portability.suites import execute_contract_suite

        execution: dict[str, Any] = {}
        for suite in suites:
            if suite == "smoke":
                execution[suite] = execute_smoke(config, args.backend)
            else:
                execution[suite] = execute_contract_suite(
                    suite,
                    structure=HERE / config["science"]["structure"],
                    output=args.output / suite,
                    live_llm=args.live_llm,
                )
        result["execution"] = execution
        failed = any(
            payload.get("return_code", 0) != 0
            or str(payload.get("status", "complete")) in {"failed", "rejected"}
            for payload in execution.values()
        )
        result["status"] = "failed" if failed else "passed"
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
