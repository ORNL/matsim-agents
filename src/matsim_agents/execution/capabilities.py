"""Registry of DFT, MLIP and LLM backends: what each supports and needs to run."""

from __future__ import annotations

import inspect
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from matsim_agents.execution.contracts import EvidenceLevel

BackendKind = Literal["dft", "mlip", "llm"]


class BackendCapability(BaseModel):
    """Static description of one backend."""

    model_config = ConfigDict(frozen=True)

    kind: BackendKind
    name: str
    tasks: tuple[str, ...]
    evidence_levels: tuple[EvidenceLevel, ...]
    uncertainty: tuple[str, ...] = ()
    fine_tuning: bool = False
    python_modules: tuple[str, ...] = ()
    extra: str | None = None
    # Env var naming the launcher command; it must resolve on PATH.
    launcher_env: str | None = None
    required_env: tuple[str, ...] = ()
    optional_env: tuple[str, ...] = ()
    # Companion venv (env var, default dir next to the active venv) for conflicting pins.
    venv_env: str | None = None
    venv_default: str | None = None
    notes: str = ""


class CapabilityCheck(BaseModel):
    """Whether a backend can run in the current environment."""

    name: str
    kind: BackendKind
    available: bool
    missing: tuple[str, ...] = ()
    python: str | None = None


_DFT = (EvidenceLevel.CONVERGED_DFT, EvidenceLevel.LOW_FIDELITY_DFT)
_MLIP = (EvidenceLevel.MLIP_RELAXATION, EvidenceLevel.MLIP_PREDICTION)
_MLIP_TASKS = ("energy-forces", "relax", "vc-relax", "md")
_MLIP_UQ = ("ensemble", "mc-dropout")

CAPABILITIES: tuple[BackendCapability, ...] = (
    BackendCapability(
        kind="dft",
        name="qe",
        tasks=("scf", "relax", "vc-relax"),
        evidence_levels=_DFT,
        launcher_env="MATSIM_QE_LAUNCHER",
        optional_env=("MATSIM_QE_PSEUDO_DIR",),
        notes="Quantum ESPRESSO pw.x; needs a pseudopotential directory.",
    ),
    BackendCapability(
        kind="dft",
        name="vasp",
        tasks=("scf", "relax", "vc-relax", "vc-relax-shape"),
        evidence_levels=_DFT,
        launcher_env="MATSIM_VASP_LAUNCHER",
        optional_env=("MATSIM_VASP_POTCAR_DIR",),
        notes="Licensed; needs a POTCAR directory.",
    ),
    BackendCapability(
        kind="mlip",
        name="hydragnn",
        tasks=_MLIP_TASKS,
        evidence_levels=_MLIP,
        uncertainty=_MLIP_UQ,
        fine_tuning=True,
        python_modules=("torch", "torch_geometric", "hydragnn"),
        extra="hydragnn",
        notes="Native MC dropout; new-head fine-tuning.",
    ),
    BackendCapability(
        kind="mlip",
        name="uma",
        tasks=_MLIP_TASKS,
        evidence_levels=_MLIP,
        uncertainty=_MLIP_UQ,
        fine_tuning=True,
        python_modules=("fairchem.core",),
        extra="uma",
        venv_env="MATSIM_FAIRCHEM_VENV",
        venv_default=".venv-uma",
        notes="Injected MC dropout; separate .venv-uma.",
    ),
    BackendCapability(
        kind="mlip",
        name="mace",
        tasks=_MLIP_TASKS,
        evidence_levels=_MLIP,
        uncertainty=_MLIP_UQ,
        fine_tuning=True,
        python_modules=("mace.calculators",),
        extra="mace",
        venv_env="MATSIM_MACE_VENV",
        venv_default=".venv-mace",
        notes="Injected MC dropout; separate .venv-mace.",
    ),
    BackendCapability(
        kind="llm",
        name="ollama",
        tasks=("chat",),
        evidence_levels=(EvidenceLevel.HYPOTHESIS,),
        python_modules=("langchain_ollama",),
        optional_env=("MATSIM_OLLAMA_BASE_URL",),
        notes="Local Ollama server.",
    ),
    BackendCapability(
        kind="llm",
        name="vllm",
        tasks=("chat",),
        evidence_levels=(EvidenceLevel.HYPOTHESIS,),
        python_modules=("openai",),
        optional_env=("MATSIM_VLLM_BASE_URL", "MATSIM_VLLM_API_KEY"),
        notes="OpenAI-compatible vLLM server.",
    ),
    BackendCapability(
        kind="llm",
        name="openai",
        tasks=("chat",),
        evidence_levels=(EvidenceLevel.HYPOTHESIS,),
        python_modules=("langchain_openai",),
        required_env=("OPENAI_API_KEY",),
    ),
    BackendCapability(
        kind="llm",
        name="anthropic",
        tasks=("chat",),
        evidence_levels=(EvidenceLevel.HYPOTHESIS,),
        python_modules=("langchain_anthropic",),
        required_env=("ANTHROPIC_API_KEY",),
    ),
    BackendCapability(
        kind="llm",
        name="huggingface",
        tasks=("chat",),
        evidence_levels=(EvidenceLevel.HYPOTHESIS,),
        python_modules=("langchain_huggingface", "transformers", "accelerate", "torch"),
        optional_env=("MATSIM_HF_MODEL_PATH",),
        notes="In-process transformers pipeline.",
    ),
)


def get_capability(name: str) -> BackendCapability:
    """The registered backend called ``name`` (case-insensitive)."""
    for capability in CAPABILITIES:
        if capability.name == name.lower():
            return capability
    known = ", ".join(c.name for c in CAPABILITIES)
    raise KeyError(f"unknown backend {name!r}; known: {known}")


def _module_found(module: str) -> bool:
    # Walks package paths instead of find_spec("a.b"), which would run a/__init__.py.
    import importlib.machinery
    import importlib.util

    top, *rest = module.split(".")
    try:
        spec = importlib.util.find_spec(top)
        for part in rest:
            path = spec.submodule_search_locations if spec else None
            if not path:
                return False
            spec = importlib.machinery.PathFinder.find_spec(part, list(path))
    except (ImportError, ValueError):
        return False
    return spec is not None


_PROBE = (
    inspect.getsource(_module_found)
    + "import sys\nsys.exit(0 if all(map(_module_found, sys.argv[1:])) else 1)\n"
)


def _venv_python(capability: BackendCapability, env: Mapping[str, str]) -> Path | None:
    if not capability.venv_env:
        return None
    venv = env.get(capability.venv_env)
    root = Path(venv) if venv else Path(sys.prefix).parent / str(capability.venv_default)
    python = root / "bin" / "python"
    return python if python.is_file() else None


def _venv_has_modules(python: Path, modules: Iterable[str]) -> bool:
    try:
        result = subprocess.run(
            [str(python), "-c", _PROBE, *modules], capture_output=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def check(
    capability: BackendCapability, environ: Mapping[str, str] | None = None
) -> CapabilityCheck:
    """Check modules, launcher and env vars without importing the backend."""
    env = os.environ if environ is None else environ
    python = None
    missing = [f"python module {m}" for m in capability.python_modules if not _module_found(m)]
    venv_python = _venv_python(capability, env) if missing else None
    if venv_python is not None and _venv_has_modules(venv_python, capability.python_modules):
        missing, python = [], str(venv_python)
    if missing and capability.venv_env:
        missing.append(f"venv ${capability.venv_env} or {capability.venv_default}")
    if capability.launcher_env:
        launcher = env.get(capability.launcher_env, "").split()
        if not launcher:
            missing.append(f"env {capability.launcher_env}")
        elif shutil.which(launcher[0], path=env.get("PATH", os.defpath)) is None:
            missing.append(f"executable {launcher[0]}")
    missing += [f"env {var}" for var in capability.required_env if not env.get(var)]
    return CapabilityCheck(
        name=capability.name,
        kind=capability.kind,
        available=not missing,
        missing=tuple(missing),
        python=python,
    )


def check_all(
    kinds: Iterable[str] | None = None, environ: Mapping[str, str] | None = None
) -> list[CapabilityCheck]:
    """Check every registered backend, optionally only of the given kinds."""
    wanted = set(kinds) if kinds else None
    return [check(c, environ) for c in CAPABILITIES if wanted is None or c.kind in wanted]


def render_markdown() -> str:
    """The static capability matrix as a markdown document."""
    lines = [
        "# Backend capabilities",
        "",
        "Generated by `matsim-agents capabilities matrix`; do not edit by hand.",
        "Run `matsim-agents capabilities check` to see what is available locally.",
        "",
        "| Kind | Backend | Tasks | Evidence | Uncertainty | Fine-tuning | Requires | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in CAPABILITIES:
        requires = [f"`{m}`" for m in c.python_modules]
        if c.extra:
            requires.append(f"extra `{c.extra}`")
        if c.venv_env:
            requires.append(f"or `${c.venv_env}`")
        if c.launcher_env:
            requires.append(f"`${c.launcher_env}`")
        requires += [f"`${var}`" for var in c.required_env]
        cells = (
            c.kind,
            f"`{c.name}`",
            ", ".join(c.tasks),
            ", ".join(level.value for level in c.evidence_levels),
            ", ".join(c.uncertainty) or "-",
            "yes" if c.fine_tuning else "no",
            ", ".join(requires) or "-",
            c.notes or "-",
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


__all__ = [
    "CAPABILITIES",
    "BackendCapability",
    "BackendKind",
    "CapabilityCheck",
    "check",
    "check_all",
    "get_capability",
    "render_markdown",
]
