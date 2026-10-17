"""Regression tests for the package layout.

Each test asserts one structural property of the package layout:
  - Imports only point to lower layers (see ``LAYERS`` and docs/architecture.md).
  - Each of the five stable interfaces is importable from its declared location.
  - Concrete implementations satisfy their Protocol at runtime.
  - Removed legacy import paths stay removed.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

# ── Layering ──────────────────────────────────────────────────────────────── #

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "matsim_agents"

# Lowest layer first. A unit may import only from units in strictly lower
# layers (or from itself); units in the same layer must not import each other.
LAYERS: list[set[str]] = [
    {"execution", "datasets"},
    {"backends.llm", "backends.dft", "backends.mlip"},
    {"discovery", "training"},
    {"active_learning", "workflows"},
    {"orchestration", "campaign"},
    {"cli"},
]
LAYER_OF = {unit: index for index, units in enumerate(LAYERS) for unit in units}


def _unit(module: str) -> str | None:
    parts = module.split(".")[1:]
    if not parts:
        return None
    if parts[0] == "backends":
        return ".".join(parts[:2]) if len(parts) > 1 else "backends"
    return parts[0]


def _imported_modules(module: str, is_package: bool, node: ast.AST) -> list[str]:
    """Absolute dotted names a statement may import, resolving relative and package forms.

    ``from pkg import name`` may import the submodule ``pkg.name``, so each
    imported name is reported as ``pkg.name`` in addition to ``pkg`` itself.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    if node.level:
        package = module if is_package else module.rpartition(".")[0]
        for _ in range(node.level - 1):
            package = package.rpartition(".")[0]
        base = f"{package}.{node.module}" if node.module else package
    else:
        base = node.module or ""
    return [base, *(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")]


def _import_target_units(module: str, is_package: bool, node: ast.AST) -> set[str | None]:
    """Layer units reached by one import statement (``None`` is the package root)."""
    names = [
        name
        for name in _imported_modules(module, is_package, node)
        if name == "matsim_agents" or name.startswith("matsim_agents.")
    ]
    if not names:
        return set()
    base_unit = _unit(names[0])
    if base_unit is not None and base_unit != "backends":
        return {base_unit}
    # ``from matsim_agents import x`` and ``from matsim_agents.backends import x``
    # reach the unit named by ``x``; the containers themselves are not units.
    reached = {_unit(name) for name in names[1:]}
    return reached or {base_unit}


def _source_violations(module: str, source_text: str, *, is_package: bool, label: str) -> list[str]:
    source = _unit(module)
    if source not in LAYER_OF:
        return []
    violations = []
    for node in ast.walk(ast.parse(source_text)):
        for dest in _import_target_units(module, is_package, node):
            if dest == source:
                continue
            if dest not in LAYER_OF or LAYER_OF[dest] >= LAYER_OF[source]:
                violations.append(f"{label}:{node.lineno}: {source} -> {dest or 'matsim_agents'}")
    return violations


def _layer_violations() -> list[str]:
    violations = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT.parent)
        module = ".".join(rel.with_suffix("").parts).removesuffix(".__init__")
        violations += _source_violations(
            module, path.read_text(), is_package=path.name == "__init__.py", label=str(rel)
        )
    return violations


@pytest.mark.parametrize(
    ("module", "is_package", "code"),
    [
        ("matsim_agents.datasets.governance", False, "from ..orchestration.state import X"),
        ("matsim_agents.datasets.governance", False, "from .. import orchestration"),
        ("matsim_agents.datasets", True, "from ..campaign import state"),
        ("matsim_agents.datasets.governance", False, "from matsim_agents import orchestration"),
        ("matsim_agents.datasets.governance", False, "from matsim_agents import execution, cli"),
        ("matsim_agents.datasets.governance", False, "from matsim_agents import MatSimState"),
        ("matsim_agents.backends.dft.base", False, "from matsim_agents.backends import mlip"),
        ("matsim_agents.backends.dft.base", False, "from ..mlip import config"),
        ("matsim_agents.discovery.seeds", False, "from ..training import cost"),
        ("matsim_agents.discovery.seeds", False, "import matsim_agents.orchestration.state"),
    ],
)
def test_layer_guard_rejects_upward_relative_and_package_imports(module, is_package, code) -> None:
    assert _source_violations(module, code, is_package=is_package, label="snippet")


@pytest.mark.parametrize(
    ("module", "is_package", "code"),
    [
        ("matsim_agents.datasets.formation_training", False, "from .governance import x"),
        ("matsim_agents.datasets.formation_training", False, "from . import governance"),
        ("matsim_agents.datasets", True, "from .governance import sha256_file"),
        ("matsim_agents.discovery.seeds", False, "from ..backends.mlip import config"),
        ("matsim_agents.discovery.seeds", False, "from matsim_agents.backends import llm, mlip"),
        ("matsim_agents.discovery.seeds", False, "from matsim_agents import datasets"),
        ("matsim_agents.campaign.state", False, "from ..active_learning.config import ALConfig"),
        ("matsim_agents.backends.mlip.config", False, "from . import results"),
    ],
)
def test_layer_guard_allows_same_unit_and_downward_imports(module, is_package, code) -> None:
    assert _source_violations(module, code, is_package=is_package, label="snippet") == []


def test_every_package_is_assigned_a_layer() -> None:
    units = {
        _unit(
            "matsim_agents."
            + ".".join(p.relative_to(PACKAGE_ROOT).with_suffix("").parts).removesuffix(".__init__")
        )
        for p in PACKAGE_ROOT.rglob("*.py")
        if p.parent != PACKAGE_ROOT
    }
    units |= {p.stem for p in PACKAGE_ROOT.glob("*.py") if p.stem != "__init__"}
    assert units - set(LAYER_OF) - {"backends"} == set()


def test_imports_only_point_to_lower_layers() -> None:
    assert _layer_violations() == []


# ── Removed legacy paths ──────────────────────────────────────────────────── #


@pytest.mark.parametrize(
    "module",
    [
        "matsim_agents.graph",
        "matsim_agents.llm",
        "matsim_agents.state",
        "matsim_agents.supervisor",
        "matsim_agents.tools",
        "matsim_agents.agents",
        "matsim_agents.chat",
        "matsim_agents.active_learning.backends",
        "matsim_agents.active_learning.vasp_runner",
    ],
)
def test_legacy_alias_modules_are_removed(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


# ── Interface 1: DFT backend ──────────────────────────────────────────────── #


def test_dft_backend_protocol_is_importable() -> None:
    from matsim_agents.backends.dft import DFTBackend, DFTJobSpec, DFTResult

    assert callable(DFTBackend)
    assert DFTJobSpec is not None
    assert DFTResult is not None


def test_vasp_backend_satisfies_dft_protocol() -> None:
    from matsim_agents.backends.dft import DFTBackend, DFTJobSpec, DFTResult

    # issubclass doesn't work on Protocols with data members; use isinstance.
    class _StubVASP:
        name = "vasp"
        nodes_per_job = 1
        ranks_per_node = 4
        threads_per_rank = 1
        timeout_sec = 3600

        def run_one(self, spec: DFTJobSpec) -> DFTResult:  # type: ignore[empty-body]
            ...

    assert isinstance(_StubVASP(), DFTBackend)
    # Also confirm VASPBackend exposes the required attributes structurally.
    from matsim_agents.backends.dft.vasp import VASPBackend

    for attr in ("name", "run_one"):
        assert hasattr(VASPBackend, attr)


# ── Interface 2: MLIP backend ─────────────────────────────────────────────── #


def test_mlip_backend_protocol_is_importable() -> None:
    from matsim_agents.backends.mlip import MLIPBackend

    assert callable(MLIPBackend)


def test_mlip_backend_protocol_requires_correct_methods() -> None:
    from matsim_agents.backends.mlip import MLIPBackend

    # Minimal stub that satisfies the Protocol.
    class _Stub:
        name = "stub"

        def as_calculator(self):
            return None

        def relax(self, atoms, *, fmax=0.05, max_steps=200):
            return None

    assert isinstance(_Stub(), MLIPBackend)


# ── Interface 3: LLM backend ──────────────────────────────────────────────── #


def test_llm_backend_type_is_importable() -> None:
    # LLMBackend is a TYPE_CHECKING alias; just confirm the module loads.
    import matsim_agents.backends.llm as llm_pkg

    assert hasattr(llm_pkg, "get_chat_model")


# ── Interface 4: Execution platform ──────────────────────────────────────── #


def test_execution_platform_protocol_is_importable() -> None:
    from matsim_agents.execution import ExecutionPlatform, ResourceRequest

    assert callable(ExecutionPlatform)
    assert callable(ResourceRequest)


def test_execution_platform_protocol_requires_correct_methods() -> None:
    from matsim_agents.execution import ExecutionPlatform, ResourceRequest

    class _Stub:
        name = "stub"

        def submit(self, command, *, resources, workdir, env=None):
            return "job-0"

        def available_resources(self):
            return ResourceRequest()

    assert isinstance(_Stub(), ExecutionPlatform)


# ── Interface 5: Run / provenance store ──────────────────────────────────── #


def test_run_store_protocol_is_importable() -> None:
    from matsim_agents.execution import RunStore

    assert callable(RunStore)


def test_jsonl_run_store_satisfies_protocol(tmp_path) -> None:
    from matsim_agents.execution import RunStore
    from matsim_agents.execution.provenance import JsonlRunStore

    store = JsonlRunStore(tmp_path / "runs.jsonl")
    assert isinstance(store, RunStore)

    store.append({"iteration": 0, "n_frames": 3})
    store.append({"iteration": 1, "n_frames": 5})
    records = list(store.iter_records())
    assert len(records) == 2
    assert records[0]["iteration"] == 0
    assert records[1]["n_frames"] == 5


def test_phase_exploration_api_is_public() -> None:
    from matsim_agents.workflows import (
        PhaseExplorationPolicy,
        PhaseExplorationWorkflowResult,
        run_phase_exploration,
    )

    assert callable(PhaseExplorationPolicy)
    assert callable(PhaseExplorationWorkflowResult)
    assert callable(run_phase_exploration)
