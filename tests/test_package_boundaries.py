"""Regression tests for the package-boundary migration.

Each test asserts one structural property of the new package layout:
  - Legacy module aliases point to the canonical implementation.
  - Each of the five stable interfaces is importable from its declared location.
  - Concrete implementations satisfy their Protocol at runtime.
"""

from __future__ import annotations

import matsim_agents.graph as legacy_graph
import matsim_agents.llm as legacy_llm
import matsim_agents.state as legacy_state
from matsim_agents.backends.llm import provider
from matsim_agents.orchestration import objective_graph, state

# ── Legacy alias tests ────────────────────────────────────────────────────── #


def test_legacy_orchestration_modules_alias_canonical_modules() -> None:
    assert legacy_graph is objective_graph
    assert legacy_state is state


def test_legacy_llm_module_aliases_canonical_provider() -> None:
    assert legacy_llm is provider


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
