"""Concurrent MLIP relaxations: one worker process per device of the ``mlip`` pool.

Each worker is started with ``spawn`` (the parent may already hold CUDA state,
for example an in-process LLM), restricted to one device by its vendor's mask,
and keeps its loaded calculators between tasks. Results come back in
submission order, so provenance does not depend on scheduling. A worker that
crashes fails only the task it was running and is replaced.

Without an ``mlip`` pool, or on CPU-only geometry, work runs in the calling
process exactly as before, with calculators reused between relaxations.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import socket
import sys
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from matsim_agents.execution.devices import child_mask_environment, visible_ordinals

if TYPE_CHECKING:
    from matsim_agents.backends.mlip.relaxation import RelaxStructureInput
    from matsim_agents.backends.mlip.results import RelaxationResult
    from matsim_agents.execution.placement import Placement

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerSlot:
    """One worker process: the device it owns and the environment that pins it there."""

    device: int
    env: dict[str, str]
    memory_fraction: float | None = None


@dataclass(frozen=True)
class WorkerOutcome:
    """Result of one task; exactly one of ``value`` and ``error`` is meaningful."""

    value: Any = None
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class _WorkerState:
    calculators: Any = None
    memory_fraction: float | None = None
    memory_fraction_applied: bool = False


_WORKER = _WorkerState()


def _initialize_worker(env: dict[str, str], sys_path: list[str], memory_fraction) -> None:
    # Runs before any task imports torch, so the device mask takes effect.
    os.environ.update(env)
    for entry in reversed(sys_path):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    _WORKER.memory_fraction = memory_fraction


def prepare_worker_device() -> None:
    """Apply the worker's device-memory budget once; call at the start of every task."""

    if _WORKER.memory_fraction_applied or _WORKER.memory_fraction is None:
        return
    import torch

    fraction = _WORKER.memory_fraction
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        setter = getattr(torch.xpu, "set_per_process_memory_fraction", None)
        if setter is None:
            log.warning("torch.xpu cannot cap device memory; memory_fraction is not enforced")
        else:
            setter(fraction, 0)
    _WORKER.memory_fraction_applied = True


def _worker_calculators():
    from matsim_agents.backends.mlip.relaxation import CalculatorCache

    if _WORKER.calculators is None:
        _WORKER.calculators = CalculatorCache()
    return _WORKER.calculators


def relax_task(args_json: str) -> dict:
    """Worker entry point: relax one structure and return a JSON-safe result."""

    from matsim_agents.backends.mlip.relaxation import RelaxStructureInput, relax_with_mlip

    prepare_worker_device()
    args = RelaxStructureInput.model_validate_json(args_json)
    return relax_with_mlip(args, calculators=_worker_calculators()).model_dump(mode="json")


def _short_host(name: str) -> str:
    return name.split(".", 1)[0]


class MLIPWorkerPool:
    """Runs independent MLIP tasks concurrently, one per device of the ``mlip`` pool."""

    def __init__(self, slots: Sequence[WorkerSlot] = ()) -> None:
        self.slots = list(slots)
        self._executors: list[ProcessPoolExecutor | None] = [None] * len(self.slots)
        self._local_calculators = None

    @classmethod
    def from_placement(cls, placement: Placement | None) -> MLIPWorkerPool:
        """Build workers for the ``mlip`` pool of *placement*; run locally without one."""

        if placement is None or "mlip" not in placement.pools:
            return cls()
        pool = placement.pool("mlip")
        if not placement.geometry.devices_per_node:
            return cls()
        if len(pool.slices) != 1:
            raise ValueError(
                "MLIP worker processes run on the node of the calling process; set "
                f"resources.pools.mlip.nodes to 1 (it resolved to {pool.hosts})"
            )
        host = pool.slices[0].host
        local = _short_host(socket.gethostname())
        if placement.scheduler not in {"local", "preview"} and _short_host(host) != local:
            raise ValueError(
                f"pool 'mlip' is placed on {host}, but this process runs on {local}; place "
                "the mlip pool on the node that runs matsim-agents"
            )
        devices = pool.slices[0].devices[: pool.max_concurrent or None]
        return cls(
            [
                WorkerSlot(
                    device=device,
                    env=child_mask_environment(
                        placement.geometry.device_mask_environment((device,)),
                        owner="pool 'mlip'",
                    ),
                    memory_fraction=pool.memory_fraction,
                )
                for device in devices
            ]
        )

    @property
    def concurrency(self) -> int:
        return max(1, len(self.slots))

    @property
    def uses_workers(self) -> bool:
        return bool(self.slots)

    def _executor(self, index: int) -> ProcessPoolExecutor:
        executor = self._executors[index]
        if executor is None:
            slot = self.slots[index]
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(slot.env, list(sys.path), slot.memory_fraction),
            )
            self._executors[index] = executor
        return executor

    def _replace(self, index: int) -> None:
        executor = self._executors[index]
        self._executors[index] = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def map(self, fn: Callable[[Any], Any], items: Iterable[Any]) -> list[WorkerOutcome]:
        """Run ``fn(item)`` for every item; outcomes are in submission order.

        In worker mode *fn* must be a picklable module-level function.
        """

        items = list(items)
        if not self.slots:
            return [self._run_local(fn, item) for item in items]
        outcomes: list[WorkerOutcome | None] = [None] * len(items)
        pending = deque(enumerate(items))
        running: dict[Future, tuple[int, int]] = {}
        idle = deque(range(len(self.slots)))
        while pending or running:
            while pending and idle:
                slot = idle.popleft()
                index, item = pending.popleft()
                running[self._executor(slot).submit(fn, item)] = (index, slot)
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                index, slot = running.pop(future)
                try:
                    outcomes[index] = WorkerOutcome(value=future.result())
                except BrokenProcessPool as exc:
                    device = self.slots[slot].device
                    log.error("MLIP worker on device %s exited; replacing it", device)
                    self._replace(slot)
                    outcomes[index] = WorkerOutcome(
                        error=RuntimeError(f"MLIP worker on device {device} exited: {exc}")
                    )
                except Exception as exc:  # noqa: BLE001 - one task's failure is its outcome
                    outcomes[index] = WorkerOutcome(error=exc)
                idle.append(slot)
        return [outcome for outcome in outcomes if outcome is not None]

    def _run_local(self, fn: Callable[[Any], Any], item: Any) -> WorkerOutcome:
        try:
            return WorkerOutcome(value=fn(item))
        except Exception as exc:  # noqa: BLE001 - one task's failure is its outcome
            return WorkerOutcome(error=exc)

    def relax(self, inputs: Sequence[RelaxStructureInput]) -> list[WorkerOutcome]:
        """Relax structures concurrently; each successful value is a ``RelaxationResult``."""

        from matsim_agents.backends.mlip.relaxation import CalculatorCache, relax_with_mlip
        from matsim_agents.backends.mlip.results import RelaxationResult

        if not self.slots:
            if self._local_calculators is None:
                self._local_calculators = CalculatorCache()
            calculators = self._local_calculators
            return self.map(lambda args: relax_with_mlip(args, calculators=calculators), inputs)
        outcomes = self.map(relax_task, [args.model_dump_json() for args in inputs])
        return [
            WorkerOutcome(value=RelaxationResult.model_validate(outcome.value))
            if outcome.ok
            else outcome
            for outcome in outcomes
        ]

    def relax_one(self, args: RelaxStructureInput) -> RelaxationResult:
        """Relax one structure on the pool and raise its error, like ``relax_with_mlip``."""

        outcome = self.relax([args])[0]
        if outcome.error is not None:
            raise outcome.error
        return outcome.value

    def close(self) -> None:
        """Stop the workers and drop loaded models; the pool restarts them on next use."""
        self._local_calculators = None
        for index in range(len(self._executors)):
            executor = self._executors[index]
            self._executors[index] = None
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> MLIPWorkerPool:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _torch_device_module(vendor: str):
    try:
        import torch
    except ImportError:
        return None
    module = torch.xpu if vendor == "xpu" and hasattr(torch, "xpu") else torch.cuda
    return module if module.is_available() else None


def release_device_memory() -> None:
    """Return freed model memory to the device so other pools' processes can use it."""

    import gc

    gc.collect()
    for vendor in ("cuda", "xpu"):
        module = _torch_device_module(vendor)
        if module is not None:
            module.empty_cache()


@contextmanager
def driver_on_mlip_device(placement: Placement | None) -> Iterator[int | None]:
    """Run this process's own MLIP work on the first device of the ``mlip`` pool.

    Calculators built with ``device="cuda"`` (or ``"xpu"``) use the current
    device, so model loading, single-seed MD, scoring, and model comparison in
    the driver honour the pool's device and ``memory_fraction`` instead of
    landing on device 0. Yields the device ordinal, or ``None`` without a pool.
    """

    pool = placement.pools.get("mlip") if placement is not None else None
    if pool is None or not pool.slices or not placement.geometry.devices_per_node:
        yield None
        return
    MLIPWorkerPool.from_placement(placement)  # same node checks as the workers
    vendor = placement.geometry.vendor
    (ordinal,) = visible_ordinals(vendor, pool.slices[0].devices[:1], owner="pool 'mlip'")
    module = _torch_device_module(vendor)
    if module is None:
        yield ordinal
        return
    previous = module.current_device()
    module.set_device(ordinal)
    capped = pool.memory_fraction is not None and hasattr(module, "set_per_process_memory_fraction")
    if capped:
        module.set_per_process_memory_fraction(pool.memory_fraction, ordinal)
    try:
        yield ordinal
    finally:
        if capped:
            module.set_per_process_memory_fraction(1.0, ordinal)
        module.set_device(previous)


__all__ = [
    "MLIPWorkerPool",
    "driver_on_mlip_device",
    "release_device_memory",
    "WorkerOutcome",
    "WorkerSlot",
    "prepare_worker_device",
    "relax_task",
]
