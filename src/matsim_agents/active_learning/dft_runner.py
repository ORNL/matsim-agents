"""Backend-agnostic concurrent DFT dispatcher inside Slurm or PBS allocations.

Mirrors the previous ``vasp_runner`` semantics but dispatches through any
:class:`DFTBackend`. Each backend is responsible for: (a) preparing its own
work_dir contents, (b) launching its own ``srun`` step via its wrapper
script, (c) parsing its outputs back into a :class:`DFTResult`. This module
only owns concurrency, timeouts, and exception capture.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from matsim_agents.active_learning.dft_backend import (
    DFTBackend,
    DFTJobSpec,
    DFTResult,
)
from matsim_agents.execution.allocation import discover_allocation, validate_dft_resources


def run_dft_batch(
    specs: Iterable[DFTJobSpec],
    backend: DFTBackend,
    max_workers: int | None = None,
    on_complete: Callable[[DFTJobSpec, DFTResult], None] | None = None,
) -> list[DFTResult]:
    """Run DFT jobs concurrently on stable, disjoint allocation partitions."""
    specs_list = list(specs)
    if not specs_list:
        return []

    allocation = discover_allocation()
    validate_dft_resources(
        allocation,
        nodes_per_job=backend.nodes_per_job,
        ranks_per_node=backend.ranks_per_node,
    )
    groups = allocation.groups(backend.nodes_per_job)
    available_workers = len(groups)
    nworkers = max_workers if max_workers is not None else available_workers
    nworkers = max(1, min(nworkers, available_workers, len(specs_list)))

    for s in specs_list:
        Path(s.work_dir).mkdir(parents=True, exist_ok=True)

    queues = [specs_list[index::nworkers] for index in range(nworkers)]

    def run_partition(
        node_group: tuple[str, ...], queue: list[DFTJobSpec]
    ) -> list[tuple[DFTJobSpec, DFTResult]]:
        completed: list[tuple[DFTJobSpec, DFTResult]] = []
        # A partition processes its queue serially, so a node group can never
        # be reused while its previous scheduler step is still running.
        for spec in queue:
            spec.assigned_nodes = node_group
            try:
                result = backend.run_one(spec)
            except Exception as exc:  # noqa: BLE001
                result = DFTResult(
                    backend=backend.name,
                    work_dir=spec.work_dir,
                    return_code=-1,
                    converged=False,
                    energy_eV=None,
                    forces_eV_per_A=None,
                    stress_eV_per_A3=None,
                    n_atoms=len(spec.atoms),
                    wall_time_sec=None,
                    final_atoms=None,
                    notes=f"Python-level exception: {exc!r}",
                )
            completed.append((spec, result))
        return completed

    results: list[DFTResult] = []
    with ThreadPoolExecutor(max_workers=nworkers) as pool:
        futures = [
            pool.submit(run_partition, groups[index], queue) for index, queue in enumerate(queues)
        ]
        for future in as_completed(futures):
            for spec, result in future.result():
                results.append(result)
                if on_complete is not None:
                    on_complete(spec, result)

    return results
