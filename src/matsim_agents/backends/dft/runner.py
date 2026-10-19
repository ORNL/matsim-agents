"""Backend-agnostic concurrent DFT dispatcher inside Slurm or PBS allocations.

Dispatches single-point jobs through any :class:`DFTBackend`. Each backend is
responsible for: (a) preparing its own work_dir contents, (b) launching its own
``srun`` step via its wrapper script, (c) parsing its outputs back into a
:class:`DFTResult`. This module only owns concurrency, timeouts, and exception
capture.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from matsim_agents.backends.dft.base import (
    DFTBackend,
    DFTJobSpec,
    DFTResult,
)
from matsim_agents.execution.placement import PoolPlacement, ProcessLaunch


def run_dft_batch(
    specs: Iterable[DFTJobSpec],
    backend: DFTBackend,
    pool: PoolPlacement,
    on_complete: Callable[[DFTJobSpec, DFTResult], None] | None = None,
) -> list[DFTResult]:
    """Run DFT jobs concurrently on the disjoint job groups of the ``dft`` pool."""
    specs_list = list(specs)
    if not specs_list:
        return []
    if not pool.launches:
        raise ValueError(f"pool {pool.name!r} has no DFT launches")

    groups = pool.launches
    nworkers = max(1, min(pool.max_concurrent or len(groups), len(groups), len(specs_list)))

    for s in specs_list:
        Path(s.work_dir).mkdir(parents=True, exist_ok=True)

    queues = [specs_list[index::nworkers] for index in range(nworkers)]

    def run_partition(
        launch: ProcessLaunch, queue: list[DFTJobSpec]
    ) -> list[tuple[DFTJobSpec, DFTResult]]:
        completed: list[tuple[DFTJobSpec, DFTResult]] = []
        # A partition processes its queue serially, so a node group can never
        # be reused while its previous scheduler step is still running.
        for spec in queue:
            spec.launch = launch
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
