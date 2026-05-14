"""Backend-agnostic concurrent DFT job dispatcher (inside one SLURM allocation).

Mirrors the previous ``vasp_runner`` semantics but dispatches through any
:class:`DFTBackend`. Each backend is responsible for: (a) preparing its own
work_dir contents, (b) launching its own ``srun`` step via its wrapper
script, (c) parsing its outputs back into a :class:`DFTResult`. This module
only owns concurrency, timeouts, and exception capture.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from matsim_agents.active_learning.dft_backend import (
    DFTBackend,
    DFTJobSpec,
    DFTResult,
)


def _max_concurrent_jobs(nodes_per_job: int) -> int:
    """Cap concurrency from the SLURM allocation size, falling back to 1."""
    nnodes_str = os.environ.get("SLURM_JOB_NUM_NODES")
    try:
        nnodes = int(nnodes_str) if nnodes_str else 1
    except ValueError:
        nnodes = 1
    return max(1, nnodes // max(1, nodes_per_job))


def run_dft_batch(
    specs: Iterable[DFTJobSpec],
    backend: DFTBackend,
    max_workers: int | None = None,
    on_complete: Callable[[DFTJobSpec, DFTResult], None] | None = None,
) -> list[DFTResult]:
    """Run a batch of DFT jobs concurrently inside one SLURM allocation."""
    specs_list = list(specs)
    if not specs_list:
        return []

    nworkers = (
        max_workers if max_workers is not None else _max_concurrent_jobs(backend.nodes_per_job)
    )
    nworkers = max(1, min(nworkers, len(specs_list)))

    for s in specs_list:
        Path(s.work_dir).mkdir(parents=True, exist_ok=True)

    results: list[DFTResult] = []
    with ThreadPoolExecutor(max_workers=nworkers) as pool:
        future_to_spec = {pool.submit(backend.run_one, s): s for s in specs_list}
        for fut in as_completed(future_to_spec):
            spec = future_to_spec[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                res = DFTResult(
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
            results.append(res)
            if on_complete is not None:
                on_complete(spec, res)

    return results
