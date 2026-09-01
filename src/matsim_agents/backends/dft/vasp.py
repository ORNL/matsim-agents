"""VASP backend: thin adapter around :mod:`active_learning.vasp_io`.

The job orchestration logic (concurrency, ThreadPoolExecutor, timeouts) lives
in :mod:`active_learning.dft_runner`. This module only knows how to run *one*
VASP single-point: prepare INCAR/KPOINTS/POTCAR/POSCAR, exec the wrapper
script, parse vasprun.xml/OUTCAR.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import TYPE_CHECKING

from matsim_agents.active_learning.dft_backend import DFTJobSpec, DFTResult
from matsim_agents.active_learning.vasp_io import (
    parse_vasp_workdir,
    prepare_vasp_workdir,
)

if TYPE_CHECKING:
    from matsim_agents.active_learning.config import VASPConfig


class VASPBackend:
    """DFTBackend implementation for VASP 6.x on Frontier."""

    name = "vasp"

    def __init__(self, cfg: VASPConfig) -> None:
        self.cfg = cfg
        self.nodes_per_job = cfg.nodes_per_job
        self.ranks_per_node = cfg.ranks_per_node
        self.threads_per_rank = cfg.threads_per_rank
        self.timeout_sec = cfg.timeout_sec

    def run_one(self, spec: DFTJobSpec) -> DFTResult:
        cfg = self.cfg

        prepare_vasp_workdir(
            spec.atoms,
            spec.work_dir,
            incar_template=cfg.incar_template,
            potcar_dir=cfg.potcar_dir,
            kpoints_template=cfg.kpoints_template,
            extra_incar={**(cfg.extra_incar or {}), **(spec.extra or {})},
        )

        # Wrapper contract: <work_dir> <vasp_bin> <nodes> <ranks_per_node> <threads_per_rank>
        argv = [
            "bash",
            str(cfg.vasp_wrapper),
            spec.work_dir,
            str(cfg.vasp_bin),
            str(cfg.nodes_per_job),
            str(cfg.ranks_per_node),
            str(cfg.threads_per_rank),
        ]

        log_path = os.path.join(spec.work_dir, "vasp.out")
        t0 = time.time()
        with open(log_path, "w") as logf:
            try:
                proc = subprocess.run(
                    argv,
                    cwd=spec.work_dir,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    timeout=cfg.timeout_sec,
                    check=False,
                )
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = 124
        elapsed = time.time() - t0

        legacy = parse_vasp_workdir(spec.work_dir, return_code=rc)
        return DFTResult(
            backend=self.name,
            work_dir=legacy.work_dir,
            return_code=legacy.return_code,
            converged=legacy.converged,
            energy_eV=legacy.energy_eV,
            forces_eV_per_A=legacy.forces_eV_per_A,
            stress_eV_per_A3=legacy.stress_eV_per_A3,
            n_atoms=legacy.n_atoms,
            wall_time_sec=legacy.wall_time_sec if legacy.wall_time_sec is not None else elapsed,
            final_atoms=legacy.final_atoms,
            notes=legacy.notes,
        )
