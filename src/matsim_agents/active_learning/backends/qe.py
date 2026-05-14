"""Quantum ESPRESSO ``pw.x`` backend (single-point SCF for AL labelling).

Reuses :mod:`matsim_agents.tools.qe_relax` to write the input deck (with
``calculation='scf'`` so we get one SCF per call, no geometry steps) and
parses the resulting ``pw.out`` with ASE's ``espresso-out`` reader, which
gives us ``Atoms`` plus a SinglePointCalculator carrying energy, forces, and
stress.

Energy / force / stress units returned to the caller match those of the VASP
backend (eV, eV/Å, eV/Å³) so the rest of the AL loop never has to special-
case the backend.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import TYPE_CHECKING, Any

from ase.io import read as ase_read

from matsim_agents.active_learning.dft_backend import DFTJobSpec, DFTResult
from matsim_agents.tools.qe_relax import (
    QESettings,
    recommend_settings,
    write_pw_input,
)

if TYPE_CHECKING:
    from matsim_agents.active_learning.config import QEBackendConfig

log = logging.getLogger(__name__)


class QEBackend:
    """DFTBackend implementation for Quantum ESPRESSO ``pw.x``."""

    name = "qe"

    def __init__(self, cfg: "QEBackendConfig") -> None:
        self.cfg = cfg
        self.nodes_per_job = cfg.nodes_per_job
        self.ranks_per_node = cfg.ranks_per_node
        self.threads_per_rank = cfg.threads_per_rank
        self.timeout_sec = cfg.timeout_sec

    # ----- input prep -------------------------------------------------------

    def _settings_for(self, atoms) -> QESettings:
        """Build a per-job QESettings from cfg + composition-aware defaults."""
        cfg = self.cfg
        # Auto-fill ecutwfc/ecutrho/kpts/pseudos from element table, then
        # override with whatever the user pinned in the YAML.
        s = recommend_settings(atoms, str(cfg.pseudo_dir))
        s.calculation = "scf"  # AL only needs a single SCF; never relax here
        if cfg.ecutwfc_ry is not None:
            s.ecutwfc_ry = cfg.ecutwfc_ry
        if cfg.ecutrho_ry is not None:
            s.ecutrho_ry = cfg.ecutrho_ry
        if cfg.kpts is not None:
            s.kpts = tuple(cfg.kpts)  # type: ignore[assignment]
        if cfg.koffset is not None:
            s.koffset = tuple(cfg.koffset)  # type: ignore[assignment]
        if cfg.occupations is not None:
            s.occupations = cfg.occupations
        s.smearing = cfg.smearing
        s.degauss_ry = cfg.degauss_ry
        if cfg.pseudopotentials:
            s.pseudopotentials = dict(cfg.pseudopotentials)
        if cfg.extra_control:
            s.extra_control = {**s.extra_control, **cfg.extra_control}
        if cfg.extra_system:
            s.extra_system = {**s.extra_system, **cfg.extra_system}
        if cfg.extra_electrons:
            s.extra_electrons = {**s.extra_electrons, **cfg.extra_electrons}
        return s

    # ----- run --------------------------------------------------------------

    def run_one(self, spec: DFTJobSpec) -> DFTResult:
        cfg = self.cfg
        os.makedirs(spec.work_dir, exist_ok=True)

        settings = self._settings_for(spec.atoms)
        input_path = os.path.join(spec.work_dir, "pw.in")
        write_pw_input(spec.atoms, settings, input_path,
                       prefix="pwscf", outdir="./tmp")

        # Wrapper contract:
        #   <work_dir> <pw_bin> <input_file> <nodes> <ranks_per_node> <threads_per_rank>
        argv = [
            "bash",
            str(cfg.pw_wrapper),
            spec.work_dir,
            str(cfg.pw_bin),
            input_path,
            str(cfg.nodes_per_job),
            str(cfg.ranks_per_node),
            str(cfg.threads_per_rank),
        ]

        stdout_path = os.path.join(spec.work_dir, "pw.out")
        t0 = time.time()
        with open(stdout_path, "w") as logf:
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

        return _parse_pw_out(
            stdout_path=stdout_path,
            work_dir=spec.work_dir,
            return_code=rc,
            elapsed_sec=elapsed,
            backend_name=self.name,
        )


# --------------------------------------------------------------------------- #
# Output parsing (eV / eV·Å⁻¹ / eV·Å⁻³)                                       #
# --------------------------------------------------------------------------- #


def _parse_pw_out(
    stdout_path: str,
    work_dir: str,
    return_code: int,
    elapsed_sec: float,
    backend_name: str,
) -> DFTResult:
    """Use ASE's espresso-out reader to extract energy/forces/stress."""
    import numpy as np

    energy: float | None = None
    forces: Any | None = None
    stress: Any | None = None
    final_atoms = None
    converged = False
    notes_list: list[str] = []
    n_atoms: int | None = None

    if not os.path.isfile(stdout_path) or os.path.getsize(stdout_path) == 0:
        notes_list.append("pw.out missing or empty")
    else:
        try:
            final_atoms = ase_read(stdout_path, index=-1, format="espresso-out")
            n_atoms = len(final_atoms)
            calc = final_atoms.calc
            if calc is not None:
                results = calc.results
                if "energy" in results:
                    energy = float(results["energy"])
                elif "free_energy" in results:
                    energy = float(results["free_energy"])
                if "forces" in results:
                    forces = np.asarray(results["forces"], dtype=np.float64)
                if "stress" in results:
                    stress = np.asarray(results["stress"], dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 — ASE raises many subclasses
            notes_list.append(f"espresso-out parse failed: {exc}")

        # Final convergence sanity check: pw.x prints "JOB DONE." on success.
        try:
            with open(stdout_path) as f:
                tail = f.read()[-4096:]
            converged = (
                return_code == 0
                and energy is not None
                and forces is not None
                and "JOB DONE." in tail
            )
            if "convergence NOT achieved" in tail:
                converged = False
                notes_list.append("SCF did not converge")
        except OSError:
            pass

    return DFTResult(
        backend=backend_name,
        work_dir=os.path.abspath(work_dir),
        return_code=return_code,
        converged=converged,
        energy_eV=energy,
        forces_eV_per_A=forces,
        stress_eV_per_A3=stress,
        n_atoms=n_atoms,
        wall_time_sec=elapsed_sec,
        final_atoms=final_atoms,
        notes="; ".join(notes_list) if notes_list else None,
    )
