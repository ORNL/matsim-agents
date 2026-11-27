"""Real LAMMPS run of a W EAM single point and short NVE, checked against ASE's EAM calculator.

Skips unless a LAMMPS executable is found (LAMMPS_EXECUTABLE or lmp on PATH) and
LAMMPS_POTENTIALS contains W_zhou.eam.alloy (see docs/external/lammps.md). This checks the
binary and the adapter for one potential, not the potential's physics.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.eam import EAM

from matsim_agents.backends.external.lammps.availability import Status, check_availability
from matsim_agents.backends.external.lammps.inputs import (
    FINAL_DUMP,
    LOG_NAME,
    LammpsTask,
    potential_from_file,
    write_inputs,
)
from matsim_agents.backends.external.lammps.parser import parse_log_file, read_final_atoms
from matsim_agents.backends.external.lammps.runner import run_local
from matsim_agents.core.seeding import numpy_rng

AV = check_availability()
POT_FILE = Path(os.environ.get("LAMMPS_POTENTIALS", "")) / "W_zhou.eam.alloy"
if AV.status is not Status.READY or not POT_FILE.is_file():
    pytest.skip(
        f"LAMMPS not ready ({AV.status.value}) or {POT_FILE} missing", allow_module_level=True
    )

# Optional MPI launcher, e.g. "srun -n 8"; empty runs the executable directly.
LAUNCHER = tuple(shlex.split(os.environ.get("LAMMPS_TEST_LAUNCHER", "")))
# Captured at import, before conftest strips SLURM_*; srun needs the job's allocation.
ENV = dict(os.environ)


def _potential():
    return potential_from_file("eam/alloy", POT_FILE, ["W"])


def test_single_point_matches_ase_eam(tmp_path):
    atoms = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
    atoms.positions += numpy_rng(11, "lammps-it").normal(0, 0.05, atoms.positions.shape)
    write_inputs(atoms, tmp_path, _potential(), LammpsTask())
    outcome = run_local(Path(AV.executable), tmp_path, 300, launcher=LAUNCHER, env=ENV)
    assert outcome.process_succeeded, outcome.stderr_tail

    ref = atoms.copy()
    ref.calc = EAM(potential=str(POT_FILE), form="alloy")
    energy = parse_log_file(tmp_path / LOG_NAME)[-1].last("PotEng")
    forces = read_final_atoms(tmp_path / FINAL_DUMP).get_forces()
    assert abs(energy - ref.get_potential_energy()) / len(atoms) < 1e-6
    assert np.abs(forces - ref.get_forces()).max() < 1e-4


def test_nve_conserves_energy(tmp_path):
    atoms = bulk("W", "bcc", a=3.165, cubic=True).repeat(6)
    task = LammpsTask(
        "nve", steps=500, timestep_ps=0.001, temperature_k=300.0, seed=11, thermo_every=100
    )
    write_inputs(atoms, tmp_path, _potential(), task)
    outcome = run_local(Path(AV.executable), tmp_path, 600, launcher=LAUNCHER, env=ENV)
    assert outcome.process_succeeded, outcome.stderr_tail

    etot = parse_log_file(tmp_path / LOG_NAME)[-1].column("TotEng")
    assert abs(etot[-1] - etot[0]) / len(atoms) < 1e-3
