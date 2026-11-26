#!/bin/bash
#SBATCH -J lammps-validation
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -N 1
#SBATCH -p batch
# ---------------------------------------------------------------------------
# LAMMPS adapter check on a compute node with srun (MPI). Runs a W EAM single point compared
# with ASE's EAM calculator, then a short NVE run whose energy drift is reported. This checks
# the adapter and the binary, not any physical result.
#
# LAMMPS_SOURCE=module (default) loads the Frontier module (LAMMPS_MODULE overrides it);
# LAMMPS_SOURCE=build sources external/lammps/activate.sh from
# deployments/frontier/setup/lammps/build-lammps-frontier.sh (LAMMPS_ROOT overrides the directory).
#
# Submit: LAMMPS_OUT=<dir> [LAMMPS_SOURCE=build] sbatch -A <ACCOUNT> -t 00:20:00 -q debug \
#   deployments/frontier/jobs/lammps/job-lammps-validation-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO="${PROJECT_ROOT:-$(d="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." 2>/dev/null && pwd)"; [[ -f "$d/pyproject.toml" ]] && echo "$d" || echo "${SLURM_SUBMIT_DIR:?}")}"
OUT="${LAMMPS_OUT:?set LAMMPS_OUT}"
NTASKS="${LAMMPS_NTASKS:-56}"

case "${LAMMPS_SOURCE:-module}" in
    module)
        module load gcc-native/14.2 cray-mpich/8.1.32 "${LAMMPS_MODULE:-lammps/20240829.1-mpi-omp}"
        ;;
    build)
        # shellcheck disable=SC1091
        source "${LAMMPS_ROOT:-$REPO/external/lammps}/activate.sh"
        ;;
    *) echo "LAMMPS_SOURCE must be module or build" >&2; exit 2 ;;
esac
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 XALT_EXECUTABLE_TRACKING=no OMP_NUM_THREADS=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

"$REPO/.venv/bin/python" - "$OUT" "$NTASKS" <<'EOF'
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.calculators.eam import EAM

from matsim_agents.backends.external.lammps.availability import check_availability
from matsim_agents.backends.external.lammps.inputs import (
    FINAL_DUMP, LOG_NAME, LammpsTask, potential_from_file, write_inputs,
)
from matsim_agents.backends.external.lammps.parser import parse_log_file, read_final_atoms
from matsim_agents.backends.external.lammps.runner import run_local
from matsim_agents.core.seeding import numpy_rng

out, ntasks = Path(sys.argv[1]), int(sys.argv[2])
av = check_availability()
pot_file = Path(os.environ["LAMMPS_POTENTIALS"]) / "W_zhou.eam.alloy"
pot = potential_from_file("eam/alloy", pot_file, ["W"])
launcher = ("srun", "-n", str(ntasks), "--cpu-bind=cores")
report = {
    "availability": av.as_dict(),
    "ntasks": ntasks,
    "lammps_source": os.environ.get("LAMMPS_SOURCE", "module"),
}

atoms = bulk("W", "bcc", a=3.165, cubic=True).repeat(6)
atoms.positions += numpy_rng(11, "lammps-validation").normal(0, 0.05, atoms.positions.shape)
sp = out / "single_point"
write_inputs(atoms, sp, pot, LammpsTask())
o = run_local(Path(av.executable), sp, 600, launcher=launcher)
ref = atoms.copy()
ref.calc = EAM(potential=str(pot_file), form="alloy")
e_l = parse_log_file(sp / LOG_NAME)[-1].last("PotEng")
f_l = read_final_atoms(sp / FINAL_DUMP).get_forces()
report["single_point"] = {
    "process_succeeded": o.process_succeeded,
    "n_atoms": len(atoms),
    "lammps_energy_ev": e_l,
    "ase_energy_ev": float(ref.get_potential_energy()),
    "abs_energy_diff_per_atom_ev": abs(e_l - float(ref.get_potential_energy())) / len(atoms),
    "max_abs_force_diff_ev_per_a": float(np.abs(f_l - ref.get_forces()).max()),
}

big = bulk("W", "bcc", a=3.165, cubic=True).repeat(20)
md = out / "nve"
write_inputs(
    big, md, pot,
    LammpsTask("nve", steps=2000, timestep_ps=0.001, temperature_k=600.0, seed=11,
               thermo_every=200),
)
o = run_local(Path(av.executable), md, 1200, launcher=launcher)
block = parse_log_file(md / LOG_NAME)[-1]
etot = block.column("TotEng")
report["nve"] = {
    "process_succeeded": o.process_succeeded,
    "n_atoms": len(big),
    "steps": 2000,
    "wall_seconds": o.wall_seconds,
    "etot_first_ev": etot[0],
    "etot_last_ev": etot[-1],
    "etot_drift_per_atom_ev": (etot[-1] - etot[0]) / len(big),
    "temp_last_k": block.last("Temp"),
}
(out / "lammps_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "availability"}, indent=2))
EOF
