#!/bin/bash
#SBATCH -J matsim-integration-tests
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -t 01:30:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# Dependency-enabled regression run on a compute node, one pytest pass per environment:
# .venv (full suite), .venv-mace (MACE tests), .venv-uma (fairchem/UMA tests) and, when
# build-openmc-frontier.sh has been run, the OpenMC activation test in its own venv; when
# setup/lammps/build-lammps-frontier.sh has been run, the LAMMPS EAM test against external/lammps.
#   sbatch -A <allocation> deployments/frontier/jobs/job-integration-tests-frontier.sh
# Warm-start integration tests still skip unless their launcher, pseudopotential, checkpoint
# and model variables are set; weights are not downloaded on compute nodes.

set -uo pipefail
# Slurm runs a spooled copy of this script, so fall back to the submit directory.
REPO="${PROJECT_ROOT:-$(d="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)"; [[ -f "$d/pyproject.toml" ]] && echo "$d" || echo "${SLURM_SUBMIT_DIR:-$d}")}"
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$REPO/deployments/frontier/setup/frontier-module-stack.sh"
load_frontier_rocm72_modules
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 XALT_EXECUTABLE_TRACKING=no
export PYTORCH_ROCM_ARCH=gfx90a ROCM_ARCH=gfx90a
export MIOPEN_DISABLE_CACHE=1 MIOPEN_USER_DB_PATH="/tmp/miopen-${SLURM_JOB_ID:-$$}"
export PYTHONPATH="$REPO/src:$REPO"
mkdir -p "$MIOPEN_USER_DB_PATH"

OUT="${TEST_OUTPUT_DIR:-$REPO/.hpc-build/frontier/integration-tests/${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT"
cd "$REPO"

MLIP_TESTS=(tests/test_mlip_*.py tests/test_e3_mlip_*.py tests/test_e4_mlip.py
  tests/test_al_mlip_config.py tests/integration)

run_phase() {
  local name="$1" venv="$2"
  shift 2
  local py="$venv/bin/python"
  [[ -x "$py" ]] || { echo "$name: missing $py" | tee -a "$OUT/summary.txt"; return 2; }
  "$py" - <<'EOF' | tee "$OUT/$name-environment.txt"
import importlib.util as u
import torch
print("torch", torch.__version__, "hip", torch.version.hip, "devices", torch.cuda.device_count())
for m in ("torch_geometric", "hydragnn", "mcp", "pymatgen", "ase", "langgraph", "mace", "fairchem"):
    print(m, bool(u.find_spec(m)))
EOF
  "$py" -m pytest "$@" -q -rs -p no:cacheprovider --junitxml="$OUT/$name-junit.xml" \
    2>&1 | tee "$OUT/$name-pytest.txt"
  local status=${PIPESTATUS[0]}
  echo "$name pytest exit status: $status" | tee -a "$OUT/summary.txt"
  return "$status"
}

# OpenMC runs in its own venv (build-openmc-frontier.sh); compute nodes cannot download data.
run_openmc_phase() {
  local activate="${OPENMC_ACTIVATE:-$REPO/.hpc-build/frontier/openmc/activate.sh}"
  [[ -f "$activate" ]] || { echo "openmc: missing $activate, phase skipped" | tee -a "$OUT/summary.txt"; return 0; }
  (
    # shellcheck disable=SC1090
    source "$activate"
    python -m pytest tests/integration/test_openmc_activation.py -q -rs -p no:cacheprovider \
      --junitxml="$OUT/openmc-junit.xml" 2>&1 | tee "$OUT/openmc-pytest.txt"
    exit "${PIPESTATUS[0]}"
  )
  local status=$?
  echo "openmc pytest exit status: $status" | tee -a "$OUT/summary.txt"
  return "$status"
}

run_lammps_phase() {
  local activate="${LAMMPS_ACTIVATE:-$REPO/external/lammps/activate.sh}"
  [[ -f "$activate" ]] || { echo "lammps: missing $activate, phase skipped" | tee -a "$OUT/summary.txt"; return 0; }
  (
    # shellcheck disable=SC1090
    source "$activate"
    OMP_NUM_THREADS=1 LAMMPS_TEST_LAUNCHER="${LAMMPS_TEST_LAUNCHER:-srun -n 8}" \
      "${MATSIM_FRONTIER_VENV:-$REPO/.venv}/bin/python" -m pytest \
      tests/integration/test_lammps_eam.py -q -rs -p no:cacheprovider \
      --junitxml="$OUT/lammps-junit.xml" 2>&1 | tee "$OUT/lammps-pytest.txt"
    exit "${PIPESTATUS[0]}"
  )
  local status=$?
  echo "lammps pytest exit status: $status" | tee -a "$OUT/summary.txt"
  return "$status"
}

overall=0
run_phase main "${MATSIM_FRONTIER_VENV:-$REPO/.venv}" tests || overall=1
run_phase mace "${MATSIM_MACE_VENV:-$REPO/.venv-mace}" tests/test_mace_*.py "${MLIP_TESTS[@]}" \
  || overall=1
run_phase uma "${MATSIM_FAIRCHEM_VENV:-$REPO/.venv-uma}" tests/test_finetune_uma.py \
  "${MLIP_TESTS[@]}" || overall=1
run_openmc_phase || overall=1
run_lammps_phase || overall=1
exit "$overall"
