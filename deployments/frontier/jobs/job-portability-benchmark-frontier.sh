#!/bin/bash
#SBATCH -J matsim-portability
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
RUN_DIR="${RUNS_ROOT}/portability/frontier-${SLURM_JOB_ID:-$$}"

source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source activate "${VENV}"

PYTHON="${VENV}/bin/python3"

"${PYTHON}" "${REPO}/benchmarks/portability/run.py" \
  --facility frontier --suite all --backend qe --execute --output "${RUN_DIR}"
"${PYTHON}" "${REPO}/benchmarks/portability/validate.py" "${RUN_DIR}"
