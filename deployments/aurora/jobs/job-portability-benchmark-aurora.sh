#!/bin/bash
#PBS -N matsim-portability
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe

set -eo pipefail

REPO="${PROJECT_ROOT:?export PROJECT_ROOT to the matsim-agents checkout}"
PROJ="$(dirname "${REPO}")"
VENV_PATH="${VENV_PATH:-${PROJ}/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Aurora/hydragnn_venv}"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
RUN_DIR="${RUNS_ROOT}/portability/aurora-${PBS_JOBID:-$$}"

module load frameworks
[[ -f "${VENV_PATH}/bin/activate" ]] && source "${VENV_PATH}/bin/activate"

PYTHON="${VENV_PATH}/bin/python3"

"${PYTHON}" "${REPO}/benchmarks/portability/run.py" \
  --facility aurora --suite all --backend qe --execute --output "${RUN_DIR}"
"${PYTHON}" "${REPO}/benchmarks/portability/validate.py" "${RUN_DIR}"
