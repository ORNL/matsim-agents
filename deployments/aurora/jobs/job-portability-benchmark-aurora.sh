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
QUALIFICATION="${MATSIM_PORTABILITY_QUALIFICATION:-contract}"
ARGS=(--facility aurora --suite all --backend qe --execute
  --qualification "${QUALIFICATION}" --output "${RUN_DIR}")
if [[ "${QUALIFICATION}" == "compute" ]]; then
  : "${MATSIM_PORTABILITY_RELAXATION_CONFIGS:?set a colon-separated MLIP and QE config list}"
  IFS=: read -r -a CONFIGS <<< "${MATSIM_PORTABILITY_RELAXATION_CONFIGS}"
  for config in "${CONFIGS[@]}"; do
    ARGS+=(--relaxation-config "${config}")
  done
fi

module load frameworks
[[ -f "${VENV_PATH}/bin/activate" ]] && source "${VENV_PATH}/bin/activate"

PYTHON="${VENV_PATH}/bin/python3"

"${PYTHON}" "${REPO}/benchmarks/portability/run.py" "${ARGS[@]}"
"${PYTHON}" "${REPO}/benchmarks/portability/validate.py" "${RUN_DIR}"
