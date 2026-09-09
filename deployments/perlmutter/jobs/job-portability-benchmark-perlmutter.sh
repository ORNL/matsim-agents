#!/bin/bash
#SBATCH -J matsim-portability
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH --gpus-per-node=4
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

REPO="${PROJECT_ROOT:?export PROJECT_ROOT to the matsim-agents checkout}"
PROJ="$(dirname "${REPO}")"
INSTALL_ROOT=$REPO/.hpc-build/perlmutter
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
RUN_DIR="${RUNS_ROOT}/portability/perlmutter-${SLURM_JOB_ID:-$$}"
QUALIFICATION="${MATSIM_PORTABILITY_QUALIFICATION:-contract}"
# Compute qualification exercises the real UMA MLIP case (config/relaxation/
# uma-si.yaml), which requires fairchem-core; that only lives in the
# matsim-owned .venv-uma compatibility environment, not the main .venv used
# by the deterministic contract gate.
if [[ "${QUALIFICATION}" == "compute" ]]; then
  VENV="${MATSIM_FAIRCHEM_VENV:-${REPO}/.venv-uma}"
else
  VENV="${MATSIM_PERLMUTTER_VENV:-${REPO}/.venv}"
fi
export MATSIM_PERLMUTTER_VENV="${VENV}"
ARGS=(--facility perlmutter --suite all --backend qe --execute
  --qualification "${QUALIFICATION}" --output "${RUN_DIR}")
if [[ "${QUALIFICATION}" == "compute" ]]; then
  : "${MATSIM_PORTABILITY_RELAXATION_CONFIGS:?set a colon-separated MLIP and QE config list}"
  IFS=: read -r -a CONFIGS <<< "${MATSIM_PORTABILITY_RELAXATION_CONFIGS}"
  for config in "${CONFIGS[@]}"; do
    ARGS+=(--relaxation-config "${config}")
  done
fi

source "${REPO}/deployments/perlmutter/setup/setup_matsim_perlmutter.sh" --gpu

PYTHON="${VENV}/bin/python3"

"${PYTHON}" "${REPO}/benchmarks/portability/run.py" "${ARGS[@]}"
"${PYTHON}" "${REPO}/benchmarks/portability/validate.py" "${RUN_DIR}"
if [[ "${MATSIM_RUN_ALL_MODEL_SCIENTIFIC_DEBATE:-0}" == "1" ]]; then
  "${PYTHON}" "${REPO}/benchmarks/portability/all_model_scientific_debate.py" \
    --rounds "${MATSIM_DEBATE_ROUNDS:-2}" --output "${RUN_DIR}/all-model-scientific-debate"
fi
