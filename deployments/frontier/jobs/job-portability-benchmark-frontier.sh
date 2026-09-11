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
VENV=$REPO/.venv
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
RUN_DIR="${RUNS_ROOT}/portability/frontier-${SLURM_JOB_ID:-$$}"
QUALIFICATION="${MATSIM_PORTABILITY_QUALIFICATION:-contract}"
ARGS=(--facility frontier --suite all --backend qe --execute
  --qualification "${QUALIFICATION}" --output "${RUN_DIR}")
if [[ "${QUALIFICATION}" == "compute" ]]; then
  : "${MATSIM_PORTABILITY_RELAXATION_CONFIGS:?set a colon-separated MLIP and QE config list}"
  IFS=: read -r -a CONFIGS <<< "${MATSIM_PORTABILITY_RELAXATION_CONFIGS}"
  for config in "${CONFIGS[@]}"; do
    ARGS+=(--relaxation-config "${config}")
  done
fi

source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "${REPO}/deployments/frontier/setup/frontier-module-stack.sh"
load_frontier_rocm72_modules
source activate "${VENV}"

# Real MLIP (--qualification compute) runs torch on-GPU; the QE step launcher
# (deployments/frontier/launchers/run-pw-gpu-frontier.sh) does its own module
# reset to the PrgEnv-cray/rocm-6.2.4 QE toolchain. Note LD_LIBRARY_PATH below
# still leaks into that subprocess (module reset doesn't clear exported env
# vars), so the launcher itself unsets it before loading QE's module stack.
export PYTHONNOUSERSITE=1
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="/tmp/miopen-${SLURM_JOB_ID:-$$}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

# Make HydraGNN example utilities importable (inference_fused, etc.), which the
# HydraGNN ASE calculator in backends/mlip/relaxation.py imports at build time.
HYDRAGNN_EXAMPLE="${HYDRAGNN_DIR:-$PROJ/HydraGNN}/examples/multidataset_hpo_sc26"
export PYTHONPATH="$HYDRAGNN_EXAMPLE:${HYDRAGNN_DIR:-$PROJ/HydraGNN}:${PYTHONPATH:-}"
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_ARCH=gfx90a
export LD_LIBRARY_PATH="${VENV}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON="${VENV}/bin/python3"

"${PYTHON}" "${REPO}/benchmarks/portability/run.py" "${ARGS[@]}"
"${PYTHON}" "${REPO}/benchmarks/portability/validate.py" "${RUN_DIR}"
if [[ "${MATSIM_RUN_ALL_MODEL_SCIENTIFIC_DEBATE:-0}" == "1" ]]; then
  "${PYTHON}" "${REPO}/benchmarks/portability/all_model_scientific_debate.py" \
    --rounds "${MATSIM_DEBATE_ROUNDS:-2}" --output "${RUN_DIR}/all-model-scientific-debate"
fi
