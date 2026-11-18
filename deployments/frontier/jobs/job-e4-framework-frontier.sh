#!/bin/bash
#SBATCH -J e4-framework
#SBATCH -p batch
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# One E4 framework measurement (E4.1 / E4.2 stage level) in this allocation.
# Supply -A, -N, -t and --export=ALL,E4_* to sbatch;
# `benchmarks/e4/framework/cli.py jobs ... --facility frontier` prints them.
# Required: PROJECT_ROOT, E4_CONFIG, E4_EXPERIMENT, E4_OUTPUT_ROOT.
# Optional: E4_REPETITION, E4_POLICY, E4_CLASSES, VENV_PATH, CONDA_SH, HYDRAGNN_DIR,
# and for real classes E4_VASP_BIN, E4_VASP_POTCAR_DIR, E4_HYDRAGNN_LOGDIR.
# Resubmitting the same command resumes the run without repeating finished tasks.
set -euo pipefail
E4_ALLOCATION_START_EPOCH=$(date +%s.%N)
export E4_ALLOCATION_START_EPOCH

REPO="${PROJECT_ROOT:?export PROJECT_ROOT to the matsim-agents checkout}"
PROJ="$(dirname "${REPO}")"
VENV_PATH="${VENV_PATH:-${REPO}/.venv}"
: "${E4_CONFIG:?}" "${E4_EXPERIMENT:?}" "${E4_OUTPUT_ROOT:?}"

if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
fi
source "${REPO}/deployments/frontier/setup/frontier-module-stack.sh"
load_frontier_rocm72_modules
if [[ -n "${CONDA_SH:-}" ]]; then
  source activate "${VENV_PATH}"
fi

export PYTHONNOUSERSITE=1
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="/tmp/miopen-${SLURM_JOB_ID:-$$}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
HYDRAGNN_ROOT="${HYDRAGNN_DIR:-${PROJ}/HydraGNN}"
export PYTHONPATH="${REPO}:${REPO}/src:${HYDRAGNN_ROOT}/examples/multidataset_hpo_sc26:${HYDRAGNN_ROOT}:${PYTHONPATH:-}"
# ROCm PyTorch exposes AMD GPUs through the cuda device API.
export E4_MLP_DEVICE="${E4_MLP_DEVICE:-cuda}"
cd "${REPO}"

ARGS=(run "${E4_CONFIG}" --facility frontier --experiment "${E4_EXPERIMENT}"
  --repetition "${E4_REPETITION:-0}" --output-root "${E4_OUTPUT_ROOT}"
  --policy "${E4_POLICY:-concurrent}" --auto-resume)
[[ -n "${E4_CLASSES:-}" ]] && ARGS+=(--classes "${E4_CLASSES}")
exec "${VENV_PATH}/bin/python3" "${REPO}/benchmarks/e4/framework/cli.py" "${ARGS[@]}"
