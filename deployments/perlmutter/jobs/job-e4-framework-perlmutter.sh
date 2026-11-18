#!/bin/bash
#SBATCH -J e4-framework
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# One E4 framework measurement (E4.1 / E4.2 stage level) in this allocation.
# Supply -A, -q, -N, -t and --export=ALL,E4_* to sbatch;
# `benchmarks/e4/framework/cli.py jobs ... --facility perlmutter` prints them.
# Required: PROJECT_ROOT, E4_CONFIG, E4_EXPERIMENT, E4_OUTPUT_ROOT.
# Optional: E4_REPETITION, E4_POLICY, E4_CLASSES, MATSIM_PERLMUTTER_VENV,
# and for real classes E4_VASP_BIN, E4_VASP_POTCAR_DIR, E4_HYDRAGNN_LOGDIR.
# Resubmitting the same command resumes the run without repeating finished tasks.
set -euo pipefail
E4_ALLOCATION_START_EPOCH=$(date +%s.%N)
export E4_ALLOCATION_START_EPOCH

REPO="${PROJECT_ROOT:?export PROJECT_ROOT to the matsim-agents checkout}"
VENV="${MATSIM_PERLMUTTER_VENV:-${REPO}/.venv}"
export MATSIM_PERLMUTTER_VENV="${VENV}"
: "${E4_CONFIG:?}" "${E4_EXPERIMENT:?}" "${E4_OUTPUT_ROOT:?}"

source "${REPO}/deployments/perlmutter/setup/setup_matsim_perlmutter.sh" --gpu
export PYTHONPATH="${REPO}:${REPO}/src:${PYTHONPATH:-}"
export E4_MLP_DEVICE="${E4_MLP_DEVICE:-cuda}"
cd "${REPO}"

ARGS=(run "${E4_CONFIG}" --facility perlmutter --experiment "${E4_EXPERIMENT}"
  --repetition "${E4_REPETITION:-0}" --output-root "${E4_OUTPUT_ROOT}"
  --policy "${E4_POLICY:-concurrent}" --auto-resume)
[[ -n "${E4_CLASSES:-}" ]] && ARGS+=(--classes "${E4_CLASSES}")
exec "${VENV}/bin/python3" "${REPO}/benchmarks/e4/framework/cli.py" "${ARGS[@]}"
