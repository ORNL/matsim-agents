#!/bin/bash
#SBATCH -J matsim-llm-check
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
# ---------------------------------------------------------------------------
# Qualify one locally stored model through a live vLLM server on Frontier
# (AMD MI250X, ROCm 7.2).
#
# `matsim-agents resources serve` starts the server described by a resource
# layout on the GPUs it assigns, waits for the exact served model, runs the
# shared qualification steps (deployments/common/run-llm-check.sh: six-stage
# LLM check, optional live scientific portability run), and stops the server.
#
# Required at submission:
#   PROJECT_ROOT       matsim-agents checkout (needed when the scheduler spools the script)
#
# Optional:
#   MATSIM_RESOURCES           layout with one server (default:
#                              deployments/frontier/resources/llm-check.yaml)
#   MODEL_ROOT                 directory of model snapshots (default: $PROJ/models)
#   MATSIM_LLM_CONCURRENCY     simultaneous qualification requests (default: 2)
#   MATSIM_RUN_PORTABILITY=1   run live scientific portability after qualification
#
# Preview the placement first: matsim-agents resources plan <layout>
#
# Submit with the site allocation supplied to sbatch, not embedded here:
#   PROJECT_ROOT=$PWD sbatch -A <allocation> \
#     deployments/frontier/jobs/job-llm-check-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
RUNTIME_ENV="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}/deployments/common/runtime-env.sh"
[[ -f "${RUNTIME_ENV}" ]] || { echo "ERROR: export PROJECT_ROOT before submission" >&2; exit 2; }
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}")"
PROJ="$(dirname "${REPO}")"
VENV="${REPO}/.venv"
init_run_dirs "${PROJ}" "llm-check-frontier" "${SLURM_JOB_ID:-$$}"
export PROJECT_ROOT="${REPO}" MODEL_ROOT="${MODEL_ROOT:-${PROJ}/models}"
RESOURCES="${MATSIM_RESOURCES:-${REPO}/deployments/frontier/resources/llm-check.yaml}"
[[ -d "${MODEL_ROOT}" ]] || { echo "ERROR: model directory not found: ${MODEL_ROOT}" >&2; exit 2; }
[[ -f "${RESOURCES}" ]] || { echo "ERROR: resource layout not found: ${RESOURCES}" >&2; exit 2; }

set +u
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "${REPO}/deployments/frontier/setup/frontier-module-stack.sh"
load_frontier_rocm72_modules
source activate "${VENV}"
set -u
cd "${REPO}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*' NO_PROXY='*'

matsim-agents resources plan "${RESOURCES}"
FACILITY=frontier RUN_DIR="${RUN_DIR}" \
  matsim-agents resources serve "${RESOURCES}" --output "${RUN_DIR}/llm-servers" -- \
  bash "${REPO}/deployments/common/run-llm-check.sh"

echo "[$(date)] LLM deployment qualification complete"
echo "  Run root:    ${RUN_DIR}"
echo "  Server logs: ${RUN_DIR}/llm-servers"
