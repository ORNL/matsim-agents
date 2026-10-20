#!/bin/bash
#PBS -N matsim-llm-check
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=01:30:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# Qualify one locally stored model through a live vLLM server on Aurora.
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
#                              deployments/aurora/resources/llm-check.yaml)
#   MODEL_ROOT                 directory of model snapshots (default: $PROJ/models)
#   MATSIM_LLM_CONCURRENCY     simultaneous qualification requests (default: 2)
#   MATSIM_RUN_PORTABILITY=1   run live scientific portability after qualification
#
# Preview the placement first: matsim-agents resources plan <layout>
#
# Submit with the site allocation supplied to qsub, not embedded here:
#   qsub -A <project> -v PROJECT_ROOT=$PWD deployments/aurora/jobs/job-llm-check-aurora.sh
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
RUNTIME_ENV="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}/deployments/common/runtime-env.sh"
[[ -f "${RUNTIME_ENV}" ]] || { echo "ERROR: export PROJECT_ROOT before submission" >&2; exit 2; }
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}")"
PROJ="$(dirname "${REPO}")"
VENV="${MATSIM_AURORA_VENV:-${REPO}/.venv}"
init_run_dirs "${PROJ}" "llm-check-aurora" "${PBS_JOBID:-$$}"
export PROJECT_ROOT="${REPO}" MODEL_ROOT="${MODEL_ROOT:-${PROJ}/models}"
RESOURCES="${MATSIM_RESOURCES:-${REPO}/deployments/aurora/resources/llm-check.yaml}"
[[ -d "${MODEL_ROOT}" ]] || { echo "ERROR: model directory not found: ${MODEL_ROOT}" >&2; exit 2; }
[[ -f "${RESOURCES}" ]] || { echo "ERROR: resource layout not found: ${RESOURCES}" >&2; exit 2; }

source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_frameworks_modules
[[ -f "${VENV}/bin/activate" ]] || { echo "ERROR: client environment not found: ${VENV}" >&2; exit 2; }
source "${VENV}/bin/activate"
cd "${REPO}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*' NO_PROXY='*'

matsim-agents resources plan "${RESOURCES}"
FACILITY=aurora RUN_DIR="${RUN_DIR}" \
  matsim-agents resources serve "${RESOURCES}" --output "${RUN_DIR}/llm-servers" -- \
  bash "${REPO}/deployments/common/run-llm-check.sh"

echo "[$(date)] LLM deployment qualification complete"
echo "  Run root:    ${RUN_DIR}"
echo "  Server logs: ${RUN_DIR}/llm-servers"
