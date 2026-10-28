#!/bin/bash
#PBS -N matsim-e1-dev
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=01:30:00
#PBS -l filesystems=home:flare
#PBS -r n
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# Development-only E1 response collection. This file is a template; it does
# not submit itself. Supply an approved one-server resource plan and a
# development-only manifest before use. Held-out evaluation is not implemented.
#
# Required at submission:
#   PROJECT_ROOT, E1_MANIFEST, MATSIM_RESOURCES
#   MODEL_ROOT may be required by the selected resource plan.
#
# Select only a queue that currently allows 90 minutes. The Aurora debug queue
# is capped at 60 minutes and is not eligible for this template. The requested
# ceiling is one node x 1.5 hours = 1.5 node-hours.
# ---------------------------------------------------------------------------

set -eo pipefail
JOB_START_EPOCH="$(date +%s)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
RUNTIME_ENV="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}/deployments/common/runtime-env.sh"
[[ -f "${RUNTIME_ENV}" ]] || { echo "ERROR: export PROJECT_ROOT before submission" >&2; exit 2; }
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}")"
PROJ="$(dirname "${REPO}")"
VENV="${MATSIM_AURORA_VENV:-${REPO}/.venv}"
init_run_dirs "${PROJ}" "e1-development-aurora" "${PBS_JOBID:-$$}"
export PROJECT_ROOT="${REPO}" MODEL_ROOT="${MODEL_ROOT:-${PROJ}/models}"
[[ -f "${E1_MANIFEST:-}" ]] || { echo "ERROR: set E1_MANIFEST to an approved development manifest" >&2; exit 2; }
[[ -f "${MATSIM_RESOURCES:-}" ]] || { echo "ERROR: set MATSIM_RESOURCES to an approved one-server plan" >&2; exit 2; }
[[ -f "${VENV}/bin/activate" ]] || { echo "ERROR: client environment not found: ${VENV}" >&2; exit 2; }

source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_frameworks_modules
source "${VENV}/bin/activate"
cd "${REPO}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MATSIM_VLLM_TIMEOUT_SECONDS=300
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*' NO_PROXY='*'

matsim-agents resources plan "${MATSIM_RESOURCES}"
ELAPSED_SECONDS=$(( $(date +%s) - JOB_START_EPOCH ))
REMAINING_SECONDS=$((5400 - ELAPSED_SECONDS))
if (( REMAINING_SECONDS <= 300 )); then
  echo "ERROR: less than 5 minutes remain for orderly shutdown and artifact collection" >&2
  exit 124
fi
WORKFLOW_TIMEOUT_SECONDS=$((REMAINING_SECONDS - 300))
FACILITY=aurora RUN_DIR="${RUN_DIR}" \
  timeout --signal=INT --kill-after=120s "${WORKFLOW_TIMEOUT_SECONDS}s" \
  matsim-agents resources serve "${MATSIM_RESOURCES}" \
    --output "${RUN_DIR}/llm-servers" -- \
    matsim-agents e1-development "${E1_MANIFEST}" --output "${OUTPUT_DIR}"

JOB_START_EPOCH="${JOB_START_EPOCH}" python3 - "${RUN_DIR}/allocation-runtime.json" <<'PY'
import json, os, pathlib, time
started = int(os.environ["JOB_START_EPOCH"])
pathlib.Path(__import__("sys").argv[1]).write_text(json.dumps({
    "job_id": os.environ.get("PBS_JOBID"),
    "queue": os.environ.get("PBS_O_QUEUE", "unknown"),
    "requested_nodes": 1,
    "maximum_walltime_seconds": 5400,
    "maximum_node_hours": 1.5,
    "job_script_elapsed_seconds": max(0, int(time.time()) - started),
    "facility_charge": None,
}, indent=2) + "\n")
PY