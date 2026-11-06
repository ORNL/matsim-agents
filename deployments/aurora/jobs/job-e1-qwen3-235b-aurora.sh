#!/bin/bash
#PBS -N e1-qwen3-235b
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:flare
#PBS -r n
#PBS -k doe
#PBS -j oe

set -eo pipefail
JOB_START_EPOCH="$(date +%s)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" && pwd)"
REPO="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RUNTIME_ENV="${REPO}/deployments/common/runtime-env.sh"
[[ -f "${REPO}/pyproject.toml" && -f "${RUNTIME_ENV}" ]] || { echo "ERROR: invalid PROJECT_ROOT: ${REPO}" >&2; exit 2; }
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}")"
PROJ="$(dirname "${REPO}")"
VENV="${MATSIM_AURORA_VENV:-${REPO}/.venv}"
MODEL_ROOT="${MODEL_ROOT:-${PROJ}/models}"
export PROJECT_ROOT="${REPO}" MODEL_ROOT

[[ -n "${E1_MODEL_KEY:-}" && "${E1_MODEL_KEY}" =~ ^[a-z0-9-]+$ ]] || { echo "ERROR: set E1_MODEL_KEY to a lowercase model slug" >&2; exit 2; }
[[ -f "${E1_MANIFEST:-}" ]] || { echo "ERROR: set E1_MANIFEST to a development manifest" >&2; exit 2; }
case "${E1_MODEL_KEY}" in
  qwen3-235b-instruct-2507)
    DEFAULT_RESOURCES="${REPO}/deployments/aurora/resources/qwen3-235b-instruct-xpu-graph.yaml"
    ;;
  qwen3-235b-thinking-2507)
    DEFAULT_RESOURCES="${REPO}/deployments/aurora/resources/qwen3-235b-thinking-xpu-graph.yaml"
    ;;
  *)
    echo "ERROR: unsupported Qwen E1 model key: ${E1_MODEL_KEY}" >&2
    exit 2
    ;;
esac
MATSIM_RESOURCES="${MATSIM_RESOURCES:-${DEFAULT_RESOURCES}}"
[[ -f "${MATSIM_RESOURCES:-}" ]] || { echo "ERROR: set MATSIM_RESOURCES to a one-server resource plan" >&2; exit 2; }
if grep -Fq -- "--enforce-eager" "${MATSIM_RESOURCES}"; then
  echo "ERROR: Qwen E1 graph runs cannot use an eager-mode resource plan: ${MATSIM_RESOURCES}" >&2
  exit 2
fi
if ! grep -Eq '"cudagraph_mode"[[:space:]]*:[[:space:]]*"PIECEWISE"' "${MATSIM_RESOURCES}"; then
  echo "ERROR: Qwen E1 graph runs require PIECEWISE XPU Graphs: ${MATSIM_RESOURCES}" >&2
  exit 2
fi
[[ -f "${VENV}/bin/activate" ]] || { echo "ERROR: Aurora virtual environment not found: ${VENV}" >&2; exit 2; }
export VLLM_XPU_ENABLE_XPU_GRAPH=1

RUN_DIR="${REPO}/runs/e1/local-2026-11-03/${E1_MODEL_KEY}-${PBS_JOBID:-local}"
OUTPUT_DIR="${RUN_DIR}/outputs"
mkdir -p "${OUTPUT_DIR}"

finalize_job() {
  local status=$?
  trap - EXIT
  if [[ -n "${PBS_JOBID:-}" ]] && command -v qstat >/dev/null 2>&1; then
    qstat -fx "${PBS_JOBID}" >"${RUN_DIR}/qstat-final.txt" 2>&1 || true
  fi
  JOB_START_EPOCH="${JOB_START_EPOCH}" JOB_STATUS="${status}" python3 - "${RUN_DIR}/allocation-runtime.json" <<'PY'
import json, os, pathlib, re, sys, time

started = int(os.environ["JOB_START_EPOCH"])
accounting = pathlib.Path(sys.argv[1]).with_name("qstat-final.txt")
text = accounting.read_text(errors="replace") if accounting.exists() else ""
match = re.search(r"resources_used\.walltime\s*=\s*(\d+):(\d+):(\d+)", text)
walltime = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3)) if match else None
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "job_id": os.environ.get("PBS_JOBID"),
    "queue": os.environ.get("PBS_O_QUEUE", "unknown"),
    "requested_nodes": 1,
    "maximum_walltime_seconds": 14400,
    "maximum_node_hours": 4.0,
    "job_script_elapsed_seconds": max(0, int(time.time()) - started),
    "pbs_reported_walltime_seconds": walltime,
    "job_script_status": int(os.environ["JOB_STATUS"]),
    "facility_charge": None,
}, indent=2) + "\n")
PY
  exit "${status}"
}
trap finalize_job EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_frameworks_modules
source "${REPO}/deployments/aurora/setup/aurora-python-env.sh"
activate_matsim_aurora_python "${REPO}" "${VENV}"
cd "${REPO}"

export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export MATSIM_VLLM_TIMEOUT_SECONDS="${MATSIM_VLLM_TIMEOUT_SECONDS:-3600}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*' NO_PROXY='*'

echo "[$(date)] Running E1 ${E1_MODEL_KEY}"
echo "[$(date)] Manifest: ${E1_MANIFEST}"
echo "[$(date)] Resource plan: ${MATSIM_RESOURCES}"
echo "[$(date)] Output: ${OUTPUT_DIR}"
matsim-agents resources validate "${MATSIM_RESOURCES}"
matsim-agents resources plan "${MATSIM_RESOURCES}"
ELAPSED_SECONDS=$(( $(date +%s) - JOB_START_EPOCH ))
REMAINING_SECONDS=$((14400 - ELAPSED_SECONDS))
if (( REMAINING_SECONDS <= 300 )); then
  echo "ERROR: less than 5 minutes remain for orderly shutdown and artifact collection" >&2
  exit 124
fi
WORKFLOW_TIMEOUT_SECONDS=$((REMAINING_SECONDS - 300))
E1_COMMAND=(matsim-agents e1-development "${E1_MANIFEST}" --output "${OUTPUT_DIR}")
FACILITY=aurora RUN_DIR="${RUN_DIR}" \
  timeout --signal=INT --kill-after=120s "${WORKFLOW_TIMEOUT_SECONDS}s" \
  matsim-agents resources serve "${MATSIM_RESOURCES}" \
    --output "${RUN_DIR}/llm-servers" -- \
    "${E1_COMMAND[@]}"
echo "[$(date)] E1 ${E1_MODEL_KEY} collection finished"