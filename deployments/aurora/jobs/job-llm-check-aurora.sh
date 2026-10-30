#!/bin/bash
#PBS -N matsim-llm-check
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -r n
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# Qualify one locally stored model through a live vLLM server on Aurora.
#
# `matsim-agents resources serve` starts the server described by a resource
# layout on the GPUs it assigns, waits for the exact served model, runs the
# shared qualification steps (deployments/common/run-llm-check.sh: six-stage
# LLM check), and stops the server. No scientific portability suite is run.
#
# Required at submission:
#   PROJECT_ROOT       matsim-agents checkout (needed when the scheduler spools the script)
#
# Optional:
#   MATSIM_RESOURCES           layout with one server (default:
#                              deployments/aurora/resources/llm-check.yaml)
#   MODEL_ROOT                 directory of model snapshots (default: $PROJ/models)
#   MODEL_DIR                  checkpoint directory name under MODEL_ROOT
#   MATSIM_LLM_CONCURRENCY     simultaneous qualification requests (fixed at 1)
#
# Preview the placement first. The debug queue allows one node for up to 60 min;
# verify HydraGNN eligibility and allocation balance with the facility first.
#
# Submit with the site allocation supplied to qsub, not embedded here:
#   qsub -A HydraGNN -v PROJECT_ROOT=$PWD deployments/aurora/jobs/job-llm-check-aurora.sh
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset
JOB_START_EPOCH="$(date +%s)"

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
MODEL_DIR="${MODEL_DIR:-Mistral-Small-24B-Instruct-2501}"
MODEL_PATH="${MODEL_ROOT}/${MODEL_DIR}"
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "ERROR: local model snapshot not found: ${MODEL_PATH}" >&2; exit 2; }
[[ -f "${RESOURCES}" ]] || { echo "ERROR: resource layout not found: ${RESOURCES}" >&2; exit 2; }

source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_frameworks_modules
[[ -f "${VENV}/bin/activate" ]] || { echo "ERROR: client environment not found: ${VENV}" >&2; exit 2; }
source "${REPO}/deployments/aurora/setup/aurora-python-env.sh"
activate_matsim_aurora_python "${REPO}" "${VENV}"
cd "${REPO}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*' NO_PROXY='*'
export MATSIM_LLM_CONCURRENCY=1
LONG_CONTEXT_DEFAULT=8192
if [[ "${MODEL_DIR}" == "Mistral-Small-24B-Instruct-2501" ]]; then
  LONG_CONTEXT_DEFAULT=16384
elif [[ "${MODEL_DIR}" == "Gemma-4-26B-A4B-it" || "${MODEL_DIR}" == "Gemma-4-31B-it" ]]; then
  LONG_CONTEXT_DEFAULT=32768
fi
export MATSIM_LONG_CONTEXT_OUTPUT_TOKENS="${MATSIM_LONG_CONTEXT_OUTPUT_TOKENS:-${LONG_CONTEXT_DEFAULT}}"
export MATSIM_LONG_CONTEXT_PROMPT_TOKENS="${MATSIM_LONG_CONTEXT_PROMPT_TOKENS:-${LONG_CONTEXT_DEFAULT}}"
export MATSIM_LONG_CONTEXT_TOKENIZER_PATH="${MATSIM_LONG_CONTEXT_TOKENIZER_PATH:-${MODEL_PATH}}"

MEMORY_LOG="${RUN_DIR}/xpu-memory-samples.jsonl"
MEMORY_PID=""
python3 -u - "${MEMORY_LOG}" <<'PY' &
import json, signal, sys, time
from datetime import datetime, timezone

stopping = False
def stop(_signum, _frame):
    global stopping
    stopping = True
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
path = sys.argv[1]
try:
    import torch
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available() or not hasattr(xpu, "mem_get_info"):
        raise RuntimeError("torch.xpu.mem_get_info is unavailable")
    with open(path, "a", buffering=1) as output:
        while not stopping:
            devices = []
            for index in range(xpu.device_count()):
                free, total = xpu.mem_get_info(index)
                devices.append({"visible_device": index, "free_bytes": int(free), "total_bytes": int(total)})
            output.write(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "devices": devices}) + "\n")
            for _ in range(10):
                if stopping:
                    break
                time.sleep(1)
except Exception as exc:
    with open(path, "a", buffering=1) as output:
        output.write(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "telemetry_error": str(exc)}) + "\n")
PY
MEMORY_PID=$!

finalize_job() {
  status=$?
  trap - EXIT
  if [[ -n "${MEMORY_PID}" ]]; then
    kill -TERM "${MEMORY_PID}" 2>/dev/null || true
    wait "${MEMORY_PID}" 2>/dev/null || true
  fi
  if [[ -n "${PBS_JOBID:-}" ]] && command -v qstat >/dev/null 2>&1; then
    qstat -fx "${PBS_JOBID}" >"${RUN_DIR}/qstat-final.txt" 2>&1 || true
  fi
  JOB_START_EPOCH="${JOB_START_EPOCH}" JOB_STATUS="${status}" python3 - "${RUN_DIR}/allocation-runtime.json" <<'PY'
import json, os, pathlib, re, sys, time
started = int(os.environ["JOB_START_EPOCH"])
ended = int(time.time())
accounting = pathlib.Path(sys.argv[1]).with_name("qstat-final.txt")
accounting_text = accounting.read_text(errors="replace") if accounting.exists() else ""
match = re.search(r"resources_used\.walltime\s*=\s*(\d+):(\d+):(\d+)", accounting_text)
pbs_walltime_seconds = (
  int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))
  if match else None
)
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "job_id": os.environ.get("PBS_JOBID"),
    "project_id": "HydraGNN",
    "queue": os.environ.get("PBS_O_QUEUE", "debug"),
    "requested_nodes": 1,
    "maximum_walltime_seconds": 3600,
    "maximum_node_hours": 1.0,
    "job_script_elapsed_seconds": max(0, ended - started),
    "pbs_reported_walltime_seconds": pbs_walltime_seconds,
    "job_script_status": int(os.environ["JOB_STATUS"]),
    "facility_charge": None,
    "charge_note": "Facility accounting must supply actual chargeable reservation.",
}, indent=2) + "\n")
PY
  exit "${status}"
}
trap finalize_job EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

matsim-agents resources plan "${RESOURCES}"
ELAPSED_SECONDS=$(( $(date +%s) - JOB_START_EPOCH ))
WORKFLOW_TIMEOUT_SECONDS=$((3300 - ELAPSED_SECONDS))
if (( WORKFLOW_TIMEOUT_SECONDS <= 0 )); then
  echo "ERROR: less than 5 minutes remain for orderly shutdown and artifact collection" >&2
  exit 124
fi
FACILITY=aurora RUN_DIR="${RUN_DIR}" \
  timeout --signal=INT --kill-after=120s "${WORKFLOW_TIMEOUT_SECONDS}s" \
  matsim-agents resources serve "${RESOURCES}" --output "${RUN_DIR}/llm-servers" -- \
  bash "${REPO}/deployments/common/run-llm-check.sh"

echo "[$(date)] LLM deployment qualification complete"
echo "  Run root:    ${RUN_DIR}"
echo "  Server logs: ${RUN_DIR}/llm-servers"
