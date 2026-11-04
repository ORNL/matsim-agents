#!/bin/bash
#PBS -N e1-deepseek-hf
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:flare
#PBS -r n
#PBS -k doe
#PBS -j oe

set -eo pipefail
JOB_START_EPOCH="$(date +%s)"

REPO="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PROJ="$(dirname "${REPO}")"
VENV="${MATSIM_AURORA_VENV:-${REPO}/.venv}"
MODEL_PATH="${MODEL_ROOT:-${PROJ}/models}/DeepSeek-R1-Distill-Qwen-32B"
RUN_DIR="${E1_RUN_DIR:-${REPO}/runs/e1/local-2026-11-02/deepseek-r1-distill-qwen-32b-hf}"
OUTPUT_DIR="${RUN_DIR}/outputs"
MANIFEST="${RUN_DIR}/manifest.json"

[[ -f "${REPO}/pyproject.toml" ]] || { echo "ERROR: invalid PROJECT_ROOT: ${REPO}" >&2; exit 2; }
[[ -f "${VENV}/bin/activate" ]] || { echo "ERROR: Aurora virtual environment not found: ${VENV}" >&2; exit 2; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "ERROR: local model snapshot not found: ${MODEL_PATH}" >&2; exit 2; }
[[ -f "${MANIFEST}" ]] || { echo "ERROR: E1 manifest not found: ${MANIFEST}" >&2; exit 2; }

source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_frameworks_modules
source "${REPO}/deployments/aurora/setup/aurora-python-env.sh"
activate_matsim_aurora_python "${REPO}" "${VENV}"
cd "${REPO}"

export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset ZE_AFFINITY_MASK http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*' NO_PROXY='*'
export MATSIM_LLM_PROVIDER=huggingface MATSIM_HF_MODEL_PATH="${MODEL_PATH}"
mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}"

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

echo "[$(date)] Running E1 DeepSeek collection on ${MODEL_PATH}"
echo "[$(date)] Output: ${OUTPUT_DIR}/responses.jsonl"
mpiexec -n 1 --ppn 1 matsim-agents e1-development "${MANIFEST}" --output "${OUTPUT_DIR}"
echo "[$(date)] E1 DeepSeek collection finished"