#!/bin/bash
#SBATCH -J matsim-llm-check
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -A m5216_g
#SBATCH --gpus-per-node=4
#SBATCH -c 32
# ---------------------------------------------------------------------------
# Qualify one locally stored model through a live vLLM server on Perlmutter.
#
# The job owns the complete server lifecycle: launch, exact-model readiness,
# matsim-agents' six-stage LLM qualification, optional live scientific
# portability run, and cleanup. Keeping this separate from model-quality
# benchmarking makes deployment failures independently reproducible.
#
# Required at submission:
#   PROJECT_ROOT       matsim-agents checkout (needed when Slurm spools script)
#
# Optional:
#   MATSIM_MODEL_DIR           local checkpoint directory
#   MATSIM_MODEL_NAME          served identifier (default: checkpoint basename)
#   MATSIM_VLLM_TP             tensor-parallel/GPU count (default: 4)
#   MATSIM_VLLM_GPUS           visible CUDA ids (default: 0,1,2,3)
#   MATSIM_VLLM_PORT           server port (default: 8000)
#   MATSIM_VLLM_MAXLEN         context length (default: 8192)
#   MATSIM_VLLM_GPU_UTIL       vLLM memory fraction (default: 0.90)
#   MATSIM_LLM_CONCURRENCY     simultaneous qualification requests (default: 2)
#   MATSIM_VLLM_READY_TIMEOUT  server startup timeout in seconds (default: 2400)
#   MATSIM_RUN_PORTABILITY=1   run live scientific portability after qualification
#
# Submit with the site allocation supplied to sbatch, not embedded here:
#   PROJECT_ROOT=$PWD sbatch -A <allocation> \
#     deployments/perlmutter/jobs/job-llm-check-perlmutter.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
RUNTIME_ENV="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}/deployments/common/runtime-env.sh"
[[ -f "${RUNTIME_ENV}" ]] || {
  echo "ERROR: export PROJECT_ROOT before submission" >&2
  exit 2
}
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}")"
PROJ="$(dirname "${REPO}")"
INSTALL_ROOT="${REPO}/.hpc-build/perlmutter"
CLIENT_VENV="${REPO}/.venv"
VLLM_VENV="${INSTALL_ROOT}/vllm_venv"
MODEL_DIR="${MATSIM_MODEL_DIR:-${PROJ}/models/Qwen2.5-14B-Instruct}"
MODEL_NAME="${MATSIM_MODEL_NAME:-$(basename "${MODEL_DIR}")}"
init_run_dirs "${PROJ}" "llm-check-perlmutter" "${SLURM_JOB_ID:-$$}"

VLLM_HOST="${MATSIM_VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${MATSIM_VLLM_PORT:-8000}"
VLLM_TP="${MATSIM_VLLM_TP:-4}"
VLLM_GPUS="${MATSIM_VLLM_GPUS:-0,1,2,3}"
VLLM_MAXLEN="${MATSIM_VLLM_MAXLEN:-8192}"
VLLM_GPU_UTIL="${MATSIM_VLLM_GPU_UTIL:-0.90}"
READY_TIMEOUT="${MATSIM_VLLM_READY_TIMEOUT:-2400}"
CONCURRENCY="${MATSIM_LLM_CONCURRENCY:-2}"
BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
VLLM_LOG="${RUN_DIR}/vllm-server.log"
CHECK_ROOT="${RUN_DIR}/llm-check"
CHECK_CONFIG="${RUN_DIR}/llm-check-config.json"

[[ -d "${MODEL_DIR}" ]] || { echo "ERROR: model directory not found: ${MODEL_DIR}" >&2; exit 2; }
[[ -x "${VLLM_VENV}/bin/vllm" ]] || { echo "ERROR: vLLM executable not found" >&2; exit 2; }
[[ -x "${CLIENT_VENV}/bin/python3" ]] || { echo "ERROR: client environment not found" >&2; exit 2; }
mkdir -p "${CHECK_ROOT}"

echo "[$(date)] Starting ${MODEL_NAME} from ${MODEL_DIR} on GPU(s) ${VLLM_GPUS} (tp=${VLLM_TP})"
(
  export CUDA_VISIBLE_DEVICES="${VLLM_GPUS}"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export VLLM_DO_NOT_TRACK=1
  export VLLM_NO_USAGE_STATS=1
  export PYTHONNOUSERSITE=1
  export VLLM_USE_FLASHINFER_SAMPLER=0
  JIT_TMP="/tmp/vllm-jit.${USER}.${SLURM_JOB_ID:-$$}"
  mkdir -p "${JIT_TMP}"
  export FLASHINFER_WORKSPACE_BASE="${JIT_TMP}"
  export TRITON_CACHE_DIR="${JIT_TMP}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${JIT_TMP}/inductor"
  export VLLM_CACHE_ROOT="${JIT_TMP}/vllm"
  # CFS does not support fcntl.flock on compute nodes; redirect all caches to tmpfs.
  export XDG_CACHE_HOME="${JIT_TMP}/xdg-cache"
  export HF_HOME="${JIT_TMP}/hf-home"
  export TORCH_HOME="${JIT_TMP}/torch-home"
  export TMPDIR="${JIT_TMP}/tmp"
  mkdir -p "${TMPDIR}"
  PYTHON_HEADERS="${CLIENT_VENV}/include/python3.11"
  export CPATH="${PYTHON_HEADERS}:${CPATH:-}"
  export C_INCLUDE_PATH="${PYTHON_HEADERS}:${C_INCLUDE_PATH:-}"
  exec "${VLLM_VENV}/bin/vllm" serve "${MODEL_DIR}" \
    --served-model-name "${MODEL_NAME}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${VLLM_TP}" \
    --max-model-len "${VLLM_MAXLEN}" \
    --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
    --trust-remote-code \
    --no-enable-log-requests \
    --enforce-eager
) >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "[$(date)] Stopping vLLM server (pid ${VLLM_PID})"
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[$(date)] Waiting for exact model ${MODEL_NAME} at ${BASE_URL}/models"
READY=0
ELAPSED=0
while (( ELAPSED < READY_TIMEOUT )); do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "ERROR: vLLM exited before readiness; tail of ${VLLM_LOG}:" >&2
    tail -n 80 "${VLLM_LOG}" >&2 || true
    exit 1
  fi
  if SERVED_MODELS=$(curl -fsS "${BASE_URL}/models" 2>/dev/null); then
    if SERVED_MODELS="${SERVED_MODELS}" EXPECTED_MODEL="${MODEL_NAME}" \
      "${CLIENT_VENV}/bin/python3" -c \
      'import json, os, sys; payload=json.loads(os.environ["SERVED_MODELS"]); sys.exit(0 if os.environ["EXPECTED_MODEL"] in [item.get("id") for item in payload.get("data", [])] else 1)'; then
      READY=1
      break
    fi
  fi
  sleep 10
  ELAPSED=$((ELAPSED + 10))
done
if [[ "${READY}" -ne 1 ]]; then
  echo "ERROR: exact model was not ready within ${READY_TIMEOUT}s; tail of ${VLLM_LOG}:" >&2
  tail -n 80 "${VLLM_LOG}" >&2 || true
  exit 1
fi
printf '%s\n' "${SERVED_MODELS}" >"${RUN_DIR}/models.json"
echo "[$(date)] Exact-model readiness passed after ${ELAPSED}s"

# Load the HydraGNN-aligned client environment only after vLLM has started in
# its isolated environment. The client communicates through HTTP and does not
# import the server's torch stack.
source "${REPO}/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CLIENT_VENV}"
cd "${REPO}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MATSIM_LLM_PROVIDER=vllm
export MATSIM_LLM_MODEL="${MODEL_NAME}"
export MATSIM_VLLM_BASE_URL="${BASE_URL}"
export MATSIM_VLLM_API_KEY=EMPTY
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*'
export NO_PROXY='*'

MODEL_NAME="${MODEL_NAME}" BASE_URL="${BASE_URL}" CHECK_ROOT="${CHECK_ROOT}" \
VLLM_TP="${VLLM_TP}" VLLM_MAXLEN="${VLLM_MAXLEN}" CONCURRENCY="${CONCURRENCY}" \
CHECK_CONFIG="${CHECK_CONFIG}" \
  "${CLIENT_VENV}/bin/python3" -c \
  'import json, os, pathlib; cfg={"provider":"vllm","model":os.environ["MODEL_NAME"],"base_url":os.environ["BASE_URL"],"api_key":"EMPTY","output_root":os.environ["CHECK_ROOT"],"temperature":0.0,"timeout_sec":120,"expected_accelerators":int(os.environ["VLLM_TP"]),"tensor_parallel_size":int(os.environ["VLLM_TP"]),"context_length":int(os.environ["VLLM_MAXLEN"]),"concurrent_requests":int(os.environ["CONCURRENCY"])}; pathlib.Path(os.environ["CHECK_CONFIG"]).write_text(json.dumps(cfg, indent=2)+"\n", encoding="utf-8")'

echo "[$(date)] Running six-stage LLM qualification"
matsim-agents llm-check "${CHECK_CONFIG}" | tee "${RUN_DIR}/llm-check-console.log"

shopt -s nullglob
CHECK_RUNS=("${CHECK_ROOT}"/*/)
shopt -u nullglob
[[ "${#CHECK_RUNS[@]}" -eq 1 ]] || {
  echo "ERROR: expected exactly one qualification run in ${CHECK_ROOT}, found ${#CHECK_RUNS[@]}" >&2
  exit 1
}
CHECK_RUN="${CHECK_RUNS[0]}"
CHECK_RUN="${CHECK_RUN}" "${CLIENT_VENV}/bin/python3" -c \
  'import json, os, pathlib, sys; result=json.loads((pathlib.Path(os.environ["CHECK_RUN"])/"result.json").read_text()); required={"readiness","load","generation","structured","discussion","distributed"}; ok=result.get("status")=="complete" and required.issubset({k for k,v in result.get("stages",{}).items() if v}); sys.exit(0 if ok else 1)'
ln -s "${CHECK_RUN}" "${RUN_DIR}/successful-llm-check"
echo "[$(date)] Qualification passed: ${CHECK_RUN}"

if [[ "${MATSIM_RUN_PORTABILITY:-0}" == "1" ]]; then
  PORTABILITY_DIR="${RUN_DIR}/portability-live"
  echo "[$(date)] Running live scientific portability suite"
  "${CLIENT_VENV}/bin/python3" benchmarks/portability/run.py \
    --facility perlmutter \
    --suite all \
    --backend qe \
    --execute \
    --live-llm \
    --llm-check-run "${CHECK_RUN}" \
    --output "${PORTABILITY_DIR}"
fi

echo "[$(date)] LLM deployment qualification complete"
echo "  Run root:      ${RUN_DIR}"
echo "  Qualification: ${CHECK_RUN}"
echo "  Server log:    ${VLLM_LOG}"
