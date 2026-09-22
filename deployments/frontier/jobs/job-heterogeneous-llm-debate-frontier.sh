#!/bin/bash
#SBATCH -J matsim-hetero-debate
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
RUNTIME_ENV="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}/deployments/common/runtime-env.sh"
[[ -f "${RUNTIME_ENV}" ]] || { echo "ERROR: export PROJECT_ROOT before submission" >&2; exit 2; }
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}")"
PROJ="$(dirname "${REPO}")"
VENV="${REPO}/.venv"
MODELS_ROOT="${MODEL_ROOT:-${PROJ}/models}"
CATALOG="${REPO}/deployments/frontier/config/heterogeneous-debate-models.json"
init_run_dirs "${PROJ}" "heterogeneous-llm-debate-frontier" "${SLURM_JOB_ID:-$$}"

source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "${REPO}/deployments/frontier/setup/frontier-module-stack.sh"
load_frontier_rocm72_modules
source activate "${VENV}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_CUDART_SO_PATH=/opt/rocm-7.2.0/lib/libamdhip64.so
export VLLM_NCCL_SO_PATH=/opt/rocm-7.2.0/lib/librccl.so.1
export LD_LIBRARY_PATH="${VENV}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export HSA_NO_SCRATCH_RECLAIM=1
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_ARCH=gfx90a
export RCCL_UNROLL_FACTOR=0
export RCCL_P2P_BATCH_ENABLE=0
export RCCL_P2P_BATCH_THRESHOLD=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*'
export NO_PROXY='*'

source "${REPO}/deployments/frontier/setup/tvm-ffi-artifact.sh"
require_tvm_ffi_artifact "${REPO}" "${VENV}"

MAX_MODEL_LEN="${MATSIM_VLLM_MAXLEN:-4096}"
GPU_UTIL="${MATSIM_VLLM_GPU_UTIL:-0.85}"
READY_TIMEOUT="${MATSIM_VLLM_READY_TIMEOUT:-2400}"
declare -a SERVER_PIDS=()
declare -a SERVER_LOGS=()

launch_server() {
  local visible_gpus="$1" tensor_parallel="$2" port="$3" model_dir="$4" model_name="$5"
  local log="${RUN_DIR}/vllm-${port}.log"
  [[ -d "${model_dir}" ]] || { echo "ERROR: model directory not found: ${model_dir}" >&2; return 2; }
  echo "[$(date)] Serving ${model_name} on GCDs ${visible_gpus}, port ${port}, TP=${tensor_parallel}"
  (
    export HIP_VISIBLE_DEVICES="${visible_gpus}"
    unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    local jit="/tmp/vllm-jit.${USER}.${SLURM_JOB_ID:-$$}.${port}"
    rm -rf "${jit}"
    mkdir -p "${jit}/tmp" "${jit}/miopen"
    export TRITON_CACHE_DIR="${jit}/triton"
    export TORCHINDUCTOR_CACHE_DIR="${jit}/inductor"
    export VLLM_CACHE_ROOT="${jit}/vllm"
    export XDG_CACHE_HOME="${jit}/xdg"
    export TORCH_HOME="${jit}/torch"
    export TMPDIR="${jit}/tmp"
    export MIOPEN_DISABLE_CACHE=1
    export MIOPEN_USER_DB_PATH="${jit}/miopen"
    exec "${VENV}/bin/vllm" serve "${model_dir}" \
      --served-model-name "${model_name}" \
      --host 127.0.0.1 \
      --port "${port}" \
      --tensor-parallel-size "${tensor_parallel}" \
      --dtype bfloat16 \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_UTIL}" \
      --trust-remote-code \
      --no-enable-log-requests \
      --enforce-eager
  ) >"${log}" 2>&1 &
  SERVER_PIDS+=("$!")
  SERVER_LOGS+=("${log}")
}

cleanup() {
  echo "[$(date)] Stopping ${#SERVER_PIDS[@]} vLLM servers"
  for pid in "${SERVER_PIDS[@]}"; do
    pkill -TERM -P "${pid}" 2>/dev/null || true
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT

launch_server "0" "1" "8000" "${MODELS_ROOT}/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-14B-Instruct"
launch_server "1,2" "2" "8001" "${MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-32B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
launch_server "3" "1" "8002" "${MODELS_ROOT}/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.1-8B-Instruct"

wait_ready() {
  local port="$1" expected_model="$2" elapsed=0 response
  while (( elapsed < READY_TIMEOUT )); do
    for index in "${!SERVER_PIDS[@]}"; do
      if ! kill -0 "${SERVER_PIDS[$index]}" 2>/dev/null; then
        echo "ERROR: vLLM server exited before readiness: ${SERVER_LOGS[$index]}" >&2
        tail -n 100 "${SERVER_LOGS[$index]}" >&2 || true
        return 1
      fi
    done
    if response="$(curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null)" && \
      RESPONSE="${response}" EXPECTED_MODEL="${expected_model}" "${VENV}/bin/python3" -c \
      'import json, os, sys; data=json.loads(os.environ["RESPONSE"]); sys.exit(0 if os.environ["EXPECTED_MODEL"] in [item.get("id") for item in data.get("data", [])] else 1)'; then
      echo "[$(date)] ${expected_model} ready on port ${port}"
      return 0
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  echo "ERROR: ${expected_model} was not ready after ${READY_TIMEOUT}s" >&2
  return 1
}

wait_ready "8000" "Qwen/Qwen2.5-14B-Instruct"
wait_ready "8001" "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
wait_ready "8002" "meta-llama/Llama-3.1-8B-Instruct"

export MATSIM_VLLM_QWEN25_14B_BASE_URL="http://127.0.0.1:8000/v1"
export MATSIM_VLLM_DEEPSEEK_R1_32B_BASE_URL="http://127.0.0.1:8001/v1"
export MATSIM_VLLM_LLAMA31_8B_BASE_URL="http://127.0.0.1:8002/v1"
export MATSIM_VLLM_API_KEY=EMPTY

"${VENV}/bin/python3" "${REPO}/benchmarks/portability/all_model_scientific_debate.py" \
  --catalog "${CATALOG}" \
  --models-root "${MODELS_ROOT}" \
  --rounds "${MATSIM_DEBATE_ROUNDS:-2}" \
  --output "${OUTPUT_DIR}" \
  | tee "${RUN_DIR}/heterogeneous-debate.log"

echo "[$(date)] Heterogeneous multi-LLM debate passed: ${OUTPUT_DIR}"