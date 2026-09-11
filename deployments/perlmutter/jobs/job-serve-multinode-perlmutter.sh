#!/bin/bash
#SBATCH -J vllm-multinode
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -N 4
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH --gpus-per-node=4
#SBATCH -c 64
# ---------------------------------------------------------------------------
# Multi-node vLLM serve on Perlmutter (NVIDIA A100 80GB, CUDA 13).
#
# Bootstraps a Ray cluster across all allocated nodes, then starts a vLLM
# server with tensor parallelism spanning every GPU across all nodes. For a
# single-node allocation (-N 1), Ray is skipped entirely and vLLM's native
# multiprocessing TP is used instead.
#
# Designed for the 10 open-model-catalog models that don't fit a single
# 4x A100 80GB node. Provisional sizing (weights-on-disk / ~56GB per GPU,
# leaving headroom for KV cache), all confirmed against actual model sizes
# on CFS:
#   kimi-k2.5                        555G  -> 4 nodes (TP=16)
#   glm-4.7                          668G  -> 4 nodes (TP=16)
#   glm-4.7-flash                     59G  -> 1 node  (TP=4)
#   deepseek-v3.2                    643G  -> 4 nodes (TP=16)
#   mistral-large-3                  635G  -> 4 nodes (TP=16, native format)
#   qwen3-235b-a22b-instruct-2507     438G -> 2 nodes (TP=8)
#   qwen3-235b-a22b-thinking-2507     438G -> 2 nodes (TP=8)
#   devstral-2                       239G  -> 2 nodes (TP=8)
#   gemma-4-31b-it                    59G  -> 1 node  (TP=4)
#   gemma-4-26b-a4b-it                49G  -> 1 node  (TP=4)
#   (total: 25 nodes if all 10 run concurrently)
#
# Required env var at submission:
#   SERVE_MODEL_PATH   - absolute path to model weights directory
#
# Optional env vars:
#   SERVE_MODEL_NAME   - served model name (default: dir basename)
#   SERVE_PORT         - vLLM HTTP port (default 8000)
#   SERVE_TP_SIZE       - tensor parallel size; default = SLURM_NNODES * 4
#   SERVE_DTYPE        - model dtype (default: bfloat16)
#   SERVE_MAX_MODEL_LEN - max context length in tokens (default: 32768)
#   SERVE_GPU_UTIL     - --gpu-memory-utilization (default 0.90)
#   SERVE_ENFORCE_EAGER - 1 (default) to skip CUDA-graph capture on first light-up
#   SERVE_EXTRA_ARGS   - extra flags passed verbatim to vllm serve (e.g. native
#                        Mistral format: "--tokenizer-mode mistral
#                        --config-format mistral --load-format mistral")
#   RAY_PORT           - Ray head port (default 6379)
#
# Example (4 nodes, GLM-4.7):
#   SERVE_MODEL_PATH=$PROJ/models/GLM-4.7 \
#   sbatch --nodes=4 deployments/perlmutter/jobs/job-serve-multinode-perlmutter.sh
#
# Example (1 node, gemma-4-31B-it):
#   SERVE_MODEL_PATH=$PROJ/models/gemma-4-31B-it \
#   sbatch --nodes=1 deployments/perlmutter/jobs/job-serve-multinode-perlmutter.sh
#
# Example (Mistral-Large-3, native format):
#   SERVE_MODEL_PATH=$PROJ/models/Mistral-Large-3-675B-Instruct-2512 \
#   SERVE_EXTRA_ARGS="--tokenizer-mode mistral --config-format mistral --load-format mistral" \
#   sbatch --nodes=4 deployments/perlmutter/jobs/job-serve-multinode-perlmutter.sh
#
# The server stays alive until the job time limit. Connect clients to:
#   http://<head_node_ip>:${SERVE_PORT}/v1
# ---------------------------------------------------------------------------

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
RUNTIME_ENV="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}/deployments/common/runtime-env.sh"
[[ -f "${RUNTIME_ENV}" ]] || { echo "ERROR: export PROJECT_ROOT before submission" >&2; exit 2; }
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}")"
PROJ="$(dirname "${REPO}")"
VLLM_VENV=$REPO/venv_vllm
init_run_dirs "$PROJ" "vllm-multinode" "${SLURM_JOB_ID:-$$}"

if [[ -z "${SERVE_MODEL_PATH:-}" ]]; then
  echo "ERROR: SERVE_MODEL_PATH is required." >&2
  echo "  SERVE_MODEL_PATH=/path/to/model sbatch $0" >&2
  exit 2
fi
if [[ ! -d "${SERVE_MODEL_PATH}" ]]; then
  echo "ERROR: SERVE_MODEL_PATH does not exist: ${SERVE_MODEL_PATH}" >&2
  exit 2
fi

SERVE_PORT=${SERVE_PORT:-8000}
SERVE_DTYPE=${SERVE_DTYPE:-bfloat16}
SERVE_MAX_MODEL_LEN=${SERVE_MAX_MODEL_LEN:-32768}
SERVE_GPU_UTIL=${SERVE_GPU_UTIL:-0.90}
SERVE_ENFORCE_EAGER=${SERVE_ENFORCE_EAGER:-1}
RAY_PORT=${RAY_PORT:-6379}
MODEL_NAME=${SERVE_MODEL_NAME:-$(basename "$SERVE_MODEL_PATH")}

GPUS_PER_NODE=4
N_NODES=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}
SERVE_TP_SIZE=${SERVE_TP_SIZE:-$(( N_NODES * GPUS_PER_NODE ))}

echo "=========================================="
echo "vLLM multi-node serve on Perlmutter"
echo "Date:          $(date)"
echo "Nodes:         $N_NODES  ($SLURM_JOB_NODELIST)"
echo "Model:         $MODEL_NAME"
echo "Path:          $SERVE_MODEL_PATH"
echo "TP size:       $SERVE_TP_SIZE"
echo "dtype:         $SERVE_DTYPE"
echo "max_model_len: $SERVE_MAX_MODEL_LEN"
echo "Port:          $SERVE_PORT"
echo "Run dir:       $RUN_DIR"
echo "=========================================="

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_IMPORT_WARNING=1
# Large multi-node models can take well over vLLM's 600s default to load
# weights from CFS and stand up their engine core processes.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-2400}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*' NO_PROXY='*'

# vLLM JIT-compiles a CUDA helper at startup and needs Python.h; the base
# python3.11 has no dev headers, so point at the matsim .venv headers.
PYHDR="$REPO/.venv/include/python3.11"
export CPATH="${PYHDR}:${CPATH:-}" C_INCLUDE_PATH="${PYHDR}:${C_INCLUDE_PATH:-}"
JIT="/tmp/vllm-jit.${USER}.${SLURM_JOB_ID:-$$}"
mkdir -p "$JIT"
export FLASHINFER_WORKSPACE_BASE="$JIT" TRITON_CACHE_DIR="$JIT/triton"
export TORCHINDUCTOR_CACHE_DIR="$JIT/inductor" VLLM_CACHE_ROOT="$JIT/vllm"

# Perlmutter Slingshot: NCCL matches the "hsn" prefix across hsn0-hsn3, but
# Gloo (used by Ray/torch for CPU-side bootstrap) requires an *exact*
# interface name and errors ("Unable to find address for: hsn") if given the
# bare prefix. Leave GLOO_SOCKET_IFNAME unset so Gloo auto-selects a real
# interface; only override NCCL's.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-PHB}
[[ -n "${GLOO_SOCKET_IFNAME:-}" ]] && export GLOO_SOCKET_IFNAME

ENFORCE_EAGER_FLAG=()
[[ "$SERVE_ENFORCE_EAGER" == "1" ]] && ENFORCE_EAGER_FLAG=(--enforce-eager)

VLLM_PID=""
RAY_HEAD_PID=""
WORKER_PIDS=()

ray_gpu_count() {
  timeout 15s env RAY_ADDRESS="$RAY_ADDRESS" "$VLLM_VENV/bin/python" -c \
    'import os, ray; ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR"); print(int(ray.cluster_resources().get("GPU", 0))); ray.shutdown()' \
    2>/dev/null || echo 0
}

wait_for_ray_gpus() {
  local expected=$1 max_wait=$2 elapsed=0 total=0
  while (( elapsed < max_wait )); do
    total=$(ray_gpu_count | tail -1)
    [[ "$total" =~ ^[0-9]+$ ]] || total=0
    if (( total >= expected )); then
      echo "[ray] $total/$expected GPUs available after ${elapsed}s."
      return 0
    fi
    sleep 5
    (( elapsed += 5 ))
  done
  echo "[ray] ERROR: only $total/$expected GPUs available after ${max_wait}s." >&2
  return 1
}

cleanup() {
  echo ""
  echo "[cleanup] Stopping vLLM and Ray cluster ..."
  if [[ -n "$VLLM_PID" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    for _ in {1..12}; do
      kill -0 "$VLLM_PID" 2>/dev/null || break
      sleep 5
    done
    kill -9 "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  if [[ "$N_NODES" -gt 1 ]]; then
    "$VLLM_VENV/bin/ray" stop --force 2>/dev/null || true
    for pid in "${WORKER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    [[ -n "$RAY_HEAD_PID" ]] && kill "$RAY_HEAD_PID" 2>/dev/null || true
  fi
  echo "[cleanup] Done."
}
trap cleanup EXIT

if [[ "$N_NODES" -gt 1 ]]; then
  # ── multi-node: bootstrap a Ray cluster across the allocation ──────────────
  mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  HEAD_NODE=${ALL_NODES[0]}
  HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
  RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
  NCPUS=${SLURM_CPUS_ON_NODE:-64}

  echo ""
  echo "Head node: $HEAD_NODE  ($HEAD_NODE_IP)"
  echo "Worker nodes: ${ALL_NODES[*]:1}"
  echo ""

  echo "[ray] Starting head node at $RAY_ADDRESS ..."
  "$VLLM_VENV/bin/ray" start \
    --head \
    --node-ip-address="$HEAD_NODE_IP" \
    --port="$RAY_PORT" \
    --include-dashboard=false \
    --num-cpus="$NCPUS" \
    --num-gpus="$GPUS_PER_NODE" \
    --block &
  RAY_HEAD_PID=$!

  echo "[ray] Waiting for the head node ..."
  wait_for_ray_gpus "$GPUS_PER_NODE" 180 || exit 1

  for node in "${ALL_NODES[@]:1}"; do
    echo "[ray] Starting worker on $node ..."
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
         -w "$node" \
         --export=ALL \
      "$VLLM_VENV/bin/ray" start \
        --address="$RAY_ADDRESS" \
        --num-cpus="$NCPUS" \
        --num-gpus="$GPUS_PER_NODE" \
        --block &
    WORKER_PIDS+=($!)
  done

  EXPECTED_GPUS=$(( N_NODES * GPUS_PER_NODE ))
  echo "[ray] Waiting for all $EXPECTED_GPUS GPUs to join the cluster ..."
  wait_for_ray_gpus "$EXPECTED_GPUS" 600 || exit 1

  echo ""
  echo "[ray] Cluster status:"
  "$VLLM_VENV/bin/ray" status --address="$RAY_ADDRESS" || true
  echo ""

  echo "[vllm] Starting server TP=${SERVE_TP_SIZE} on port ${SERVE_PORT} (ray backend) ..."
  "$VLLM_VENV/bin/vllm" serve "$SERVE_MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size "$SERVE_TP_SIZE" \
    --distributed-executor-backend ray \
    --dtype "$SERVE_DTYPE" \
    --max-model-len "$SERVE_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$SERVE_GPU_UTIL" \
    --port "$SERVE_PORT" \
    --trust-remote-code \
    "${ENFORCE_EAGER_FLAG[@]}" \
    ${SERVE_EXTRA_ARGS:-} \
    > "$RUN_DIR/vllm-serve.log" 2>&1 &
  VLLM_PID=$!
else
  # ── single node: vLLM's native multiprocessing TP, no Ray needed ──────────
  HEAD_NODE=$(hostname -s)
  HEAD_NODE_IP=$(hostname -I | awk '{print $1}')

  echo "[vllm] Starting server TP=${SERVE_TP_SIZE} on port ${SERVE_PORT} (single node) ..."
  "$VLLM_VENV/bin/vllm" serve "$SERVE_MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size "$SERVE_TP_SIZE" \
    --dtype "$SERVE_DTYPE" \
    --max-model-len "$SERVE_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$SERVE_GPU_UTIL" \
    --port "$SERVE_PORT" \
    --trust-remote-code \
    "${ENFORCE_EAGER_FLAG[@]}" \
    ${SERVE_EXTRA_ARGS:-} \
    > "$RUN_DIR/vllm-serve.log" 2>&1 &
  VLLM_PID=$!
fi

echo "[vllm] Server PID=$VLLM_PID, waiting for /health ..."

MAX_WAIT=5400
ELAPSED=0
INTERVAL=10
while true; do
  if curl -sf "http://localhost:${SERVE_PORT}/health" > /dev/null 2>&1; then
    echo "[vllm] Server ready after ${ELAPSED}s."
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[vllm] ERROR: server process exited early." >&2
    tail -60 "$RUN_DIR/vllm-serve.log" >&2
    exit 1
  fi
  if (( ELAPSED >= MAX_WAIT )); then
    echo "[vllm] ERROR: server did not become ready within ${MAX_WAIT}s." >&2
    echo "[vllm] Last 60 lines of log:" >&2
    tail -60 "$RUN_DIR/vllm-serve.log" >&2
    exit 1
  fi
  sleep $INTERVAL
  (( ELAPSED += INTERVAL ))
done

echo ""
echo "=========================================="
echo "vLLM server is READY"
echo "  HEAD NODE:   $HEAD_NODE  ($HEAD_NODE_IP)"
echo "  BASE URL:    http://${HEAD_NODE_IP}:${SERVE_PORT}/v1"
echo "  MODEL NAME:  $MODEL_NAME"
echo "  TP SIZE:     $SERVE_TP_SIZE  (${N_NODES} node(s))"
echo "  JOB ID:      ${SLURM_JOB_ID:-N/A}"
echo "=========================================="
echo ""
echo "Point the matching base_url_env at this endpoint, e.g.:"
echo "  export MATSIM_VLLM_..._BASE_URL=http://${HEAD_NODE_IP}:${SERVE_PORT}/v1"
echo ""

echo "[serve] Server running. Waiting for job time limit or cancellation ..."
wait "$VLLM_PID"
echo "[serve] vLLM process exited."
