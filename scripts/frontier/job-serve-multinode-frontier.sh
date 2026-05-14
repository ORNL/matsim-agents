#!/bin/bash
#SBATCH -A mat746
#SBATCH -J vllm-multinode
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/vllm-multinode-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/vllm-multinode-%j/job-%j.out
#SBATCH -t 02:00:00
#SBATCH -N 4
#SBATCH -p batch
#SBATCH -q debug
# ---------------------------------------------------------------------------
# Multi-node vLLM serve on Frontier (AMD MI250X, ROCm 7.2)
#
# Bootstraps a Ray cluster across all allocated nodes, then starts a vLLM
# server with tensor parallelism spanning every GPU across all nodes.
#
# Designed for very large models that exceed a single node's GPU memory.
# Default config targets DeepSeek-V4-0324 (1.6T params, FP8 ~1.6 TB):
#   - 4 nodes × 8 GCDs × 64 GB = 2 TB total — fits with headroom for KV cache
#   - TP = 32 (--tensor-parallel-size = N_NODES * 8)
#
# Required env var at submission:
#   SERVE_MODEL_PATH   – absolute path to model weights directory
#
# Optional env vars:
#   SERVE_MODEL_NAME   – HF model id for --served-model-name (default: dir basename)
#   SERVE_PORT         – vLLM HTTP port (default 8000)
#   SERVE_TP_SIZE      – tensor parallel size; default = SLURM_NNODES * 8
#   SERVE_DTYPE        – model dtype: bfloat16|float16 (default: bfloat16; MI250X has no FP8 support)
#   SERVE_MAX_MODEL_LEN – max context length in tokens (default: 32768)
#   SERVE_EXTRA_ARGS   – extra flags passed verbatim to vllm serve
#   RAY_PORT           – Ray head port (default 6379)
#
# Example submission (4 nodes, DeepSeek-V4-Pro):
#   SERVE_MODEL_PATH=/lustre/orion/mat746/proj-shared/models/DeepSeek-V4-Pro \
#   sbatch --nodes=4 scripts/frontier/job-serve-multinode-frontier.sh
#
# Example submission (2 nodes, Mixtral 8x22B sanity check):
#   SERVE_MODEL_PATH=/lustre/orion/mat746/proj-shared/models/Mixtral-8x22B-Instruct-v0.1 \
#   sbatch --nodes=2 scripts/frontier/job-serve-multinode-frontier.sh
#
# The server stays alive until the job time limit. Connect clients to:
#   http://<head_node>:${SERVE_PORT}/v1
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ=/lustre/orion/mat746/proj-shared
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
RUN_DIR=$PROJ/runs/vllm-multinode-$SLURM_JOB_ID
mkdir -p "$RUN_DIR"

# ---------------------------------------------------------------------------
# Validate required inputs
# ---------------------------------------------------------------------------
if [[ -z "${SERVE_MODEL_PATH:-}" ]]; then
  echo "ERROR: SERVE_MODEL_PATH is required." >&2
  echo "  SERVE_MODEL_PATH=/path/to/model sbatch $0" >&2
  exit 2
fi
if [[ ! -d "${SERVE_MODEL_PATH}" ]]; then
  echo "ERROR: SERVE_MODEL_PATH does not exist: ${SERVE_MODEL_PATH}" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVE_PORT=${SERVE_PORT:-8000}
SERVE_DTYPE=${SERVE_DTYPE:-bfloat16}
SERVE_MAX_MODEL_LEN=${SERVE_MAX_MODEL_LEN:-32768}
RAY_PORT=${RAY_PORT:-6379}
MODEL_NAME=${SERVE_MODEL_NAME:-$(basename "$SERVE_MODEL_PATH")}

# Tensor parallel: cover every GCD across all nodes by default
N_NODES=$SLURM_NNODES
GPUS_PER_NODE=8   # MI250X presents 8 GCDs per node to ROCm
SERVE_TP_SIZE=${SERVE_TP_SIZE:-$(( N_NODES * GPUS_PER_NODE ))}

echo "=========================================="
echo "vLLM multi-node serve on Frontier"
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

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source activate "$VENV"

module load rocm/7.2.0

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# ── No outbound connections (HPC: no internet access) ────────────────────────
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# vLLM usage telemetry
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# Ray telemetry
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_IMPORT_WARNING=1
# Triton: disable remote kernel downloads
export TRITON_DISABLE_AUTOTUNE_CACHE=1
# Block all HTTP/S proxies
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*'
export NO_PROXY='*'

# Allow Ray/vLLM to use all GCDs on each node
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Improve RCCL inter-node performance over Slingshot (libfabric)
export NCCL_SOCKET_IFNAME=hsn
export GLOO_SOCKET_IFNAME=hsn
export FI_CXI_ATS=0                    # disable address translation services
export MIOPEN_USER_DB_PATH="$RUN_DIR/miopen-cache"
mkdir -p "$MIOPEN_USER_DB_PATH"

# ---------------------------------------------------------------------------
# Discover nodes
# ---------------------------------------------------------------------------
mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE=${ALL_NODES[0]}
HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

echo ""
echo "Head node: $HEAD_NODE  ($HEAD_NODE_IP)"
echo "Worker nodes: ${ALL_NODES[*]:1}"
echo ""

# ---------------------------------------------------------------------------
# Start Ray head on this node (node 0)
# ---------------------------------------------------------------------------
RAY="$VENV/bin/ray"

echo "[ray] Starting head node at $RAY_ADDRESS ..."
"$RAY" start \
  --head \
  --node-ip-address="$HEAD_NODE_IP" \
  --port="$RAY_PORT" \
  --num-cpus=56 \
  --num-gpus="$GPUS_PER_NODE" \
  --block &
RAY_HEAD_PID=$!

sleep 10   # give head a moment to fully initialize

# ---------------------------------------------------------------------------
# Start Ray workers on remaining nodes
# ---------------------------------------------------------------------------
WORKER_PIDS=()
for node in "${ALL_NODES[@]:1}"; do
  echo "[ray] Starting worker on $node ..."
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
       -w "$node" \
       --export=ALL \
    "$VENV/bin/ray" start \
      --address="$RAY_ADDRESS" \
      --num-cpus=56 \
      --num-gpus="$GPUS_PER_NODE" \
      --block &
  WORKER_PIDS+=($!)
done

sleep 15   # wait for workers to join the cluster

# Verify cluster is assembled
echo ""
echo "[ray] Cluster status:"
"$RAY" status --address="$RAY_ADDRESS" || true
echo ""

# ---------------------------------------------------------------------------
# Cleanup function
# ---------------------------------------------------------------------------
cleanup() {
  echo ""
  echo "[cleanup] Stopping vLLM and Ray cluster ..."
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  for pid in "${WORKER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  kill "$RAY_HEAD_PID" 2>/dev/null || true
  echo "[cleanup] Done."
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Start vLLM server
# ---------------------------------------------------------------------------
echo "[vllm] Starting server TP=${SERVE_TP_SIZE} on port ${SERVE_PORT} ..."
"$VENV/bin/vllm" serve "$SERVE_MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size "$SERVE_TP_SIZE" \
  --distributed-executor-backend ray \
  --dtype "$SERVE_DTYPE" \
  --max-model-len "$SERVE_MAX_MODEL_LEN" \
  --port "$SERVE_PORT" \
  --trust-remote-code \
  --disable-log-requests \
  ${SERVE_EXTRA_ARGS:-} \
  > "$RUN_DIR/vllm-serve.log" 2>&1 &
VLLM_PID=$!

echo "[vllm] Server PID=$VLLM_PID, waiting for /health ..."

# ---------------------------------------------------------------------------
# Wait for vLLM to be ready
# ---------------------------------------------------------------------------
MAX_WAIT=600   # large models can take 5-10 min to load
ELAPSED=0
INTERVAL=10
while true; do
  if curl -sf "http://localhost:${SERVE_PORT}/health" > /dev/null 2>&1; then
    echo "[vllm] Server ready after ${ELAPSED}s."
    break
  fi
  if (( ELAPSED >= MAX_WAIT )); then
    echo "[vllm] ERROR: server did not become ready within ${MAX_WAIT}s." >&2
    echo "[vllm] Last 30 lines of log:" >&2
    tail -30 "$RUN_DIR/vllm-serve.log" >&2
    exit 1
  fi
  sleep $INTERVAL
  (( ELAPSED += INTERVAL ))
done

# ---------------------------------------------------------------------------
# Emit connection info for external clients
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "vLLM server is READY"
echo "  HEAD NODE:   $HEAD_NODE  ($HEAD_NODE_IP)"
echo "  BASE URL:    http://${HEAD_NODE_IP}:${SERVE_PORT}/v1"
echo "  MODEL NAME:  $MODEL_NAME"
echo "  TP SIZE:     $SERVE_TP_SIZE  (${N_NODES} nodes)"
echo "  JOB ID:      $SLURM_JOB_ID"
echo "=========================================="
echo ""
echo "To run eval against this server, on the head node:"
echo "  export MATSIM_VLLM_BASE_URL=http://${HEAD_NODE_IP}:${SERVE_PORT}/v1"
echo "  export MATSIM_VLLM_API_KEY=EMPTY"
echo ""

# ---------------------------------------------------------------------------
# Keep server alive until job time limit or manual cancel
# ---------------------------------------------------------------------------
echo "[serve] Server running. Waiting for job time limit or cancellation ..."
wait "$VLLM_PID"
echo "[serve] vLLM process exited."
