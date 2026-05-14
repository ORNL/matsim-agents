#!/bin/bash
# ---------------------------------------------------------------------------
# smoke-vllm-multinode-frontier.sh
#
# Smoke test: boot a multi-node vLLM Ray cluster on Frontier, load
# DeepSeek-V4-Pro (789 GB, FP8), verify /health, then run one inference
# request to confirm end-to-end generation works.
#
# Default: 4 nodes × 8 GCDs = 32 GCDs total (TP=32), ~2 TB GPU memory.
#
# Submit:
#   sbatch --nodes=4 scripts/frontier/smoke-vllm-multinode-frontier.sh
#
# Override model:
#   SMOKE_MODEL_PATH=... SMOKE_MODEL_NAME=... \
#   sbatch --nodes=4 scripts/frontier/smoke-vllm-multinode-frontier.sh
# ---------------------------------------------------------------------------
#SBATCH -A mat746
#SBATCH -J smoke-multinode
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/smoke-multinode-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/smoke-multinode-%j/job-%j.out
#SBATCH -t 01:00:00
#SBATCH -N 4
#SBATCH -p batch
#SBATCH -q debug

set -uo pipefail

PROJ=/lustre/orion/mat746/proj-shared
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
SMOKE_MODEL_PATH=${SMOKE_MODEL_PATH:-$PROJ/models/DeepSeek-V4-Pro}
SMOKE_MODEL_NAME=${SMOKE_MODEL_NAME:-deepseek-ai/DeepSeek-V4-Pro}
SMOKE_PORT=${SMOKE_PORT:-8000}
SMOKE_DTYPE=${SMOKE_DTYPE:-auto}                # auto → picks FP8 if available
SMOKE_MAX_MODEL_LEN=${SMOKE_MAX_MODEL_LEN:-4096}  # short context for smoke test
RAY_PORT=${RAY_PORT:-6379}
GPUS_PER_NODE=8

RUN_DIR=$PROJ/runs/smoke-multinode-${SLURM_JOB_ID:-local}
mkdir -p "$RUN_DIR"

N_NODES=$SLURM_NNODES
TP_SIZE=$(( N_NODES * GPUS_PER_NODE ))

echo "=========================================="
echo "Multi-node vLLM smoke test"
echo "Date:    $(date)"
echo "Nodes:   $N_NODES  ($SLURM_JOB_NODELIST)"
echo "Model:   $SMOKE_MODEL_NAME"
echo "Path:    $SMOKE_MODEL_PATH"
echo "TP:      $TP_SIZE"
echo "dtype:   $SMOKE_DTYPE"
echo "Run dir: $RUN_DIR"
echo "=========================================="

# ── Environment ─────────────────────────────────────────────────────────────
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$PROJ/matsim-agents/scripts/frontier/frontier-module-stack.sh"
load_frontier_rocm72_modules
source activate "$VENV"

module load rocm/7.2.0

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export MIOPEN_USER_DB_PATH="$RUN_DIR/miopen-cache"
mkdir -p "$MIOPEN_USER_DB_PATH"

export VLLM_CUDART_SO_PATH=/opt/rocm-7.2.0/lib/libamdhip64.so

# Inter-node networking over Slingshot (libfabric / HSN)
export NCCL_SOCKET_IFNAME=hsn
export GLOO_SOCKET_IFNAME=hsn
export FI_CXI_ATS=0

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# ── Discover nodes ───────────────────────────────────────────────────────────
mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE=${ALL_NODES[0]}
HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

echo ""
echo "Head: $HEAD_NODE ($HEAD_NODE_IP)"
echo "Workers: ${ALL_NODES[*]:1}"
echo ""

RAY="$VENV/bin/ray"

# ── Start Ray head ───────────────────────────────────────────────────────────
echo "[ray] Starting head ..."
"$RAY" start --head \
  --node-ip-address="$HEAD_NODE_IP" \
  --port="$RAY_PORT" \
  --num-cpus=56 \
  --num-gpus="$GPUS_PER_NODE" \
  --block &
RAY_HEAD_PID=$!
sleep 10

# ── Start Ray workers ────────────────────────────────────────────────────────
WORKER_PIDS=()
for node in "${ALL_NODES[@]:1}"; do
  echo "[ray] Starting worker on $node ..."
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 -w "$node" --export=ALL \
    "$VENV/bin/ray" start --address="$RAY_ADDRESS" \
      --num-cpus=56 --num-gpus="$GPUS_PER_NODE" --block &
  WORKER_PIDS+=($!)
done
sleep 20

echo "[ray] Cluster status:"
"$RAY" status --address="$RAY_ADDRESS" 2>&1 | head -20 || true
echo ""

# ── Cleanup trap ─────────────────────────────────────────────────────────────
cleanup() {
  echo "[cleanup] Stopping vLLM and Ray ..."
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  "$RAY" stop --force 2>/dev/null || true
  for pid in "${WORKER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  kill "$RAY_HEAD_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Start vLLM ───────────────────────────────────────────────────────────────
echo "[vllm] Starting server TP=${TP_SIZE} ..."
"$VENV/bin/vllm" serve "$SMOKE_MODEL_PATH" \
  --served-model-name "$SMOKE_MODEL_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --distributed-executor-backend ray \
  --dtype "$SMOKE_DTYPE" \
  --max-model-len "$SMOKE_MAX_MODEL_LEN" \
  --port "$SMOKE_PORT" \
  --trust-remote-code \
  --disable-log-requests \
  > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

# ── Wait for /health ─────────────────────────────────────────────────────────
echo "[vllm] Waiting for server (up to 20 min for weight loading) ..."
MAX_WAIT=1200
ELAPSED=0
INTERVAL=15
while true; do
  if curl -sf "http://localhost:${SMOKE_PORT}/health" > /dev/null 2>&1; then
    echo "[vllm] Server ready after ${ELAPSED}s."
    break
  fi
  if (( ELAPSED >= MAX_WAIT )); then
    echo "[FAIL] vLLM did not become ready within ${MAX_WAIT}s." >&2
    echo "Last 40 lines of vllm.log:" >&2
    tail -40 "$RUN_DIR/vllm.log" >&2
    exit 1
  fi
  sleep $INTERVAL
  (( ELAPSED += INTERVAL ))
done

echo ""
echo "=========================================="
echo "PASS: vLLM server is up"
echo "=========================================="
echo ""

# ── Inference smoke test ──────────────────────────────────────────────────────
echo "[smoke] Sending test inference request ..."
RESPONSE=$(curl -sf "http://localhost:${SMOKE_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d "{
    \"model\": \"${SMOKE_MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly three words: the sky is\"}],
    \"max_tokens\": 16,
    \"temperature\": 0.0
  }" 2>&1)

echo "[smoke] Response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"

# Check for non-empty content in the response
if echo "$RESPONSE" | grep -q '"content"'; then
  echo ""
  echo "=========================================="
  echo "PASS: inference returned a response"
  echo "  Model:  $SMOKE_MODEL_NAME"
  echo "  Nodes:  $N_NODES"
  echo "  TP:     $TP_SIZE"
  echo "  Job:    $SLURM_JOB_ID"
  echo "=========================================="
  EXIT_CODE=0
else
  echo ""
  echo "=========================================="
  echo "FAIL: inference response missing 'content' field"
  echo "=========================================="
  EXIT_CODE=1
fi

echo ""
echo "Full vLLM log: $RUN_DIR/vllm.log"
exit $EXIT_CODE
