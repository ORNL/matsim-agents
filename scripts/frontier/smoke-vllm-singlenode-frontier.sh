#!/bin/bash
# ---------------------------------------------------------------------------
# smoke-vllm-singlenode-frontier.sh
#
# Single-node smoke test: verify vLLM + ROCm works on Frontier (1 node,
# 8 GCDs, multiprocessing backend) before attempting multi-node runs.
#
# Submit:
#   sbatch --nodes=1 scripts/frontier/smoke-vllm-singlenode-frontier.sh
#
# Override model:
#   SMOKE_MODEL_PATH=... SMOKE_MODEL_NAME=... \
#   sbatch --nodes=1 scripts/frontier/smoke-vllm-singlenode-frontier.sh
# ---------------------------------------------------------------------------
#SBATCH -A mat746
#SBATCH -J smoke-singlenode
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/smoke-singlenode-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/smoke-singlenode-%j/job-%j.out
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug

set -uo pipefail

PROJ=/lustre/orion/mat746/proj-shared
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
# Default: small model that loads quickly
SMOKE_MODEL_PATH=${SMOKE_MODEL_PATH:-$PROJ/models/Llama-3.1-8B-Instruct}
SMOKE_MODEL_NAME=${SMOKE_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}
SMOKE_PORT=${SMOKE_PORT:-8000}
SMOKE_DTYPE=${SMOKE_DTYPE:-bfloat16}
SMOKE_MAX_MODEL_LEN=${SMOKE_MAX_MODEL_LEN:-4096}
GPUS_PER_NODE=8

RUN_DIR=$PROJ/runs/smoke-singlenode-${SLURM_JOB_ID:-local}
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "Single-node vLLM smoke test"
echo "Date:    $(date)"
echo "Node:    $(hostname)"
echo "Model:   $SMOKE_MODEL_NAME"
echo "Path:    $SMOKE_MODEL_PATH"
echo "TP:      $GPUS_PER_NODE"
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
export MIOPEN_DISABLE_CACHE=1
rmdir "$RUN_DIR/miopen-cache" 2>/dev/null; mkdir -p "$MIOPEN_USER_DB_PATH"

export VLLM_CUDART_SO_PATH=/opt/rocm-7.2.0/lib/libamdhip64.so

# Use ROCm 7.2.0 system RCCL (not PyTorch-bundled)
export VLLM_NCCL_SO_PATH=/opt/rocm-7.2.0/lib/librccl.so.1
TORCH_LIB=$VENV/lib/python3.11/site-packages/torch/lib
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

# HSA scratch reclaim fix (prevents RCCL collective init crash on Frontier)
export HSA_NO_SCRATCH_RECLAIM=1

# Target correct GPU architecture
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_ARCH=gfx90a

export NCCL_SOCKET_IFNAME=hsn
export GLOO_SOCKET_IFNAME=hsn
export FI_CXI_ATS=0

# Wipe stale compiled kernel cache (old gfx900 kernels cause HSA illegal instruction)
export VLLM_CACHE_ROOT=$RUN_DIR/vllm-cache
export TRITON_CACHE_DIR=$RUN_DIR/vllm-cache/triton
rm -rf "$VLLM_CACHE_ROOT"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# ── Cleanup trap ─────────────────────────────────────────────────────────────
TAIL_PID=""
cleanup() {
  echo "[cleanup] Stopping vLLM ..."
  [[ -n "${TAIL_PID:-}" ]] && kill "$TAIL_PID" 2>/dev/null || true
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Diagnostics ─────────────────────────────────────────────────────────────
echo "Python:  $(which python)  ($(python --version 2>&1))"
echo "torch:   $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())' 2>&1)"
echo "vllm:    $(python -c 'import vllm; print(vllm.__version__)' 2>&1)"
echo ""

# ── GPU warm-up (post-reboot KFD/HSA driver init) ────────────────────────────
# After a fresh node reboot the HSA runtime can hang on the first multi-process
# HIP init. Running a lightweight GPU job via srun first warms up /dev/kfd so
# subsequent srun steps (vLLM) don't deadlock.
echo "[warmup] Warming up GPUs on $(hostname) ..."
srun -N1 -n1 -c56 --gpus-per-task=${GPUS_PER_NODE} --gpu-bind=closest \
  "$VENV/bin/python" -c "
import torch, time
n = torch.cuda.device_count()
print(f'  HIP devices: {n}')
for i in range(n):
    x = torch.zeros(1024, device=f'cuda:{i}')
    _ = x + 1
print('  Warm-up tensors OK')
"
echo "[warmup] Done."
echo ""

# ── Start vLLM ───────────────────────────────────────────────────────────────
echo "[vllm] Starting server TP=${GPUS_PER_NODE} ..."
srun -N1 -n1 -c56 --gpus-per-task=${GPUS_PER_NODE} --gpu-bind=closest \
  "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$SMOKE_MODEL_PATH" \
    --served-model-name "$SMOKE_MODEL_NAME" \
    --tensor-parallel-size "$GPUS_PER_NODE" \
    --dtype "$SMOKE_DTYPE" \
    --max-model-len "$SMOKE_MAX_MODEL_LEN" \
    --port "$SMOKE_PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --disable-log-requests \
    --enforce-eager \
  > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
# Mirror log to stdout in real time
tail -f "$RUN_DIR/vllm.log" &
TAIL_PID=$!

# ── Wait for /health ─────────────────────────────────────────────────────────
echo "[vllm] Waiting for server (up to 10 min) ..."
MAX_WAIT=600
ELAPSED=0
INTERVAL=10
while true; do
  if curl -sf "http://localhost:${SMOKE_PORT}/health" > /dev/null 2>&1; then
    echo "[vllm] Server ready after ${ELAPSED}s."
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[FAIL] vLLM process (PID $VLLM_PID) exited before becoming ready." >&2
    echo "Last 40 lines of vllm.log:" >&2
    tail -40 "$RUN_DIR/vllm.log" >&2
    exit 1
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

if echo "$RESPONSE" | grep -q '"content"'; then
  echo ""
  echo "=========================================="
  echo "PASS: inference returned a response"
  echo "  Model:  $SMOKE_MODEL_NAME"
  echo "  Node:   $(hostname)"
  echo "  TP:     $GPUS_PER_NODE"
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
