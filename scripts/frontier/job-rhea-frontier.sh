#!/bin/bash
#SBATCH -A mat746
#SBATCH -J matsim-rhea
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/rhea-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/rhea-%j/job-%j.out
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
# ---------------------------------------------------------------------------
# matsim-agents: RHEA discovery run on Frontier (1 node, 8 GCDs)
#
# Layout:
#   • vLLM server  : all 8 GCDs, tensor-parallel-size=8
#   • matsim-agents: CPU-only (chat --auto-confirm, piped query)
#
# Usage:
#   sbatch scripts/frontier/job-rhea-frontier.sh
#
# Override model at submission:
#   MATSIM_MODEL_DIR=.../Qwen3-32B MATSIM_MODEL_NAME=Qwen/Qwen3-32B \
#     sbatch scripts/frontier/job-rhea-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ───────────────────────────────────────────────────────────────────
PROJ=/lustre/orion/mat746/proj-shared
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
HYDRAGNN_EXAMPLE=$PROJ/HydraGNN/examples/multidataset_hpo_sc26
LOGDIR=$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64
MLP_CHECKPOINT=$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt
MODEL_DIR=${MATSIM_MODEL_DIR:-$PROJ/models/Qwen2.5-72B-Instruct}
MODEL_NAME=${MATSIM_MODEL_NAME:-Qwen/Qwen2.5-72B-Instruct}
RUN_DIR=$PROJ/runs/rhea-$SLURM_JOB_ID
OUTPUT_DIR=$RUN_DIR/outputs

mkdir -p "$RUN_DIR" "$OUTPUT_DIR"

# ── modules & conda env ──────────────────────────────────────────────────────
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh

source "$PROJ/matsim-agents/scripts/frontier/frontier-module-stack.sh"
load_frontier_rocm72_modules

source activate "$VENV"

# Make HydraGNN example utilities importable (inference_fused, etc.)
export PYTHONPATH=$HYDRAGNN_EXAMPLE:$PROJ/HydraGNN:${PYTHONPATH:-}

# Suppress ROCm / MIOpen noise
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/miopen-$SLURM_JOB_ID
mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Hard-block all remote fetches (compute nodes have no outbound internet)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ROCm / vLLM env vars established during smoke tests
export VLLM_CUDART_SO_PATH=/opt/rocm-7.2.0/lib/libamdhip64.so
export VLLM_NCCL_SO_PATH=/opt/rocm-7.2.0/lib/librccl.so.1
TORCH_LIB=$VENV/lib/python3.11/site-packages/torch/lib
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export HSA_NO_SCRATCH_RECLAIM=1
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_ARCH=gfx90a
export NCCL_DEBUG=WARN   # reduce noise; set INFO for debugging
export RCCL_UNROLL_FACTOR=0
export RCCL_P2P_BATCH_ENABLE=0
export RCCL_P2P_BATCH_THRESHOLD=0

# Compile / kernel cache on Lustre (avoids NFS Stale file handle errors)
export TVM_FFI_CACHE_DIR=$PROJ/cache/tvm-ffi
export VLLM_CACHE_ROOT=$PROJ/cache/vllm-cache
export TRITON_CACHE_DIR=$PROJ/cache/vllm-cache/triton
rm -rf "$VLLM_CACHE_ROOT"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

# ── vLLM server ──────────────────────────────────────────────────────────────
VLLM_PORT=8000
VLLM_LOG=$RUN_DIR/vllm-server.log

# For Qwen3 models: strip <think> tokens and route them to reasoning_content
REASONING_ARGS=""
if [[ "$MODEL_NAME" == *"Qwen3"* ]]; then
    REASONING_ARGS="--reasoning-parser deepseek_r1"
fi

echo "[$(date)] Starting vLLM server (TP=8, model=$MODEL_NAME) ..."
srun -N1 -n1 -c56 --gpus-per-task=8 --gpu-bind=closest \
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR" \
        --served-model-name "$MODEL_NAME" \
        --tensor-parallel-size 8 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --enforce-eager \
        --port $VLLM_PORT \
        --host 0.0.0.0 \
        $REASONING_ARGS \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "[$(date)] vLLM PID: $VLLM_PID"

# Kill vLLM on script exit (normal or error)
trap 'echo "[$(date)] Killing vLLM (PID $VLLM_PID)"; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null' EXIT

# ── wait for vLLM to be ready (up to 10 min) ────────────────────────────────
# Loading 133 GB Qwen2.5-72B from Lustre onto 8 GCDs takes ~5-10 min.
echo "[$(date)] Waiting for vLLM server on port $VLLM_PORT ..."
READY=0
for i in $(seq 1 120); do
    if curl -fsS --max-time 3 "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[$(date)] vLLM server is ready (attempt $i)."
        READY=1
        break
    fi
    sleep 5
done
if [[ $READY -eq 0 ]]; then
    echo "[$(date)] ERROR: vLLM server did not start within 10 minutes. Check $VLLM_LOG"
    exit 1
fi

export MATSIM_VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"

# ── RHEA query ───────────────────────────────────────────────────────────────
QUERY="Propose 4 to 5 refractory high-entropy alloy compositions using elements \
from Mo, Nb, Ta, W, V, Cr, Hf, Zr, Ti that are known for combined \
high-temperature resistance and mechanical strength. For each composition \
specify the relevant crystal phases (e.g. BCC, B2, HCP) and explain the \
physical justification. Then relax each proposed structure using the MLFF \
and report the final energies and which phases are most stable."

echo "[$(date)] Submitting RHEA query to matsim-agents ..."
echo "$QUERY" | matsim-agents chat \
    --logdir          "$LOGDIR" \
    --mlp-checkpoint  "$MLP_CHECKPOINT" \
    --output-dir      "$OUTPUT_DIR" \
    --llm-provider    vllm \
    --llm-model       "$MODEL_NAME" \
    --llm-base-url    "http://localhost:${VLLM_PORT}/v1" \
    --ase-structure-optimizer FIRE \
    --maxiter         500 \
    --fmax            0.02 \
    --min-atoms       64 \
    --n-orderings     2 \
    --auto-confirm \
    2>&1 | tee "$RUN_DIR/matsim-agents.log"

echo "[$(date)] matsim-agents finished. Artifacts in $OUTPUT_DIR"
