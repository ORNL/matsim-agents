#!/bin/bash
#SBATCH -A mat746
#SBATCH -J matsim-rhea
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/rhea-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/rhea-%j/job-%j.out
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
# ---------------------------------------------------------------------------
# matsim-agents: RHEA discovery run on Frontier (1 node, 8 GCDs)
#
# Layout:
#   • vLLM server  : all 8 GCDs, tensor-parallel-size=8  (Qwen2.5-72B-Instruct)
#   • matsim-agents: CPU-only (chat --auto-confirm, piped query)
#
# Usage:
#   sbatch scripts/job-rhea-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ───────────────────────────────────────────────────────────────────
PROJ=/lustre/orion/mat746/proj-shared
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv
HYDRAGNN_EXAMPLE=$PROJ/HydraGNN/examples/multidataset_hpo_sc26
LOGDIR=$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64
MLP_CHECKPOINT=$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt
MODEL_DIR=$PROJ/models/Qwen2.5-72B-Instruct
RUN_DIR=$PROJ/runs/rhea-$SLURM_JOB_ID
OUTPUT_DIR=$RUN_DIR/outputs

mkdir -p "$RUN_DIR" "$OUTPUT_DIR"

# ── proxy (needed for any network calls on Frontier compute nodes) ───────────
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

# ── modules & conda env ──────────────────────────────────────────────────────
module reset
module load rocm/7.1.1
module load amd-mixed/7.1.1
module load PrgEnv-gnu
module load miniforge3/23.11.0-0
module unload darshan-runtime        # avoids ADIOS2 / darshan conflicts

conda activate "$VENV"

# Make HydraGNN example utilities importable (inference_fused, etc.)
export PYTHONPATH=$HYDRAGNN_EXAMPLE:$PROJ/HydraGNN:${PYTHONPATH:-}

# Suppress ROCm / MIOpen noise
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/miopen-$SLURM_JOB_ID
mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1   # flush vLLM / matsim-agents output immediately

# ── hard-block all remote model / dataset fetches ───────────────────────────
# These env-vars make HuggingFace transformers / hub / datasets raise an error
# immediately if any code attempts a network download, instead of silently
# hitting the internet (which would also fail on Frontier compute nodes).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── vLLM server ──────────────────────────────────────────────────────────────
VLLM_PORT=8000
VLLM_LOG=$RUN_DIR/vllm-server.log

echo "[$(date)] Starting vLLM server (tensor-parallel-size=8) ..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port $VLLM_PORT \
    --host 0.0.0.0 \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "[$(date)] vLLM PID: $VLLM_PID"

# Kill vLLM on script exit (normal or error)
trap 'echo "[$(date)] Killing vLLM (PID $VLLM_PID)"; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null' EXIT

# ── wait for vLLM to be ready (up to 20 min) ────────────────────────────────
# Loading 133 GB Qwen2.5-72B from Lustre onto 8 GCDs takes ~15 min.
echo "[$(date)] Waiting for vLLM server on port $VLLM_PORT ..."
READY=0
for i in $(seq 1 240); do
    if curl -fsS --max-time 3 "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[$(date)] vLLM server is ready (attempt $i)."
        READY=1
        break
    fi
    sleep 5
done
if [[ $READY -eq 0 ]]; then
    echo "[$(date)] ERROR: vLLM server did not start within 20 minutes. Check $VLLM_LOG"
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
    --llm-model       Qwen/Qwen2.5-72B-Instruct \
    --llm-base-url    "http://localhost:${VLLM_PORT}/v1" \
    --optimizer       FIRE \
    --maxiter         500 \
    --fmax            0.02 \
    --min-atoms       64 \
    --n-orderings     2 \
    --auto-confirm \
    2>&1 | tee "$RUN_DIR/matsim-agents.log"

echo "[$(date)] matsim-agents finished. Artifacts in $OUTPUT_DIR"
