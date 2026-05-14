#!/bin/bash
# ---------------------------------------------------------------------------
# smoke-vllm-frontier.sh
#
# Smoke test: just bring up vLLM with Qwen2.5-72B on a single Frontier node
# (8 GCDs, tensor-parallel-size=8) and curl /health. Nothing else.
#
# Submit:
#   sbatch scripts/smoke-vllm-frontier.sh
#
# Or run interactively after `salloc`:
#   bash   scripts/smoke-vllm-frontier.sh
# ---------------------------------------------------------------------------
#SBATCH -A mat746
#SBATCH -J smoke-vllm
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/smoke-vllm-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/smoke-vllm-%j/job-%j.out
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug

set -uo pipefail

PROJ=/lustre/orion/mat746/proj-shared
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv
MODEL_DIR=$PROJ/models/Qwen2.5-72B-Instruct
RUN_DIR=$PROJ/runs/smoke-vllm-${SLURM_JOB_ID:-local}
mkdir -p "$RUN_DIR"

# ── conda + modules ─────────────────────────────────────────────────────────
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$PROJ/matsim-agents/scripts/frontier-module-stack.sh"
load_frontier_rocm_modules
source activate "$VENV"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/miopen-${SLURM_JOB_ID:-local}
mkdir -p "$MIOPEN_USER_DB_PATH"

# flashinfer/comm/cuda_ipc.py instantiates CudaRTLibrary() at module-import
# time (before torch loads in worker subprocs), so /proc/self/maps has neither
# libcudart nor libamdhip64 yet.  LD_PRELOAD forces libamdhip64 into the maps
# immediately on process start, satisfying both flashinfer and vLLM's check.
# VLLM_CUDART_SO_PATH is kept as a belt-and-suspenders fallback.
export LD_PRELOAD=/opt/rocm-7.1.1/lib/libamdhip64.so${LD_PRELOAD:+:${LD_PRELOAD}}
export VLLM_CUDART_SO_PATH=/opt/rocm-7.1.1/lib/libamdhip64.so

# Use the project-shared, prebuilt tvm-ffi torch<->DLPack ROCm bridge.
# Built once by install_matsim_frontier.sh; skips a ~5 min on-the-fly compile.
export TVM_FFI_CACHE_DIR=$PROJ/cache/tvm-ffi

VLLM_PORT=8000
VLLM_LOG=$RUN_DIR/vllm-server.log

echo "[$(date)] === Diagnostics ==="
echo "Hostname:       $(hostname)"
echo "SLURM_JOB_ID:   ${SLURM_JOB_ID:-N/A}"
echo "ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-unset}"
echo "HIP_VISIBLE_DEVICES:  ${HIP_VISIBLE_DEVICES:-unset}"
echo "Python:         $(which python)  ($(python --version 2>&1))"
echo "torch:          $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())' 2>&1)"
echo "vLLM:           $(python -c 'import vllm; print(vllm.__version__)' 2>&1)"
echo "rocm-smi:"
rocm-smi --showproductname 2>&1 | head -20 || true
echo "================================="

echo ""
echo "[$(date)] Launching vLLM (TP=8) under srun ..."
echo "Log:           $VLLM_LOG"

# Single task, all 8 GCDs of the node visible to that task.
# vLLM spawns 8 worker subprocesses internally (TP=8), each binds 1 GCD.
srun -N1 -n1 -c56 --gpus-per-task=8 --gpu-bind=closest \
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

trap 'echo "[$(date)] EXIT trap: killing vLLM pid $VLLM_PID"; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null' EXIT

# Tail the log so we see real-time output in the job stdout
( tail -f "$VLLM_LOG" & echo $! > "$RUN_DIR/.tail.pid" )
TAIL_PID=$(cat "$RUN_DIR/.tail.pid")
trap 'kill $TAIL_PID 2>/dev/null; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null' EXIT

echo "[$(date)] Polling /health on port $VLLM_PORT ..."
READY=0
for i in $(seq 1 240); do      # 240 * 5s = 20 min
    # Print a heartbeat every 30s
    (( i % 6 == 0 )) && echo "[$(date)] still waiting... attempt $i / 240"
    if curl -fsS --max-time 3 "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[$(date)] vLLM /health OK after $((i*5))s"
        READY=1
        break
    fi
    # If vLLM died, bail out
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[$(date)] ERROR: vLLM process exited. Last 60 lines:"
        tail -60 "$VLLM_LOG"
        exit 2
    fi
    sleep 5
done

if [[ $READY -eq 0 ]]; then
    echo "[$(date)] ERROR: vLLM did not become healthy in 20 min."
    echo "----- last 80 lines of $VLLM_LOG -----"
    tail -80 "$VLLM_LOG"
    exit 1
fi

echo ""
echo "[$(date)] === Smoke success: vLLM is up. Sending one tiny completion ==="
curl -sS -X POST "http://localhost:${VLLM_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen2.5-72B-Instruct","messages":[{"role":"user","content":"Reply with the single word: OK"}],"max_tokens":4,"temperature":0}' \
    | tee "$RUN_DIR/completion.json"

echo ""
echo "[$(date)] DONE."
