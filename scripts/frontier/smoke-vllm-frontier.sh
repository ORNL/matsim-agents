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
# Run J: switched to ROCm 7.2 venv (ROCm 7.1 venv was $PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv)
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
MODEL_DIR=${MATSIM_MODEL_DIR:-$PROJ/models/Qwen2.5-72B-Instruct}
MODEL_NAME=${MATSIM_MODEL_NAME:-Qwen/Qwen2.5-72B-Instruct}
RUN_DIR=$PROJ/runs/smoke-vllm-${SLURM_JOB_ID:-local}
mkdir -p "$RUN_DIR"

# ── conda + modules ─────────────────────────────────────────────────────────
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$PROJ/matsim-agents/scripts/frontier/frontier-module-stack.sh"
# Run J: full ROCm 7.2 environment (previously load_frontier_rocm711_modules)
load_frontier_rocm72_modules
source activate "$VENV"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/miopen-${SLURM_JOB_ID:-local}
mkdir -p "$MIOPEN_USER_DB_PATH"

# LD_PRELOAD of libamdhip64.so causes hipErrorInvalidKernelFile on Frontier
# because it interferes with HIP fat binary extraction in RCCL worker processes.
# Instead, point VLLM_CUDART_SO_PATH so flashinfer can find the HIP runtime.
# Run J: updated to ROCm 7.2 (was /opt/rocm-7.1.1/lib/libamdhip64.so)
export VLLM_CUDART_SO_PATH=/opt/rocm-7.2.0/lib/libamdhip64.so

# ── RCCL crash history: HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION in ncclDevKernel_Generic_4 ──
#
# All runs used TP=8 on a single Frontier node (8× MI250X GCDs, gfx90a),
# RCCL 2.27.7-HEAD:84d2752, vLLM with Qwen2.5-72B-Instruct (BF16).
#
# Run A (job 4509130) — hypothesis: wrong unroll variant selected at runtime
#   Tried:  RCCL_UNROLL_FACTOR=0
#   Result: FAILED — kernel name unchanged (_Generic_4 still appeared).
#   Why:    RCCL_UNROLL_FACTOR only affects runtime dispatch in RCCL builds
#           that support runtime switching. The PyTorch-bundled librccl fat
#           binary for gfx90a has the unroll-4 variant baked in at compile
#           time; the env var is silently ignored.
#
# Run B (job 4509146) — hypothesis: new P2P batching kernel path introduced in ROCm 7.1.0
#   Tried:  RCCL_P2P_BATCH_ENABLE=0, RCCL_P2P_BATCH_THRESHOLD=0
#   Result: FAILED — same HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION in _Generic_4.
#   Why:    The P2P batching path was not involved in the crash; disabling it
#           had no effect on the collective that crashed.
#
# Run C (job 4509167) — hypothesis: system librccl has a code-gen bug for gfx90a
#   Tried:  unset VLLM_NCCL_SO_PATH, prepend torch/lib so bundled librccl is used
#           (RCCL 2.27.7-HEAD:84d2752 from PyTorch wheel, same version as system)
#   Result: FAILED — bundled librccl crashed identically.
#   Why:    Both libraries are the same version and appear to share the same
#           gfx90a code object. The crash is not library-specific.
#
# Run D (job 4509240) — hypothesis: HSA scratch memory reclaimed mid-kernel
#   Tried:  HSA_NO_SCRATCH_RECLAIM=1
#   Result: PARTIAL — RCCL comm init now completed on all 8 ranks (previously
#           crashed during init). However, the first actual AllGather after init
#           (opCount 0, BF16, 4866048 elems ≈ 9.3 MB) still crashed with the
#           same _Generic_4 illegal instruction.
#   Why:    HSA_NO_SCRATCH_RECLAIM fixed a secondary masking issue (init crash),
#           revealing that the illegal instruction persists during the first real
#           collective. The root cause was not yet identified at this point.
#
# Run E (job 4510276) — hypothesis: NCCL_PROTO=LL crashes on large messages
#   Tried:  NCCL_PROTO=Simple, removed NCCL_ALGO=Ring pin
#   Result: FAILED — RCCL log confirmed "Algo=RING proto=SIMPLE", but the same
#           illegal instruction in ncclDevKernel_Generic_4 occurred immediately
#           after. Protocol choice is irrelevant; the crash is transport-level.
#   Why:    NCCL_P2P_DISABLE=1 + NCCL_SHM_DISABLE=1 was still set, which strips
#           RCCL of both xGMI (native MI250X inter-GCD interconnect) and shared
#           memory, forcing all intra-node collectives through NET/Socket. That
#           network kernel dispatch path is broken on Frontier gfx90a and causes
#           the illegal instruction regardless of protocol.
#
# Run F (job 4510596) — hypothesis: NET/Socket fallback (forced by P2P+SHM disable) is broken
#   Tried:  removed NCCL_P2P_DISABLE and NCCL_SHM_DISABLE entirely
#   Result: FAILED — RCCL switched to P2P/IPC (xGMI) transport as expected
#           (Channel 00: 0→1 via P2P/IPC), but same illegal instruction in
#           ncclDevKernel_Generic_4. Transport path is not the cause.
#   Why:    NCCL_PROTO=Simple always dispatches ncclDevKernel_Generic_*.
#           That kernel binary is broken on gfx90a in RCCL 2.27.7-HEAD:84d2752
#           regardless of which transport carries the data.
#
# Run G (job 4510778) — hypothesis: ncclDevKernel_Generic_4 is broken; LL128 uses different kernel
#   Tried:  NCCL_PROTO=LL128
#   Result: FAILED — different error: "no algorithm/protocol available for
#           function AllReduce with datatype ncclFloat32. NCCL_PROTO was set
#           to LL128." LL128 is not supported for fp32 AllReduce in this RCCL
#           build. No HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION this time.
#   Why:    Pinning any single NCCL_PROTO fails because vLLM issues multiple
#           collective types (AllGather and AllReduce) with different datatypes
#           (BF16 and FP32). No single pinned protocol satisfies all of them.
#           LL/Simple → Generic_4 crash on AllGather; LL128 → no valid combo
#           for fp32 AllReduce.
#
# Run H (job 4510825) — hypothesis: RCCL auto-select avoids the broken kernel path
#   Tried:  unset NCCL_PROTO (auto-select per collective + datatype)
#   Result: FAILED — RCCL correctly auto-selected LL for small AllReduce (passed),
#           then auto-selected RING+SIMPLE for 9.3MB AllGather → same crash in
#           ncclDevKernel_Generic_4. RCCL always picks Simple for messages above
#           the LL threshold (~8MB with 16 channels), and Simple always dispatches
#           Generic_4. There is no way to avoid Generic_4 for large AllGather
#           without using a different RCCL binary.
#   Why:    ncclDevKernel_Generic_4 in RCCL 2.27.7-HEAD:84d2752 has a broken
#           gfx90a code object. LL also dispatches Generic_4 for large messages,
#           and LL128 is not supported for all collective+dtype combos.
#           The only fix is a different RCCL build.
#
# Run I (job 4511391, CANCELLED before running)
#   Rationale: Test ROCm 7.2.0 RCCL via VLLM_NCCL_SO_PATH while keeping the
#              ROCm 7.1 venv. ROCm 7.2.0 RCCL is commit fc0010cf6a vs ROCm 7.1.1
#              commit 26aae437f6 — different build, may fix Generic_4 on gfx90a.
#   Result:    CANCELLED by user; migrated to full ROCm 7.2 environment instead.
#
# Run J: full ROCm 7.2 environment
#   Rationale: Rather than just swapping the RCCL .so, rebuild the entire venv
#              against ROCm 7.2 + torch+rocm7.2 + vLLM from source.
#              No pre-built vLLM pip wheels exist for ROCm 7.2 (only Docker images).
#              ROCm 7.2.0 RCCL (commit fc0010cf6a, build 7.2.0.0-43) differs from
#              ROCm 7.1.1 RCCL (commit 26aae437f6, build 7.1.1.0-38); may fix
#              ncclDevKernel_Generic_4 HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION.
# ──────────────────────────────────────────────────────────────────────────────

# Use PyTorch-bundled RCCL path for LD_LIBRARY_PATH (keeps torch HIP deps aligned).
# VLLM_NCCL_SO_PATH: point to ROCm 7.2.0 RCCL (loaded from the ROCm 7.2 venv's
# system path). Both the venv and RCCL are now ROCm 7.2 (consistent).
TORCH_LIB=$VENV/lib/python3.11/site-packages/torch/lib
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export VLLM_NCCL_SO_PATH=/opt/rocm-7.2.0/lib/librccl.so.1

# HSA scratch reclaim fix (established in Run D): prevents HSA from reclaiming
# kernel private scratch memory during RCCL collective init.
export HSA_NO_SCRATCH_RECLAIM=1

# Run J: full ROCm 7.2 environment; NCCL_PROTO auto-selected.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,TUNING
export RCCL_LOG_LEVEL=3

# Retained from Run A/B (harmless, kept for reproducibility):
export RCCL_UNROLL_FACTOR=0
export RCCL_P2P_BATCH_ENABLE=0
export RCCL_P2P_BATCH_THRESHOLD=0

# Use the project-shared, prebuilt tvm-ffi torch<->DLPack ROCm bridge.
# Built once by install_matsim_frontier.sh; skips a ~5 min on-the-fly compile.
export TVM_FFI_CACHE_DIR=$PROJ/cache/tvm-ffi

# Redirect torch.compile / vLLM compile cache to Lustre scratch.
# The home directory (/ccs/home) is NFS-mounted on compute nodes and can
# produce "Errno 116: Stale file handle" errors when multiple TP workers
# write temp files concurrently. Lustre handles this correctly.
export VLLM_CACHE_ROOT=$PROJ/cache/vllm-cache
export TRITON_CACHE_DIR=$PROJ/cache/vllm-cache/triton
# Wipe any stale compiled kernels before each run. Past runs with wrong
# HSA_OVERRIDE_GFX_VERSION=9.0.0 left gfx900 binaries in cache that cause
# HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION on the actual gfx90a hardware.
rm -rf "$VLLM_CACHE_ROOT"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

# Ensure torch.compile / triton JIT targets the correct GPU architecture.
# Without this, triton may auto-detect or compile for a wrong arch and
# cause HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION at runtime.
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_ARCH=gfx90a
# Do NOT set HSA_OVERRIDE_GFX_VERSION. Frontier's MI250X already reports
# gfx90a (major=9, minor=0, stepping=0xa=10). Setting it to 9.0.0 (gfx900)
# causes RCCL to look for a non-existent gfx900 code object → hipErrorInvalidKernelFile.

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
# Qwen3 models use a hybrid thinking mode; vLLM needs --reasoning-parser to
# strip <think>...</think> tokens and expose reasoning_content separately.
REASONING_ARGS=""
if [[ "$MODEL_NAME" == *"Qwen3"* ]]; then
    REASONING_ARGS="--reasoning-parser deepseek_r1"
fi

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
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: OK\"}],\"max_tokens\":4,\"temperature\":0}" \
    | tee "$RUN_DIR/completion.json"

echo ""
echo "[$(date)] DONE."
