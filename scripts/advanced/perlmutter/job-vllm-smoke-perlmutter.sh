#!/bin/bash
#SBATCH -A amsc001
#SBATCH -J vllm-smoke
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:45:00
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
# ---------------------------------------------------------------------------
# Minimal single-GPU smoke test for the isolated vLLM server (vllm_venv).
#
# Uses the `shared` QOS so the scheduler can backfill a single GPU quickly,
# instead of waiting for a whole node. Validates the riskiest unknown:
# that `vllm serve` actually starts on a Perlmutter compute node, the
# OpenAI-compatible /v1 endpoint comes up, and a completion round-trips.
#
# Submit:
#   sbatch scripts/advanced/perlmutter/job-vllm-smoke-perlmutter.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-${REPO_DEFAULT}}"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/global/cfs/projectdirs/m5216/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
INSTALL_ROOT=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter
VLLM_VENV=$INSTALL_ROOT/vllm_venv
MODEL_DIR=${MATSIM_MODEL_DIR:-$PROJ/models/Qwen2.5-14B-Instruct}
MODEL_NAME=${MATSIM_MODEL_NAME:-$(basename "$MODEL_DIR")}

PORT=${MATSIM_VLLM_PORT:-8000}
BASE_URL="http://127.0.0.1:${PORT}/v1"
RUN_DIR=$RUNS_ROOT/vllm-smoke-${SLURM_JOB_ID:-$$}
mkdir -p "$RUN_DIR"
VLLM_LOG="$RUN_DIR/vllm-server.log"

echo "[$(date)] Host $(hostname); serving $MODEL_NAME on 1 GPU (vllm_venv) ..."

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_DO_NOT_TRACK=1
export PYTHONNOUSERSITE=1
# vLLM defaults to FlashInfer's top-k/top-p sampler, which JIT-compiles a kernel
# and takes a file lock (fcntl.flock) inside its cache dir. CFS/GPFS does not
# support flock -> OSError [Errno 524]. Use the native torch sampler (no JIT,
# no lock), which is plenty for a smoke test.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Any remaining JIT/compile caches must live on node-local tmpfs (flock-capable),
# never on CFS or $HOME (both GPFS).
_JIT_TMP="/tmp/vllm-jit.${USER}.${SLURM_JOB_ID:-$$}"
mkdir -p "$_JIT_TMP"
export FLASHINFER_WORKSPACE_BASE="$_JIT_TMP"
export TRITON_CACHE_DIR="$_JIT_TMP/triton"
export TORCHINDUCTOR_CACHE_DIR="$_JIT_TMP/inductor"
export VLLM_CACHE_ROOT="$_JIT_TMP/vllm"
# Triton/inductor may JIT-compile a CUDA stub needing Python.h, which is absent
# in the system-python vllm_venv; borrow the conda py3.11 headers from hydragnn_venv.
_PY_HDR="$INSTALL_ROOT/hydragnn_venv/include/python3.11"
export CPATH="${_PY_HDR}:${CPATH:-}"
export C_INCLUDE_PATH="${_PY_HDR}:${C_INCLUDE_PATH:-}"

"$VLLM_VENV/bin/vllm" serve "$MODEL_DIR" \
    --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len "${MATSIM_VLLM_MAXLEN:-8192}" \
    --gpu-memory-utilization "${MATSIM_VLLM_GPU_UTIL:-0.90}" \
    --enforce-eager >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'echo "[$(date)] stopping vLLM ($VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true' EXIT

echo "[$(date)] Waiting for $BASE_URL ..."
READY=0
for i in $(seq 1 240); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[ERROR] vLLM exited early. Tail:" >&2; tail -n 40 "$VLLM_LOG" >&2 || true; exit 1
  fi
  if curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; then
    READY=1; echo "[$(date)] endpoint ready after ~$((i*10))s."; break
  fi
  sleep 10
done
[[ "$READY" -ne 1 ]] && { echo "[ERROR] endpoint not ready. Tail:" >&2; tail -n 40 "$VLLM_LOG" >&2; exit 1; }

echo "[$(date)] /v1/models:"
curl -fsS "${BASE_URL}/models" | head -c 600; echo

echo "[$(date)] one completion round-trip:"
curl -fsS "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, name a refractory carbide and its crystal family.\"}],\"max_tokens\":64,\"temperature\":0}" \
  | tee "$RUN_DIR/completion.json" | head -c 1200; echo

echo "[$(date)] SMOKE TEST PASSED — vLLM serves on a compute node."
