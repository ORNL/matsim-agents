#!/bin/bash
#PBS -N matsim-smoke-vllm-singlenode
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# smoke-vllm-singlenode-aurora.sh
#
# Single-node smoke test: verify vLLM + Intel PVC (XPU) works on Aurora
# (1 node, up to 12 tiles, multiprocessing backend) before attempting
# multi-node runs.
#
# Prerequisite:
#   - vLLM XPU venv already built via:
#       bash deployments/aurora/setup/install-vllm-xpu-aurora.sh
#
# Submit (default model = Mistral-Small-24B):
#   qsub deployments/aurora/smoke-tests/smoke-vllm-singlenode-aurora.sh
#
# Override model:
#   qsub -v SMOKE_MODEL_PATH=$PROJ/models/Qwen2.5-32B-Instruct \
#        deployments/aurora/smoke-tests/smoke-vllm-singlenode-aurora.sh
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"

# vLLM is provided by the `frameworks` module (vLLM 0.15 + PyTorch 2.10/XPU as
# of frameworks/2025.3.1). We then activate the matsim-owned .venv (built with
# --system-site-packages on top of that same Python 3.12) so HydraGNN +
# matsim-agents are importable alongside vLLM.
VENV_PATH="${VENV_PATH:-${REPO}/.venv}"

SMOKE_MODEL_PATH="${SMOKE_MODEL_PATH:-${PROJ}/models/Mistral-Small-24B-Instruct-2501}"
SMOKE_MODEL_NAME="${SMOKE_MODEL_NAME:-$(basename "$SMOKE_MODEL_PATH")}"
SMOKE_PORT="${SMOKE_PORT:-8000}"
SMOKE_DTYPE="${SMOKE_DTYPE:-bfloat16}"
SMOKE_MAX_MODEL_LEN="${SMOKE_MAX_MODEL_LEN:-4096}"
# PVC: 6 GPUs × 2 tiles = 12 tiles/node.  Default TP=2 keeps the smoke test
# small + fast; bump up for larger models.
TP_SIZE="${TP_SIZE:-2}"

JOBID="${PBS_JOBID:-local-$$}"
RUN_DIR="${PROJ}/runs/smoke-vllm-singlenode-${JOBID}"
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "Single-node vLLM-XPU smoke test (Aurora)"
echo "Date:    $(date)"
echo "Node:    $(hostname)"
echo "Model:   $SMOKE_MODEL_NAME"
echo "Path:    $SMOKE_MODEL_PATH"
echo "TP:      $TP_SIZE"
echo "dtype:   $SMOKE_DTYPE"
echo "Run dir: $RUN_DIR"
echo "=========================================="

# ── Environment ─────────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
  module reset
  module load frameworks
fi
if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
else
  echo "WARN: $VENV_PATH/bin/activate not found; using bare frameworks Python" >&2
fi


export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# Compute nodes have no outbound internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# vLLM telemetry off
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# Ray telemetry off
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_IMPORT_WARNING=1
# Kill any stray proxy env that the build/run might inherit
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*'
export NO_PROXY='*'

# Aurora oneCCL / fabric tunings.
# NOTE: do NOT set CCL_KVS_MODE=mpi or CCL_PROCESS_LAUNCHER=pmix here. Those
# are valid only when ranks are MPI-launched (HydraGNN training pattern).
# vLLM's multiproc_executor fork()s its TP workers — they are NOT MPI ranks,
# so oneCCL must use its default internal-KVS over TCP. Setting MPI/PMIx
# modes triggers: "internal kvs should be used with pmi kvs mode or ofi
# transport with pmi kvs mode and pmix launcher" and aborts the workers.
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_RX_MATCH_MODE=hybrid
export TORCH_DISTRIBUTED_USE_TORCHCOMMS=1

# Restrict to first $TP_SIZE PVC tiles via Level Zero affinity mask.
# Use FLAT hierarchy: each tile becomes its own root device, so
# torch.xpu.device_count() == #tiles, which is what vLLM TP requires.
# (In COMPOSITE, both tiles of one PVC count as ONE root device with
# 2 subdevices, so device_count()=1 and TP=2 fails with
# "device index out of range. It must be in [0, 1), but got 1".)
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
TILE_TABLE=(0 1 2 3 4 5 6 7 8 9 10 11)
MASK=""
for ((i=0; i<TP_SIZE; i++)); do
  [[ $i -gt 0 ]] && MASK+=","
  MASK+="${TILE_TABLE[$i]}"
done
export ZE_AFFINITY_MASK="$MASK"
# IMPORTANT: do NOT override ONEAPI_DEVICE_SELECTOR. The frameworks module sets
# it to "opencl:gpu;level_zero:gpu" which Triton-XPU + vLLM require. The legacy
# value "level_zero:gpu" disables Triton and breaks vLLM device discovery.

# Cache dirs in the run dir so a stale cache can never poison the smoke test
export VLLM_CACHE_ROOT="$RUN_DIR/vllm-cache"
export TRITON_CACHE_DIR="$RUN_DIR/triton-cache"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

# Override TMPDIR — PBS sets it to a long path under /var/tmp/pbs.<jobid>/<uuid>/
# which exceeds the 107-char Unix-socket limit and breaks vLLM's ZMQ IPC.
export TMPDIR=/tmp

# Prevent core dumps from filling Lustre (~38 GB per rank per crash).
# If a crash is suspected, comment this out and inspect with: gdb python core.<pid>
ulimit -c 0

# Enable Python's faulthandler so SIGSEGV prints a Python + C stack trace
# into the job log instead of just "Segmentation fault (core dumped)".
export PYTHONFAULTHANDLER=1

# ── Cleanup trap ─────────────────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
  echo "[cleanup] Stopping vLLM ..."
  [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Diagnostics ──────────────────────────────────────────────────────────────
# All XPU-touching probes go through mpiexec — bare python on Aurora doesn't
# see the PVCs.
echo "Python:  $(which python)  ($(python --version 2>&1))"
echo "torch:   $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "ipex:    $(python -c 'import intel_extension_for_pytorch as ipex; print(ipex.__version__)' 2>&1)"
echo "vllm:    $(python -c 'import vllm; print(vllm.__version__)' 2>&1)"
echo -n "xpu:     "
mpiexec -n 1 --ppn 1 python -c 'import torch; print(torch.xpu.is_available(), torch.xpu.device_count())' 2>&1 | tail -1
echo "ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"
echo "ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-<unset>}"
echo "ONEAPI_DEVICE_SELECTOR=$ONEAPI_DEVICE_SELECTOR"
echo ""

# ── Pre-flight: test the subprocess that vLLM will spawn for model inspection ─
# vLLM's registry._run_in_subprocess() launches:
#   python -m vllm.model_executor.models.registry
# as a plain subprocess.run() child of the mpiexec'd server process.  That
# child inherits PMI_RANK / PMI_FD / PALS_* env vars set by mpiexec and, when
# IPEX triggers the oneCCL / deepspeed initialisation chain, oneCCL finds PMI
# env vars and attempts to join the MPI job as a real rank — which fails with
# SIGSEGV because the child is not a real MPI rank.
#
# Updated strategy: provide proper pickled stdin input to the subprocess so
# it actually executes fn() = _ModelInfo.from_model_cls(MistralForCausalLM).
# Use faulthandler to a file (not captured) to see the C stack on SIGSEGV.
# Test both with PMI vars and without to identify the trigger.
echo "[preflight] Testing vllm.model_executor.models.registry subprocess ..."

# Write the pickled (fn, output_file) that vLLM would send via stdin.
PREFLIGHT_PY="$RUN_DIR/preflight_input.py"
PREFLIGHT_IN="$RUN_DIR/preflight_input.bin"
cat > "$PREFLIGHT_PY" << 'PYEOF'
import sys, cloudpickle, tempfile, os
tmpdir = sys.argv[1]
output_file = os.path.join(tmpdir, "preflight_out.tmp")
def fn():
    from vllm.model_executor.models.registry import _ModelInfo
    import importlib
    mod = importlib.import_module("vllm.model_executor.models.mistral")
    cls = getattr(mod, "MistralForCausalLM")
    return _ModelInfo.from_model_cls(cls)
sys.stdout.buffer.write(cloudpickle.dumps((fn, output_file)))
PYEOF
python "$PREFLIGHT_PY" "$RUN_DIR" > "$PREFLIGHT_IN"
PREFLIGHT_IN_SIZE=$(wc -c < "$PREFLIGHT_IN")
echo "  pickled input: ${PREFLIGHT_IN_SIZE} bytes"

run_preflight() {
  local label="$1"; shift
  local out="$RUN_DIR/preflight_${label}"
  echo "  [$label] $*"
  # faulthandler to a file so it is not swallowed by capture_output
  local fh_file="$RUN_DIR/faulthandler_${label}.txt"
  "$@" bash -c "
    python -c 'import faulthandler,sys; faulthandler.enable(open(\"${fh_file}\",\"w\"))'
    PYTHONFAULTHANDLER=1 python -m vllm.model_executor.models.registry \
      < '${PREFLIGHT_IN}'
  " > "${out}.stdout" 2>"${out}.stderr" \
    && echo "  [$label] OK" || echo "  [$label] FAILED (exit $?)"
[[ -s "${out}.stderr" ]] && { echo "  --- stderr ---"; cat "${out}.stderr"; } || true
    [[ -s "$fh_file" ]]      && { echo "  --- faulthandler ---"; cat "$fh_file"; } || true
}

echo "  [pmi]   subprocess with PMI vars inherited from mpiexec ..."
run_preflight "pmi" mpiexec -n 1 --ppn 1

echo "  [nopmi] subprocess with PMI/PALS vars unset ..."
run_preflight "nopmi" mpiexec -n 1 --ppn 1 \
  env -u PMI_RANK -u PMI_SIZE -u PMI_FD \
      -u PALS_APID -u PALS_RANKID -u PALS_NODEID -u PALS_SPOOL_DIR \
      -u MPI_LOCALRANKID -u MPI_LOCALNRANKS \
      -u OMPI_COMM_WORLD_RANK -u OMPI_COMM_WORLD_SIZE

echo "  [noipex] subprocess with PMI unset + IPEX deepspeed disabled ..."
run_preflight "noipex" mpiexec -n 1 --ppn 1 \
  env -u PMI_RANK -u PMI_SIZE -u PMI_FD \
      -u PALS_APID -u PALS_RANKID -u PALS_NODEID -u PALS_SPOOL_DIR \
      CCL_PROCESS_LAUNCHER=none DS_SKIP_CUDA_CHECK=1

# [plain] mimics exactly what vLLM does: subprocess.run() without mpiexec.
# If this crashes (SIGSEGV), it confirms the root cause is that plain
# subprocess children lack PALS Level Zero device-fabric permissions.
echo "  [plain] plain subprocess.run() without mpiexec (mimics vLLM) ..."
PYTHONFAULTHANDLER=1 \
  python -m vllm.model_executor.models.registry \
  < "$PREFLIGHT_IN" \
  > "$RUN_DIR/preflight_plain.stdout" 2>"$RUN_DIR/preflight_plain.stderr" \
  && echo "  [plain] OK" || echo "  [plain] FAILED (exit $?)"
[[ -s "$RUN_DIR/preflight_plain.stderr" ]] && { echo "  --- stderr ---"; cat "$RUN_DIR/preflight_plain.stderr"; } || true
PLAIN_FH="$RUN_DIR/faulthandler_plain.txt"
[[ -s "$PLAIN_FH" ]] && { echo "  --- faulthandler ---"; cat "$PLAIN_FH"; } || true

# [cpudev] plain subprocess + ONEAPI_DEVICE_SELECTOR=opencl:cpu.
# The registry only inspects class attributes -- no XPU needed.
# opencl:cpu is the canonical backend:device_type form; avoids Level Zero.
# 'cpu' and 'cpu:*' are rejected by some Aurora node SYCL versions.
echo "  [cpudev] plain subprocess, ONEAPI_DEVICE_SELECTOR=opencl:cpu ..."
ONEAPI_DEVICE_SELECTOR=opencl:cpu PYTHONFAULTHANDLER=1 \
  python -m vllm.model_executor.models.registry \
  < "$PREFLIGHT_IN" \
  > "$RUN_DIR/preflight_cpudev.stdout" 2>"$RUN_DIR/preflight_cpudev.stderr" \
  && echo "  [cpudev] OK" || echo "  [cpudev] FAILED (exit $?)"
[[ -s "$RUN_DIR/preflight_cpudev.stderr" ]] && { echo "  --- stderr ---"; cat "$RUN_DIR/preflight_cpudev.stderr"; } || true
CPUDEV_FH="$RUN_DIR/faulthandler_cpudev.txt"
[[ -s "$CPUDEV_FH" ]] && { echo "  --- faulthandler ---"; cat "$CPUDEV_FH"; } || true

echo "[preflight] done."
echo ""

# ── Sanity check that the model dir exists ───────────────────────────────────
if [[ ! -d "$SMOKE_MODEL_PATH" ]]; then
  echo "ERROR: SMOKE_MODEL_PATH does not exist: $SMOKE_MODEL_PATH" >&2
  exit 2
fi

# ── Start vLLM server ────────────────────────────────────────────────────────
# IMPORTANT: launch via mpiexec (PALS) — bare python on Aurora compute nodes
# does NOT get PVC visibility (xpu.device_count() returns 0).  HydraGNN's
# Aurora scripts always wrap python in mpiexec for the same reason.
# vLLM's mp backend will then spawn its TP workers as Python multiprocessing
# children that inherit the GPU env from this rank-0 process.
#
# PMI/PALS vars (PMI_RANK, PMI_FD, PALS_APID, etc.) are unset for the server
# process itself.  The server is not an MPI rank — only mpiexec is used here
# to obtain PALS device-fabric permissions.
#
# aurora_vllm_entrypoint.py patches vLLM's _run_in_subprocess to set
# ONEAPI_DEVICE_SELECTOR=cpu for the model-registry subprocess.  That
# subprocess is a plain fork+exec child (not an mpiexec rank) and therefore
# lacks PALS Level Zero device-fabric permissions; trying to init the XPU
# there causes SIGSEGV.  The registry only inspects class attributes so cpu
# selector is sufficient and avoids Level Zero init entirely.
echo "[vllm] Starting server (TP=$TP_SIZE, device=xpu) via mpiexec ..."
mpiexec -n 1 --ppn 1 \
  env -u PMI_RANK -u PMI_SIZE -u PMI_FD \
      -u PALS_APID -u PALS_RANKID -u PALS_NODEID -u PALS_SPOOL_DIR \
      -u MPI_LOCALRANKID -u MPI_LOCALNRANKS \
      -u OMPI_COMM_WORLD_RANK -u OMPI_COMM_WORLD_SIZE \
  python "$REPO/deployments/aurora/smoke-tests/aurora_vllm_entrypoint.py" \
    --model "$SMOKE_MODEL_PATH" \
    --served-model-name "$SMOKE_MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype "$SMOKE_DTYPE" \
    --max-model-len "$SMOKE_MAX_MODEL_LEN" \
    --port "$SMOKE_PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --enforce-eager \
    --distributed-executor-backend mp \
  > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

echo "[vllm] PID=$VLLM_PID, log=$RUN_DIR/vllm.log"
tail -F "$RUN_DIR/vllm.log" &
TAIL_PID=$!

# ── Wait for /v1/models to respond ───────────────────────────────────────────
echo "[probe] Waiting for vLLM to come up on http://localhost:$SMOKE_PORT ..."
ready=0
for i in $(seq 1 90); do  # 90 × 10 s = 15 min
  if curl -sf "http://localhost:$SMOKE_PORT/v1/models" > "$RUN_DIR/models.json" 2>/dev/null; then
    ready=1
    echo "[probe] vLLM is up after ${i}0 s"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[probe] vLLM process died unexpectedly." >&2
    break
  fi
  sleep 10
done
kill "$TAIL_PID" 2>/dev/null || true

if [[ $ready -ne 1 ]]; then
  echo "[FAIL] vLLM did not become ready within 15 min" >&2
  echo "Last 80 lines of vllm.log:" >&2
  tail -80 "$RUN_DIR/vllm.log" >&2
  exit 1
fi

# ── Issue one chat completion ───────────────────────────────────────────────
echo "[probe] Sending one chat-completion request ..."
curl -sf -X POST "http://localhost:$SMOKE_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$SMOKE_MODEL_NAME\",
    \"messages\": [{\"role\":\"user\",\"content\":\"What is 2+2? Reply with only the number.\"}],
    \"max_tokens\": 16,
    \"temperature\": 0.0
  }" | tee "$RUN_DIR/response.json"
echo
echo "[OK] vLLM-XPU single-node smoke test PASSED."
echo "Run dir: $RUN_DIR"
