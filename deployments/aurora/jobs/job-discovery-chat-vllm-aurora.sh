#!/bin/bash
#PBS -N discovery-chat-vllm
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# matsim-agents: end-to-end discovery run on ALCF Aurora using vLLM-XPU.
#
# Aurora analog of deployments/frontier/jobs/job-discovery-chat-vllm-frontier.sh
#
# Layout (single node):
#   • vLLM server   : TP=2 (2 PVC tiles), Mistral-Small-24B via aurora_vllm_entrypoint.py
#   • matsim-agents : CPU-side chat client (--llm-provider vllm → localhost:8000)
#
# Aurora geometry used here: 1 node, 2 tiles (matches the smoke test).
# For larger models bump to select=2 and set CHAT_TP_SIZE=24.
#
# Prerequisites:
#   - vLLM XPU stack (frameworks/2025.3.1, no source build needed).
#     Verify: bash deployments/aurora/setup/install-vllm-xpu-aurora.sh
#   - HydraGNN BEST6 fp64 checkpoint:
#       $PROJ/HydraGNN/examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64/
#   - MLP branch weights:
#       $PROJ/HydraGNN/examples/multidataset_hpo_sc26/mlp_branch_weights.pt
#   - Local model weights:
#       $PROJ/models/Mistral-Small-24B-Instruct-2501/
#
# Submit:
#   qsub deployments/aurora/jobs/job-discovery-chat-vllm-aurora.sh
#
# Override model at submission:
#   MATSIM_MODEL_DIR=$PROJ/models/Qwen2.5-72B-Instruct \
#   qsub -l select=2 deployments/aurora/jobs/job-discovery-chat-vllm-aurora.sh
#
# See docs/vllm-aurora.md for full bring-up notes and known challenges.
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

# ── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"

VENV_PATH="${VENV_PATH:-${REPO}/.venv}"
HYDRAGNN_EXAMPLE="${PROJ}/HydraGNN/examples/multidataset_hpo_sc26"
LOGDIR="${MATSIM_HYDRAGNN_LOGDIR:-${CHAT_HYDRAGNN_LOGDIR:-${HYDRAGNN_EXAMPLE}/multidataset_hpo-BEST6-fp64}}"
HYDRAGNN_BRANCH_MLP_CHECKPOINT="${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-${CHAT_HYDRAGNN_BRANCH_MLP_CHECKPOINT:-${HYDRAGNN_EXAMPLE}/mlp_branch_weights.pt}}"

MODEL_DIR="${MATSIM_MODEL_DIR:-${CHAT_MODEL_PATH:-${PROJ}/models/Mistral-Small-24B-Instruct-2501}}"
MODEL_NAME="${MATSIM_MODEL_NAME:-${CHAT_MODEL_NAME:-$(basename "${MODEL_DIR}")}}"
VLLM_PORT="${MATSIM_VLLM_PORT:-${CHAT_PORT:-8000}}"
TP_SIZE="${MATSIM_VLLM_TP_SIZE:-${CHAT_TP_SIZE:-2}}"
VLLM_DTYPE="${MATSIM_VLLM_DTYPE:-${CHAT_DTYPE:-bfloat16}}"
VLLM_MAX_MODEL_LEN="${MATSIM_VLLM_MAX_MODEL_LEN:-${CHAT_MAX_MODEL_LEN:-32768}}"

JOBID="${PBS_JOBID:-local-$$}"
RUN_DIR="${PROJ}/runs/discovery-chat-vllm-aurora-${JOBID}"
OUTPUT_DIR="${RUN_DIR}/outputs"
mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}"

# ── modules + venv ──────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module reset
    module load frameworks
fi
# shellcheck disable=SC1091
[[ -f "${VENV_PATH}/bin/activate" ]] && source "${VENV_PATH}/bin/activate"

export PYTHONPATH="${HYDRAGNN_EXAMPLE}:${PROJ}/HydraGNN:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
# PBS sets TMPDIR to /var/tmp/pbs.<jobid>.<full-hostname>/<uuid>/ which is
# longer than the 107-char Unix socket limit.  vLLM uses TMPDIR for ZMQ IPC
# sockets (EngineCore ↔ APIServer), so we override it before vLLM starts.
export TMPDIR=/tmp

# executor_node reads these env vars as fallback when config injection is
# unavailable (e.g. across LangGraph checkpoint boundaries).
export MATSIM_HYDRAGNN_LOGDIR="${LOGDIR}"
export HYDRAGNN_BRANCH_MLP_CHECKPOINT="${HYDRAGNN_BRANCH_MLP_CHECKPOINT}"

# Compute nodes have no outbound internet
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# vLLM / Ray telemetry off
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export RAY_USAGE_STATS_ENABLED=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*'
export NO_PROXY='*'

# Aurora oneCCL / fabric tunings (keep ONEAPI_DEVICE_SELECTOR from module).
# NOTE: do NOT set CCL_KVS_MODE=mpi or CCL_PROCESS_LAUNCHER=pmix here. Those
# are valid only when ranks are MPI-launched (HydraGNN training pattern).
# vLLM's multiproc_executor fork()s its TP workers — they are NOT MPI ranks,
# so oneCCL must use its default internal-KVS over TCP. Setting MPI/PMIx
# modes triggers: "kvs_set_value: condition can_use_internal_kvs() failed"
# and "WorkerProc initialization failed" (first seen in job 8508267).
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_RX_MATCH_MODE=hybrid
export TORCH_DISTRIBUTED_USE_TORCHCOMMS=1

# ── diagnostics ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Aurora] matsim-agents discovery-chat (vLLM-XPU)"
echo "Date:          $(date)"
echo "Job ID:        ${JOBID}"
echo "Host:          $(hostname)"
echo "Run dir:       ${RUN_DIR}"
echo "Repo:          ${REPO}"
echo "Venv:          ${VENV_PATH}"
echo "LLM model:     ${MODEL_NAME}  (${MODEL_DIR})"
echo "TP size:       ${TP_SIZE}"
echo "HydraGNN log:  ${LOGDIR}"
echo "MLP ckpt:      ${HYDRAGNN_BRANCH_MLP_CHECKPOINT}"
echo "=========================================="

python - <<'PY'
import torch
xpu = getattr(torch, "xpu", None)
print(f"[torch] {torch.__version__}  xpu_available={xpu is not None and xpu.is_available()}")
try:
    import hydragnn  # noqa
    import matsim_agents  # noqa
    print("[imports] hydragnn + matsim_agents OK")
except Exception as e:
    print(f"[imports] FAILED: {e}")
    raise
PY

# ── start vLLM server ────────────────────────────────────────────────────────
VLLM_LOG="${RUN_DIR}/vllm-server.log"
echo "[$(date)] Starting vLLM server (TP=${TP_SIZE}, model=${MODEL_NAME}) ..."

# aurora_vllm_entrypoint.py patches _run_in_subprocess to avoid SIGSEGV on
# nodes where the registry subprocess (plain fork+exec) triggers Level Zero init
# without PALS permissions. mpiexec gives us PALS; unset PMI/PALS rank vars
# so vLLM does not interpret itself as an MPI job rank.
mpiexec -n 1 --ppn 1 \
    env -u PMI_RANK -u PMI_SIZE -u PMI_FD \
        -u PALS_APID -u PALS_RANKID -u PALS_NODEID -u PALS_SPOOL_DIR \
        -u MPI_LOCALRANKID -u MPI_LOCALNRANKS \
        -u OMPI_COMM_WORLD_RANK -u OMPI_COMM_WORLD_SIZE \
    python "${REPO}/deployments/aurora/smoke-tests/aurora_vllm_entrypoint.py" \
        --model "${MODEL_DIR}" \
        --served-model-name "${MODEL_NAME}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --dtype "${VLLM_DTYPE}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --port "${VLLM_PORT}" \
        --host 0.0.0.0 \
        --trust-remote-code \
        --no-enable-log-requests \
        --enforce-eager \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo "[$(date)] vLLM PID: ${VLLM_PID}"

# Kill vLLM on exit (normal or error)
# shellcheck disable=SC2064
trap "echo '[cleanup] Stopping vLLM (PID ${VLLM_PID}) ...'; kill ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null; echo '[cleanup] Done.'" EXIT

# ── wait for vLLM to be ready ────────────────────────────────────────────────
echo "[$(date)] Waiting for vLLM on http://localhost:${VLLM_PORT} (up to 10 min) ..."
MAX_WAIT=600
ELAPSED=0
INTERVAL=10
while true; do
    if curl -fsS --max-time 3 "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[$(date)] vLLM server ready after ${ELAPSED}s."
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[$(date)] ERROR: vLLM process died. Last 40 lines of ${VLLM_LOG}:" >&2
        tail -40 "${VLLM_LOG}" >&2
        exit 1
    fi
    if (( ELAPSED >= MAX_WAIT )); then
        echo "[$(date)] ERROR: vLLM did not start within ${MAX_WAIT}s." >&2
        tail -40 "${VLLM_LOG}" >&2
        exit 1
    fi
    sleep ${INTERVAL}
    (( ELAPSED += INTERVAL ))
done

export MATSIM_VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"

# ── multi-turn discovery dialogue ────────────────────────────────────────────
# Three user turns + 'exit'. This keeps the script generic while still
# exercising discovery, result-grounded comparison, and direct formula input.
# Empty lines are silently skipped by the REPL; 'exit' closes it cleanly.
echo "[$(date)] Submitting multi-turn discovery dialogue to matsim-agents (vLLM) ..."
matsim-agents chat \
    --logdir          "${LOGDIR}" \
    --hydragnn-branch-mlp-checkpoint "${HYDRAGNN_BRANCH_MLP_CHECKPOINT}" \
    --output-dir      "${OUTPUT_DIR}" \
    --llm-provider    vllm \
    --llm-model       "${MODEL_NAME}" \
    --llm-base-url    "http://localhost:${VLLM_PORT}/v1" \
    --ase-structure-optimizer FIRE \
    --maxiter         500 \
    --fmax            0.02 \
    --n-random        10 \
    --random-seed     42 \
    --auto-confirm \
    <<'CHAT_INPUT' 2>&1 | tee "${RUN_DIR}/matsim-agents.log"
Propose exactly TWO candidate materials in a single concise reply: (1) one oxide composition and (2) one intermetallic composition. For each, give the formula on its own line followed by a one-sentence physical justification. Keep the reply under 200 words.
The atomistic exploration results for the two compositions above were just appended to your context as [discovery] system messages. Using only those reported energies and |F|max values, answer: (a) which composition reached the lower minimum |F|max, suggesting better dynamical stability under the MLFF? (b) which composition has the lower total energy per atom? Be quantitative and quote the numbers.
Good. Now also evaluate MgO as a simple reference oxide. Briefly explain in two sentences why MgO is a useful baseline.
exit
CHAT_INPUT

echo "[$(date)] Discovery-chat finished. Artifacts in ${OUTPUT_DIR}"
