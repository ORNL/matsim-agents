#!/bin/bash
#PBS -A CM2US
#PBS -N discovery-chat-vllm
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# matsim-agents: end-to-end RHEA discovery run on ALCF Aurora using vLLM-XPU.
#
# Aurora analog of scripts/advanced/frontier/job-discovery-chat-vllm-frontier.sh
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
#     Verify: bash scripts/setup/aurora/install-vllm-xpu-aurora.sh
#   - HydraGNN BEST6 fp64 checkpoint:
#       $PROJ/HydraGNN/examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64/
#   - MLP branch weights:
#       $PROJ/HydraGNN/examples/multidataset_hpo_sc26/mlp_branch_weights.pt
#   - Local model weights:
#       $PROJ/models/Mistral-Small-24B-Instruct-2501/
#
# Submit:
#   qsub scripts/advanced/aurora/job-discovery-chat-vllm-aurora.sh
#
# Override model at submission:
#   CHAT_MODEL_PATH=$PROJ/models/Qwen2.5-72B-Instruct \
#   qsub -l select=2 scripts/advanced/aurora/job-discovery-chat-vllm-aurora.sh
#
# See docs/vllm-aurora.md for full bring-up notes and known challenges.
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

# ── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=/lus/flare/projects/CM2US/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"

VENV_PATH="${VENV_PATH:-${PROJ}/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Aurora/hydragnn_venv}"
HYDRAGNN_EXAMPLE="${PROJ}/HydraGNN/examples/multidataset_hpo_sc26"
LOGDIR="${CHAT_HYDRAGNN_LOGDIR:-${HYDRAGNN_EXAMPLE}/multidataset_hpo-BEST6-fp64}"
MLP_CHECKPOINT="${CHAT_HYDRAGNN_MLP_CKPT:-${HYDRAGNN_EXAMPLE}/mlp_branch_weights.pt}"

CHAT_MODEL_PATH="${CHAT_MODEL_PATH:-${PROJ}/models/Mistral-Small-24B-Instruct-2501}"
CHAT_MODEL_NAME="${CHAT_MODEL_NAME:-$(basename "${CHAT_MODEL_PATH}")}"
CHAT_PORT="${CHAT_PORT:-8000}"
CHAT_TP_SIZE="${CHAT_TP_SIZE:-2}"
CHAT_DTYPE="${CHAT_DTYPE:-bfloat16}"
CHAT_MAX_MODEL_LEN="${CHAT_MAX_MODEL_LEN:-32768}"

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
export MATSIM_HYDRAGNN_MLP_CKPT="${MLP_CHECKPOINT}"

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
echo "LLM model:     ${CHAT_MODEL_NAME}  (${CHAT_MODEL_PATH})"
echo "TP size:       ${CHAT_TP_SIZE}"
echo "HydraGNN log:  ${LOGDIR}"
echo "MLP ckpt:      ${MLP_CHECKPOINT}"
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
echo "[$(date)] Starting vLLM server (TP=${CHAT_TP_SIZE}, model=${CHAT_MODEL_NAME}) ..."

# aurora_vllm_entrypoint.py patches _run_in_subprocess to avoid SIGSEGV on
# nodes where the registry subprocess (plain fork+exec) triggers Level Zero init
# without PALS permissions. mpiexec gives us PALS; unset PMI/PALS rank vars
# so vLLM does not interpret itself as an MPI job rank.
mpiexec -n 1 --ppn 1 \
    env -u PMI_RANK -u PMI_SIZE -u PMI_FD \
        -u PALS_APID -u PALS_RANKID -u PALS_NODEID -u PALS_SPOOL_DIR \
        -u MPI_LOCALRANKID -u MPI_LOCALNRANKS \
        -u OMPI_COMM_WORLD_RANK -u OMPI_COMM_WORLD_SIZE \
    python "${REPO}/scripts/smoke-tests/aurora/aurora_vllm_entrypoint.py" \
        --model "${CHAT_MODEL_PATH}" \
        --served-model-name "${CHAT_MODEL_NAME}" \
        --tensor-parallel-size "${CHAT_TP_SIZE}" \
        --dtype "${CHAT_DTYPE}" \
        --max-model-len "${CHAT_MAX_MODEL_LEN}" \
        --port "${CHAT_PORT}" \
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
echo "[$(date)] Waiting for vLLM on http://localhost:${CHAT_PORT} (up to 10 min) ..."
MAX_WAIT=600
ELAPSED=0
INTERVAL=10
while true; do
    if curl -fsS --max-time 3 "http://localhost:${CHAT_PORT}/health" > /dev/null 2>&1; then
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

export MATSIM_VLLM_BASE_URL="http://localhost:${CHAT_PORT}/v1"

# ── multi-turn discovery dialogue ────────────────────────────────────────────
# Three user turns + 'exit'. Each turn exercises a different chat feature:
#
#   Turn 1  →  multi-formula extraction from an *assistant* reply
#              (one Li-Mn-O cathode  +  one refractory HEA)
#              exercises:  AFLOW-prototype decoration  (oxide)
#                          pyXtal random search        (HEA, no AFLOW match)
#                          --auto-confirm path, novelty flagging
#   Turn 2  →  asks the LLM to reason over the [discovery] SystemMessages
#              that were injected after Turn 1's relaxations.
#              exercises:  feedback loop, multi-round chat history
#   Turn 3  →  the user *themself* writes a formula (BaTiO3) in the prompt.
#              exercises:  composition extraction from *user* text
#                          AFLOW prototype path (perovskite, many SGs hit)
#
# Empty lines are silently skipped by the REPL; 'exit' closes it cleanly.
echo "[$(date)] Submitting multi-turn discovery dialogue to matsim-agents (vLLM) ..."
matsim-agents chat \
    --logdir          "${LOGDIR}" \
    --mlp-checkpoint  "${MLP_CHECKPOINT}" \
    --output-dir      "${OUTPUT_DIR}" \
    --llm-provider    vllm \
    --llm-model       "${CHAT_MODEL_NAME}" \
    --llm-base-url    "http://localhost:${CHAT_PORT}/v1" \
    --ase-structure-optimizer FIRE \
    --maxiter         500 \
    --fmax            0.02 \
    --n-random        10 \
    --random-seed     42 \
    --auto-confirm \
    <<'CHAT_INPUT' 2>&1 | tee "${RUN_DIR}/matsim-agents.log"
Propose exactly TWO candidate materials in a single concise reply: (1) one Li-Mn-O layered cathode written as a clean formula such as Li2MnO3, and (2) one body-centered refractory high-entropy alloy drawn from {Mo, Nb, Ta, W, V, Cr}. For each, give the formula on its own line followed by a one-sentence physical justification (oxidation states, ionic radii, or Hume-Rothery rules). Keep the reply under 200 words.
The atomistic exploration results for the two compositions above were just appended to your context as [discovery] system messages. Using only those reported energies and |F|max values, answer: (a) which composition reached the lower minimum |F|max, suggesting better dynamical stability under the MLFF? (b) which composition has the lower total energy per atom? Be quantitative and quote the numbers.
Good. Now I would like to also evaluate the canonical perovskite BaTiO3 as a reference oxide. Please briefly explain why BaTiO3 is a useful baseline, in two sentences.
exit
CHAT_INPUT

echo "[$(date)] Discovery-chat finished. Artifacts in ${OUTPUT_DIR}"
