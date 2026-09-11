#!/bin/bash
#PBS -N discovery-chat-rhea
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# matsim-agents: Aurora discovery-chat that screens FOUR refractory
# high-entropy alloys (RHEAs) in a single multi-turn conversation
# against the HydraGNN BEST6 MLFF.
#
# The four compositions (equiatomic, integer stoichiometries → regex-safe):
#   1) MoNbTaW          Senkov 4-component BCC (1st RHEA, 2010)
#   2) MoNbTaVW         Senkov 5-component BCC (V-substituted)
#   3) HfNbTaTiZr       Senkov 5-component BCC (group-IV/V family)
#   4) CrMoNbTaW        Cr-substituted 5-component variant (smoke-test composition)
#
# All formulas appear explicitly in the USER turns. AFLOW prototypes
# generally do NOT match these equiatomic n-element signatures
# (1,1,1,1)/(1,1,1,1,1), so the seed pipeline falls through to the
# pyXtal random search across all 230 space groups. Every successful
# relaxation is therefore flagged with NOVELTY ALERT and reported as
# requiring DFT verification — exactly the intended HEA-discovery use
# case. The final turn asks the LLM to rank the four from the cumulative
# [discovery] feedback messages.
#
# Submit:  qsub deployments/aurora/jobs/job-discovery-chat-rhea-aurora.sh
# ---------------------------------------------------------------------------

set -eo pipefail

# ── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"

VENV_PATH="${VENV_PATH:-${REPO}/.venv}"
HYDRAGNN_EXAMPLE="${PROJ}/HydraGNN/examples/multidataset_hpo_sc26"
LOGDIR="${CHAT_HYDRAGNN_LOGDIR:-${HYDRAGNN_EXAMPLE}/multidataset_hpo-BEST6-fp64}"
HYDRAGNN_BRANCH_MLP_CHECKPOINT="${CHAT_HYDRAGNN_BRANCH_MLP_CHECKPOINT:-${HYDRAGNN_EXAMPLE}/mlp_branch_weights.pt}"

CHAT_MODEL_PATH="${CHAT_MODEL_PATH:-${PROJ}/models/Mistral-Small-24B-Instruct-2501}"
CHAT_MODEL_NAME="${CHAT_MODEL_NAME:-$(basename "${CHAT_MODEL_PATH}")}"
CHAT_PORT="${CHAT_PORT:-8000}"
CHAT_TP_SIZE="${CHAT_TP_SIZE:-2}"
CHAT_DTYPE="${CHAT_DTYPE:-bfloat16}"
CHAT_MAX_MODEL_LEN="${CHAT_MAX_MODEL_LEN:-32768}"

JOBID="${PBS_JOBID:-local-$$}"
RUN_DIR="${PROJ}/runs/discovery-chat-rhea-aurora-${JOBID}"
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
export TMPDIR=/tmp

export MATSIM_HYDRAGNN_LOGDIR="${LOGDIR}"
export HYDRAGNN_BRANCH_MLP_CHECKPOINT="${HYDRAGNN_BRANCH_MLP_CHECKPOINT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export RAY_USAGE_STATS_ENABLED=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*'
export NO_PROXY='*'

export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_RX_MATCH_MODE=hybrid
export TORCH_DISTRIBUTED_USE_TORCHCOMMS=1

# ── diagnostics ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Aurora] matsim-agents discovery-chat — RHEAs"
echo "Date:          $(date)"
echo "Job ID:        ${JOBID}"
echo "Run dir:       ${RUN_DIR}"
echo "LLM model:     ${CHAT_MODEL_NAME}  (TP=${CHAT_TP_SIZE}, bf16)"
echo "Compositions:  MoNbTaW, MoNbTaVW, HfNbTaTiZr, CrMoNbTaW"
echo "=========================================="

# ── start vLLM server ────────────────────────────────────────────────────────
VLLM_LOG="${RUN_DIR}/vllm-server.log"
echo "[$(date)] Starting vLLM server ..."

mpiexec -n 1 --ppn 1 \
    env -u PMI_RANK -u PMI_SIZE -u PMI_FD \
        -u PALS_APID -u PALS_RANKID -u PALS_NODEID -u PALS_SPOOL_DIR \
        -u MPI_LOCALRANKID -u MPI_LOCALNRANKS \
        -u OMPI_COMM_WORLD_RANK -u OMPI_COMM_WORLD_SIZE \
    python "${REPO}/deployments/aurora/smoke-tests/aurora_vllm_entrypoint.py" \
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

# shellcheck disable=SC2064
trap "echo '[cleanup] Stopping vLLM (PID ${VLLM_PID}) ...'; kill ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null; echo '[cleanup] Done.'" EXIT

# ── wait for vLLM ────────────────────────────────────────────────────────────
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
        echo "[$(date)] ERROR: vLLM process died." >&2
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

# ── multi-turn RHEA dialogue ────────────────────────────────────────────────
# Each turn states ONE equiatomic RHEA formula explicitly in the user
# text (regex-guaranteed: no fractional decimals, no element-name spelling,
# no Unicode subscripts). The composition extractor fires immediately;
# the seed pipeline produces 0 prototypes + 8 pyXtal random structures;
# the relaxed [discovery] summary is fed back for the next turn. Turn 5
# asks the LLM to rank the four from the cumulative feedback.
#
# n_random=8 keeps the per-composition cost predictable (~30 s at our
# eager-mode XPU throughput) so all four fit comfortably in the 1 h cap.
echo "[$(date)] Submitting 4-RHEA dialogue to matsim-agents ..."
matsim-agents chat \
    --logdir          "${LOGDIR}" \
    --hydragnn-branch-mlp-checkpoint "${HYDRAGNN_BRANCH_MLP_CHECKPOINT}" \
    --output-dir      "${OUTPUT_DIR}" \
    --llm-provider    vllm \
    --llm-model       "${CHAT_MODEL_NAME}" \
    --llm-base-url    "http://localhost:${CHAT_PORT}/v1" \
    --ase-structure-optimizer FIRE \
    --maxiter         500 \
    --fmax            1e-3 \
    --relative-increase-threshold 10.0 \
    --n-random        32 \
    --random-seed     42 \
    --auto-confirm \
    <<'CHAT_INPUT' 2>&1 | tee "${RUN_DIR}/matsim-agents.log"
We will screen four equiatomic refractory high-entropy alloys (RHEAs) for predicted dynamical stability under the HydraGNN MLFF. We begin with MoNbTaW, the original Senkov 4-component BCC RHEA reported in 2010. In two sentences, explain why MoNbTaW forms a stable single-phase BCC solid solution despite the four constituent metals having different crystal structures in their pure forms.
Next we evaluate MoNbTaVW, the 5-component variant obtained by adding V to MoNbTaW. In two sentences, explain how V (the smallest of the five) affects the lattice strain, configurational entropy, and high-temperature softening of the alloy.
Now we evaluate HfNbTaTiZr, the canonical Senkov refractory family member built from group-IV and group-V transition metals. In two sentences, explain why HfNbTaTiZr stays single-phase BCC despite the larger size mismatch between Hf/Zr and Ti, and what trade-offs that introduces vs MoNbTaW.
Finally we evaluate CrMoNbTaW, obtained by substituting V in MoNbTaVW with Cr. In two sentences, explain how Cr alters the electronic and chemical-stability picture (sigma-phase tendency, oxidation resistance) compared with MoNbTaVW.
The four RHEAs above have all been screened under the MLFF and the per-composition stability reports were appended to your context as [discovery] messages. Using only those numbers, produce a single table with columns: Formula, predicted-GS space group, E/atom (eV), |F|max (eV/A), dynamically_stable_proxy, near_degenerate_within_10meV. Then in one paragraph rank the four from most to least confidently stable, justify the ranking quantitatively, and explicitly note that every ground-state assignment is a NOVELTY ALERT requiring DFT verification (since no AFLOW prototype matched these signatures and only the pyXtal random search produced seeds).
exit
CHAT_INPUT

echo "[$(date)] RHEA discovery-chat finished. Artifacts in ${OUTPUT_DIR}"
