#!/bin/bash
#PBS -N discovery-chat-perovskites
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# matsim-agents: Aurora discovery-chat that screens FIVE perovskites in a
# single multi-turn conversation against the HydraGNN BEST6 MLFF.
#
# The five compositions (all integer stoichiometries → regex-safe):
#   1) SrTiO3          cubic Pm-3m reference
#   2) BaTiO3          ferroelectric, tetragonal P4mm at RT
#   3) CaTiO3          orthorhombic Pbnm, mineral perovskite parent
#   4) KNbO3           lead-free ferroelectric, multiple polar phases
#   5) Cs2AgBiBr6      double perovskite (A2BB'X6), lead-free PV candidate
#
# All formulas are written explicitly in the USER turns so the
# extract_compositions() regex picks them up deterministically (no
# reliance on the LLM to echo them back exactly). The final turn asks
# the LLM to rank the five by E/atom and |F|max from the [discovery]
# feedback messages.
#
# Submit:  qsub deployments/aurora/jobs/job-discovery-chat-perovskites-aurora.sh
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
RUN_DIR="${PROJ}/runs/discovery-chat-perovskites-aurora-${JOBID}"
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
echo "[Aurora] matsim-agents discovery-chat — PEROVSKITES"
echo "Date:          $(date)"
echo "Job ID:        ${JOBID}"
echo "Run dir:       ${RUN_DIR}"
echo "LLM model:     ${CHAT_MODEL_NAME}  (TP=${CHAT_TP_SIZE}, bf16)"
echo "Compositions:  SrTiO3, BaTiO3, CaTiO3, KNbO3, Cs2AgBiBr6"
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

# ── multi-turn perovskite dialogue ───────────────────────────────────────────
# Each of the first five turns explicitly mentions ONE perovskite formula in
# the USER text (regex-guaranteed) and asks the LLM for one sentence of
# physical reasoning. The composition extractor fires immediately; the MLFF
# relaxation runs; the [discovery] feedback is appended for the next turn.
# Turn 6 asks the LLM to rank the five from the cumulative feedback.
echo "[$(date)] Submitting 5-perovskite dialogue to matsim-agents ..."
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
    --n-random        8 \
    --random-seed     42 \
    --auto-confirm \
    <<'CHAT_INPUT' 2>&1 | tee "${RUN_DIR}/matsim-agents.log"
Let's screen a family of perovskite oxides and halides for thermodynamic stability under our MLFF. We start with SrTiO3, the canonical cubic-perovskite reference (Pm-3m, a~3.905 A). In two sentences, explain why SrTiO3 is the right starting point for benchmarking an MLFF on perovskite chemistry.
Next we consider BaTiO3, the prototypical ferroelectric perovskite that undergoes cubic-to-tetragonal-to-orthorhombic-to-rhombohedral transitions on cooling. In two sentences, summarize what makes BaTiO3 a more demanding test than SrTiO3 for an MLFF.
Now we evaluate CaTiO3, the mineral perovskite (orthorhombic Pbnm) for which the perovskite family is named. In two sentences, explain how the smaller Ca cation drives octahedral tilting and why that is a good MLFF stress test.
Next, KNbO3: a lead-free ferroelectric isostructural to BaTiO3 but with a much larger spontaneous polarization. In two sentences, explain why KNbO3 is interesting as a Pb-free piezoelectric and what makes its MLFF description harder than BaTiO3.
Finally, the double perovskite Cs2AgBiBr6 (A2BB'X6 ordered rock-salt sublattice on the B site), a lead-free photovoltaic absorber. In two sentences, explain why double perovskites are an attractive substitute for MAPbI3 and what new chemistry the Ag+/Bi3+ pair introduces.
The five compositions above have all been relaxed under the HydraGNN MLFF and the per-composition stability reports were appended to your context as [discovery] messages. Using only those numbers, produce a single table with columns: Formula, predicted-GS source (prototype/random), predicted-GS space group, E/atom (eV), |F|max (eV/A), dynamically_stable_proxy, near_degenerate_within_10meV. Then in one paragraph rank the five from most to least confidently stable and justify the ranking quantitatively.
exit
CHAT_INPUT

echo "[$(date)] Perovskite discovery-chat finished. Artifacts in ${OUTPUT_DIR}"
