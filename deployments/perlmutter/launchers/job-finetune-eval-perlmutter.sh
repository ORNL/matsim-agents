#!/bin/bash
#SBATCH -J ft-eval
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out
#SBATCH -t 04:00:00
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: one endpoints FINE-TUNE + EVAL campaign for a single
# (backend, dataset) pair on a single Perlmutter GPU.
#
# Runs src/matsim_agents/active_learning/finetune_eval.py, which produces the
# two learning-curve endpoints the paper reports -- *before* (baseline
# foundation model) and *after* (fine-tuned on all AL-corrected data) -- on a
# fixed held-out test set, plus a cost.json with wall-time / GPU-hours.
#
# GPU visibility is left to Slurm (a `shared` allocation with `-G 1` exports
# CUDA_VISIBLE_DEVICES for the single bound GPU); the campaign auto-detects it
# via torch, matching the "let Slurm manage the GPU" policy used elsewhere.
#
# Submit the whole 2x5 matrix with the companion wrapper:
#   deployments/perlmutter/launchers/submit-finetune-eval-matrix-perlmutter.sh
# or a single campaign directly, e.g.:
#   BACKEND=hydragnn CASE=lifepo4-al-001 \
#     sbatch deployments/perlmutter/launchers/job-finetune-eval-perlmutter.sh
#   BACKEND=uma CASE=zn-formate-mof-uma-al-001 UMA_TASK=omol \
#     sbatch deployments/perlmutter/launchers/job-finetune-eval-perlmutter.sh
#   BACKEND=mace CASE=lifepo4-al-001 MACE_FAMILY=mace_mp MACE_MODEL=medium \
#     sbatch deployments/perlmutter/launchers/job-finetune-eval-perlmutter.sh
#
# Required env:
#   BACKEND   hydragnn | uma | mace
#   CASE      dataset case dir under $RUNS_ROOT (contains dataset.extxyz)
# Optional env (with defaults):
#   UMA_TASK      omat | omol | oc20 | odac | omc   (uma only; default omat)
#   UMA_LORA      1 = LoRA fine-tune of UMA backbone scalar linears (default 0)
#   MACE_FAMILY   mace_mp | mace_off | checkpoint   (mace only; default mace_mp)
#   MACE_MODEL    small | medium | large | tag/URL | .model path
#                                                   (mace only; default medium)
#   MACE_MODEL_ID curated MACE_MODELS id (overrides MACE_FAMILY/MACE_MODEL)
#   MACE_LORA     1 = native mace_run_train LoRA fine-tune (default 0)
#   MACE_PRECISION  fp32 | fp64                      (mace only; default fp64)
#   EPOCHS        fine-tune epochs                  (default 20)
#   BATCH_SIZE    training batch size               (default 4)
#   DEVICE        override torch device             (default: auto = cuda)
#   HYDRAGNN_STRATEGY  routed | unfrozen | frozen | scratch (hydragnn only;
#                      default routed = branch-MLP head reuse. The other three
#                      drop all 16 heads and grow one fresh head.)
#   HYDRAGNN_HEAD  fine-tune a PRETRAINED head instead of a random one: a branch
#                  index 0..15 or dataset name (e.g. MPTrj, Alexandria, OMat24).
#                  Requires HYDRAGNN_STRATEGY=unfrozen|frozen. (default: random)
#   FT_REPO       ORNL HydraGNN_GFM_FineTuning4Materials checkout (new-head only)
#   FTEVAL_ROOT   output root                       (default $RUNS_ROOT/finetune-eval)
#   PROJECT_ROOT / RUNS_ROOT / HYDRAGNN_ROOT / GFM_LOGDIR / BRANCH_MLP / FAIRCHEM_SRC
# ---------------------------------------------------------------------------
set -euo pipefail

# ── paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-${REPO_DEFAULT}}"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"

HYDRAGNN_ROOT="${HYDRAGNN_ROOT:-${PROJ}/HydraGNN}"
VENV_ROOT="${HYDRAGNN_ROOT}/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter"

# ── required inputs ─────────────────────────────────────────────────────────
BACKEND="${BACKEND:?set BACKEND=hydragnn|uma|mace}"
CASE="${CASE:?set CASE=<dataset case dir under RUNS_ROOT>}"
DATASET="${DATASET:-${RUNS_ROOT}/${CASE}/dataset.extxyz}"
[[ ! -f "${DATASET}" ]] && { echo "ERROR: dataset not found: ${DATASET}" >&2; exit 2; }

# ── knobs ────────────────────────────────────────────────────────────────────
UMA_TASK="${UMA_TASK:-omat}"
MACE_FAMILY="${MACE_FAMILY:-mace_mp}"
MACE_MODEL="${MACE_MODEL:-medium}"
MACE_PRECISION="${MACE_PRECISION:-fp64}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
HYDRAGNN_STRATEGY="${HYDRAGNN_STRATEGY:-routed}"
HYDRAGNN_HEAD="${HYDRAGNN_HEAD:-}"
FTEVAL_ROOT="${FTEVAL_ROOT:-${RUNS_ROOT}/finetune-eval}"
# Distinct runs (head strategy, MACE size/variant, LoRA/frozen) write to their
# own subdir via a composed variant tag so they never collide.
VARIANT_PARTS=()
if [[ "${BACKEND}" == "hydragnn" ]]; then
  [[ "${HYDRAGNN_STRATEGY}" != "routed" ]] && VARIANT_PARTS+=("${HYDRAGNN_STRATEGY}")
  [[ -n "${HYDRAGNN_HEAD}" ]] && VARIANT_PARTS+=("$(echo "${HYDRAGNN_HEAD}" | tr '/ ' '__')")
elif [[ "${BACKEND}" == "mace" ]]; then
  __mace_variant="${MACE_MODEL_ID:-${MACE_MODEL}}"
  [[ "${__mace_variant}" != "medium" ]] && VARIANT_PARTS+=("$(basename "${__mace_variant}")")
  [[ "${MACE_LORA:-0}" == "1" ]] && VARIANT_PARTS+=("lora")
  [[ "${MACE_FREEZE_BACKBONE:-0}" == "1" ]] && VARIANT_PARTS+=("frozen")
elif [[ "${BACKEND}" == "uma" ]]; then
  [[ "${UMA_LORA:-0}" == "1" ]] && VARIANT_PARTS+=("lora")
  [[ "${UMA_FREEZE_BACKBONE:-0}" == "1" ]] && VARIANT_PARTS+=("frozen")
fi
if [[ ${#VARIANT_PARTS[@]} -gt 0 ]]; then
  VARIANT_TAG="$(IFS=-; echo "${VARIANT_PARTS[*]}")"
  OUT_DIR="${FTEVAL_ROOT}/${BACKEND}-${VARIANT_TAG}/${CASE}"
else
  OUT_DIR="${FTEVAL_ROOT}/${BACKEND}/${CASE}"
fi
mkdir -p "${OUT_DIR}"

# HydraGNN GFM assets (only used for BACKEND=hydragnn)
GFM_LOGDIR="${GFM_LOGDIR:-${HYDRAGNN_ROOT}/examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64}"
BRANCH_MLP="${BRANCH_MLP:-${HYDRAGNN_ROOT}/examples/multidataset_hpo_sc26/mlp_branch_weights.pt}"
# ORNL fine-tuning utilities checkout (only used for new-head strategies).
FT_REPO="${FT_REPO:-${PROJ}/HydraGNN_GFM_FineTuning4Materials}"

# ── modules + venv ────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "${REPO}/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu

if [[ "${BACKEND}" == "uma" ]]; then
  VENV="${MATSIM_FAIRCHEM_VENV:-${VENV_ROOT}/hydragnn_venv}"
elif [[ "${BACKEND}" == "hydragnn" ]]; then
  VENV="${MATSIM_HYDRAGNN_VENV:-${VENV_ROOT}/hydragnn_venv}"
elif [[ "${BACKEND}" == "mace" ]]; then
  VENV="${MATSIM_MACE_VENV:-${VENV_ROOT}/mace_venv}"
else
  echo "ERROR: BACKEND must be 'hydragnn', 'uma', or 'mace' (got '${BACKEND}')" >&2
  exit 2
fi
[[ ! -d "${VENV}" ]] && { echo "ERROR: venv not found: ${VENV}" >&2; exit 2; }
# mace_venv is a plain Python venv; the unified hydragnn_venv is conda and may
# need activation through the conda hook when bin/activate is unavailable.
if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
else
  __conda_base="$(conda info --base 2>/dev/null)"
  [[ -z "${__conda_base}" ]] && { echo "ERROR: 'conda' not available to activate ${VENV}" >&2; exit 2; }
  # shellcheck disable=SC1091
  source "${__conda_base}/etc/profile.d/conda.sh"
  conda activate "${VENV}" || { echo "ERROR: conda activate failed for ${VENV}" >&2; exit 2; }
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# matsim_agents importable from the repo src tree in either venv.
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

# ── backend-specific env ──────────────────────────────────────────────────────
declare -a EXTRA_ARGS=()
if [[ "${BACKEND}" == "hydragnn" ]]; then
  # HydraGNN package + example dir (inference_fused / branch MLP) on the path.
  export PYTHONPATH="${HYDRAGNN_ROOT}:${PYTHONPATH}"
  [[ ! -d "${GFM_LOGDIR}" ]] && { echo "ERROR: GFM logdir not found: ${GFM_LOGDIR}" >&2; exit 2; }
  [[ ! -f "${BRANCH_MLP}" ]] && { echo "ERROR: branch MLP not found: ${BRANCH_MLP}" >&2; exit 2; }
  EXTRA_ARGS+=(--gfm-logdir "${GFM_LOGDIR}" --branch-mlp "${BRANCH_MLP}" --hydragnn-root "${HYDRAGNN_ROOT}")
  EXTRA_ARGS+=(--hydragnn-strategy "${HYDRAGNN_STRATEGY}")
  [[ -n "${HYDRAGNN_HEAD}" ]] && EXTRA_ARGS+=(--hydragnn-head "${HYDRAGNN_HEAD}")
  [[ -n "${HYDRAGNN_LR:-}" ]] && EXTRA_ARGS+=(--lr "${HYDRAGNN_LR}")
  if [[ "${HYDRAGNN_STRATEGY}" != "routed" ]]; then
    [[ ! -d "${FT_REPO}" ]] && { echo "ERROR: FT_REPO (ORNL fine-tune utils) not found: ${FT_REPO}" >&2; exit 2; }
    EXTRA_ARGS+=(--ft-repo "${FT_REPO}")
  fi
elif [[ "${BACKEND}" == "uma" ]]; then
  # UMA fine-tuning uses a self-contained custom PyTorch loop (no fairchem
  # config templates needed), plus a HuggingFace / fairchem model cache for the
  # base UMA weights.
  export HF_HOME="${HF_HOME:-${PROJ}/models/hf_cache}"
  mkdir -p "${HF_HOME}"
  if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
    HF_TOKEN="$(< "${HOME}/.cache/huggingface/token")"; export HF_TOKEN
  fi
  export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-${SCRATCH:-/tmp}/matsim-agents/fairchem_cache}"
  mkdir -p "${FAIRCHEM_CACHE_DIR}"
  EXTRA_ARGS+=(--uma-task-name "${UMA_TASK}")
  # Fine-tune recipe knobs (Adam lr=1e-4, force-weighted MSE loss; see
  # finetune_uma). Defaults live in the Python CLI; override via env.
  [[ -n "${UMA_EPOCHS:-}" ]]         && EXTRA_ARGS+=(--uma-epochs "${UMA_EPOCHS}")
  [[ -n "${UMA_LR:-}" ]]             && EXTRA_ARGS+=(--uma-lr "${UMA_LR}")
  [[ -n "${UMA_FORCE_WEIGHT:-}" ]]   && EXTRA_ARGS+=(--uma-force-weight "${UMA_FORCE_WEIGHT}")
  [[ -n "${UMA_WEIGHT_DECAY:-}" ]]   && EXTRA_ARGS+=(--uma-weight-decay "${UMA_WEIGHT_DECAY}")
  [[ "${UMA_FREEZE_BACKBONE:-0}" == "1" ]] && EXTRA_ARGS+=(--uma-freeze-backbone)
  [[ "${UMA_LORA:-0}" == "1" ]]     && EXTRA_ARGS+=(--uma-lora)
  [[ -n "${UMA_LORA_R:-}" ]]        && EXTRA_ARGS+=(--uma-lora-r "${UMA_LORA_R}")
  [[ -n "${UMA_LORA_ALPHA:-}" ]]    && EXTRA_ARGS+=(--uma-lora-alpha "${UMA_LORA_ALPHA}")
else
  # MACE fine-tuning is delegated to mace_run_train (reference recipe); base
  # foundation weights are fetched to an in-project MACE cache. All MACE
  # versions are selectable via MACE_MODEL_ID or MACE_FAMILY / MACE_MODEL, with
  # optional native LoRA (MACE_LORA=1).
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJ}/models/mace_cache}"
  export MACE_CACHE="${MACE_CACHE:-${XDG_CACHE_HOME}/mace}"
  mkdir -p "${MACE_CACHE}"
  if [[ -n "${MACE_MODEL_ID:-}" ]]; then
    EXTRA_ARGS+=(--mace-model-id "${MACE_MODEL_ID}")
  else
    EXTRA_ARGS+=(--mace-family "${MACE_FAMILY}" --mace-model "${MACE_MODEL}")
  fi
  EXTRA_ARGS+=(--mace-precision "${MACE_PRECISION}")
  [[ -n "${MACE_EPOCHS:-}" ]]               && EXTRA_ARGS+=(--mace-epochs "${MACE_EPOCHS}")
  [[ "${MACE_DISPERSION:-0}" == "1" ]]      && EXTRA_ARGS+=(--mace-dispersion)
  [[ -n "${MACE_LR:-}" ]]                   && EXTRA_ARGS+=(--mace-lr "${MACE_LR}")
  [[ -n "${MACE_FORCE_WEIGHT:-}" ]]         && EXTRA_ARGS+=(--mace-force-weight "${MACE_FORCE_WEIGHT}")
  [[ -n "${MACE_WEIGHT_DECAY:-}" ]]         && EXTRA_ARGS+=(--mace-weight-decay "${MACE_WEIGHT_DECAY}")
  [[ "${MACE_FREEZE_BACKBONE:-0}" == "1" ]] && EXTRA_ARGS+=(--mace-freeze-backbone)
  [[ "${MACE_LORA:-0}" == "1" ]]            && EXTRA_ARGS+=(--mace-lora)
  [[ -n "${MACE_LORA_RANK:-}" ]]            && EXTRA_ARGS+=(--mace-lora-rank "${MACE_LORA_RANK}")
  [[ -n "${MACE_LORA_ALPHA:-}" ]]           && EXTRA_ARGS+=(--mace-lora-alpha "${MACE_LORA_ALPHA}")
fi

# Optional explicit device override; default leaves detection to torch, which
# picks the Slurm-bound GPU.
[[ -n "${DEVICE:-}" ]] && EXTRA_ARGS+=(--device "${DEVICE}")

# Leakage-free RE-SCORE mode: skip training, reuse the existing split +
# fine-tuned checkpoint, and re-score both endpoints with the per-element energy
# reference fit on the TRAIN partition (writes eval/iter*_trainref.json).
[[ "${RESCORE:-0}" == "1" ]] && EXTRA_ARGS+=(--eval-only)

echo "=========================================="
echo "[Perlmutter FT+eval campaign]"
echo "Date:       $(date)"
echo "Job ID:     ${SLURM_JOB_ID:-N/A}"
echo "Backend:    ${BACKEND}"
echo "Case:       ${CASE}"
echo "Dataset:    ${DATASET}"
echo "Strategy:   ${HYDRAGNN_STRATEGY}   (hydragnn only)"
echo "Head:       ${HYDRAGNN_HEAD:-<random>}   (hydragnn new-head only)"
echo "Task (UMA): ${UMA_TASK}   (uma LoRA: ${UMA_LORA:-0})"
echo "MACE:       ${MACE_MODEL_ID:-${MACE_FAMILY}:${MACE_MODEL}} (${MACE_PRECISION})  (LoRA: ${MACE_LORA:-0})   (mace only)"
echo "Epochs:     ${EPOCHS}   Batch: ${BATCH_SIZE}"
echo "Out dir:    ${OUT_DIR}"
echo "Venv:       ${VENV}"
echo "Python:     $(which python)"
echo "=========================================="

cd "${REPO}"
python -m matsim_agents.active_learning.finetune_eval \
  --backend "${BACKEND}" \
  --dataset "${DATASET}" \
  --output-dir "${OUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${OUT_DIR}/campaign.log"

echo "[$(date)] Campaign complete -> ${OUT_DIR}"
