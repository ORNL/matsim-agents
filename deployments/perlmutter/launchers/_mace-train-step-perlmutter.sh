#!/bin/bash
# ---------------------------------------------------------------------------
# _mace-train-step-perlmutter.sh
#
# Per-iteration MACE fine-tune step for the active-learning loop, invoked by
# matsim_agents.active_learning.trainer.retrain_mace via TrainerConfig.
# train_launcher. It activates the MACE venv (mace-torch + e3nn) and delegates
# to the finetune_mace CLI, which drives the reference `mace_run_train` recipe.
#
# Argument order is fixed by retrain_mace:
#   $1  train_script   (ignored; HydraGNN-specific, kept for launcher symmetry)
#   $2  dataset_path   (extxyz with accumulated DFT labels)
#   $3  out_model_dir  (writes mace_finetuned.model here)
#   $4  family         (mace_mp | mace_off | checkpoint)
#   $5  base_model     (small | medium | large | tag/URL | .model path)
#   $6  epochs
#   $7  nodes          (ignored; single-GPU fine-tune)
#   $8  ranks_per_node (ignored)
#
# Optional env: MACE_PRECISION (fp64), MACE_LORA=1 (+ MACE_LORA_RANK/ALPHA),
#   MACE_LR, MACE_WEIGHT_DECAY.
# ---------------------------------------------------------------------------
set -euo pipefail

TRAIN_SCRIPT="${1:-}"          # unused (HydraGNN symmetry)
DATASET="${2:?dataset path required}"
OUT_DIR="${3:?out model dir required}"
FAMILY="${4:-mace_mp}"
BASE_MODEL="${5:-medium}"
EPOCHS="${6:-}"
# $7 nodes, $8 ranks: single-GPU MACE fine-tune, ignored.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
VENV_ROOT="${REPO}/.hpc-build/perlmutter"
VENV="${MATSIM_MACE_VENV:-${VENV_ROOT}/mace_venv}"
[[ ! -d "${VENV}" ]] && { echo "ERROR: MACE venv not found: ${VENV}" >&2; exit 2; }

# shellcheck disable=SC1091
source "${REPO}/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
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
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

# MACE foundation weights + fine-tune outputs cache (shared project dir).
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJ}/models/mace_cache}"
export MACE_CACHE="${MACE_CACHE:-${XDG_CACHE_HOME}/mace}"
mkdir -p "${MACE_CACHE}" "${OUT_DIR}"

ARGS=(
  --dataset "${DATASET}"
  --output-dir "${OUT_DIR}"
  --family "${FAMILY}"
  --base-model "${BASE_MODEL}"
  --precision "${MACE_PRECISION:-fp64}"
)
[[ -n "${EPOCHS}" ]]                && ARGS+=(--epochs "${EPOCHS}")
[[ -n "${MACE_LR:-}" ]]            && ARGS+=(--lr "${MACE_LR}")
[[ -n "${MACE_WEIGHT_DECAY:-}" ]]  && ARGS+=(--weight-decay "${MACE_WEIGHT_DECAY}")
[[ "${MACE_LORA:-0}" == "1" ]]     && ARGS+=(--lora)
[[ -n "${MACE_LORA_RANK:-}" ]]     && ARGS+=(--lora-rank "${MACE_LORA_RANK}")
[[ -n "${MACE_LORA_ALPHA:-}" ]]    && ARGS+=(--lora-alpha "${MACE_LORA_ALPHA}")

echo "[mace-train-step] family=${FAMILY} model=${BASE_MODEL} epochs=${EPOCHS:-default} lora=${MACE_LORA:-0}"
echo "[mace-train-step] dataset=${DATASET}"
echo "[mace-train-step] out=${OUT_DIR}"

exec python -m matsim_agents.active_learning.finetune_mace "${ARGS[@]}"
