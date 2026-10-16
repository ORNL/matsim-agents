#!/bin/bash
# =============================================================================
# _hydragnn-train-step-aurora.sh
#
# Inner-step launcher for one matsim-agents HydraGNN fine-tuning round on
# Aurora. The AL driver has already loaded the frameworks module stack and
# activated the HydraGNN environment; this step runs within the same PBS
# allocation on one XPU tile.
#
# Arguments (same contract as the Frontier and Perlmutter launchers):
#   <train_script> <dataset> <out_logdir> <resume_logdir> <epochs>
#   <nodes> <ranks_per_node> <elemental_manifest> [checkpoint] [branch_mlp]
#   [ft_repo] [charge] [spin] [hydragnn_root]
# =============================================================================

set -euo pipefail

if [[ $# -lt 8 ]]; then
  echo "Usage: $0 <train_script> <dataset> <out_logdir> <resume_logdir> <epochs> <nodes> <ranks_per_node> <elemental_manifest>" >&2
  exit 2
fi

TRAIN_SCRIPT="$1"
DATASET="$2"
OUT_LOGDIR="$3"
RESUME_LOGDIR="$4"
EPOCHS="$5"
NNODES="$6"
RANKS_PER_NODE="$7"
ELEMENTAL_MANIFEST="$8"
TOTAL_RANKS=$((NNODES * RANKS_PER_NODE))
SCRIPT_NAME="$(basename "${TRAIN_SCRIPT}")"

[[ -f "${TRAIN_SCRIPT}" ]] || { echo "train script not found: ${TRAIN_SCRIPT}" >&2; exit 2; }
[[ -f "${ELEMENTAL_MANIFEST}" ]] || { echo "elemental manifest missing: ${ELEMENTAL_MANIFEST}" >&2; exit 2; }
[[ "${TOTAL_RANKS}" == 1 ]] || { echo "Built-in HydraGNN trainers require one process" >&2; exit 2; }
[[ "${SCRIPT_NAME}" == "finetune_hydragnn.py" || "${SCRIPT_NAME}" == "finetune_hydragnn_newhead.py" ]] || {
  echo "unsupported HydraGNN trainer: ${SCRIPT_NAME}" >&2
  exit 2
}

EXTRA_ARGS=(--device xpu)
if [[ -n "${9:-}" ]]; then
  EXTRA_ARGS+=(--gfm-checkpoint "${9}")
fi
EXTRA_ARGS+=(--charge "${12:-0.0}" --spin "${13:-0.0}")
if [[ "${SCRIPT_NAME}" == "finetune_hydragnn_newhead.py" && -n "${11:-}" ]]; then
  EXTRA_ARGS+=(--ft-repo "${11}")
fi
if [[ -n "${14:-}" ]]; then
  EXTRA_ARGS+=(--hydragnn-root "${14}")
fi
if [[ "${SCRIPT_NAME}" == "finetune_hydragnn.py" ]]; then
  BRANCH_MLP="${10:-${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-}}"
  [[ -f "${BRANCH_MLP}" ]] || { echo "branch MLP checkpoint missing: ${BRANCH_MLP}" >&2; exit 2; }
  EXTRA_ARGS+=(--branch-mlp "${BRANCH_MLP}")
fi

mkdir -p "${OUT_LOGDIR}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
export ZE_AFFINITY_MASK="${MATSIM_TRAIN_TILE:-0}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"

echo "[hydragnn-train] $(date) host=$(hostname) tile=${ZE_AFFINITY_MASK}"
echo "[hydragnn-train] mpiexec -n 1 --ppn 1 python ${TRAIN_SCRIPT} ..."

exec mpiexec -n 1 --ppn 1 \
  python "${TRAIN_SCRIPT}" \
    --dataset "${DATASET}" \
    --output-dir "${OUT_LOGDIR}" \
    --gfm-logdir "${RESUME_LOGDIR}" \
    --epochs "${EPOCHS}" \
    --elemental-reference-manifest "${ELEMENTAL_MANIFEST}" \
    "${EXTRA_ARGS[@]}"
