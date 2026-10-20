#!/bin/bash
# =============================================================================
# _hydragnn-train-step-perlmutter.sh
#
# Inner-step launcher for one matsim-agents HydraGNN fine-tuning round.
# The AL driver has already activated the HydraGNN environment and this step
# runs within the same Perlmutter allocation.
#
# Arguments:
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

EXTRA_ARGS=()
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MPICH_GPU_SUPPORT_ENABLED=1

echo "[hydragnn-train] $(date) host=$(hostname)"
# The training pool's host comes from the placement (MATSIM_ASSIGNED_NODES).
NODE_ARGS=()
if [[ -n "${MATSIM_ASSIGNED_NODES:-}" ]]; then
  NODE_ARGS=(--nodelist "${MATSIM_ASSIGNED_NODES}")
fi
echo "[hydragnn-train] srun -N 1 -n 1 python ${TRAIN_SCRIPT} ..."

# Slurm picks which GPUs a step receives, so a GPU count alone cannot honour the
# placement's device indices. When the placement uses only some of the node's
# GPUs, take them all for the step and expose only the placed ones inside it.
STEP_GPUS="1"
GPU_ARGS=(--gpus-per-node="${STEP_GPUS}" --gpu-bind=closest)
STEP_ENV=()
STEP_SHARE=()
if [[ -n "${MATSIM_DEVICE_MASK:-}" && -n "${MATSIM_NODE_DEVICES:-}" \
      && "${STEP_GPUS}" != "${MATSIM_NODE_DEVICES}" ]]; then
  GPU_ARGS=(--gpus-per-node="${MATSIM_NODE_DEVICES}" --gpu-bind=none)
  # Other pools may hold the node's remaining GPUs (e.g. a vLLM step); the
  # placement keeps devices disjoint, so overlap instead of waiting for them.
  STEP_SHARE=(--overlap)
  STEP_ENV=(env "CUDA_VISIBLE_DEVICES=${MATSIM_DEVICE_MASK}")
fi
echo "[step] gpu_args=${GPU_ARGS[*]} placed_devices=${MATSIM_DEVICE_MASK:-all}"
exec srun \
  "${STEP_SHARE[@]}" \
  "${NODE_ARGS[@]}" \
  -N 1 \
  -n 1 \
  -c "${OMP_NUM_THREADS}" \
  "${GPU_ARGS[@]}" \
  "${STEP_ENV[@]}" \
  python "${TRAIN_SCRIPT}" \
    --dataset "${DATASET}" \
    --output-dir "${OUT_LOGDIR}" \
    --gfm-logdir "${RESUME_LOGDIR}" \
    --epochs "${EPOCHS}" \
    --elemental-reference-manifest "${ELEMENTAL_MANIFEST}" \
    "${EXTRA_ARGS[@]}"
