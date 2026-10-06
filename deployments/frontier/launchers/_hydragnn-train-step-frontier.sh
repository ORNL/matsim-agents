#!/bin/bash
# =============================================================================
# _hydragnn-train-step-frontier.sh
#
# Inner-step launcher for ONE HydraGNN training round inside the AL loop.
# Called by trainer.retrain_hydragnn() with positional args:
#
#   bash _hydragnn-train-step-frontier.sh \
#        <train_script> <dataset_path> <out_logdir> <resume_logdir> \
#        <epochs> <nodes_for_train> <ranks_per_node> <elemental_manifest> \
#        [checkpoint] [branch_mlp]
#
# Runs the built-in CLI or the legacy custom --logdir/--resume_from CLI,
# always passing --elemental-reference-manifest.
# inside the SAME allocation as the AL driver (no separate sbatch). The
# training script must accept these flags; if your script uses a different
# CLI, edit the `srun python ...` line at the bottom.
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
TOTAL_RANKS=$(( NNODES * RANKS_PER_NODE ))

[[ -f "${TRAIN_SCRIPT}" ]] || { echo "train script not found: ${TRAIN_SCRIPT}" >&2; exit 2; }
[[ -f "${ELEMENTAL_MANIFEST}" ]] || { echo "elemental manifest missing" >&2; exit 2; }
EXTRA_ARGS=()
OUTPUT_FLAG=--logdir
RESUME_FLAG=--resume_from
SCRIPT_NAME="$(basename "${TRAIN_SCRIPT}")"
if [[ "${SCRIPT_NAME}" == "finetune_hydragnn.py" || "${SCRIPT_NAME}" == "finetune_hydragnn_newhead.py" ]]; then
  [[ "${TOTAL_RANKS}" == 1 ]] || { echo "Built-in HydraGNN trainers require one process" >&2; exit 2; }
  OUTPUT_FLAG=--output-dir
  RESUME_FLAG=--gfm-logdir
  if [[ -n "${9:-}" ]]; then
    EXTRA_ARGS+=(--gfm-checkpoint "${9}")
  fi
fi
if [[ "${SCRIPT_NAME}" == "finetune_hydragnn.py" ]]; then
  BRANCH_MLP="${10:-${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-}}"
  [[ -f "${BRANCH_MLP}" ]] || { echo "branch MLP checkpoint missing" >&2; exit 2; }
  EXTRA_ARGS+=(--branch-mlp "${BRANCH_MLP}")
fi

mkdir -p "${OUT_LOGDIR}"

# We are already inside the AL driver's environment (PrgEnv-gnu + rocm/7.2.0
# + the HydraGNN venv). No module swap needed for HydraGNN training — only
# the VASP step needs PrgEnv-cray. We just exec srun against the allocation.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"
export MPICH_GPU_SUPPORT_ENABLED=1

echo "[hydragnn-train] $(date) host=$(hostname)"
echo "[hydragnn-train] srun -N ${NNODES} -n ${TOTAL_RANKS} python ${TRAIN_SCRIPT} ..."

exec srun \
  -N "${NNODES}" \
  -n "${TOTAL_RANKS}" \
  -c "${OMP_NUM_THREADS}" \
  --gpus-per-node="${RANKS_PER_NODE}" \
  --gpu-bind=closest \
  python "${TRAIN_SCRIPT}" \
    --dataset "${DATASET}" \
    "${OUTPUT_FLAG}" "${OUT_LOGDIR}" \
    "${RESUME_FLAG}" "${RESUME_LOGDIR}" \
    --epochs "${EPOCHS}" \
    --elemental-reference-manifest "${ELEMENTAL_MANIFEST}" \
    "${EXTRA_ARGS[@]}"
