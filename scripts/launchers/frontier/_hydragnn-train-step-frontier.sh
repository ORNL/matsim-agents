#!/bin/bash
# =============================================================================
# _hydragnn-train-step-frontier.sh
#
# Inner-step launcher for ONE HydraGNN training round inside the AL loop.
# Called by trainer.retrain_hydragnn() with positional args:
#
#   bash _hydragnn-train-step-frontier.sh \
#        <train_script> <dataset_path> <out_logdir> <resume_logdir> \
#        <epochs> <nodes_for_train> <ranks_per_node>
#
# Runs `srun python <train_script> --dataset --logdir --resume_from --epochs`
# inside the SAME allocation as the AL driver (no separate sbatch). The
# training script must accept these flags; if your script uses a different
# CLI, edit the `srun python ...` line at the bottom.
# =============================================================================

set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 <train_script> <dataset> <out_logdir> <resume_logdir> <epochs> <nodes> <ranks_per_node>" >&2
  exit 2
fi

TRAIN_SCRIPT="$1"
DATASET="$2"
OUT_LOGDIR="$3"
RESUME_LOGDIR="$4"
EPOCHS="$5"
NNODES="$6"
RANKS_PER_NODE="$7"
TOTAL_RANKS=$(( NNODES * RANKS_PER_NODE ))

[[ -f "${TRAIN_SCRIPT}" ]] || { echo "train script not found: ${TRAIN_SCRIPT}" >&2; exit 2; }

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
    --logdir "${OUT_LOGDIR}" \
    --resume_from "${RESUME_LOGDIR}" \
    --epochs "${EPOCHS}"
