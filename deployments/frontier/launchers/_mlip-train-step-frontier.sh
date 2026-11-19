#!/bin/bash
# =============================================================================
# _mlip-train-step-frontier.sh
#
# Site launcher for one active-learning fine-tune (trainer.train_launcher).
# The trainer runs, inside the AL driver's Frontier allocation:
#
#   bash _mlip-train-step-frontier.sh <backend> <nodes> <ranks_per_node> \
#        -m <module> <args...>
#
# The MLIP configuration builds <module> and <args> (mlip.fine_tuner); this
# script does not interpret them. It applies the training pool's placement and
# runs `python -m <module> <args...>` under srun.
#
# The AL driver's environment (PrgEnv-gnu + rocm/7.2.0) is already active.
# HydraGNN trains with the driver's venv; UMA and MACE use the interpreters of
# the .venv-uma / .venv-mace environments built by setup/install.sh.
# =============================================================================

set -euo pipefail

if [[ $# -lt 5 || "$4" != "-m" ]]; then
  echo "Usage: $0 <backend> <nodes> <ranks_per_node> -m <module> [args...]" >&2
  exit 2
fi
BACKEND="$1"
NNODES="$2"
RANKS_PER_NODE="$3"
shift 3
TOTAL_RANKS=$((NNODES * RANKS_PER_NODE))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ -f "${REPO}/pyproject.toml" ]] || REPO="${PROJECT_ROOT:?export PROJECT_ROOT}"

activate_environment() {
  local venv="$1"
  [[ -f "${venv}/bin/activate" ]] || {
    echo "ERROR: ${BACKEND} environment not found: ${venv} (run setup/install.sh)" >&2
    exit 2
  }
  # shellcheck disable=SC1091
  source "${venv}/bin/activate"
}

case "${BACKEND}" in
  hydragnn) ;;
  uma) activate_environment "${MATSIM_FAIRCHEM_VENV:-${REPO}/.venv-uma}" ;;
  mace) activate_environment "${MATSIM_MACE_VENV:-${REPO}/.venv-mace}" ;;
  *)
    echo "ERROR: unsupported MLIP backend on Frontier: ${BACKEND} (hydragnn, uma, mace)" >&2
    exit 2
    ;;
esac

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"
export MPICH_GPU_SUPPORT_ENABLED=1

# The training pool's host comes from the placement (MATSIM_ASSIGNED_NODES).
NODE_ARGS=()
if [[ -n "${MATSIM_ASSIGNED_NODES:-}" ]]; then
  NODE_ARGS=(--nodelist "${MATSIM_ASSIGNED_NODES}")
fi

# Slurm picks which GPUs a step receives, so a GPU count alone cannot honour the
# placement's device indices. When the placement uses only some of the node's
# GPUs, take them all for the step and expose only the placed ones inside it.
STEP_GPUS="${RANKS_PER_NODE}"
GPU_ARGS=(--gpus-per-node="${STEP_GPUS}" --gpu-bind=closest)
STEP_ENV=()
STEP_SHARE=()
if [[ -n "${MATSIM_DEVICE_MASK:-}" && -n "${MATSIM_NODE_DEVICES:-}" \
      && "${STEP_GPUS}" != "${MATSIM_NODE_DEVICES}" ]]; then
  GPU_ARGS=(--gpus-per-node="${MATSIM_NODE_DEVICES}" --gpu-bind=none)
  # Other pools may hold the node's remaining GPUs (e.g. a vLLM step); the
  # placement keeps devices disjoint, so overlap instead of waiting for them.
  STEP_SHARE=(--overlap)
  STEP_ENV=(env -u HIP_VISIBLE_DEVICES "ROCR_VISIBLE_DEVICES=${MATSIM_DEVICE_MASK}")
fi

echo "[mlip-train] $(date) host=$(hostname) backend=${BACKEND} nodes=${NNODES} ranks=${TOTAL_RANKS}"
echo "[mlip-train] gpu_args=${GPU_ARGS[*]} placed_devices=${MATSIM_DEVICE_MASK:-all}"
echo "[mlip-train] python $*"
exec srun \
  ${STEP_SHARE[@]+"${STEP_SHARE[@]}"} \
  ${NODE_ARGS[@]+"${NODE_ARGS[@]}"} \
  -N "${NNODES}" \
  -n "${TOTAL_RANKS}" \
  -c "${OMP_NUM_THREADS}" \
  "${GPU_ARGS[@]}" \
  ${STEP_ENV[@]+"${STEP_ENV[@]}"} \
  python "$@"
