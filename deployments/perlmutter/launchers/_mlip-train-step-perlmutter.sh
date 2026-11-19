#!/bin/bash
# =============================================================================
# _mlip-train-step-perlmutter.sh
#
# Site launcher for one active-learning fine-tune (trainer.train_launcher).
# The trainer runs, inside the AL driver's Perlmutter allocation:
#
#   bash _mlip-train-step-perlmutter.sh <backend> <nodes> <ranks_per_node> \
#        -m <module> <args...>
#
# The MLIP configuration builds <module> and <args> (mlip.fine_tuner); this
# script does not interpret them. It prepares <backend>'s environment, applies
# the training pool's placement, and runs `python -m <module> <args...>`.
#
# Environments:
#   hydragnn  the AL driver's active HydraGNN environment (.venv)
#   uma       MATSIM_FAIRCHEM_VENV (default .venv-uma); the durable UMA bundle
#             location (MATSIM_UMA_ARTIFACT_DIR) is inherited from the driver
#   mace      MATSIM_MACE_VENV (default .venv-mace) and the durable MACE
#             foundation-model artifacts
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
  [[ -d "${venv}" ]] || { echo "ERROR: ${BACKEND} environment not found: ${venv}" >&2; exit 2; }
  # shellcheck disable=SC1091
  source "${REPO}/deployments/perlmutter/setup/perlmutter-module-stack.sh"
  load_perlmutter_modules_gpu
  if [[ -f "${venv}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${venv}/bin/activate"
  else
    local conda_base
    conda_base="$(conda info --base 2>/dev/null)"
    [[ -n "${conda_base}" ]] || { echo "ERROR: 'conda' not available to activate ${venv}" >&2; exit 2; }
    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "${venv}" || { echo "ERROR: conda activate failed for ${venv}" >&2; exit 2; }
  fi
}

case "${BACKEND}" in
  hydragnn) ;;
  uma) activate_environment "${MATSIM_FAIRCHEM_VENV:-${REPO}/.venv-uma}" ;;
  mace)
    activate_environment "${MATSIM_MACE_VENV:-${REPO}/.venv-mace}"
    # shellcheck disable=SC1091
    source "${REPO}/deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
    configure_mace_model_artifacts "${REPO}"
    ;;
  *)
    echo "ERROR: unsupported MLIP backend on Perlmutter: ${BACKEND} (hydragnn, uma, mace)" >&2
    exit 2
    ;;
esac

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
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
  STEP_ENV=(env "CUDA_VISIBLE_DEVICES=${MATSIM_DEVICE_MASK}")
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
