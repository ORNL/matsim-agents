#!/bin/bash
# =============================================================================
# _mlip-train-step-aurora.sh
#
# Site launcher for one active-learning fine-tune (trainer.train_launcher).
# The trainer runs, inside the AL driver's PBS allocation:
#
#   bash _mlip-train-step-aurora.sh <backend> <nodes> <ranks_per_node> \
#        -m <module> <args...>
#
# The MLIP configuration builds <module> and <args> (mlip.fine_tuner); this
# script does not interpret them. It prepares <backend>'s environment, applies
# the training pool's placement (hosts and XPU tile), and runs
# `python -m <module> <args...>` under mpiexec.
#
# Environments (deployments/aurora/setup/install.sh):
#   hydragnn  the AL driver's active frameworks + HydraGNN environment (.venv)
#   uma       MATSIM_FAIRCHEM_VENV (default .venv-uma)
#   mace      MATSIM_MACE_VENV (default .venv-mace)
# =============================================================================

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

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
  [[ -f "${venv}/bin/activate" ]] || { echo "ERROR: ${BACKEND} environment not found: ${venv}" >&2; exit 2; }
  # shellcheck disable=SC1091
  source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
  load_aurora_frameworks_modules
  # shellcheck disable=SC1091
  source "${venv}/bin/activate"
}

case "${BACKEND}" in
  hydragnn) ;;
  uma) activate_environment "${MATSIM_FAIRCHEM_VENV:-${REPO}/.venv-uma}" ;;
  mace) activate_environment "${MATSIM_MACE_VENV:-${REPO}/.venv-mace}" ;;
  *)
    echo "ERROR: unsupported MLIP backend on Aurora: ${BACKEND} (hydragnn, uma, mace)" >&2
    exit 2
    ;;
esac

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
# The training pool sets ZE_AFFINITY_MASK to its tiles.
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:?resources.pools.training must place the trainer on XPU tiles}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"

# The training pool's hosts come from the placement.
HOST_ARGS=()
if [[ -n "${MATSIM_ASSIGNED_NODES:-}" ]]; then
  HOST_ARGS=(--hosts "${MATSIM_ASSIGNED_NODES}")
fi

echo "[mlip-train] $(date) host=$(hostname) backend=${BACKEND} tiles=${ZE_AFFINITY_MASK} ranks=${TOTAL_RANKS}"
echo "[mlip-train] python $*"
exec mpiexec "${HOST_ARGS[@]}" -n "${TOTAL_RANKS}" --ppn "${RANKS_PER_NODE}" python "$@"
