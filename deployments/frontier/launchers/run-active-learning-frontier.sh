#!/bin/bash
#SBATCH -J matsim-al
#SBATCH -p batch
#SBATCH -N 64
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# =============================================================================
# run-active-learning-frontier.sh
#
# Single-job orchestrator for the HydraGNN <-> VASP active-learning loop.
#
# The driver (Python) runs in the parent shell under PrgEnv-gnu + the HydraGNN
# rocm/7.2.0 venv. Each VASP single-point is dispatched as a separate srun
# step via _vasp-step-frontier.sh, which does its own module swap to
# PrgEnv-cray + rocm/7.1.1. The two stacks never coexist in the same process.
#
# Required environment overrides (or edit below):
#   AL_CONFIG     Path to the AL YAML config
#
# Optional:
#   PROJECT_ROOT  Repo root (default: ${PROJECT_ROOT:?export PROJECT_ROOT})
#   VENV_ROOT     HydraGNN venv (default: rocm72 venv)
#   LOG_LEVEL     Python logging level (default: INFO)
#
# Usage:
#   sbatch --export=ALL,AL_CONFIG=/path/to/al.yaml \
#       deployments/frontier/launchers/run-active-learning-frontier.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}"
[[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]] && PROJECT_ROOT=${PROJECT_ROOT:?export PROJECT_ROOT}

AL_CONFIG="${AL_CONFIG:-}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
VENV_ROOT="${VENV_ROOT:-${PROJECT_ROOT}/.venv}"

if [[ -z "${AL_CONFIG}" ]]; then
  echo "ERROR: AL_CONFIG must be set (path to AL YAML config)." >&2
  exit 2
fi
if [[ ! -f "${AL_CONFIG}" ]]; then
  echo "ERROR: AL_CONFIG file not found: ${AL_CONFIG}" >&2
  exit 2
fi

# ── Parent-shell environment: HydraGNN side (PrgEnv-gnu + rocm/7.2.0) ───────
module reset
module load PrgEnv-gnu
module load rocm/7.2.0
module load amd-mixed/7.2.0
module load miniforge3/23.11.0-0

if [[ ! -d "${VENV_ROOT}" ]]; then
  echo "ERROR: HydraGNN venv not found: ${VENV_ROOT}" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "${VENV_ROOT}/bin/activate" || conda activate "${VENV_ROOT}"

cd "${PROJECT_ROOT}"

# Make sure the AL package is importable even if the user hasn't `pip -e`'d
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "=========================================="
echo "matsim-agents active-learning loop"
echo "Date:           $(date)"
echo "Host:           $(hostname)"
echo "Allocation:     ${SLURM_JOB_NUM_NODES:-?} node(s), JobID=${SLURM_JOB_ID:-local}"
echo "PROJECT_ROOT:   ${PROJECT_ROOT}"
echo "AL_CONFIG:      ${AL_CONFIG}"
echo "VENV_ROOT:      ${VENV_ROOT}"
echo "LOG_LEVEL:      ${LOG_LEVEL}"
echo "=========================================="

exec python -m matsim_agents.cli al run "${AL_CONFIG}" --log-level "${LOG_LEVEL}"
