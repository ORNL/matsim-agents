#!/bin/bash
#SBATCH -J qe-warmstart-bench
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# =============================================================================
# Run the HydraGNN warm-start vs Quantum ESPRESSO cold-start benchmark
# (tests/integration/test_qe_warmstart.py) on a single Perlmutter GPU node.
#
# Architecture (Perlmutter is simpler than Frontier here):
#   * HydraGNN ASE relax phase  : cudatoolkit/12.9 + hydragnn_venv (torch cu129)
#   * pw.x phase                : NVHPC 25.5 (also CUDA 12.9) — same major.minor
#                                 as the PyTorch wheel, so no module conflict.
#   The pw.x launcher handles the PrgEnv-nvidia swap inside its own subshell
#   so the parent environment stays HydraGNN-aligned throughout.
#
# Required overrides (set via environment or edit below):
#   PROJECT_ROOT        repo root (default: ${PROJECT_ROOT:?export PROJECT_ROOT})
#   PSEUDO_DIR          directory of .UPF pseudopotentials (REQUIRED)
#   HYDRAGNN_LOGDIR     HydraGNN logdir with config.json + checkpoint
#   HYDRAGNN_BRANCH_MLP_CHECKPOINT   BranchWeightMLP .pt checkpoint
#
# Optional:
#   FIXTURES            comma-separated fixture names (default: Si_diamond)
#   QE_TIMEOUT_SEC      per-pw.x-run timeout in seconds (default: 3600)
#
# Usage:
#   sbatch \
#     --export=ALL,PSEUDO_DIR=/path/to/pseudos,HYDRAGNN_LOGDIR=/path,HYDRAGNN_BRANCH_MLP_CHECKPOINT=/path/mlp.pt \
#     deployments/perlmutter/launchers/run-qe-warmstart-benchmark-perlmutter.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}"
[[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]] && \
  PROJECT_ROOT=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${PROJECT_ROOT}")"

PSEUDO_DIR="${PSEUDO_DIR:-${PROJECT_ROOT}/external/quantum-espresso/src/pseudo}"
HYDRAGNN_LOGDIR="${HYDRAGNN_LOGDIR:-$PROJ/HydraGNN/examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64}"
HYDRAGNN_BRANCH_MLP_CHECKPOINT="${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-$PROJ/HydraGNN/examples/multidataset_hpo_sc26/mlp_branch_weights.pt}"
FIXTURES="${FIXTURES:-Si_diamond}"
QE_TIMEOUT_SEC="${QE_TIMEOUT_SEC:-3600}"

if [[ ! -d "${PSEUDO_DIR}" || ! -d "${HYDRAGNN_LOGDIR}" || ! -f "${HYDRAGNN_BRANCH_MLP_CHECKPOINT}" ]]; then
  echo "ERROR: One of PSEUDO_DIR / HYDRAGNN_LOGDIR / HYDRAGNN_BRANCH_MLP_CHECKPOINT is missing." >&2
  echo "  PSEUDO_DIR        = ${PSEUDO_DIR}" >&2
  echo "  HYDRAGNN_LOGDIR   = ${HYDRAGNN_LOGDIR}" >&2
  echo "  HYDRAGNN_BRANCH_MLP_CHECKPOINT = ${HYDRAGNN_BRANCH_MLP_CHECKPOINT}" >&2
  exit 2
fi

QE_LAUNCHER="${PROJECT_ROOT}/deployments/perlmutter/launchers/run-pw-gpu-perlmutter.sh"
if [[ ! -x "${QE_LAUNCHER}" ]]; then
  echo "ERROR: QE launcher not found or not executable: ${QE_LAUNCHER}" >&2
  exit 2
fi

# ── HydraGNN-aligned module stack + venv ────────────────────────────────────
source "${PROJECT_ROOT}/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu

VENV_ROOT="$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv"
if [[ ! -d "${VENV_ROOT}" ]]; then
  echo "ERROR: HydraGNN venv not found: ${VENV_ROOT}" >&2
  exit 2
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VENV_ROOT}"

cd "${PROJECT_ROOT}"

export MATSIM_QE_LAUNCHER="${QE_LAUNCHER}"
export MATSIM_QE_PSEUDO_DIR="${PSEUDO_DIR}"
export MATSIM_HYDRAGNN_LOGDIR="${HYDRAGNN_LOGDIR}"
export HYDRAGNN_BRANCH_MLP_CHECKPOINT="${HYDRAGNN_BRANCH_MLP_CHECKPOINT}"
export MATSIM_QE_TIMEOUT_SEC="${QE_TIMEOUT_SEC}"
export MATSIM_QE_MLP_DEVICE="cuda"
export MATSIM_WARMSTART_FIXTURES="${FIXTURES}"

# inference_fused.py lives in the HydraGNN SC26 examples dir; add it so
# HydraGNN's warm-start path can import it.
HYDRAGNN_SC26_DIR="$PROJ/HydraGNN/examples/multidataset_hpo_sc26"
export PYTHONPATH="${HYDRAGNN_SC26_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Disable GPU-aware MPI for the pw.x sub-launch: the GTL library is not
# linked in this batch job context, which causes all MPI ranks to abort.
# QE's GPU acceleration (CUDA kernels) still works; only GPU-direct
# inter-rank communication is affected, which this benchmark does not test.
export MPICH_GPU_SUPPORT_ENABLED=0

echo "=========================================="
echo "QE warm-start benchmark on Perlmutter"
echo "Date:               $(date)"
echo "Host:               $(hostname)"
echo "PROJECT_ROOT:       ${PROJECT_ROOT}"
echo "QE launcher:        ${MATSIM_QE_LAUNCHER}"
echo "PSEUDO_DIR:         ${MATSIM_QE_PSEUDO_DIR}"
echo "HYDRAGNN_LOGDIR:    ${MATSIM_HYDRAGNN_LOGDIR}"
echo "HYDRAGNN_BRANCH_MLP_CHECKPOINT:  ${HYDRAGNN_BRANCH_MLP_CHECKPOINT}"
echo "FIXTURES:           ${FIXTURES}"
echo "QE_TIMEOUT_SEC:     ${MATSIM_QE_TIMEOUT_SEC}"
echo "=========================================="

RUN_DIR="${PROJ}/runs/qe-warmstart-${SLURM_JOB_ID:-local-$$}/pytest-tmp"
mkdir -p "${RUN_DIR}"

set +e
python -m pytest -ra -s \
  --basetemp="${RUN_DIR}" \
  tests/integration/test_qe_warmstart.py
status=$?
set -e

echo
echo "=========================================="
echo "JSON comparison summaries:"
find "${RUN_DIR}" -name "comparison.json" -print
echo "=========================================="

exit "${status}"
