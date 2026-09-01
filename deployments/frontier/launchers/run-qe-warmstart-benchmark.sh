#!/bin/bash
#SBATCH -J qe-warmstart-bench
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# =============================================================================
# Run the HydraGNN warm-start vs Quantum ESPRESSO cold-start benchmark
# (tests/integration/test_qe_warmstart.py) on a single Frontier node.
#
# Architecture:
#   * The HydraGNN ASE relax phase needs rocm/7.2.0 + the rocm72 venv.
#   * The pw.x phase needs rocm/6.2.4 + cray-mpich (see build-qe-gpu-frontier.sh
#     for why ROCm is *forced* to 6.2.4 for QE).
#   * These two ROCm pins cannot coexist in one shell.
#
# We therefore launch pytest with rocm/7.2.0 loaded in the *parent* shell, and
# point MATSIM_QE_LAUNCHER at the GPU pw.x launcher script. That launcher
# does its own `module reset && module load rocm/6.2.4 ...` inside its child
# bash, so the QE subprocess sees a clean environment regardless of what the
# parent loaded.
#
# Required overrides (set via environment or edit below):
#   PROJECT_ROOT          repo root (default: ${PROJECT_ROOT:?export PROJECT_ROOT})
#   PSEUDO_DIR            directory of .UPF pseudopotentials (REQUIRED)
#   HYDRAGNN_LOGDIR       HydraGNN logdir with config.json + checkpoint
#   HYDRAGNN_BRANCH_MLP_CHECKPOINT     BranchWeightMLP .pt checkpoint
#
# Optional:
#   FIXTURES              comma-separated fixture names (default: all)
#   QE_TIMEOUT_SEC        per-pw.x-run timeout in seconds (default: 3600)
#
# Usage:
#   sbatch \
#     --export=ALL,PSEUDO_DIR=/path/to/pseudos,HYDRAGNN_LOGDIR=/path/to/logdir,HYDRAGNN_BRANCH_MLP_CHECKPOINT=/path/to/mlp.pt \
#     deployments/frontier/launchers/run-qe-warmstart-benchmark.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}"
[[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]] && PROJECT_ROOT=${PROJECT_ROOT:?export PROJECT_ROOT}

PSEUDO_DIR="${PSEUDO_DIR:-}"
HYDRAGNN_LOGDIR="${HYDRAGNN_LOGDIR:-}"
HYDRAGNN_BRANCH_MLP_CHECKPOINT="${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-}"
FIXTURES="${FIXTURES:-}"
QE_TIMEOUT_SEC="${QE_TIMEOUT_SEC:-3600}"

if [[ -z "${PSEUDO_DIR}" || -z "${HYDRAGNN_LOGDIR}" || -z "${HYDRAGNN_BRANCH_MLP_CHECKPOINT}" ]]; then
  echo "ERROR: PSEUDO_DIR, HYDRAGNN_LOGDIR, and HYDRAGNN_BRANCH_MLP_CHECKPOINT must all be set." >&2
  echo "Got: PSEUDO_DIR='${PSEUDO_DIR}' HYDRAGNN_LOGDIR='${HYDRAGNN_LOGDIR}' HYDRAGNN_BRANCH_MLP_CHECKPOINT='${HYDRAGNN_BRANCH_MLP_CHECKPOINT}'" >&2
  exit 2
fi

QE_LAUNCHER="${PROJECT_ROOT}/deployments/frontier/launchers/run-pw-gpu-frontier.sh"
if [[ ! -x "${QE_LAUNCHER}" ]]; then
  echo "ERROR: QE launcher not found or not executable: ${QE_LAUNCHER}" >&2
  exit 2
fi

# ── HydraGNN side: rocm/7.2.0 + rocm72 venv ────────────────────────────────
module reset
module load PrgEnv-gnu
module load rocm/7.2.0
module load amd-mixed/7.2.0
module load miniforge3/23.11.0-0

VENV_ROOT="${VENV_ROOT:-$(dirname "${PROJECT_ROOT}")/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72}"
if [[ ! -d "${VENV_ROOT}" ]]; then
  echo "ERROR: HydraGNN rocm/7.2.0 venv not found: ${VENV_ROOT}" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "${VENV_ROOT}/bin/activate" || conda activate "${VENV_ROOT}"

cd "${PROJECT_ROOT}"

export MATSIM_QE_LAUNCHER="${QE_LAUNCHER}"
export MATSIM_QE_PSEUDO_DIR="${PSEUDO_DIR}"
export MATSIM_HYDRAGNN_LOGDIR="${HYDRAGNN_LOGDIR}"
export HYDRAGNN_BRANCH_MLP_CHECKPOINT="${HYDRAGNN_BRANCH_MLP_CHECKPOINT}"
export MATSIM_QE_TIMEOUT_SEC="${QE_TIMEOUT_SEC}"
export MATSIM_QE_MLP_DEVICE="cuda"
[[ -n "${FIXTURES}" ]] && export MATSIM_WARMSTART_FIXTURES="${FIXTURES}"

echo "=========================================="
echo "QE warm-start benchmark on Frontier"
echo "Date:               $(date)"
echo "Host:               $(hostname)"
echo "PROJECT_ROOT:       ${PROJECT_ROOT}"
echo "QE launcher:        ${MATSIM_QE_LAUNCHER}"
echo "PSEUDO_DIR:         ${MATSIM_QE_PSEUDO_DIR}"
echo "HYDRAGNN_LOGDIR:    ${MATSIM_HYDRAGNN_LOGDIR}"
echo "HYDRAGNN_BRANCH_MLP_CHECKPOINT:  ${HYDRAGNN_BRANCH_MLP_CHECKPOINT}"
echo "FIXTURES:           ${FIXTURES:-<all>}"
echo "QE_TIMEOUT_SEC:     ${MATSIM_QE_TIMEOUT_SEC}"
echo "=========================================="

# Persist artefacts (the test uses tmp_path which lives under TMPDIR by default;
# we override pytest's basetemp so all per-fixture work dirs land under a
# stable, slurm-job-scoped location).
RUN_DIR="${PROJECT_ROOT}/runs/qe-warmstart-${SLURM_JOB_ID:-local-$$}/pytest-tmp"
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
