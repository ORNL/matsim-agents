#!/bin/bash
#PBS -A CM2US
#PBS -N matsim-qe-warmstart
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# matsim-agents: HydraGNN warm-start vs Quantum ESPRESSO cold-start
# benchmark on ALCF Aurora.
#
# Runs tests/integration/test_qe_warmstart.py, which:
#   1. Relaxes each fixture with HydraGNN MLFF (warm start).
#   2. Runs pw.x cold-start and pw.x warm-start (initial coords from MLFF).
#   3. Reports SCF iterations / wall-time speed-up.
#
# Submit:
#   qsub scripts/advanced/aurora/job-qe-warmstart-aurora.sh
#
# Override fixtures (comma-separated, see test file for available names):
#   qsub -v MATSIM_WARMSTART_FIXTURES=Si_diamond \
#        scripts/advanced/aurora/job-qe-warmstart-aurora.sh
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

# ── repo / paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/lus/flare/projects/CM2US/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"

VENV="${MATSIM_AURORA_VENV:-${PROJ}/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Aurora/hydragnn_venv}"
HYDRAGNN_EXAMPLE="${PROJ}/HydraGNN/examples/multidataset_hpo_sc26"
LOGDIR="${MATSIM_HYDRAGNN_LOGDIR:-${HYDRAGNN_EXAMPLE}/multidataset_hpo-BEST6-fp64}"
MLP_CHECKPOINT="${MATSIM_HYDRAGNN_MLP_CKPT:-${HYDRAGNN_EXAMPLE}/mlp_branch_weights.pt}"

QE_LAUNCHER="${MATSIM_QE_LAUNCHER:-${REPO}/scripts/launchers/aurora/run-pw-gpu-aurora.sh}"
QE_PSEUDO_DIR="${MATSIM_QE_PSEUDO_DIR:-${REPO}/external/quantum-espresso/src/pseudo}"

JOBID="${PBS_JOBID:-$$}"
RUN_DIR="${PROJ}/runs/qe-warmstart-aurora-${JOBID}"
WARMSTART_DIR="${RUN_DIR}/qe-warmstart"
mkdir -p "${RUN_DIR}" "${WARMSTART_DIR}"

# ── modules + venv ──────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
  module reset
  module load frameworks
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONPATH="${HYDRAGNN_EXAMPLE}:${PROJ}/HydraGNN:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"

# ── QE / warmstart env ──────────────────────────────────────────────────────
export MATSIM_QE_LAUNCHER="${QE_LAUNCHER}"
export MATSIM_QE_PSEUDO_DIR="${QE_PSEUDO_DIR}"
export MATSIM_HYDRAGNN_LOGDIR="${LOGDIR}"
export MATSIM_HYDRAGNN_MLP_CKPT="${MLP_CHECKPOINT}"
export MATSIM_QE_MLP_DEVICE="${MATSIM_QE_MLP_DEVICE:-cuda}"
export MATSIM_QE_TIMEOUT_SEC="${MATSIM_QE_TIMEOUT_SEC:-3600}"
# Restrict to fixtures whose elements have UPFs in QE_PSEUDO_DIR (Si_r.upf
# ships with the QE source tree; heavier fixtures need extra UPFs).
export MATSIM_WARMSTART_FIXTURES="${MATSIM_WARMSTART_FIXTURES:-Si_diamond}"

# ── diagnostics ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Aurora QE warm-start benchmark]"
echo "Date:         $(date)"
echo "Job ID:       ${JOBID}"
echo "Host:         $(hostname)"
echo "Repo:         ${REPO}"
echo "Venv:         ${VENV}"
echo "HydraGNN log: ${LOGDIR}"
echo "MLP ckpt:     ${MLP_CHECKPOINT}"
echo "QE launcher:  ${QE_LAUNCHER}"
echo "QE pseudos:   ${QE_PSEUDO_DIR}"
echo "Fixtures:     ${MATSIM_WARMSTART_FIXTURES}"
echo "=========================================="

echo "[$(date)] Python: $(which python) ($(python --version 2>&1))"

# ── benchmark ───────────────────────────────────────────────────────────────
pushd "${REPO}" >/dev/null
python -m pytest -xvs tests/integration/test_qe_warmstart.py \
  --basetemp="${WARMSTART_DIR}" \
  2>&1 | tee "${RUN_DIR}/qe-warmstart.log"
popd >/dev/null

echo "[$(date)] QE warm-start benchmark complete. Artifacts in ${WARMSTART_DIR}"
