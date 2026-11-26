#!/bin/bash
#SBATCH -J neb-escalation-smoke
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 00:10:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# ---------------------------------------------------------------------------
# Uncertainty-gated NEB escalation to a real QE neb.x run on one Frontier node.
#
# Checks the launch and ingest path only: the MLIP outcome is a stand-in, and the QE settings are
# far below any converged setup. A zero exit means neb.x was launched and its output was ingested,
# not that a barrier was computed or validated.
#
# Submit:
#   sbatch -A <ACCOUNT> deployments/frontier/jobs/job-neb-escalation-smoke-frontier.sh
#
# Override at submit time: PROJECT_ROOT, QE_PREFIX, NEB_BIN, MATSIM_PSEUDO_DIR, NEB_SMOKE_DIR.
# ---------------------------------------------------------------------------

set -euo pipefail

# Slurm runs a spooled copy of this script, so fall back to the submit directory.
REPO="${PROJECT_ROOT:-$(d="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)"; [[ -f "$d/pyproject.toml" ]] && echo "$d" || echo "${SLURM_SUBMIT_DIR:-$d}")}"
PROJ="$(dirname "${REPO}")"
QE_PREFIX="${QE_PREFIX:-${REPO}/external/quantum-espresso}"
NEB_BIN="${NEB_BIN:-${QE_PREFIX}/install-gpu/bin/neb.x}"
PSEUDO_DIR="${MATSIM_PSEUDO_DIR:-${PROJ}/cache/pseudopotentials/sssp-1.3.0-pbe-precision}"
WORK_DIR="${NEB_SMOKE_DIR:-${REPO}/.hpc-build/frontier/neb-escalation-smoke/${SLURM_JOB_ID:-local}}"
RANKS="${NEB_SMOKE_RANKS:-8}"
THREADS="${OMP_NUM_THREADS:-7}"
LAUNCHER="${REPO}/deployments/frontier/launchers/_qe-neb-step-frontier.sh"
PYTHON="${REPO}/.venv/bin/python"

[[ -x "${NEB_BIN}" ]] || { echo "neb.x not found: ${NEB_BIN}" >&2; exit 2; }
[[ -f "${LAUNCHER}" ]] || { echo "launcher not found: ${LAUNCHER}" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }
[[ -d "${PSEUDO_DIR}" ]] || { echo "pseudopotential dir not found: ${PSEUDO_DIR}" >&2; exit 2; }
mkdir -p "${WORK_DIR}"

echo "NEB escalation smoke on $(hostname), job ${SLURM_JOB_ID:-local}"
echo "neb.x: ${NEB_BIN}"
echo "pseudo dir: ${PSEUDO_DIR}"
echo "work dir: ${WORK_DIR}"
echo "scope: launch/ingest plumbing only; not a physical or numerical validation"

export XALT_EXECUTABLE_TRACKING=no
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" "${REPO}/deployments/common/neb_escalation_smoke.py" \
  --workdir "${WORK_DIR}/run" \
  --pseudo-dir "${PSEUDO_DIR}" \
  --neb-bin "${NEB_BIN}" \
  --launcher bash "${LAUNCHER}" "${RANKS}" "${THREADS}" \
  --nimage "${NEB_SMOKE_NIMAGE:-1}"

echo "NEB escalation smoke finished; summary: ${WORK_DIR}/run/escalation-summary.json"
