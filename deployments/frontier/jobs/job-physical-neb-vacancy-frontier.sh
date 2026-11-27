#!/bin/bash
#SBATCH -J physical-neb-vacancy
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 01:30:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# ---------------------------------------------------------------------------
# Vacancy-migration NEB with QE on one Frontier node: unit-cell vc-relax for the lattice constant,
# fixed-cell relaxation of both hop endpoints, then a climbing-image neb.x run.
#
# The default system is fcc Al in a 2x2x2 conventional supercell. The hop distance is derived from
# the structure, so MATSIM_NEB_STRUCTURE / MATSIM_NEB_ELEMENT / MATSIM_NEB_PSEUDO_FILE select
# another cubic crystal.
#
# The result is a single-setting barrier. Cutoff, k-point, supercell and image-count convergence
# are not tested, so it is at best converged_unqualified and not a validated barrier.
#
# Submit:
#   sbatch -A <ACCOUNT> deployments/frontier/jobs/job-physical-neb-vacancy-frontier.sh
#
# Override at submit time: PROJECT_ROOT, QE_PREFIX, MATSIM_PSEUDO_DIR, PHYS_NEB_DIR,
# MATSIM_NEB_STRUCTURE, MATSIM_NEB_ELEMENT, MATSIM_NEB_PSEUDO_FILE, MATSIM_NEB_LATTICE_GUESS,
# MATSIM_NEB_ARGS (extra arguments for the driver).
# ---------------------------------------------------------------------------

set -euo pipefail

# Slurm runs a spooled copy of this script, so fall back to the submit directory.
REPO="${PROJECT_ROOT:-$(d="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)"; [[ -f "$d/pyproject.toml" ]] && echo "$d" || echo "${SLURM_SUBMIT_DIR:-$d}")}"
PROJ="$(dirname "${REPO}")"
QE_PREFIX="${QE_PREFIX:-${REPO}/external/quantum-espresso}"
PW_BIN="${PW_BIN:-${QE_PREFIX}/install-gpu/bin/pw.x}"
NEB_BIN="${NEB_BIN:-${QE_PREFIX}/install-gpu/bin/neb.x}"
PSEUDO_DIR="${MATSIM_PSEUDO_DIR:-${PROJ}/cache/pseudopotentials/sssp-1.3.0-pbe-precision}"
WORK_DIR="${PHYS_NEB_DIR:-${REPO}/.hpc-build/frontier/physical-neb-vacancy/${SLURM_JOB_ID:-local}}"
RANKS="${PHYS_NEB_RANKS:-8}"
THREADS="${OMP_NUM_THREADS:-7}"
STRUCTURE="${MATSIM_NEB_STRUCTURE:-fcc}"
ELEMENT="${MATSIM_NEB_ELEMENT:-Al}"
PSEUDO_FILE="${MATSIM_NEB_PSEUDO_FILE:-Al.pbe-n-kjpaw_psl.1.0.0.UPF}"
LATTICE_GUESS="${MATSIM_NEB_LATTICE_GUESS:-4.05}"
PW_LAUNCHER="${REPO}/deployments/frontier/launchers/_qe-step-frontier.sh"
NEB_LAUNCHER="${REPO}/deployments/frontier/launchers/_qe-neb-step-frontier.sh"
PYTHON="${REPO}/.venv/bin/python"

[[ -x "${PW_BIN}" ]] || { echo "pw.x not found: ${PW_BIN}" >&2; exit 2; }
[[ -x "${NEB_BIN}" ]] || { echo "neb.x not found: ${NEB_BIN}" >&2; exit 2; }
[[ -f "${PW_LAUNCHER}" ]] || { echo "launcher not found: ${PW_LAUNCHER}" >&2; exit 2; }
[[ -f "${NEB_LAUNCHER}" ]] || { echo "launcher not found: ${NEB_LAUNCHER}" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }
[[ -f "${PSEUDO_DIR}/${PSEUDO_FILE}" ]] || { echo "pseudopotential not found: ${PSEUDO_DIR}/${PSEUDO_FILE}" >&2; exit 2; }
mkdir -p "$(dirname "${WORK_DIR}")"

echo "Vacancy-migration NEB on $(hostname), job ${SLURM_JOB_ID:-local}"
echo "system: ${ELEMENT} ${STRUCTURE}; pw.x: ${PW_BIN}; neb.x: ${NEB_BIN}"
echo "pseudo: ${PSEUDO_DIR}/${PSEUDO_FILE}"
echo "work dir: ${WORK_DIR}"
echo "scope: single-setting barrier; not converged in cutoff, k-points, supercell or images"

export XALT_EXECUTABLE_TRACKING=no
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

# shellcheck disable=SC2086
"${PYTHON}" "${REPO}/scripts/neb/physical_neb_vacancy.py" \
  --workdir "${WORK_DIR}" \
  --pseudo-dir "${PSEUDO_DIR}" \
  --pseudo-file "${PSEUDO_FILE}" \
  --pw-bin "${PW_BIN}" \
  --neb-bin "${NEB_BIN}" \
  --element "${ELEMENT}" \
  --structure "${STRUCTURE}" \
  --lattice-guess "${LATTICE_GUESS}" \
  --pw-launcher bash "${PW_LAUNCHER}" "{work_dir}" "${PW_BIN}" "{input}" 1 "${RANKS}" "${THREADS}" \
  --neb-launcher bash "${NEB_LAUNCHER}" "${RANKS}" "${THREADS}" \
  ${MATSIM_NEB_ARGS:-}

echo "Vacancy NEB finished; summary: ${WORK_DIR}/physical-neb-summary.json"
