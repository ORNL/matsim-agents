#!/bin/bash
#PBS -N al-dft-scaling
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=01:30:00
#PBS -l filesystems=home:flare
#PBS -q prod
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# matsim-agents: DFT-concurrency STRONG-SCALING harness on ALCF Aurora.
#
# Runs ONE active-learning iteration with a FIXED 32-job DFT workload
# (examples/paper_cases/al_hea_fcc_scaling_aurora.yaml, n_select=32,
# HydraGNN + VASP). The AL driver dispatches up to
# (allocated nodes / nodes_per_job=1) VASP single-points concurrently, so
# submitting this script at several node counts produces a strong-scaling curve
# for identical work.
#
# Submit the full sweep with the companion wrapper:
#   deployments/aurora/jobs/submit-al-dft-scaling-sweep.sh
# or a single point directly:
#   qsub -l select=4 deployments/aurora/jobs/job-al-dft-scaling-aurora.sh
#
# Each submission writes to a per-(N,repeat) RUN_TAG
# (hea-fcc-scaling-aurora-N<nodes>-r<rep>) so points never collide. REP/AL_SEED
# randomize the workload per repeat (fixed across node counts within a repeat).
#
# Paths default to the checkout containing this script and are env-overridable:
#   PROJECT_ROOT, RUNS_ROOT, HYDRAGNN_DIR.
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

# ── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}"
[[ -f "${REPO}/pyproject.toml" ]] || REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
export PROJ_ROOT="${REPO}"
export RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
export HYDRAGNN_DIR="${HYDRAGNN_DIR:-${PROJ}/HydraGNN}"

AL_CONFIG="${REPO}/examples/paper_cases/al_hea_fcc_scaling_aurora.yaml"
[[ -f "${AL_CONFIG}" ]] || { echo "ERROR: missing ${AL_CONFIG}" >&2; exit 2; }

# ── per-(N,repeat) run tag (consumed by the YAML via ${SCALE_TAG}) ───────────
NNODES="$(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | wc -l)"
[[ "${NNODES}" -ge 1 ]] || NNODES=1
REP="${REP:-1}"
export AL_SEED="${AL_SEED:-$((12 + REP))}"
export SCALE_TAG="hea-fcc-scaling-aurora-N${NNODES}-r${REP}"
RUN_DIR="${RUNS_ROOT}/al-scaling-aurora-N${NNODES}-r${REP}-${PBS_JOBID:-$$}"
mkdir -p "${RUN_DIR}"

export MLIP_BACKEND="${MLIP_BACKEND:-hydragnn}"
export DFT_BACKEND="${DFT_BACKEND:-vasp}"

# ── modules & venv ──────────────────────────────────────────────────────────
source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_frameworks_modules
VENV="${MATSIM_AURORA_VENV:-${REPO}/.venv}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONPATH="${HYDRAGNN_DIR}/examples/multidataset_hpo_sc26:${HYDRAGNN_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"

echo "=========================================="
echo "[Aurora AL DFT-scaling point]"
echo "Date:      $(date)"
echo "Job ID:    ${PBS_JOBID:-N/A}"
echo "Nodes:     ${NNODES}  (=> ${NNODES}-way DFT concurrency, nodes_per_job=1)"
echo "Repeat:    ${REP}  (AL_SEED=${AL_SEED})"
echo "SCALE_TAG: ${SCALE_TAG}"
echo "AL config: ${AL_CONFIG}"
echo "Run dir:   ${RUN_DIR}"
echo "=========================================="

# ── run one AL iteration ────────────────────────────────────────────────────
cd "${REPO}"
matsim-agents al validate-config "${AL_CONFIG}" >"${RUN_DIR}/al-config.resolved.json" 2>&1 || {
  echo "[ERROR] config validation failed; see ${RUN_DIR}/al-config.resolved.json" >&2
  tail -n 30 "${RUN_DIR}/al-config.resolved.json" >&2 || true
  exit 1
}

echo "[$(date)] Starting scaling point (N=${NNODES}) ..."
matsim-agents al run "${AL_CONFIG}" --log-level "${LOG_LEVEL:-INFO}" \
    2>&1 | tee "${RUN_DIR}/scaling.log"

echo "[$(date)] Scaling point N=${NNODES} complete. State under ${RUNS_ROOT}/${SCALE_TAG}."
