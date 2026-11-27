#!/usr/bin/env bash
# Submit a repeatable warm-start check matrix on Perlmutter.
#
# HydraGNN uses only branch-7 (OMat24).
#
# Each submitted job runs the existing integration benchmark, which executes
# both cold and warm QE vc-relax and writes comparison.json artifacts.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: submit-qe-warmstart-check-matrix.sh

Environment overrides:
  PROJECT_ROOT, RUNS_ROOT, FIXTURE, REPEATS, TIME_LIMIT, QOS, ACCOUNT
  MATSIM_QE_PSEUDO_DIR, MATSIM_HYDRAGNN_LOGDIR
  MATSIM_QE_LAUNCHER, MATSIM_QE_TIMEOUT_SEC, MATSIM_QE_MLP_DEVICE

Example:
  REPEATS=3 FIXTURE=MoNbTaW_HEA \
  deployments/perlmutter/jobs/submit-qe-warmstart-check-matrix.sh
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
JOB_SCRIPT="${REPO_ROOT}/deployments/perlmutter/jobs/job-qe-warmstart-perlmutter.sh"

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "ERROR: job script not found: ${JOB_SCRIPT}" >&2
  exit 2
fi

PROJ="$(dirname "${REPO_ROOT}")"

RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
FIXTURE="${FIXTURE:-MoNbTaW_HEA}"
REPEATS="${REPEATS:-5}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
QOS="${QOS:-regular}"
ACCOUNT="${ACCOUNT:?set ACCOUNT to your NERSC allocation}"

PSEUDO_DIR="${MATSIM_QE_PSEUDO_DIR:-${REPO_ROOT}/external/quantum-espresso/src/pseudo}"
HYDRAGNN_LOGDIR="${MATSIM_HYDRAGNN_LOGDIR:-${PROJ}/HydraGNN/examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64}"
QE_LAUNCHER="${MATSIM_QE_LAUNCHER:-${REPO_ROOT}/deployments/perlmutter/launchers/run-pw-gpu-perlmutter.sh}"
QE_TIMEOUT="${MATSIM_QE_TIMEOUT_SEC:-3600}"
MLP_DEVICE="${MATSIM_QE_MLP_DEVICE:-cuda}"

for req in "${PSEUDO_DIR}" "${HYDRAGNN_LOGDIR}" "${QE_LAUNCHER}"; do
  if [[ ! -e "${req}" ]]; then
    echo "ERROR: required path is missing: ${req}" >&2
    exit 2
  fi
done

mkdir -p "${RUNS_ROOT}"

submit_group() {
  local label="$1"
  local i job_name export_vars jid

  for ((i = 1; i <= REPEATS; i++)); do
    job_name="qe-ws-${label}-${i}"
    export_vars="ALL,PROJECT_ROOT=${REPO_ROOT},RUNS_ROOT=${RUNS_ROOT},MATSIM_WARMSTART_FIXTURES=${FIXTURE},MATSIM_QE_PSEUDO_DIR=${PSEUDO_DIR},MATSIM_HYDRAGNN_LOGDIR=${HYDRAGNN_LOGDIR},MATSIM_QE_LAUNCHER=${QE_LAUNCHER},MATSIM_QE_TIMEOUT_SEC=${QE_TIMEOUT},MATSIM_QE_MLP_DEVICE=${MLP_DEVICE}"

    jid="$(sbatch --parsable \
      --account="${ACCOUNT}" \
      --qos="${QOS}" \
      --time="${TIME_LIMIT}" \
      --job-name="${job_name}" \
      --export="${export_vars}" \
      "${JOB_SCRIPT}")"

    echo "submitted ${job_name}: ${jid}"
  done
}

echo "Submitting warm-start check matrix"
echo "  repo:      ${REPO_ROOT}"
echo "  runs root: ${RUNS_ROOT}"
echo "  fixture:   ${FIXTURE}"
echo "  repeats:   ${REPEATS}"
echo ""

submit_group "omat24"

echo ""
echo "Next checks"
echo "  Queue status:"
echo "    squeue -u ${USER} | grep 'qe-ws-'"
echo ""
echo "  After completion, evaluate thresholds:"
echo "    python ${REPO_ROOT}/scripts/diagnostics/check_warmstart_thresholds.py --runs-root ${RUNS_ROOT} --fixture ${FIXTURE} --energy-tol-ev 0.05"
