#!/usr/bin/env bash
# Submit a repeatable warm-start check matrix on Perlmutter — VASP backend.
#
# Mirror of submit-qe-warmstart-check-matrix.sh for VASP.
#
# HydraGNN uses only branch-7 (OMat24).
#
# Results land in:
#   $RUNS_ROOT/vasp-warmstart-<jobid>/          ← top-level run dir
#       vasp-warmstart/                          ← pytest --basetemp
#           test_hydragnn_warmstart_helps_0/
#               <FixtureName>/
#                   cold/  warm/  hydragnn/  comparison.json
#       vasp-warmstart.log
#
# This is intentionally separate from qe-warmstart-<jobid>/ so QE and VASP
# results can be analysed independently or compared side-by-side under the
# same $RUNS_ROOT.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: submit-vasp-warmstart-check-matrix.sh

Environment overrides:
  PROJECT_ROOT, RUNS_ROOT, FIXTURE, REPEATS, TIME_LIMIT, QOS, ACCOUNT
  MATSIM_VASP_POTCAR_DIR, MATSIM_HYDRAGNN_LOGDIR
  MATSIM_VASP_LAUNCHER, MATSIM_VASP_TIMEOUT_SEC, MATSIM_VASP_MLP_DEVICE

Example:
  REPEATS=3 FIXTURE=MoNbTaW_HEA \
  deployments/perlmutter/jobs/submit-vasp-warmstart-check-matrix.sh
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
JOB_SCRIPT="${REPO_ROOT}/deployments/perlmutter/jobs/job-vasp-warmstart-perlmutter.sh"

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

POTCAR_DIR="${MATSIM_VASP_POTCAR_DIR:-${REPO_ROOT}/external/vasp6/potcar/potpaw_PBE.64}"
source "${REPO_ROOT}/deployments/common/hydragnn-env.sh"
HYDRAGNN_LOGDIR="${MATSIM_HYDRAGNN_LOGDIR:-${HYDRAGNN_BEST6_LOGDIR}}"
VASP_LAUNCHER="${MATSIM_VASP_LAUNCHER:-${REPO_ROOT}/deployments/perlmutter/launchers/run-vasp-gpu-perlmutter.sh}"
VASP_TIMEOUT="${MATSIM_VASP_TIMEOUT_SEC:-3600}"
MLP_DEVICE="${MATSIM_VASP_MLP_DEVICE:-cuda}"

for req in "${POTCAR_DIR}" "${HYDRAGNN_LOGDIR}" "${VASP_LAUNCHER}"; do
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
    job_name="vasp-ws-${label}-${i}"
    export_vars="ALL,PROJECT_ROOT=${REPO_ROOT},RUNS_ROOT=${RUNS_ROOT},MATSIM_WARMSTART_FIXTURES=${FIXTURE},MATSIM_VASP_POTCAR_DIR=${POTCAR_DIR},MATSIM_HYDRAGNN_LOGDIR=${HYDRAGNN_LOGDIR},MATSIM_VASP_LAUNCHER=${VASP_LAUNCHER},MATSIM_VASP_TIMEOUT_SEC=${VASP_TIMEOUT},MATSIM_VASP_MLP_DEVICE=${MLP_DEVICE}"

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

echo "Submitting VASP warm-start check matrix"
echo "  repo:      ${REPO_ROOT}"
echo "  runs root: ${RUNS_ROOT}"
echo "  fixture:   ${FIXTURE}"
echo "  repeats:   ${REPEATS}"
echo ""

submit_group "omat24"

echo ""
echo "Results will appear under:"
echo "  ${RUNS_ROOT}/vasp-warmstart-<jobid>/vasp-warmstart/<fixture>/comparison.json"
echo ""
echo "Queue status:"
echo "  squeue -u ${USER} | grep 'vasp-ws-'"
echo ""
echo "After completion, evaluate thresholds:"
echo "  python ${REPO_ROOT}/scripts/diagnostics/check_warmstart_thresholds.py \\"
echo "    --runs-root ${RUNS_ROOT} --fixture ${FIXTURE} --energy-tol-ev 0.05 --backend vasp"
