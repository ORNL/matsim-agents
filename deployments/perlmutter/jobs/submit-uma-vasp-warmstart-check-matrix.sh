#!/usr/bin/env bash
# Submit a repeatable UMA+VASP warm-start benchmark matrix on Perlmutter.
#
# Mirror of submit-uma-warmstart-check-matrix.sh for the VASP backend.
#
# Results land in:
#   $RUNS_ROOT/uma-vasp-warmstart-<jobid>/
#       uma-vasp-warmstart/
#           test_uma_warmstart_helps_0/
#               <FixtureName>/
#                   cold/  warm/  uma/  comparison.json
#       uma-vasp-warmstart.log

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: submit-uma-vasp-warmstart-check-matrix.sh

Environment overrides:
  PROJECT_ROOT, RUNS_ROOT, FIXTURE, REPEATS, TIME_LIMIT, QOS, ACCOUNT
  MATSIM_VASP_POTCAR_DIR, MATSIM_VASP_LAUNCHER, MATSIM_VASP_TIMEOUT_SEC
  MATSIM_VASP_MLP_DEVICE, MATSIM_UMA_MODEL_NAME, MATSIM_UMA_TASK
  MATSIM_FAIRCHEM_VENV

Example:
  REPEATS=3 FIXTURE=MoNbTaW_HEA \
  deployments/perlmutter/jobs/submit-uma-vasp-warmstart-check-matrix.sh
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
JOB_SCRIPT="${REPO_ROOT}/deployments/perlmutter/jobs/job-uma-vasp-warmstart-perlmutter.sh"

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "ERROR: job script not found: ${JOB_SCRIPT}" >&2
  exit 2
fi

PROJ="$(dirname "${REPO_ROOT}")"

RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
FIXTURE="${FIXTURE:-MoNbTaW_HEA}"
REPEATS="${REPEATS:-3}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
QOS="${QOS:-regular}"
ACCOUNT="${ACCOUNT:-m5216}"

POTCAR_DIR="${MATSIM_VASP_POTCAR_DIR:-${REPO_ROOT}/external/vasp6/potcar/potpaw_PBE.64}"
VASP_LAUNCHER="${MATSIM_VASP_LAUNCHER:-${REPO_ROOT}/deployments/perlmutter/launchers/run-vasp-gpu-perlmutter.sh}"
VASP_TIMEOUT="${MATSIM_VASP_TIMEOUT_SEC:-3600}"
MLP_DEVICE="${MATSIM_VASP_MLP_DEVICE:-cuda}"
UMA_MODEL="${MATSIM_UMA_MODEL_NAME:-uma-s-1p1}"
UMA_TASK="${MATSIM_UMA_TASK:-omat}"

VENV_ROOT="${PROJ}/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter"
FAIRCHEM_VENV="${MATSIM_FAIRCHEM_VENV:-${VENV_ROOT}/fairchem_venv}"

for req in "${POTCAR_DIR}" "${VASP_LAUNCHER}" "${FAIRCHEM_VENV}"; do
  if [[ ! -e "${req}" ]]; then
    echo "ERROR: required path is missing: ${req}" >&2
    exit 2
  fi
done

mkdir -p "${RUNS_ROOT}"

HF_HOME_DIR="${PROJ}/models/hf_cache"
mkdir -p "${HF_HOME_DIR}"

echo "Submitting UMA+VASP warm-start matrix"
echo "  repo:          ${REPO_ROOT}"
echo "  runs root:     ${RUNS_ROOT}"
echo "  fixture:       ${FIXTURE}"
echo "  repeats:       ${REPEATS}"
echo "  uma model:     ${UMA_MODEL}"
echo "  uma task:      ${UMA_TASK}"
echo "  fairchem venv: ${FAIRCHEM_VENV}"
echo ""

for ((i = 1; i <= REPEATS; i++)); do
  job_name="uma-vasp-ws-${i}"
  export_vars="ALL,PROJECT_ROOT=${REPO_ROOT},RUNS_ROOT=${RUNS_ROOT}"
  export_vars+=",MATSIM_WARMSTART_FIXTURES=${FIXTURE}"
  export_vars+=",MATSIM_VASP_POTCAR_DIR=${POTCAR_DIR}"
  export_vars+=",MATSIM_VASP_LAUNCHER=${VASP_LAUNCHER}"
  export_vars+=",MATSIM_VASP_TIMEOUT_SEC=${VASP_TIMEOUT}"
  export_vars+=",MATSIM_VASP_MLP_DEVICE=${MLP_DEVICE}"
  export_vars+=",MATSIM_UMA_MODEL_NAME=${UMA_MODEL}"
  export_vars+=",MATSIM_UMA_TASK=${UMA_TASK}"
  export_vars+=",MATSIM_FAIRCHEM_VENV=${FAIRCHEM_VENV}"
  export_vars+=",HF_HOME=${HF_HOME_DIR}"
  export_vars+=",FAIRCHEM_CACHE_DIR=${FAIRCHEM_CACHE_DIR:-${SCRATCH:-/tmp}/matsim-agents/fairchem_cache}"

  jid="$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --qos="${QOS}" \
    --time="${TIME_LIMIT}" \
    --job-name="${job_name}" \
    --export="${export_vars}" \
    "${JOB_SCRIPT}")"

  echo "submitted ${job_name}: ${jid}"
done

echo ""
echo "Results will appear under:"
echo "  ${RUNS_ROOT}/uma-vasp-warmstart-<jobid>/uma-vasp-warmstart/<fixture>/comparison.json"
echo ""
echo "Queue status:"
echo "  squeue -u ${USER} | grep 'uma-vasp-ws-'"
echo ""
echo "After completion, evaluate thresholds:"
echo "  python ${REPO_ROOT}/scripts/diagnostics/check_warmstart_thresholds.py \\"
echo "    --runs-root ${RUNS_ROOT} --fixture ${FIXTURE} --energy-tol-ev 0.05 --backend vasp"
