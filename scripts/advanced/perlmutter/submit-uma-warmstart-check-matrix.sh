#!/usr/bin/env bash
# Submit a repeatable UMA warm-start benchmark matrix on Perlmutter.
#
# Each submitted job runs the UMA integration benchmark, which executes
# both cold and warm QE vc-relax and writes comparison.json artifacts.
#
# Unlike the HydraGNN matrix, there is no routing variant to sweep —
# UMA always uses the same model (uma-s-1p1) and task (omat).

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: submit-uma-warmstart-check-matrix.sh

Environment overrides:
  PROJECT_ROOT, RUNS_ROOT, FIXTURE, REPEATS, TIME_LIMIT, QOS, ACCOUNT
  MATSIM_QE_PSEUDO_DIR, MATSIM_QE_LAUNCHER, MATSIM_QE_TIMEOUT_SEC
  MATSIM_QE_MLP_DEVICE, MATSIM_UMA_MODEL_NAME, MATSIM_UMA_TASK
  MATSIM_FAIRCHEM_VENV   # override fairchem_venv path

Example:
  REPEATS=3 FIXTURE=MoNbTaW_HEA QOS=premium ACCOUNT=m5216_g \
  scripts/advanced/perlmutter/submit-uma-warmstart-check-matrix.sh
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
JOB_SCRIPT="${REPO_ROOT}/scripts/advanced/perlmutter/job-uma-warmstart-perlmutter.sh"

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
ACCOUNT="${ACCOUNT:-amsc001}"

PSEUDO_DIR="${MATSIM_QE_PSEUDO_DIR:-${REPO_ROOT}/external/quantum-espresso/src/pseudo}"
QE_LAUNCHER="${MATSIM_QE_LAUNCHER:-${REPO_ROOT}/scripts/launchers/perlmutter/run-pw-gpu-perlmutter.sh}"
QE_TIMEOUT="${MATSIM_QE_TIMEOUT_SEC:-3600}"
MLP_DEVICE="${MATSIM_QE_MLP_DEVICE:-cuda}"
UMA_MODEL="${MATSIM_UMA_MODEL_NAME:-uma-s-1p1}"
UMA_TASK="${MATSIM_UMA_TASK:-omat}"

VENV_ROOT="${PROJ}/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter"
FAIRCHEM_VENV="${MATSIM_FAIRCHEM_VENV:-${VENV_ROOT}/fairchem_venv}"

for req in "${PSEUDO_DIR}" "${QE_LAUNCHER}" "${FAIRCHEM_VENV}"; do
  if [[ ! -e "${req}" ]]; then
    echo "ERROR: required path is missing: ${req}" >&2
    exit 2
  fi
done

mkdir -p "${RUNS_ROOT}"

# Shared HF cache so the model is downloaded once and reused across jobs.
HF_HOME_DIR="${PROJ}/models/hf_cache"
mkdir -p "${HF_HOME_DIR}"

# facebook/UMA is gated — ensure HF_TOKEN is available for the SLURM job.
# Prefer an explicit env var; fall back to the cached login token.
if [[ -z "${HF_TOKEN:-}" ]]; then
  _TOKEN_FILE="${HOME}/.cache/huggingface/token"
  if [[ -f "${_TOKEN_FILE}" ]]; then
    HF_TOKEN="$(< "${_TOKEN_FILE}")"
  else
    echo "WARNING: HF_TOKEN is unset and no cached token found." >&2
    echo "         UMA jobs will fail to download the model." >&2
  fi
fi

echo "Submitting UMA warm-start matrix"
echo "  repo:        ${REPO_ROOT}"
echo "  runs root:   ${RUNS_ROOT}"
echo "  fixture:     ${FIXTURE}"
echo "  repeats:     ${REPEATS}"
echo "  uma model:   ${UMA_MODEL}"
echo "  uma task:    ${UMA_TASK}"
echo "  fairchem venv: ${FAIRCHEM_VENV}"
echo ""

for ((i = 1; i <= REPEATS; i++)); do
  job_name="uma-ws-${i}"
  export_vars="ALL,PROJECT_ROOT=${REPO_ROOT},RUNS_ROOT=${RUNS_ROOT}"
  export_vars+=",MATSIM_WARMSTART_FIXTURES=${FIXTURE}"
  export_vars+=",MATSIM_QE_PSEUDO_DIR=${PSEUDO_DIR}"
  export_vars+=",MATSIM_QE_LAUNCHER=${QE_LAUNCHER}"
  export_vars+=",MATSIM_QE_TIMEOUT_SEC=${QE_TIMEOUT}"
  export_vars+=",MATSIM_QE_MLP_DEVICE=${MLP_DEVICE}"
  export_vars+=",MATSIM_UMA_MODEL_NAME=${UMA_MODEL}"
  export_vars+=",MATSIM_UMA_TASK=${UMA_TASK}"
  export_vars+=",MATSIM_FAIRCHEM_VENV=${FAIRCHEM_VENV}"
  export_vars+=",HF_HOME=${HF_HOME_DIR}"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export_vars+=",HF_TOKEN=${HF_TOKEN}"
  fi

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
echo "Next checks"
echo "  Queue status:"
echo "    squeue -u ${USER} | grep 'uma-ws-'"
echo ""
echo "  After completion, evaluate thresholds:"
echo "    python ${REPO_ROOT}/scripts/diagnostics/check_warmstart_thresholds.py \\"
echo "      --runs-root ${RUNS_ROOT} --fixture ${FIXTURE} --energy-tol-ev 0.05"
