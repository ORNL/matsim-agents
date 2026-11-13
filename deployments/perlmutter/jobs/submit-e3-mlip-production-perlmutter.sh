#!/bin/bash
# Run on a non-GPU host: no GPU allocation is requested until actual generation passes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
export MATSIM_E3_MLIP_ACCOUNT="${MATSIM_E3_MLIP_ACCOUNT:-m5216}"
case "${MATSIM_E3_MLIP_ACCOUNT}" in
  m5216|m5048|m4828|amsc001|amsc004|dasrepo) ;;
  *) echo "ERROR: account is not authorized for this campaign" >&2; exit 2 ;;
esac
[[ "${MATSIM_E3_MLIP_RESOURCE_AUTHORIZED:-}" == "${MATSIM_E3_MLIP_ACCOUNT}:premium:768" ]] || {
  echo "ERROR: explicit resource authorization is required" >&2
  exit 2
}
[[ $# == 1 && "$1" =~ ^[0-5]$ ]] || {
  echo "Usage: bash submit-e3-mlip-production-perlmutter.sh MATERIAL_INDEX (0-5)" >&2
  exit 2
}
[[ -z "${SLURM_JOB_ID:-}" ]] || {
  echo "ERROR: execute CPU generation before requesting a GPU allocation" >&2
  exit 2
}
cd "${REPO}"
PYTHON="${MATSIM_E3_MLIP_BASE_PYTHON:-${REPO}/.venv/bin/python}"
MANIFEST="${MATSIM_E3_MLIP_MANIFEST:-${REPO}/benchmarks/e3/manifests/e3-mlip-production.json}"
OUTPUT_ROOT="${MATSIM_E3_MLIP_OUTPUT_ROOT:-${REPO}/runs/e3-mlip-production}"
MATERIALS=(bcc_refractory fcc_cantor lifepo4 phosphorene nb_ta_o zn_formate_mof)
MATERIAL="${MATERIALS[$1]}"
export PYTHONPATH="${REPO}:${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="${HOME}/.local/bin:${PATH}"
PRIVATE_COPILOT_HOME="$(mktemp -d /tmp/matsim-e3-preflight.XXXXXX)"
cleanup() {
  rm -f -- "${PRIVATE_COPILOT_HOME}/config.json"
  # The directory is a uniquely created, private local CLI runtime.
  rm -rf -- "${PRIVATE_COPILOT_HOME}"
}
trap cleanup EXIT
if [[ -r "${HOME}/.copilot/config.json" ]]; then
  cp -p "${HOME}/.copilot/config.json" "${PRIVATE_COPILOT_HOME}/config.json"
  chmod 600 "${PRIVATE_COPILOT_HOME}/config.json"
fi
export COPILOT_HOME="${PRIVATE_COPILOT_HOME}"
source "${REPO}/deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
configure_mace_model_artifacts "${REPO}"
"${PYTHON}" -m benchmarks.e3.production prepare \
  --manifest "${MANIFEST}" --output-root "${OUTPUT_ROOT}" --material "${MATERIAL}" \
  --round-index "${MATSIM_E3_MLIP_START_ROUND:-1}"
export MATSIM_E3_MLIP_RESOURCE_STATE="${MATSIM_E3_MLIP_RESOURCE_STATE:-${OUTPUT_ROOT}/resource-budget.json}"
export MATSIM_E3_MLIP_RESERVATION_ID="submission-$("${PYTHON}" -c 'from uuid import uuid4; print(uuid4().hex)')"
"${PYTHON}" -m benchmarks.e3.production budget-reserve \
  --state "${MATSIM_E3_MLIP_RESOURCE_STATE}" \
  --reservation-id "${MATSIM_E3_MLIP_RESERVATION_ID}" \
  --reserved-gpu-seconds 460800 --cap-gpu-seconds 2764800
# A failed/ambiguous submission retains its charge until an operator verifies
# whether Slurm created an allocation. Never release based only on shell exit.
echo "Reserved allocation: ${MATSIM_E3_MLIP_RESERVATION_ID}" >&2
SUBMITTED="$(sbatch --parsable --array="$1-$1%1" --account="${MATSIM_E3_MLIP_ACCOUNT}" \
  "${SCRIPT_DIR}/job-e3-mlip-production-perlmutter.sh")"
JOB_ID="${SUBMITTED%%;*}"
"${PYTHON}" -m benchmarks.e3.production budget-bind \
  --state "${MATSIM_E3_MLIP_RESOURCE_STATE}" \
  --reservation-id "${MATSIM_E3_MLIP_RESERVATION_ID}" --job-id "${JOB_ID}"
echo "${SUBMITTED}"
echo "After Slurm is terminal, run budget-reconcile with state=${MATSIM_E3_MLIP_RESOURCE_STATE} reservation-id=${MATSIM_E3_MLIP_RESERVATION_ID}" >&2
