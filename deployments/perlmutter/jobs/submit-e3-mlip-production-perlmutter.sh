#!/bin/bash
# Run on a non-GPU host: no GPU allocation is requested until actual generation passes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
[[ "${MATSIM_E3_MLIP_RESOURCE_AUTHORIZED:-}" == "m5216:premium:768" ]] || {
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
sbatch --parsable --array="$1-$1%1" --account=m5216 \
  "${SCRIPT_DIR}/job-e3-mlip-production-perlmutter.sh"
