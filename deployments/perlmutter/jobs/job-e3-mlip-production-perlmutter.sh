#!/bin/bash
#SBATCH -J e3-mlip
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 16
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5%5
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}}"
[[ -f "${REPO}/pyproject.toml" ]] || {
  echo "ERROR: PROJECT_ROOT is not a matsim-agents checkout: ${REPO}" >&2
  exit 2
}
: "${SLURM_JOB_ID:?submit this script through Slurm}"
[[ "${SLURM_JOB_ACCOUNT:-}" == "m5216_g" ]] || {
  echo "ERROR: this campaign requires Perlmutter account m5216_g" >&2
  exit 2
}
[[ "${SLURM_JOB_QOS:-}" == "gpu_premium" ]] || {
  echo "ERROR: this campaign requires the gpu_premium QOS" >&2
  exit 2
}
[[ "${SLURM_NNODES:-}" == "16" && "${SLURM_NTASKS:-}" == "64" ]] || {
  echo "ERROR: each material campaign requires exactly 16 nodes and 64 tasks" >&2
  exit 2
}
[[ "${MATSIM_E3_MLIP_RESOURCE_AUTHORIZED:-}" == "m5216:premium:768" ]] || {
  echo "ERROR: explicit authorization required: MATSIM_E3_MLIP_RESOURCE_AUTHORIZED=m5216:premium:768" >&2
  exit 2
}

BASE_PYTHON="${MATSIM_E3_MLIP_BASE_PYTHON:-${REPO}/.venv/bin/python}"
MACE_PYTHON="${MATSIM_E3_MLIP_WORKER_PYTHON:-${REPO}/.venv-mace/bin/python}"
[[ -x "${BASE_PYTHON}" && -x "${MACE_PYTHON}" ]] || {
  echo "ERROR: prepared base and MACE Python environments are required" >&2
  exit 2
}
export PATH="${HOME}/.local/bin:${PATH}"
"${BASE_PYTHON}" -c 'import copilot' || {
  echo "ERROR: Copilot SDK is missing from ${BASE_PYTHON}" >&2
  exit 2
}
command -v copilot >/dev/null || {
  echo "ERROR: Copilot CLI is not on PATH" >&2
  exit 2
}

MANIFEST="${MATSIM_E3_MLIP_MANIFEST:-${REPO}/benchmarks/e3/manifests/e3-mlip-production.json}"
OUTPUT_ROOT="${MATSIM_E3_MLIP_OUTPUT_ROOT:-${REPO}/runs/e3-mlip-production}"
[[ -s "${MANIFEST}" ]] || { echo "ERROR: missing campaign manifest: ${MANIFEST}" >&2; exit 2; }
export PYTHONPATH="${REPO}:${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${BASE_PYTHON}" -m benchmarks.e3.production validate --manifest "${MANIFEST}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?array task id is required}"
MATERIALS=(bcc_refractory fcc_cantor lifepo4 phosphorene nb_ta_o zn_formate_mof)
[[ "${TASK_ID}" =~ ^[0-5]$ ]] || { echo "ERROR: invalid material array index" >&2; exit 2; }
MATERIAL="${MATERIALS[TASK_ID]}"
"${BASE_PYTHON}" -m benchmarks.e3.production verify-preflight \
  --manifest "${MANIFEST}" --output-root "${OUTPUT_ROOT}" --material "${MATERIAL}" \
  --round-index "${MATSIM_E3_MLIP_START_ROUND:-1}"
mkdir -p "${OUTPUT_ROOT}"
RESOURCE_STATE="${OUTPUT_ROOT}/resource-budget.json"
RESOURCE_START=${SECONDS}
"${BASE_PYTHON}" -m benchmarks.e3.production budget-reserve \
  --state "${RESOURCE_STATE}" \
  --reservation-id "${SLURM_JOB_ID}" \
  --reserved-gpu-seconds 460800 \
  --cap-gpu-seconds 2764800

cleanup_copilot_home() {
  local path="${COPILOT_HOME:-}"
  [[ -z "${path}" ]] && return 0
  local expected_prefix="${copilot_home_root}/matsim-copilot-${SLURM_JOB_ID}."
  if [[ "${path}" != "${expected_prefix}"* ]]; then
    echo "ERROR: refusing to remove unexpected Copilot home: ${path}" >&2
    return 1
  fi
  if ! rm -rf -- "${path}"; then
    echo "ERROR: failed to remove private Copilot home: ${path}" >&2
    return 1
  fi
}

release_resource_budget() {
  local status=$?
  local runtime_seconds=$((SECONDS - RESOURCE_START))
  trap - EXIT
  if ! "${BASE_PYTHON}" -m benchmarks.e3.production budget-release \
    --state "${RESOURCE_STATE}" \
    --reservation-id "${SLURM_JOB_ID}" \
    --actual-gpu-seconds "$((runtime_seconds * 64))"; then
    echo "ERROR: failed to reconcile GPU allocation; reservation remains conservatively charged" >&2
    status=1
  fi
  if ! cleanup_copilot_home; then
    status=1
  fi
  exit "${status}"
}
trap release_resource_budget EXIT

source "${REPO}/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
source "${REPO}/deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
configure_mace_model_artifacts "${REPO}"

copilot_home_root="${SLURM_TMPDIR:-/tmp}"
if [[ ! -d "${copilot_home_root}" || ! -w "${copilot_home_root}" ]]; then
  copilot_home_root="/tmp"
fi
COPILOT_HOME="$(mktemp -d "${copilot_home_root}/matsim-copilot-${SLURM_JOB_ID}.XXXXXX")"
export COPILOT_HOME
if [[ -r "${HOME}/.copilot/config.json" ]]; then
  cp -p "${HOME}/.copilot/config.json" "${COPILOT_HOME}/config.json"
  chmod 600 "${COPILOT_HOME}/config.json"
fi

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
{
  echo "campaign=e3-mlip-production"
  echo "job_id=${SLURM_JOB_ID}"
  echo "array_job_id=${SLURM_ARRAY_JOB_ID:-}"
  echo "array_task_id=${TASK_ID}"
  echo "material=${MATERIAL}"
  echo "source_revision=$(git -C "${REPO}" rev-parse HEAD)"
  echo "account=m5216_g"
  echo "qos=gpu_premium"
  echo "nodes=16"
  echo "gpus=64"
  echo "allocation_seconds=7200"
  echo "round_seconds=1800"
  echo "rounds=3"
  echo "hard_optimization_seconds=300"
  echo "strategy=llm_guided_adaptation"
  echo "seed=20261113"
  echo "dft_outputs_used=false"
  echo "dft_labels_used=false"
} | tee "${OUTPUT_ROOT}/slurm-${SLURM_JOB_ID}-${TASK_ID}.txt"

"${BASE_PYTHON}" -m benchmarks.e3.production run-campaign \
  --manifest "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --material "${MATERIAL}" \
  --workers 64 \
  --worker-python "${MACE_PYTHON}" \
  --hard-optimization-seconds 300
