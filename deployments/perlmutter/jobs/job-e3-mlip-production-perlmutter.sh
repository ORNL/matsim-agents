#!/bin/bash
#SBATCH -J e3-mlip
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 13:53:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --array=0-35%4
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
[[ -f "${REPO}/pyproject.toml" ]] || {
  echo "ERROR: PROJECT_ROOT is not a matsim-agents checkout: ${REPO}" >&2
  exit 2
}
: "${SLURM_JOB_ID:?submit this script through Slurm}"
[[ "${SLURM_JOB_ACCOUNT:-}" == "m5216_g" ]] || {
  echo "ERROR: this campaign requires Perlmutter account m5216_g" >&2
  exit 2
}
[[ "${MATSIM_E3_MLIP_RESOURCE_AUTHORIZED:-}" == "m5216:regular:2000" ]] || {
  echo "ERROR: explicit authorization required: MATSIM_E3_MLIP_RESOURCE_AUTHORIZED=m5216:regular:2000" >&2
  exit 2
}

BASE_PYTHON="${MATSIM_E3_MLIP_BASE_PYTHON:-${REPO}/.venv/bin/python}"
MACE_PYTHON="${MATSIM_E3_MLIP_WORKER_PYTHON:-${REPO}/.venv-mace/bin/python}"
[[ -x "${BASE_PYTHON}" ]] || { echo "ERROR: missing ${BASE_PYTHON}" >&2; exit 2; }
[[ -x "${MACE_PYTHON}" ]] || { echo "ERROR: missing ${MACE_PYTHON}" >&2; exit 2; }
export PATH="${HOME}/.local/bin:${PATH}"
"${BASE_PYTHON}" -c 'import copilot' || {
  echo "ERROR: Copilot SDK is missing from ${BASE_PYTHON}" >&2
  exit 2
}
command -v copilot >/dev/null || { echo "ERROR: Copilot CLI is not on PATH" >&2; exit 2; }
MANIFEST="${MATSIM_E3_MLIP_MANIFEST:-${REPO}/benchmarks/e3/manifests/e3-mlip-production-v1.json}"
OUTPUT_ROOT="${MATSIM_E3_MLIP_OUTPUT_ROOT:-${REPO}/runs/e3-mlip-production-v1}"
[[ -s "${MANIFEST}" ]] || { echo "ERROR: missing production manifest: ${MANIFEST}" >&2; exit 2; }
[[ "${SLURM_NNODES:-}" == "1" ]] || {
  echo "ERROR: this allocation is capped for exactly one GPU node per array element" >&2
  exit 2
}
mkdir -p "${OUTPUT_ROOT}"
RESOURCE_STATE="${REPO}/runs/e3-mlip-production-v1/resource-budget.json"
RESOURCE_START=${SECONDS}
export PYTHONPATH="${REPO}:${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${BASE_PYTHON}" -m benchmarks.e3.production budget-reserve \
  --state "${RESOURCE_STATE}" \
  --reservation-id "${SLURM_JOB_ID}" \
  --reserved-gpu-seconds 199920 \
  --cap-gpu-seconds 7200000

release_resource_budget() {
  local status=$?
  local runtime_seconds=$((SECONDS - RESOURCE_START + 60))
  trap - EXIT
  if ! "${BASE_PYTHON}" -m benchmarks.e3.production budget-release \
    --state "${RESOURCE_STATE}" \
    --reservation-id "${SLURM_JOB_ID}" \
    --actual-gpu-seconds "$((runtime_seconds * 4))"; then
    echo "ERROR: failed to reconcile the GPU-hour reservation; cap remains conservatively reserved" >&2
    status=1
  fi
  exit "${status}"
}
trap release_resource_budget EXIT

source "${REPO}/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
source "${REPO}/deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
configure_mace_model_artifacts "${REPO}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MATSIM_E3_MLIP_WORKERS=4
export MATSIM_E3_MLIP_WORKER_PYTHON="${MACE_PYTHON}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?array task id is required}"
MATERIALS=(bcc_refractory fcc_cantor lifepo4 phosphorene nb_ta_o zn_formate_mof)
STRATEGIES=(rule_based_adaptation llm_guided_adaptation)
SEEDS=(20261111 20261112 20261113)
material_index=$((TASK_ID / 6))
pair_index=$((TASK_ID % 6))
strategy_index=$((pair_index / 3))
seed_index=$((pair_index % 3))
MATERIAL="${MATERIALS[material_index]}"
STRATEGY="${STRATEGIES[strategy_index]}"
SEED="${SEEDS[seed_index]}"

source_revision="$(git -C "${REPO}" rev-parse HEAD)"
{
  echo "campaign=e3-mlip-production-v1"
  echo "job_id=${SLURM_JOB_ID}"
  echo "array_job_id=${SLURM_ARRAY_JOB_ID:-}"
  echo "array_task_id=${TASK_ID}"
  echo "material=${MATERIAL}"
  echo "strategy=${STRATEGY}"
  echo "seed=${SEED}"
  echo "source_revision=${source_revision}"
  echo "account=m5216_g"
  echo "submission_account=m5216"
  echo "qos=gpu_regular"
  echo "nodes=${SLURM_NNODES}"
  echo "gpus_per_node=4"
  echo "MACE_ARTIFACT_DIR=${MACE_ARTIFACT_DIR}"
  echo "dft_outputs_used=false"
  echo "dft_labels_used=false"
} | tee "${OUTPUT_ROOT}/slurm-${SLURM_JOB_ID}-${TASK_ID}.txt"

"${BASE_PYTHON}" -m benchmarks.e3.production run-replicate \
  --manifest "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --material "${MATERIAL}" \
  --strategy "${STRATEGY}" \
  --seed "${SEED}" \
  --workers 4 \
  --workers-per-node 4 \
  --worker-python "${MACE_PYTHON}"
