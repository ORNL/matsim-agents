#!/bin/bash
#SBATCH -J e3-mlip
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 20
#SBATCH -t 48:00:00
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
[[ "${MATSIM_E3_MLIP_RESOURCE_AUTHORIZED:-}" == "m5216_g:regular:38016" ]] || {
  echo "ERROR: explicit authorization required: MATSIM_E3_MLIP_RESOURCE_AUTHORIZED=m5216_g:regular:38016" >&2
  exit 2
}

BASE_PYTHON="${REPO}/.venv/bin/python"
MACE_PYTHON="${REPO}/.venv-mace/bin/python"
[[ -x "${BASE_PYTHON}" ]] || { echo "ERROR: missing ${BASE_PYTHON}" >&2; exit 2; }
[[ -x "${MACE_PYTHON}" ]] || { echo "ERROR: missing ${MACE_PYTHON}" >&2; exit 2; }
MANIFEST="${MATSIM_E3_MLIP_MANIFEST:-${REPO}/benchmarks/e3/manifests/e3-mlip-production-v1.json}"
OUTPUT_ROOT="${MATSIM_E3_MLIP_OUTPUT_ROOT:-${REPO}/runs/e3-mlip-production-v1}"
[[ -s "${MANIFEST}" ]] || { echo "ERROR: missing production manifest: ${MANIFEST}" >&2; exit 2; }

source "${REPO}/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
source "${REPO}/deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
configure_mace_model_artifacts "${REPO}"

export PYTHONPATH="${REPO}:${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MATSIM_E3_MLIP_WORKERS=80
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
  echo "qos=regular"
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
  --workers 80 \
  --workers-per-node 4 \
  --worker-python "${MACE_PYTHON}"
