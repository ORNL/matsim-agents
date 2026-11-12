#!/bin/bash
#SBATCH -J e3-mlip
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -c 64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
[[ -f "${REPO}/pyproject.toml" ]] || {
  echo "ERROR: PROJECT_ROOT is not a matsim-agents checkout: ${REPO}" >&2
  exit 2
}
[[ -x "${REPO}/.venv/bin/python" ]] || {
  echo "ERROR: missing base Python environment: ${REPO}/.venv/bin/python" >&2
  exit 2
}
[[ -x "${REPO}/.venv-mace/bin/python" ]] || {
  echo "ERROR: missing MACE Python environment: ${REPO}/.venv-mace/bin/python" >&2
  exit 2
}
: "${SLURM_JOB_ID:?submit this script with sbatch}"

BASE_PYTHON="${REPO}/.venv/bin/python"
MACE_PYTHON="${REPO}/.venv-mace/bin/python"
RUNS_ROOT="${MATSIM_RUNS_ROOT:-${REPO}/runs}"
RUN_DIR="${RUNS_ROOT}/e3-perlmutter/preparation-${SLURM_JOB_ID}"
PREPARED_DIR="${RUN_DIR}/prepared"
PREDICTIONS_DIR="${RUN_DIR}/predictions"
THREADS="${MATSIM_E3_MLIP_THREADS:-8}"
MATERIALS=(lifepo4 bcc_refractory fcc_cantor phosphorene zn_formate_mof nb_ta_o)

mkdir -p "${RUN_DIR}" "${PREDICTIONS_DIR}"
cd "${REPO}"

source "${REPO}/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules
source "${REPO}/deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
configure_mace_model_artifacts "${REPO}"

export PYTHONPATH="${REPO}:${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" OPENBLAS_NUM_THREADS=1

git -C "${REPO}" rev-parse HEAD > "${RUN_DIR}/source-revision.txt"
git -C "${REPO}" status --short > "${RUN_DIR}/source-status.txt"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "repo=${REPO}"
  echo "run_dir=${RUN_DIR}"
  echo "model=mace-omat-0-medium"
  echo "device=cpu"
  echo "materials=${MATERIALS[*]}"
  echo "threads=${THREADS}"
  echo "mace_artifacts=${MACE_ARTIFACT_DIR}"
  echo "dft_outputs_used=false"
  echo "dft_labels_used=false"
} | tee "${RUN_DIR}/run-info.txt"

"${BASE_PYTHON}" -m benchmarks.e3.run prepare-candidates \
  --output "${PREPARED_DIR}" \
  --materials "${MATERIALS[@]}" \
  2>&1 | tee "${RUN_DIR}/prepare.log"

FAILED_MATERIALS=()
for material in "${MATERIALS[@]}"; do
  input="${PREPARED_DIR}/${material}/candidates.extxyz"
  output="${PREDICTIONS_DIR}/${material}.mace-omat-0-medium.json"
  if ! "${MACE_PYTHON}" -m benchmarks.e3.mlip_predict \
      --model mace-omat-0-medium \
      --input "${input}" \
      --output "${output}" \
      --device cpu \
      2>&1 | tee -a "${RUN_DIR}/predictions.log"; then
    echo "ERROR: MACE prediction failed for ${material}" \
      | tee -a "${RUN_DIR}/predictions.log"
    FAILED_MATERIALS+=("${material}")
  fi
done

if (( ${#FAILED_MATERIALS[@]} > 0 )); then
  printf 'ERROR: E3-MLIP predictions failed for: %s\n' "${FAILED_MATERIALS[*]}" >&2
  exit 1
fi

echo "E3-MLIP geometry-only predictions complete: ${RUN_DIR}"
