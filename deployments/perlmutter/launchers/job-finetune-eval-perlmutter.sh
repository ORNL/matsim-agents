#!/bin/bash
#SBATCH -J ft-eval
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out
#SBATCH -t 04:00:00
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: one fine-tune + evaluation run of one MLIP on one dataset, on a
# single Perlmutter GPU.
#
# Runs matsim_agents.workflows.active_learning.finetune_eval, which fine-tunes the
# selected foundation model and reports the baseline ("before") and fine-tuned
# ("after") models on the declared validation/test partitions, plus a cost.json
# with wall time / GPU hours.
#
# GPU visibility is left to Slurm (a `shared` allocation with `-G 1` exports
# CUDA_VISIBLE_DEVICES for the single bound GPU).
#
# Submit the whole matrix with submit-finetune-eval-matrix-perlmutter.sh, or one
# run directly, e.g.:
#   MLIP_CONFIG=$PWD/examples/mlip/mace-mp-medium.yaml \
#   EVALUATION_CONFIG=$PWD/examples/evaluation/random-frame-interpolation.yaml \
#   CASE=lifepo4-al-001 \
#     sbatch deployments/perlmutter/launchers/job-finetune-eval-perlmutter.sh
#
# Required env:
#   MLIP_CONFIG        YAML whose `mlip:` block is the foundation model (backend,
#                      checkpoint, device, ...; examples/mlip/)
#   EVALUATION_CONFIG  evaluation YAML: objective, split, test domains,
#                      repetitions, selection, and the backend's `fine_tune:`
#                      recipe (docs/mlip-evaluation.md; examples/evaluation/)
#   CASE               dataset case dir under $RUNS_ROOT (contains dataset.extxyz)
# Optional env:
#   DATASET            default $RUNS_ROOT/$CASE/dataset.extxyz
#   ELEMENTAL_REFERENCE_MANIFEST  default $RUNS_ROOT/$CASE/elemental_references.json
#   OUT_DIR            default $FTEVAL_ROOT/<MLIP_CONFIG stem>/$CASE
#   FTEVAL_ROOT        default $RUNS_ROOT/finetune-eval
#   SEED               fine-tuning seed (default 0)
#   RESCORE=1          re-score an existing run without training (--eval-only)
#   PROJECT_ROOT / RUNS_ROOT / HYDRAGNN_ROOT, and the backend venv overrides
#   MATSIM_HYDRAGNN_VENV / MATSIM_FAIRCHEM_VENV / MATSIM_MACE_VENV
# ---------------------------------------------------------------------------
set -euo pipefail

# ── paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-${REPO_DEFAULT}}"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"

# ── required inputs ─────────────────────────────────────────────────────────
readable() {  # $1=variable name; resolve it to an absolute path (the job cds later)
  local value="${!1}"
  [[ -r "${value}" ]] || { echo "ERROR: $1 not readable: ${value}" >&2; exit 2; }
  printf -v "$1" '%s' "$(realpath "${value}")"
}
MLIP_CONFIG="${MLIP_CONFIG:?set MLIP_CONFIG=<YAML with an mlip: block>, see examples/mlip/}"
readable MLIP_CONFIG
EVALUATION_CONFIG="${EVALUATION_CONFIG:?set EVALUATION_CONFIG=<evaluation YAML>, see examples/evaluation/}"
readable EVALUATION_CONFIG
CASE="${CASE:?set CASE=<dataset case dir under RUNS_ROOT>}"
DATASET="${DATASET:-${RUNS_ROOT}/${CASE}/dataset.extxyz}"
[[ ! -f "${DATASET}" ]] && { echo "ERROR: dataset not found: ${DATASET}" >&2; exit 2; }
ELEMENTAL_REFERENCE_MANIFEST="${ELEMENTAL_REFERENCE_MANIFEST:-${RUNS_ROOT}/${CASE}/elemental_references.json}"
[[ ! -f "${ELEMENTAL_REFERENCE_MANIFEST}" ]] && {
  echo "ERROR: DFT-labelled elemental reference manifest not found: ${ELEMENTAL_REFERENCE_MANIFEST}" >&2
  exit 2
}
FTEVAL_ROOT="${FTEVAL_ROOT:-${RUNS_ROOT}/finetune-eval}"
VARIANT="$(basename "${MLIP_CONFIG}")"
OUT_DIR="${OUT_DIR:-${FTEVAL_ROOT}/${VARIANT%.*}/${CASE}}"
mkdir -p "${OUT_DIR}"

# ── modules + backend environment ──────────────────────────────────────────
# shellcheck disable=SC1091
source "${REPO}/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
# shellcheck disable=SC1091
source "${REPO}/deployments/perlmutter/setup/campaign-naming.sh"
# Any matsim Python can read the backend; the main venv has PyYAML.
BACKEND="$(campaign_mlip_backend "${REPO}" "${MATSIM_HYDRAGNN_VENV:-${REPO}/.venv}/bin/python3" "${MLIP_CONFIG}")"
case "${BACKEND}" in
  hydragnn) VENV="${MATSIM_HYDRAGNN_VENV:-${REPO}/.venv}" ;;
  uma)      VENV="${MATSIM_FAIRCHEM_VENV:-${REPO}/.venv-uma}" ;;
  mace)     VENV="${MATSIM_MACE_VENV:-${REPO}/.venv-mace}" ;;
  *) echo "ERROR: unsupported MLIP backend '${BACKEND}' in ${MLIP_CONFIG}" >&2; exit 2 ;;
esac
[[ ! -d "${VENV}" ]] && { echo "ERROR: venv not found: ${VENV}" >&2; exit 2; }
if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
else
  __conda_base="$(conda info --base 2>/dev/null)"
  [[ -z "${__conda_base}" ]] && { echo "ERROR: 'conda' not available to activate ${VENV}" >&2; exit 2; }
  # shellcheck disable=SC1091
  source "${__conda_base}/etc/profile.d/conda.sh"
  conda activate "${VENV}" || { echo "ERROR: conda activate failed for ${VENV}" >&2; exit 2; }
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

# shellcheck disable=SC1091
source "${REPO}/deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
configure_mlip_model_artifacts "${REPO}" python "${MLIP_CONFIG}"
case "${BACKEND}" in
  # HydraGNN package + example dir (inference_fused) on the path.
  hydragnn)
    source "${REPO}/deployments/common/hydragnn-env.sh"
    hydragnn_export_pythonpath
    ;;
  uma)
    if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
      HF_TOKEN="$(< "${HOME}/.cache/huggingface/token")"; export HF_TOKEN
    fi
    ;;
esac

EXTRA_ARGS=()
# Re-score an existing run with fixed elemental references, without training.
[[ "${RESCORE:-0}" == "1" ]] && EXTRA_ARGS+=(--eval-only)

echo "=========================================="
echo "[Perlmutter fine-tune + evaluation]"
echo "Date:        $(date)"
echo "Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "MLIP:        ${MLIP_CONFIG} (${BACKEND})"
echo "Evaluation:  ${EVALUATION_CONFIG}"
echo "Case:        ${CASE}"
echo "Dataset:     ${DATASET}"
echo "References:  ${ELEMENTAL_REFERENCE_MANIFEST}"
echo "Out dir:     ${OUT_DIR}"
echo "Venv:        ${VENV}"
echo "Python:      $(command -v python)"
echo "=========================================="

cd "${REPO}"
# Keep the run's inputs with its outputs (the matrix already writes them there).
[[ "${MLIP_CONFIG}" -ef "${OUT_DIR}/mlip.yaml" ]] || cp "${MLIP_CONFIG}" "${OUT_DIR}/mlip.yaml"
[[ "${EVALUATION_CONFIG}" -ef "${OUT_DIR}/evaluation.yaml" ]] \
  || cp "${EVALUATION_CONFIG}" "${OUT_DIR}/evaluation.yaml"
python -m matsim_agents.workflows.active_learning.finetune_eval \
  --mlip-config "${MLIP_CONFIG}" \
  --evaluation-config "${EVALUATION_CONFIG}" \
  --dataset "${DATASET}" \
  --elemental-reference-manifest "${ELEMENTAL_REFERENCE_MANIFEST}" \
  --output-dir "${OUT_DIR}" \
  --seed "${SEED:-0}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  2>&1 | tee "${OUT_DIR}/campaign.log"

echo "[$(date)] Run complete -> ${OUT_DIR}"
