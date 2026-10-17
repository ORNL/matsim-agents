#!/bin/bash
#PBS -N al-paper-case
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# matsim-agents: full active-learning loop for a single paper case on ALCF
# Aurora (Intel Data Center GPU Max).
#
#   MLIP (HydraGNN by default) MD sampling
#     -> mc_dropout uncertainty acquisition
#       -> VASP or QE single-points on selected structures (mpiexec steps)
#         -> append labelled structures -> retrain -> next iteration
#
# Runs `matsim-agents al run examples/paper_cases/al_<case>_aurora.yaml`.
#
# Select the case with CASE (default: hea_bcc):
#   qsub -v CASE=lifepo4        deployments/aurora/jobs/job-active-learning-paper-cases-aurora.sh
#   qsub -v CASE=hea_bcc        ...
#   qsub -v CASE=phosphorene    ...
#   qsub -v CASE=phosphorene_qe ...   (QE labelling)
#
# Override the MLIP backend (HydraGNN in .venv, UMA in .venv-uma, MACE in
# .venv-mace):
#   qsub -v CASE=hea_bcc,MLIP_BACKEND=uma ...
#
# MULTI-NODE DFT CONCURRENCY: the AL driver dispatches the selected
# single-points concurrently, up to (allocated nodes / nodes_per_job) at a time.
# Request more nodes for parallel DFT, e.g. `qsub -l select=4 ...`.
#
# Paths default to the checkout containing this script and are env-overridable:
#   PROJECT_ROOT, RUNS_ROOT, HYDRAGNN_DIR, HYDRAGNN_FT_REPO.
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

# ── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}"
[[ -f "${REPO}/pyproject.toml" ]] || REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
export PROJ_ROOT="${REPO}"
export RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
export HYDRAGNN_DIR="${HYDRAGNN_DIR:-${PROJ}/HydraGNN}"
export HYDRAGNN_FT_REPO="${HYDRAGNN_FT_REPO:-${PROJ}/HydraGNN_GFM_FineTuning4Materials}"

# ── case -> AL YAML mapping ─────────────────────────────────────────────────
CASE="${CASE:-hea_bcc}"
declare -A CASE_YAML=(
  [lifepo4]=al_lifepo4_aurora.yaml
  [hea_bcc]=al_hea_bcc_aurora.yaml
  [phosphorene]=al_phosphorene_aurora.yaml
  [phosphorene_qe]=al_phosphorene_qe_aurora.yaml
)
YAML_NAME="${CASE_YAML[$CASE]:-}"
if [[ -z "${YAML_NAME}" ]]; then
  echo "ERROR: unknown CASE='${CASE}'. Valid: ${!CASE_YAML[*]}" >&2
  exit 2
fi
AL_CONFIG="${REPO}/examples/paper_cases/${YAML_NAME}"
[[ -f "${AL_CONFIG}" ]] || { echo "ERROR: AL config not found: ${AL_CONFIG}" >&2; exit 2; }

# phosphorene_qe uses the QE backend; everything else uses VASP.
if [[ "${CASE}" == "phosphorene_qe" ]]; then
  DFT_BACKEND="${DFT_BACKEND:-qe}"
else
  DFT_BACKEND="${DFT_BACKEND:-vasp}"
fi
export MLIP_BACKEND="${MLIP_BACKEND:-hydragnn}"
export DFT_BACKEND

JOBID="${PBS_JOBID:-$$}"
RUN_DIR="${RUNS_ROOT}/al-paper-${CASE}-aurora-${JOBID}"
mkdir -p "${RUN_DIR}"

# ── modules & venv (backend-specific) ───────────────────────────────────────
source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_frameworks_modules
case "${MLIP_BACKEND}" in
  uma)  VENV="${MATSIM_FAIRCHEM_VENV:-${REPO}/.venv-uma}" ;;
  mace) VENV="${MATSIM_MACE_VENV:-${REPO}/.venv-mace}" ;;
  *)    VENV="${MATSIM_AURORA_VENV:-${REPO}/.venv}" ;;
esac
[[ -f "${VENV}/bin/activate" ]] || { echo "ERROR: venv not found: ${VENV}" >&2; exit 2; }
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONPATH="${HYDRAGNN_DIR}/examples/multidataset_hpo_sc26:${HYDRAGNN_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "=========================================="
echo "[Aurora active-learning — paper case]"
echo "Date:         $(date)"
echo "Job ID:       ${JOBID}"
echo "Host:         $(hostname)"
echo "Nodes:        $(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | wc -l)"
echo "Case:         ${CASE}"
echo "AL config:    ${AL_CONFIG}"
echo "MLIP backend: ${MLIP_BACKEND}"
echo "DFT backend:  ${DFT_BACKEND}"
echo "Venv:         ${VENV}"
echo "Run dir:      ${RUN_DIR}"
echo "=========================================="

python - <<'PY'
import torch
xpu = getattr(torch, "xpu", None)
available = bool(xpu and xpu.is_available())
print(f"[torch] {torch.__version__}  xpu={available}  devices={xpu.device_count() if available else 0}")
PY

# ── validate then run the AL loop ───────────────────────────────────────────
cd "${REPO}"
echo "[$(date)] Validating AL config ..."
matsim-agents al validate-config "${AL_CONFIG}" >"${RUN_DIR}/al-config.resolved.json" 2>&1 || {
  echo "[ERROR] AL config validation failed. See ${RUN_DIR}/al-config.resolved.json" >&2
  tail -n 30 "${RUN_DIR}/al-config.resolved.json" >&2 || true
  exit 1
}

echo "[$(date)] Starting active-learning loop for case '${CASE}' ..."
matsim-agents al run "${AL_CONFIG}" --log-level "${LOG_LEVEL:-INFO}" \
    2>&1 | tee "${RUN_DIR}/active-learning.log"

echo "[$(date)] Active-learning loop complete for '${CASE}'. Artifacts under ${RUNS_ROOT} (RUN_TAG from YAML)."
