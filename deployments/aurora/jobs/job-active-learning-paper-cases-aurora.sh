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
# Replace the paper case's model with MLIP_CONFIG, a YAML with an `mlip:` block
# (`matsim-agents al run --mlip-config`). The job activates the venv of its
# backend (HydraGNN in .venv, UMA in .venv-uma, MACE in .venv-mace):
#   qsub -v CASE=hea_bcc,MLIP_CONFIG=$PWD/examples/mlip/uma-s-1p1-omat.yaml ...
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
source "${REPO}/deployments/common/hydragnn-env.sh"
export HYDRAGNN_FT_REPO="${HYDRAGNN_FT_REPO:-${PROJ}/HydraGNN_GFM_FineTuning4Materials}"

# ── case -> AL YAML mapping ─────────────────────────────────────────────────
CASE="${CASE:-hea_bcc}"
case "${CASE}" in
  lifepo4)        YAML_NAME=al_lifepo4_aurora.yaml ;;
  hea_bcc)        YAML_NAME=al_hea_bcc_aurora.yaml ;;
  phosphorene)    YAML_NAME=al_phosphorene_aurora.yaml ;;
  phosphorene_qe) YAML_NAME=al_phosphorene_qe_aurora.yaml ;;
  *)
    echo "ERROR: unknown CASE='${CASE}'. Valid: lifepo4 hea_bcc phosphorene phosphorene_qe" >&2
    exit 2
    ;;
esac
AL_CONFIG="${REPO}/examples/paper_cases/${YAML_NAME}"
# This job reproduces a repository example on purpose (`al run` refuses examples otherwise).
export MATSIM_ALLOW_TEMPLATE_CONFIG=1
[[ -f "${AL_CONFIG}" ]] || { echo "ERROR: AL config not found: ${AL_CONFIG}" >&2; exit 2; }

# phosphorene_qe uses the QE backend; everything else uses VASP.
if [[ "${CASE}" == "phosphorene_qe" ]]; then
  DFT_BACKEND="${DFT_BACKEND:-qe}"
else
  DFT_BACKEND="${DFT_BACKEND:-vasp}"
fi
export DFT_BACKEND

# ── MLIP selection (optional `mlip:` YAML replacing the paper case's model) ──
MLIP_ARGS=()
if [[ -n "${MLIP_CONFIG:-}" ]]; then
  [[ -f "${MLIP_CONFIG}" ]] || { echo "ERROR: MLIP config not found: ${MLIP_CONFIG}" >&2; exit 2; }
  MLIP_CONFIG="$(cd "$(dirname "${MLIP_CONFIG}")" && pwd)/$(basename "${MLIP_CONFIG}")"
  MLIP_ARGS=(--mlip-config "${MLIP_CONFIG}")
fi
source "${REPO}/deployments/perlmutter/setup/campaign-naming.sh"
MLIP="$(campaign_mlip_backend "${REPO}" "${MATSIM_AURORA_VENV:-${REPO}/.venv}/bin/python" "${MLIP_CONFIG:-${AL_CONFIG}}")"

JOBID="${PBS_JOBID:-$$}"
RUN_DIR="${RUNS_ROOT}/al-paper-${CASE}-aurora-${JOBID}"
mkdir -p "${RUN_DIR}"

# ── modules & venv (backend-specific) ───────────────────────────────────────
source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_frameworks_modules
case "${MLIP}" in
  uma)  VENV="${MATSIM_FAIRCHEM_VENV:-${REPO}/.venv-uma}" ;;
  mace) VENV="${MATSIM_MACE_VENV:-${REPO}/.venv-mace}" ;;
  *)    VENV="${MATSIM_AURORA_VENV:-${REPO}/.venv}" ;;
esac
[[ -f "${VENV}/bin/activate" ]] || { echo "ERROR: venv not found: ${VENV}" >&2; exit 2; }
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

hydragnn_export_pythonpath
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
echo "MLIP backend: ${MLIP}"
echo "MLIP config:  ${MLIP_CONFIG:-${AL_CONFIG}}"
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
matsim-agents al validate-config "${AL_CONFIG}" "${MLIP_ARGS[@]}" >"${RUN_DIR}/al-config.resolved.json" 2>&1 || {
  echo "[ERROR] AL config validation failed. See ${RUN_DIR}/al-config.resolved.json" >&2
  tail -n 30 "${RUN_DIR}/al-config.resolved.json" >&2 || true
  exit 1
}

echo "[$(date)] Starting active-learning loop for case '${CASE}' ..."
matsim-agents al run "${AL_CONFIG}" "${MLIP_ARGS[@]}" --log-level "${LOG_LEVEL:-INFO}" \
    2>&1 | tee "${RUN_DIR}/active-learning.log"

echo "[$(date)] Active-learning loop complete for '${CASE}'. Artifacts under ${RUNS_ROOT} (RUN_TAG from YAML)."
