#!/bin/bash
#SBATCH -J al-qe-portability
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 4
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: DFT-engine portability demo on Perlmutter.
#
# Runs the phosphorene AL loop with the Quantum ESPRESSO backend instead of
# VASP (examples/paper_cases/al_phosphorene_qe.yaml). Demonstrates that the same
# loop is DFT-engine-agnostic on one platform: only the `dft:` block differs
# from al_phosphorene.yaml.
#
#   sbatch deployments/perlmutter/jobs/job-al-qe-portability-perlmutter.sh
#
# -N 4 gives 4-way concurrency for the nodes_per_job=1 QE single-points.
# PREREQUISITE: UMA weights prefetched (download-uma-perlmutter.sh) and the QE
# GPU build present at external/quantum-espresso/install-gpu/bin/pw.x.
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-${REPO_DEFAULT}}"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"

VENV_ROOT=$REPO/.hpc-build/perlmutter

AL_CONFIG="$REPO/examples/paper_cases/al_phosphorene_qe.yaml"
[[ ! -f "$AL_CONFIG" ]] && { echo "ERROR: missing $AL_CONFIG" >&2; exit 2; }

RUN_DIR="$RUNS_ROOT/al-qe-portability-${SLURM_JOB_ID:-$$}"
mkdir -p "$RUN_DIR"

# ── backends ──────────────────────────────────────────────────────────────────
# MLIP_CONFIG (optional) is a YAML with an `mlip:` block that replaces the AL
# config's model (UMA by default); the job activates the venv of its backend.
export DFT_BACKEND="${DFT_BACKEND:-qe}"
MLIP_ARGS=()
if [[ -n "${MLIP_CONFIG:-}" ]]; then
  [[ -f "$MLIP_CONFIG" ]] || { echo "ERROR: MLIP config not found: $MLIP_CONFIG" >&2; exit 2; }
  MLIP_CONFIG="$(cd "$(dirname "$MLIP_CONFIG")" && pwd)/$(basename "$MLIP_CONFIG")"
  MLIP_ARGS=(--mlip-config "$MLIP_CONFIG")
fi
source "$REPO/deployments/perlmutter/setup/campaign-naming.sh"
MLIP="$(campaign_mlip_backend "$REPO" "${MATSIM_PERLMUTTER_VENV:-$REPO/.venv}/bin/python3" "${MLIP_CONFIG:-$AL_CONFIG}")"
case "$MLIP" in
  uma)      VENV="${MATSIM_FAIRCHEM_VENV:-${REPO}/.venv-uma}" ;;
  mace)     VENV="${MATSIM_MACE_VENV:-${REPO}/.venv-mace}" ;;
  hydragnn) VENV="${MATSIM_HYDRAGNN_VENV:-${REPO}/.venv}" ;;
  *)        echo "ERROR: unsupported MLIP backend '$MLIP'" >&2; exit 2 ;;
esac

# ── modules and the matsim-owned venv of the MLIP backend ────────────────────
source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ── Durable MLIP model artifacts ──────────────────────────────────────────────
source "${REPO}/deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
case "$MLIP" in
  uma)
    UMA_MODEL="$(mlip_uma_model_name python "${MLIP_CONFIG:-$AL_CONFIG}")"
    configure_uma_model_artifacts "${REPO}" "${UMA_MODEL}"
    ;;
  mace) configure_mace_model_artifacts "${REPO}" ;;
esac
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(< "${HOME}/.cache/huggingface/token")"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "=========================================="
echo "[Perlmutter AL — QE portability demo]"
echo "Date:         $(date)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Nodes:        ${SLURM_JOB_NUM_NODES:-1}"
echo "AL config:    ${AL_CONFIG}"
echo "MLIP backend: ${MLIP}"
echo "MLIP config:  ${MLIP_CONFIG:-$AL_CONFIG}"
echo "DFT backend:  ${DFT_BACKEND}"
echo "Run dir:      ${RUN_DIR}"
echo "=========================================="

# ── validate then run ────────────────────────────────────────────────────────
cd "$REPO"
matsim-agents al validate-config "$AL_CONFIG" ${MLIP_ARGS[@]+"${MLIP_ARGS[@]}"} >"$RUN_DIR/al-config.resolved.json" 2>&1 || {
  echo "[ERROR] config validation failed; see $RUN_DIR/al-config.resolved.json" >&2
  tail -n 30 "$RUN_DIR/al-config.resolved.json" >&2 || true
  exit 1
}

echo "[$(date)] Starting QE-backend AL loop (phosphorene) ..."
matsim-agents al run "$AL_CONFIG" ${MLIP_ARGS[@]+"${MLIP_ARGS[@]}"} --log-level "${LOG_LEVEL:-INFO}" \
    2>&1 | tee "$RUN_DIR/active-learning.log"

echo "[$(date)] QE portability run complete. State under ${RUNS_ROOT}/phosphorene-2d-qe-al-001."
