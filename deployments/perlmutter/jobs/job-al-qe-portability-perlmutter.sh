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

VENV_ROOT=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter
VENV="${MATSIM_FAIRCHEM_VENV:-${VENV_ROOT}/fairchem_venv}"

AL_CONFIG="$REPO/examples/paper_cases/al_phosphorene_qe.yaml"
[[ ! -f "$AL_CONFIG" ]] && { echo "ERROR: missing $AL_CONFIG" >&2; exit 2; }

RUN_DIR="$RUNS_ROOT/al-qe-portability-${SLURM_JOB_ID:-$$}"
mkdir -p "$RUN_DIR"

# ── backends ──────────────────────────────────────────────────────────────────
export MLIP_BACKEND="${MLIP_BACKEND:-uma}"
export DFT_BACKEND="${DFT_BACKEND:-qe}"

# ── modules & venv (fairchem_venv for UMA MD/scoring) ─────────────────────────
source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ── UMA model cache (identical to the paper-case job) ─────────────────────────
export HF_HOME="${HF_HOME:-${PROJ}/models/hf_cache}"
mkdir -p "${HF_HOME}"
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(< "${HOME}/.cache/huggingface/token")"
fi
export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-${SCRATCH:-/tmp}/matsim-agents/fairchem_cache}"
mkdir -p "${FAIRCHEM_CACHE_DIR}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "=========================================="
echo "[Perlmutter AL — QE portability demo]"
echo "Date:         $(date)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Nodes:        ${SLURM_JOB_NUM_NODES:-1}"
echo "AL config:    ${AL_CONFIG}"
echo "MLIP backend: ${MLIP_BACKEND}"
echo "DFT backend:  ${DFT_BACKEND}"
echo "Run dir:      ${RUN_DIR}"
echo "=========================================="

# ── validate then run ────────────────────────────────────────────────────────
cd "$REPO"
matsim-agents al validate-config "$AL_CONFIG" >"$RUN_DIR/al-config.resolved.json" 2>&1 || {
  echo "[ERROR] config validation failed; see $RUN_DIR/al-config.resolved.json" >&2
  tail -n 30 "$RUN_DIR/al-config.resolved.json" >&2 || true
  exit 1
}

echo "[$(date)] Starting QE-backend AL loop (phosphorene) ..."
matsim-agents al run "$AL_CONFIG" --log-level "${LOG_LEVEL:-INFO}" \
    2>&1 | tee "$RUN_DIR/active-learning.log"

echo "[$(date)] QE portability run complete. State under ${RUNS_ROOT}/phosphorene-2d-qe-al-001."
