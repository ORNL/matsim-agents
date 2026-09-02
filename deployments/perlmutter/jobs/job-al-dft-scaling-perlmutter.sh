#!/bin/bash
#SBATCH -J al-dft-scaling
#SBATCH -o %x-N%N-%j.out
#SBATCH -e %x-N%N-%j.err
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: DFT-concurrency STRONG-SCALING harness.
#
# Runs ONE active-learning iteration with a FIXED 32-job DFT workload
# (examples/paper_cases/al_hea_fcc_scaling.yaml, n_select=32, UMA + VASP). The
# AL driver dispatches up to (SLURM_JOB_NUM_NODES / nodes_per_job=1) VASP
# single-points concurrently, so submitting this script at several node counts
# produces a strong-scaling curve for identical work.
#
# Submit the full sweep with the companion wrapper:
#   deployments/perlmutter/jobs/submit-al-dft-scaling-sweep.sh
# or a single point directly:
#   sbatch -N 4 deployments/perlmutter/jobs/job-al-dft-scaling-perlmutter.sh
#
# Each submission writes to a per-N RUN_TAG (hea-fcc-scaling-N<nodes>) so points
# never collide. Aggregate/plot with:
#   research/paper/manuscript/figures/plot_dft_scaling.py
#
# PREREQUISITE: UMA weights prefetched (see download-uma-perlmutter.sh) exactly
# as for job-active-learning-paper-cases-perlmutter.sh.
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
VENV="${MATSIM_FAIRCHEM_VENV:-${VENV_ROOT}/hydragnn_venv}"

AL_CONFIG="$REPO/examples/paper_cases/al_hea_fcc_scaling.yaml"
[[ ! -f "$AL_CONFIG" ]] && { echo "ERROR: missing $AL_CONFIG" >&2; exit 2; }

# ── per-(N,repeat) run tag (consumed by the YAML via ${SCALE_TAG}) ────────────
# REP indexes the repeat; AL_SEED randomizes the workload per repeat (held fixed
# across node counts within a repeat) so averaging over repeats dampens per-draw
# variability. Each (N,REP) writes to its own out_dir -> no resume/collision.
NNODES="${SLURM_JOB_NUM_NODES:-1}"
REP="${REP:-1}"
export AL_SEED="${AL_SEED:-$((12 + REP))}"
export SCALE_TAG="hea-fcc-scaling-N${NNODES}-r${REP}"
RUN_DIR="$RUNS_ROOT/al-scaling-N${NNODES}-r${REP}-${SLURM_JOB_ID:-$$}"
mkdir -p "$RUN_DIR"

# ── backends ──────────────────────────────────────────────────────────────────
export MLIP_BACKEND="${MLIP_BACKEND:-uma}"
export DFT_BACKEND="${DFT_BACKEND:-vasp}"

# ── modules & venv (hydragnn_venv for UMA) ────────────────────────────────────
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
echo "[Perlmutter AL DFT-scaling point]"
echo "Date:      $(date)"
echo "Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "Nodes:     ${NNODES}  (=> ${NNODES}-way DFT concurrency, nodes_per_job=1)"
echo "Repeat:    ${REP}  (AL_SEED=${AL_SEED})"
echo "SCALE_TAG: ${SCALE_TAG}"
echo "AL config: ${AL_CONFIG}"
echo "Run dir:   ${RUN_DIR}"
echo "=========================================="

# ── run one AL iteration ─────────────────────────────────────────────────────
cd "$REPO"
matsim-agents al validate-config "$AL_CONFIG" >"$RUN_DIR/al-config.resolved.json" 2>&1 || {
  echo "[ERROR] config validation failed; see $RUN_DIR/al-config.resolved.json" >&2
  tail -n 30 "$RUN_DIR/al-config.resolved.json" >&2 || true
  exit 1
}

echo "[$(date)] Starting scaling point (N=${NNODES}) ..."
matsim-agents al run "$AL_CONFIG" --log-level "${LOG_LEVEL:-INFO}" \
    2>&1 | tee "$RUN_DIR/scaling.log"

echo "[$(date)] Scaling point N=${NNODES} complete. State under ${RUNS_ROOT}/${SCALE_TAG}."
