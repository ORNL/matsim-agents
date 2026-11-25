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

VENV_ROOT=$REPO/.hpc-build/perlmutter

AL_CONFIG="$REPO/examples/paper_cases/al_hea_fcc_scaling.yaml"
# This job reproduces a repository example on purpose (`al run` refuses examples otherwise).
export MATSIM_ALLOW_TEMPLATE_CONFIG=1
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
# MLIP_CONFIG (optional) is a YAML with an `mlip:` block that replaces the AL
# config's model (UMA by default); the job activates the venv of its backend.
export DFT_BACKEND="${DFT_BACKEND:-vasp}"
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
echo "[Perlmutter AL DFT-scaling point]"
echo "Date:      $(date)"
echo "Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "Nodes:     ${NNODES}  (=> ${NNODES}-way DFT concurrency, nodes_per_job=1)"
echo "Repeat:    ${REP}  (AL_SEED=${AL_SEED})"
echo "SCALE_TAG: ${SCALE_TAG}"
echo "AL config: ${AL_CONFIG}"
echo "MLIP:      ${MLIP} (${MLIP_CONFIG:-$AL_CONFIG})"
echo "Run dir:   ${RUN_DIR}"
echo "=========================================="

# ── run one AL iteration ─────────────────────────────────────────────────────
cd "$REPO"
matsim-agents al validate-config "$AL_CONFIG" ${MLIP_ARGS[@]+"${MLIP_ARGS[@]}"} >"$RUN_DIR/al-config.resolved.json" 2>&1 || {
  echo "[ERROR] config validation failed; see $RUN_DIR/al-config.resolved.json" >&2
  tail -n 30 "$RUN_DIR/al-config.resolved.json" >&2 || true
  exit 1
}

echo "[$(date)] Starting scaling point (N=${NNODES}) ..."
matsim-agents al run "$AL_CONFIG" ${MLIP_ARGS[@]+"${MLIP_ARGS[@]}"} --log-level "${LOG_LEVEL:-INFO}" \
    2>&1 | tee "$RUN_DIR/scaling.log"

echo "[$(date)] Scaling point N=${NNODES} complete. State under ${RUNS_ROOT}/${SCALE_TAG}."
