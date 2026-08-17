#!/bin/bash
#SBATCH -A lrn070
#SBATCH -J al-dft-scaling
#SBATCH -o %x-N%N-%j.out
#SBATCH -e %x-N%N-%j.err
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -p batch
# ---------------------------------------------------------------------------
# matsim-agents: DFT-concurrency STRONG-SCALING harness on OLCF Frontier.
#
# Runs ONE active-learning iteration with a FIXED 32-job DFT workload
# (examples/paper_cases/al_hea_fcc_scaling_frontier.yaml, n_select=32,
# HydraGNN + VASP). The AL driver dispatches up to
# (SLURM_JOB_NUM_NODES / nodes_per_job=1) VASP single-points concurrently, so
# submitting this script at several node counts produces a strong-scaling curve
# for identical work.
#
# Submit the full sweep with the companion wrapper:
#   deployments/frontier/jobs/submit-al-dft-scaling-sweep.sh
# or a single point directly:
#   sbatch -N 4 deployments/frontier/jobs/job-al-dft-scaling-frontier.sh
#
# Each submission writes to a per-(N,repeat) RUN_TAG
# (hea-fcc-scaling-frontier-N<nodes>-r<rep>) so points never collide. REP/AL_SEED
# randomize the workload per repeat (fixed across node counts within a repeat).
#
# Paths default to the lrn070 project checkout but are env-overridable:
#   PROJECT_ROOT, RUNS_ROOT, VENV_ROOT.
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-${REPO_DEFAULT}}"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/lustre/orion/lrn070/proj-shared/mlupopa/matstim-folder/matsim-agents
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"

AL_CONFIG="$REPO/examples/paper_cases/al_hea_fcc_scaling_frontier.yaml"
[[ ! -f "$AL_CONFIG" ]] && { echo "ERROR: missing $AL_CONFIG" >&2; exit 2; }

# ── per-(N,repeat) run tag (consumed by the YAML via ${SCALE_TAG}) ────────────
# REP indexes the repeat; AL_SEED randomizes the workload per repeat (held fixed
# across node counts within a repeat) so averaging over repeats dampens per-draw
# variability. Each (N,REP) writes to its own out_dir -> no resume/collision.
NNODES="${SLURM_JOB_NUM_NODES:-1}"
REP="${REP:-1}"
export AL_SEED="${AL_SEED:-$((12 + REP))}"
export SCALE_TAG="hea-fcc-scaling-frontier-N${NNODES}-r${REP}"
RUN_DIR="$RUNS_ROOT/al-scaling-N${NNODES}-r${REP}-${SLURM_JOB_ID:-$$}"
mkdir -p "$RUN_DIR"

# ── backends ──────────────────────────────────────────────────────────────────
export MLIP_BACKEND="${MLIP_BACKEND:-hydragnn}"
export DFT_BACKEND="${DFT_BACKEND:-vasp}"

# ── modules & venv (HydraGNN ROCm 7.2.0) ──────────────────────────────────────
cd "$REPO"
# shellcheck disable=SC1091
source deployments/frontier/setup/setup_matsim_frontier.sh --rocm72

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Make HydraGNN example utilities importable (inference_fused, etc.), which the
# HydraGNN ASE calculator in tools/relaxation.py imports at build time.
HYDRAGNN_EXAMPLE="${HYDRAGNN_DIR:-$PROJ/HydraGNN}/examples/multidataset_hpo_sc26"
export PYTHONPATH="$HYDRAGNN_EXAMPLE:${HYDRAGNN_DIR:-$PROJ/HydraGNN}:${PYTHONPATH:-}"

echo "=========================================="
echo "[Frontier AL DFT-scaling point]"
echo "Date:      $(date)"
echo "Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "Nodes:     ${NNODES}  (=> ${NNODES}-way DFT concurrency, nodes_per_job=1)"
echo "Repeat:    ${REP}  (AL_SEED=${AL_SEED})"
echo "SCALE_TAG: ${SCALE_TAG}"
echo "AL config: ${AL_CONFIG}"
echo "Run dir:   ${RUN_DIR}"
echo "=========================================="

# ── run one AL iteration ──────────────────────────────────────────────────────
matsim-agents al validate-config "$AL_CONFIG" >"$RUN_DIR/al-config.resolved.json" 2>&1 || {
  echo "[ERROR] config validation failed; see $RUN_DIR/al-config.resolved.json" >&2
  tail -n 30 "$RUN_DIR/al-config.resolved.json" >&2 || true
  exit 1
}

echo "[$(date)] Starting scaling point (N=${NNODES}) ..."
matsim-agents al run "$AL_CONFIG" --log-level "${LOG_LEVEL:-INFO}" \
    2>&1 | tee "$RUN_DIR/scaling.log"

echo "[$(date)] Scaling point N=${NNODES} complete. State under ${RUNS_ROOT}/${SCALE_TAG}."
