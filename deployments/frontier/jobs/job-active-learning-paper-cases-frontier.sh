#!/bin/bash
#SBATCH -A lrn070
#SBATCH -J al-paper-case
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch
# ---------------------------------------------------------------------------
# matsim-agents: full active-learning loop for a single paper case on
# OLCF Frontier (AMD MI250X), using the HydraGNN MLFF + VASP reference DFT.
#
#   HydraGNN (ROCm) MD sampling
#     -> mc_dropout uncertainty acquisition
#       -> VASP vasp_std single-points on selected structures (srun steps)
#         -> retrain HydraGNN -> next iteration
#
# Runs `matsim-agents al run examples/paper_cases/al_<case>_frontier.yaml` with
#   MLIP_BACKEND=hydragnn DFT_BACKEND=vasp
# from the HydraGNN ROCm 7.2.0 venv (UMA/fairchem is NOT installed on Frontier).
#
# Select the case with the CASE env var (default: hea_bcc):
#   CASE=lifepo4       sbatch deployments/frontier/jobs/job-active-learning-paper-cases-frontier.sh
#   CASE=hea_bcc       sbatch ...
#   CASE=phosphorene   sbatch ...          (VASP backend)
#   CASE=phosphorene_qe sbatch ...         (QE backend; requires QE pw.x built)
#
# MULTI-NODE DFT CONCURRENCY — the AL driver dispatches the selected VASP
# single-points concurrently, up to (SLURM_JOB_NUM_NODES / nodes_per_job) at a
# time. This script defaults to -N 1 (serial DFT). To run K DFT jobs in
# parallel, submit with more nodes, e.g. `sbatch -N 4 ...` gives 4-way
# concurrency for the nodes_per_job=1 paper cases.
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

# ── case -> AL YAML mapping ───────────────────────────────────────────────────
CASE="${CASE:-hea_bcc}"
MLIP_BACKEND="${MLIP_BACKEND:-hydragnn}"
declare -A CASE_YAML=(
  [lifepo4]=al_lifepo4_frontier.yaml
  [hea_bcc]=al_hea_bcc_frontier.yaml
  [phosphorene]=al_phosphorene_frontier.yaml
  [phosphorene_qe]=al_phosphorene_qe_frontier.yaml
)

YAML_NAME="${CASE_YAML[$CASE]:-}"
if [[ -z "$YAML_NAME" ]]; then
  echo "ERROR: unknown CASE='$CASE'. Valid: ${!CASE_YAML[*]}" >&2
  exit 2
fi
AL_CONFIG="$REPO/examples/paper_cases/$YAML_NAME"
if [[ ! -f "$AL_CONFIG" ]]; then
  echo "ERROR: AL config not found: $AL_CONFIG" >&2
  exit 2
fi

# phosphorene_qe uses the QE backend; everything else uses VASP.
if [[ "$CASE" == "phosphorene_qe" ]]; then
  DFT_BACKEND="${DFT_BACKEND:-qe}"
else
  DFT_BACKEND="${DFT_BACKEND:-vasp}"
fi

RUN_DIR="$RUNS_ROOT/al-paper-${CASE}-${SLURM_JOB_ID:-$$}"
mkdir -p "$RUN_DIR"

# ── modules & venv (HydraGNN ROCm 7.2.0) ──────────────────────────────────────
cd "$REPO"
# shellcheck disable=SC1091
source deployments/frontier/setup/setup_matsim_frontier.sh --rocm72

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# ── AL backend selection (consumed by the YAML via ${...} env interpolation) ──
export MLIP_BACKEND
export DFT_BACKEND

echo "=========================================="
echo "[Frontier active-learning — paper case]"
echo "Date:         $(date)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Host:         $(hostname)"
echo "Nodes:        ${SLURM_JOB_NUM_NODES:-1}"
echo "Case:         $CASE"
echo "AL config:    $AL_CONFIG"
echo "MLIP backend: $MLIP_BACKEND"
echo "DFT backend:  $DFT_BACKEND"
echo "Repo:         $REPO"
echo "Run dir:      $RUN_DIR"
echo "=========================================="

python - <<'PY'
import torch
print(f"[torch] {torch.__version__}  cuda={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
PY

# ── validate then run the AL loop ─────────────────────────────────────────────
echo "[$(date)] Validating AL config ..."
matsim-agents al validate-config "$AL_CONFIG" >"$RUN_DIR/al-config.resolved.json" 2>&1 || {
  echo "[ERROR] AL config validation failed. See $RUN_DIR/al-config.resolved.json" >&2
  tail -n 30 "$RUN_DIR/al-config.resolved.json" >&2 || true
  exit 1
}

echo "[$(date)] Starting active-learning loop for case '$CASE' ..."
matsim-agents al run "$AL_CONFIG" --log-level "${LOG_LEVEL:-INFO}" \
    2>&1 | tee "$RUN_DIR/active-learning.log"

echo "[$(date)] Active-learning loop complete for '$CASE'. Artifacts under $RUNS_ROOT (RUN_TAG from YAML)."
