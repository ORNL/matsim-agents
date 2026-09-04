#!/bin/bash
#SBATCH -J al-paper-case
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: full active-learning loop for a single paper case on
# Perlmutter, using the UMA foundation MLFF + VASP reference DFT.
#
#   MLFF (UMA, fairchem-core) MD sampling
#     -> mc_dropout uncertainty acquisition
#       -> VASP vasp_std single-points on selected structures (srun steps)
#         -> append labelled structures -> next iteration
#
# Runs `matsim-agents al run examples/paper_cases/al_<case>.yaml` with
#   MLIP_BACKEND=uma DFT_BACKEND=vasp
# from the matsim-owned .venv-uma created with INSTALL_UMA=1.
#
# Select the case with the CASE env var (default: hea_bcc):
#   CASE=lifepo4    sbatch deployments/perlmutter/jobs/job-active-learning-paper-cases-perlmutter.sh
#   CASE=hea_bcc    sbatch ...
#   CASE=hea_fcc    sbatch ...
#   CASE=phosphorene sbatch ...
#   CASE=cu_bht     sbatch ...   (requires a supplied CIF seed)
#   CASE=zn_formate sbatch ...
#
# Override backends:
#   MLIP_BACKEND=hydragnn DFT_BACKEND=qe CASE=hea_bcc sbatch ...
#   MLIP_BACKEND=mace     CASE=hea_bcc sbatch ...            (frozen MACE-MP loop)
#   MLIP_BACKEND=mace MACE_MODEL=large MACE_RETRAIN=1 CASE=hea_bcc sbatch ...
# MACE runs FROZEN by default (comparable to the frozen-UMA loop); MACE_RETRAIN=1
# fine-tunes each iteration and requires `.venv-mace` plus a prefetched foundation
# model cache under $PROJ/models/mace_cache.
#
# MULTI-NODE DFT CONCURRENCY — the AL driver dispatches the selected VASP
# single-points concurrently, up to (SLURM_JOB_NUM_NODES / nodes_per_job) at a
# time. This script defaults to -N 1 (serial DFT). To run K DFT jobs in
# parallel, submit with more nodes, e.g. `sbatch -N 4 ...` gives 4-way
# concurrency for the nodes_per_job=1 cases (lifepo4, hea_bcc, hea_fcc,
# phosphorene). NOTE: al_zn_formate*.yaml uses nodes_per_job: 2 (dense
# framework cell), so zn_formate REQUIRES -N >= 2 (use -N 4 for 2-way
# concurrency); with -N 1 every step fails with
# "srun: error: Only allocated 1 nodes asked for 2".
#
# PREREQUISITE (UMA backend) — prefetch the UMA weights first. This job reads
# the shared HF cache in OFFLINE mode (HF_HUB_OFFLINE=1): compute nodes have no
# internet and CFS does not support fcntl.flock over DVS (OSError [Errno 524]),
# so UMA is NOT downloaded on first use. Run once before submitting:
#   sbatch deployments/perlmutter/download/download-uma-perlmutter.sh
# See docs/model-download.md ("UMA MLIP weights on Perlmutter").
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-${REPO_DEFAULT}}"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"

# ── case -> AL YAML mapping ──────────────────────────────────────────────────
CASE="${CASE:-hea_bcc}"
MLIP_BACKEND="${MLIP_BACKEND:-uma}"
declare -A CASE_YAML=(
  [lifepo4]=al_lifepo4.yaml
  [hea_bcc]=al_hea_bcc.yaml
  [hea_fcc]=al_hea_fcc_cantor.yaml
  [phosphorene]=al_phosphorene.yaml
  [cu_bht]=al_cu_bht_2d.yaml
  [zn_formate]=al_zn_formate.yaml
)
# zn_formate ships a dedicated UMA-tuned config.
if [[ "$CASE" == "zn_formate" && "$MLIP_BACKEND" == "uma" ]]; then
  CASE_YAML[zn_formate]=al_zn_formate_uma.yaml
fi

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

RUN_DIR=$RUNS_ROOT/al-paper-${CASE}-${SLURM_JOB_ID:-$$}
mkdir -p "$RUN_DIR"

# ── modules & venv (backend-specific) ────────────────────────────────────────
source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu

case "$MLIP_BACKEND" in
  uma)      VENV="${MATSIM_FAIRCHEM_VENV:-${REPO}/.venv-uma}" ;;
  mace)     VENV="${MATSIM_MACE_VENV:-${REPO}/.venv-mace}" ;;
  hydragnn) VENV="${MATSIM_HYDRAGNN_VENV:-${REPO}/.venv}" ;;
  *)        VENV="${MATSIM_FAIRCHEM_VENV:-${REPO}/.venv-uma}" ;;
esac
[[ ! -d "${VENV}" ]] && { echo "ERROR: venv not found: ${VENV}" >&2; exit 2; }
# Both default environments are matsim-owned Python virtual environments.
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
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# UMA model cache (shared project dir).
export HF_HOME="${HF_HOME:-${PROJ}/models/hf_cache}"
mkdir -p "${HF_HOME}"
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(< "${HOME}/.cache/huggingface/token")"
fi
# fairchem's pretrained_mlip.get_predict_unit() ignores HF_HOME entirely -- it
# always calls hf_hub_download(..., cache_dir=FAIRCHEM_CACHE_DIR), which
# defaults to ~/.cache/fairchem on $HOME (CFS/GPFS, no fcntl.flock support ->
# OSError [Errno 524], regardless of offline mode). Point it at $SCRATCH
# (flock-capable, persistent across jobs). Requires a prior successful run of
# deployments/perlmutter/download/download-uma-perlmutter.sh.
export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-${SCRATCH:-/tmp}/matsim-agents/fairchem_cache}"
mkdir -p "${FAIRCHEM_CACHE_DIR}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# ── AL backend selection (consumed by the YAML via ${...} env interpolation) ─
export MLIP_BACKEND
export DFT_BACKEND="${DFT_BACKEND:-vasp}"

# MACE backend: cache foundation weights in the shared project dir and run the
# loop FROZEN by default (directly comparable to the frozen-UMA loop). Set
# MACE_RETRAIN=1 to fine-tune each iteration via the MACE train-step launcher.
if [[ "$MLIP_BACKEND" == "mace" ]]; then
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJ}/models/mace_cache}"
  export MACE_CACHE="${MACE_CACHE:-${XDG_CACHE_HOME}/mace}"
  mkdir -p "${MACE_CACHE}"
  export MACE_FAMILY="${MACE_FAMILY:-mace_mp}"
  export MACE_MODEL="${MACE_MODEL:-medium}"
  if [[ "${MACE_RETRAIN:-0}" == "1" ]]; then
    export TRAINER_ENABLED="true"
    export TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-${REPO}/deployments/perlmutter/launchers/_mace-train-step-perlmutter.sh}"
  else
    export TRAINER_ENABLED="${TRAINER_ENABLED:-false}"
  fi
fi

echo "=========================================="
echo "[Perlmutter active-learning — paper case]"
echo "Date:         $(date)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Host:         $(hostname)"
echo "Case:         $CASE"
echo "AL config:    $AL_CONFIG"
echo "MLIP backend: $MLIP_BACKEND"
echo "DFT backend:  $DFT_BACKEND"
echo "Venv:         $VENV"
echo "Run dir:      $RUN_DIR"
echo "=========================================="

python - <<'PY'
import torch
print(f"[torch] {torch.__version__}  cuda={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
PY

# ── validate then run the AL loop ────────────────────────────────────────────
cd "$REPO"
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
