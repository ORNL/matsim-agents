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
# Runs `matsim-agents al run examples/paper_cases/al_<case>.yaml --mlip-config
# $MLIP_CONFIG` with DFT_BACKEND=vasp. MLIP_CONFIG is any YAML with an `mlip:`
# block; it replaces the paper case's own model. It defaults to
# examples/mlip/uma-s-1p1-omat.yaml (uma-s-1p1-omol.yaml for zn_formate), run
# from the matsim-owned .venv-uma created with INSTALL_UMA=1.
#
# Select the case with the CASE env var (default: hea_bcc):
#   CASE=lifepo4    sbatch deployments/perlmutter/jobs/job-active-learning-paper-cases-perlmutter.sh
#   CASE=hea_bcc    sbatch ...
#   CASE=hea_fcc    sbatch ...
#   CASE=phosphorene sbatch ...
#   CASE=cu_bht     sbatch ...   (requires a supplied CIF seed)
#   CASE=zn_formate sbatch ...   (al_zn_formate_uma.yaml when the MLIP is UMA)
#
# Select the MLIP with MLIP_CONFIG; the job activates the venv of its backend:
#   MLIP_CONFIG=$PWD/examples/mlip/mace-mp-medium.yaml CASE=hea_bcc sbatch ...
#   MLIP_CONFIG= DFT_BACKEND=qe CASE=hea_bcc sbatch ...   (the paper case's own HydraGNN block)
#   MLIP_CONFIG=$PWD/examples/mlip/mace-mp-medium.yaml RETRAIN=1 CASE=hea_bcc sbatch ...
# RETRAIN=1 fine-tunes the model each iteration (mlip.fine_tuner, run through the
# backend-neutral launchers/_mlip-train-step-perlmutter.sh); RETRAIN=0 keeps it
# frozen. HydraGNN retrains by default; UMA and MACE foundation models run
# FROZEN by default. RETRAIN sets TRAINER_ENABLED, which every case's
# trainer.enabled reads except cu_bht (always trains) and al_zn_formate_uma.yaml
# (always frozen). MACE needs prefetched foundation model artifacts under
# $PROJ/models/artifacts/mace.
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
case "$CASE" in
  lifepo4)     YAML_NAME=al_lifepo4.yaml ;;
  hea_bcc)     YAML_NAME=al_hea_bcc.yaml ;;
  hea_fcc)     YAML_NAME=al_hea_fcc_cantor.yaml ;;
  phosphorene) YAML_NAME=al_phosphorene.yaml ;;
  cu_bht)      YAML_NAME=al_cu_bht_2d.yaml ;;
  zn_formate)  YAML_NAME=al_zn_formate.yaml ;;
  *)
    echo "ERROR: unknown CASE='$CASE'. Valid: lifepo4 hea_bcc hea_fcc phosphorene cu_bht zn_formate" >&2
    exit 2
    ;;
esac
AL_CONFIG="$REPO/examples/paper_cases/$YAML_NAME"
if [[ ! -f "$AL_CONFIG" ]]; then
  echo "ERROR: AL config not found: $AL_CONFIG" >&2
  exit 2
fi

# ── MLIP selection (an `mlip:` YAML that replaces the paper case's model) ────
MLIP_CONFIG_DEFAULT="$REPO/examples/mlip/uma-s-1p1-omat.yaml"
[[ "$CASE" != "zn_formate" ]] || MLIP_CONFIG_DEFAULT="$REPO/examples/mlip/uma-s-1p1-omol.yaml"
MLIP_CONFIG="${MLIP_CONFIG-$MLIP_CONFIG_DEFAULT}"
MLIP_ARGS=()
if [[ -n "$MLIP_CONFIG" ]]; then
  [[ -f "$MLIP_CONFIG" ]] || { echo "ERROR: MLIP config not found: $MLIP_CONFIG" >&2; exit 2; }
  MLIP_CONFIG="$(cd "$(dirname "$MLIP_CONFIG")" && pwd)/$(basename "$MLIP_CONFIG")"
  MLIP_ARGS=(--mlip-config "$MLIP_CONFIG")
fi
source "$REPO/deployments/perlmutter/setup/campaign-naming.sh"
MLIP="$(campaign_mlip_backend "$REPO" "${MATSIM_PERLMUTTER_VENV:-$REPO/.venv}/bin/python3" "${MLIP_CONFIG:-$AL_CONFIG}")"
# zn_formate ships a dedicated UMA-tuned config (frozen model, dense-cell MD).
if [[ "$CASE" == "zn_formate" && "$MLIP" == "uma" ]]; then
  AL_CONFIG="$REPO/examples/paper_cases/al_zn_formate_uma.yaml"
fi

RUN_DIR=$RUNS_ROOT/al-paper-${CASE}-${SLURM_JOB_ID:-$$}
mkdir -p "$RUN_DIR"

# ── modules & venv (backend-specific) ────────────────────────────────────────
source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu

case "$MLIP" in
  uma)      VENV="${MATSIM_FAIRCHEM_VENV:-${REPO}/.venv-uma}" ;;
  mace)     VENV="${MATSIM_MACE_VENV:-${REPO}/.venv-mace}" ;;
  hydragnn) VENV="${MATSIM_HYDRAGNN_VENV:-${REPO}/.venv}" ;;
  *)        echo "ERROR: unsupported MLIP backend '$MLIP'" >&2; exit 2 ;;
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

source "${REPO}/deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
if [[ "$MLIP" == "uma" ]]; then
  UMA_MODEL="$(mlip_uma_model_name python "${MLIP_CONFIG:-$AL_CONFIG}")"
  configure_uma_model_artifacts "${REPO}" "${UMA_MODEL}"
fi
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(< "${HOME}/.cache/huggingface/token")"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# ── DFT backend (consumed by the YAML via ${...} env interpolation) ─────────
export DFT_BACKEND="${DFT_BACKEND:-vasp}"

# MACE backend: cache foundation weights in the shared project dir.
[[ "$MLIP" != "mace" ]] || configure_mace_model_artifacts "${REPO}"

# Per-iteration fine-tuning: HydraGNN retrains by default; UMA and MACE run
# FROZEN by default (directly comparable to each other).
if [[ "$MLIP" == "hydragnn" ]]; then RETRAIN="${RETRAIN:-1}"; else RETRAIN="${RETRAIN:-0}"; fi
case "$RETRAIN" in
  1) export TRAINER_ENABLED=true ;;
  0) export TRAINER_ENABLED=false ;;
  *) echo "ERROR: RETRAIN must be 0 or 1" >&2; exit 2 ;;
esac

echo "=========================================="
echo "[Perlmutter active-learning — paper case]"
echo "Date:         $(date)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Host:         $(hostname)"
echo "Case:         $CASE"
echo "AL config:    $AL_CONFIG"
echo "MLIP backend: $MLIP"
echo "MLIP config:  ${MLIP_CONFIG:-$AL_CONFIG}"
echo "DFT backend:  $DFT_BACKEND"
echo "Retrain:      $RETRAIN"
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
matsim-agents al validate-config "$AL_CONFIG" ${MLIP_ARGS[@]+"${MLIP_ARGS[@]}"} >"$RUN_DIR/al-config.resolved.json" 2>&1 || {
  echo "[ERROR] AL config validation failed. See $RUN_DIR/al-config.resolved.json" >&2
  tail -n 30 "$RUN_DIR/al-config.resolved.json" >&2 || true
  exit 1
}

echo "[$(date)] Starting active-learning loop for case '$CASE' ..."
matsim-agents al run "$AL_CONFIG" ${MLIP_ARGS[@]+"${MLIP_ARGS[@]}"} --log-level "${LOG_LEVEL:-INFO}" \
    2>&1 | tee "$RUN_DIR/active-learning.log"

echo "[$(date)] Active-learning loop complete for '$CASE'. Artifacts under $RUNS_ROOT (RUN_TAG from YAML)."
