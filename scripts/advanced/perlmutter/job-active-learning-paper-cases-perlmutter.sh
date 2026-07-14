#!/bin/bash
#SBATCH -A m5216
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
# from the fairchem_venv (UMA requires numpy>=2; not the hydragnn_venv).
#
# Select the case with the CASE env var (default: hea_bcc):
#   CASE=lifepo4    sbatch scripts/advanced/perlmutter/job-active-learning-paper-cases-perlmutter.sh
#   CASE=hea_bcc    sbatch ...
#   CASE=hea_fcc    sbatch ...
#   CASE=phosphorene sbatch ...
#   CASE=cu_bht     sbatch ...   (requires a supplied CIF seed)
#   CASE=zn_formate sbatch ...
#
# Override backends:
#   MLIP_BACKEND=hydragnn DFT_BACKEND=qe CASE=hea_bcc sbatch ...
#
# PREREQUISITE (UMA backend) — prefetch the UMA weights first. This job reads
# the shared HF cache in OFFLINE mode (HF_HUB_OFFLINE=1): compute nodes have no
# internet and CFS does not support fcntl.flock over DVS (OSError [Errno 524]),
# so UMA is NOT downloaded on first use. Run once before submitting:
#   sbatch scripts/download/perlmutter/download-uma-perlmutter.sh
# See docs/model-download.md ("UMA MLIP weights on Perlmutter").
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-${REPO_DEFAULT}}"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/global/cfs/projectdirs/m5216/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"

VENV_ROOT=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter
VENV="${MATSIM_FAIRCHEM_VENV:-${VENV_ROOT}/fairchem_venv}"

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

# ── modules & venv (fairchem_venv for UMA) ───────────────────────────────────
source "$REPO/scripts/setup/perlmutter/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

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
# scripts/download/perlmutter/download-uma-perlmutter.sh.
export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-${SCRATCH:-/tmp}/matsim-agents/fairchem_cache}"
mkdir -p "${FAIRCHEM_CACHE_DIR}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# ── AL backend selection (consumed by the YAML via ${...} env interpolation) ─
export MLIP_BACKEND
export DFT_BACKEND="${DFT_BACKEND:-vasp}"

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
