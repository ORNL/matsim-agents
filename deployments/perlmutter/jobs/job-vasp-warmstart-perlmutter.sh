#!/bin/bash
#SBATCH -A m5216
#SBATCH -J vasp-warmstart
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: HydraGNN warm-start vs VASP cold-start benchmark on Perlmutter.
#
# Mirror of job-qe-warmstart-perlmutter.sh for the VASP backend.
# Artifacts are written under:
#   $RUNS_ROOT/vasp-warmstart-$SLURM_JOB_ID/
#       vasp-warmstart/          ← pytest --basetemp (cold/warm VASP work dirs)
#       vasp-warmstart.log       ← combined stdout/stderr
#
# Submit:
#   sbatch deployments/perlmutter/jobs/job-vasp-warmstart-perlmutter.sh
#
# Override fixtures (comma-separated):
#   MATSIM_WARMSTART_FIXTURES=MoNbTaW_HEA \
#     sbatch deployments/perlmutter/jobs/job-vasp-warmstart-perlmutter.sh
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
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv
HYDRAGNN_EXAMPLE=$PROJ/HydraGNN/examples/multidataset_hpo_sc26
LOGDIR=${MATSIM_HYDRAGNN_LOGDIR:-$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64}
HYDRAGNN_BRANCH_MLP_CHECKPOINT=${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt}

VASP_LAUNCHER=${MATSIM_VASP_LAUNCHER:-$REPO/deployments/perlmutter/launchers/run-vasp-gpu-perlmutter.sh}
VASP_POTCAR_DIR=${MATSIM_VASP_POTCAR_DIR:-$REPO/external/vasp6/potcar/potpaw_PBE.64}

# Output goes under vasp-warmstart-<jobid>/ — clearly separate from qe-warmstart-<jobid>/
RUN_DIR=$RUNS_ROOT/vasp-warmstart-$SLURM_JOB_ID
WARMSTART_DIR=$RUN_DIR/vasp-warmstart
mkdir -p "$RUN_DIR" "$WARMSTART_DIR"

# ── modules & conda env ──────────────────────────────────────────────────────
source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VENV"

export PYTHONPATH=$HYDRAGNN_EXAMPLE:$PROJ/HydraGNN:${PYTHONPATH:-}
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ── VASP / warmstart env ────────────────────────────────────────────────────
export MATSIM_VASP_LAUNCHER="$VASP_LAUNCHER"
export MATSIM_VASP_POTCAR_DIR="$VASP_POTCAR_DIR"
export MATSIM_HYDRAGNN_LOGDIR="$LOGDIR"
export HYDRAGNN_BRANCH_MLP_CHECKPOINT="$HYDRAGNN_BRANCH_MLP_CHECKPOINT"
export MATSIM_VASP_MLP_DEVICE="${MATSIM_VASP_MLP_DEVICE:-cuda}"
export MATSIM_VASP_TIMEOUT_SEC="${MATSIM_VASP_TIMEOUT_SEC:-3600}"
export MATSIM_WARMSTART_FIXTURES="${MATSIM_WARMSTART_FIXTURES:-MoNbTaW_HEA}"

# Optionally pin HydraGNN routing to a fixed expert branch (e.g. 7 = OMat24).
if [[ -n "${HYDRAGNN_FORCE_BRANCH:-}" ]]; then
  export HYDRAGNN_FORCE_BRANCH
fi

# ── diagnostics ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Perlmutter VASP warm-start benchmark]"
echo "Date:         $(date)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Host:         $(hostname)"
echo "Repo:         $REPO"
echo "Venv:         $VENV"
echo "HydraGNN log: $LOGDIR"
echo "MLP ckpt:     $HYDRAGNN_BRANCH_MLP_CHECKPOINT"
echo "VASP launcher:$VASP_LAUNCHER"
echo "POTCAR dir:   $VASP_POTCAR_DIR"
echo "Fixtures:     $MATSIM_WARMSTART_FIXTURES"
echo "Force branch: ${HYDRAGNN_FORCE_BRANCH:-<unset> (softmax routing)}"
echo "Run dir:      $RUN_DIR"
echo "=========================================="

echo "[$(date)] Python: $(which python) ($(python --version 2>&1))"

# ── benchmark ───────────────────────────────────────────────────────────────
pushd "$REPO" >/dev/null
python -m pytest -xvs tests/integration/test_vasp_warmstart.py \
  --basetemp="$WARMSTART_DIR" \
  2>&1 | tee "$RUN_DIR/vasp-warmstart.log"
popd >/dev/null

echo "[$(date)] VASP warm-start benchmark complete. Artifacts in $WARMSTART_DIR"
