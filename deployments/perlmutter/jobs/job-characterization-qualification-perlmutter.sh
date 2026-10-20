#!/bin/bash
#SBATCH -J characterization-qualification
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --gpus-per-node=4
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: qualify characterization contracts with real UMA and VASP.
#
# Runs characterization_qualification.py: UMA relaxation and energy ranking of
# the MgO examples, VASP PBE references for Si, C, and 2H-SiC, the 3C-SiC
# hull-stability contract, protocol checks, and a resume that must not
# recalculate. Writes <run dir>/qualification-summary.json.
#
# Submit:
#   sbatch deployments/perlmutter/jobs/job-characterization-qualification-perlmutter.sh
# Resume an interrupted run (completed calculations are reused):
#   MATSIM_CHARACTERIZATION_DIR=<run dir> sbatch ...
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}"
[[ -f "${REPO}/pyproject.toml" ]] || {
  echo "ERROR: set PROJECT_ROOT to the matsim-agents checkout" >&2
  exit 2
}
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
# UMA runs in the isolated compatibility environment (install-uma-compat.sh).
VENV="${MATSIM_FAIRCHEM_VENV:-$REPO/.venv-uma}"
VASP_LAUNCHER="${MATSIM_VASP_LAUNCHER:-$REPO/deployments/perlmutter/launchers/run-vasp-gpu-perlmutter.sh}"
POTCAR_DIR="${MATSIM_VASP_POTCAR_DIR:-$REPO/external/vasp6/potcar/potpaw_PBE.64}"
RUN_DIR="${MATSIM_CHARACTERIZATION_DIR:-$RUNS_ROOT/characterization-qualification-${SLURM_JOB_ID:-$$}}"
mkdir -p "$RUN_DIR"

source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "$REPO/deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
configure_uma_model_artifacts "$REPO"

export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "=========================================="
echo "[Perlmutter characterization qualification]"
echo "Date:      $(date)"
echo "Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "Repo:      $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown))"
echo "Venv:      $VENV"
echo "VASP:      $VASP_LAUNCHER"
echo "POTCARs:   $POTCAR_DIR"
echo "UMA:       $MATSIM_UMA_ARTIFACT_DIR"
echo "Run dir:   $RUN_DIR"
echo "=========================================="

python "$REPO/deployments/perlmutter/jobs/characterization_qualification.py" \
  --output-root "$RUN_DIR" \
  --vasp-launcher "$VASP_LAUNCHER" \
  --potcar-dir "$POTCAR_DIR" 2>&1 | tee -a "$RUN_DIR/qualification.log"
