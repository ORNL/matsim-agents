#!/bin/bash
#SBATCH -J campaign-formula-vasp
#SBATCH -N 16
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH --gpus-per-node=4
#SBATCH -c 64
#SBATCH -t 24:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

REPO="${PROJECT_ROOT:?export PROJECT_ROOT to the matsim-agents checkout}"
export MATSIM_VASP_BIN="${MATSIM_VASP_BIN:-$REPO/external/vasp6/src/vasp.6.6.1/bin/vasp_std}"
export MATSIM_VASP_POTCAR_DIR="${MATSIM_VASP_POTCAR_DIR:-$REPO/external/vasp6/potcar/potpaw_PBE.64}"
export MATSIM_CAMPAIGN_AL_CONFIG="$REPO/deployments/perlmutter/jobs/config/campaign-nb-ta-o-uma-vasp.yaml"
export MATSIM_CAMPAIGN_REFERENCE_BACKEND=vasp
export MATSIM_CAMPAIGN_DFT_METHOD_SIGNATURE="vasp-6.6.1-pbe64-encut520-kspacing0.25-o2-triplet-v1"
export MATSIM_CAMPAIGN_RUN_TAG=campaign-formula-e2e-all-vasp

[[ -x "$MATSIM_VASP_BIN" ]] || {
  echo "ERROR: VASP binary is not executable: $MATSIM_VASP_BIN" >&2
  exit 2
}
[[ -d "$MATSIM_VASP_POTCAR_DIR" ]] || {
  echo "ERROR: VASP POTCAR directory does not exist: $MATSIM_VASP_POTCAR_DIR" >&2
  exit 2
}
for element in Nb Ta O; do
  [[ -f "$MATSIM_VASP_POTCAR_DIR/$element/POTCAR" ]] || {
    echo "ERROR: missing $element POTCAR under $MATSIM_VASP_POTCAR_DIR" >&2
    exit 2
  }
done

exec bash "$REPO/deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-perlmutter.sh"