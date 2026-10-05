#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu&hbm80g
#SBATCH --gpus-per-node=4
#SBATCH -c 64
#SBATCH -q premium
#SBATCH -t 02:00:00
set -euo pipefail
REPO="${PROJECT_ROOT:?export PROJECT_ROOT}"
ROOT="${QUALIFICATION_ROOT:?export QUALIFICATION_ROOT}"
STAGE="${QUALIFICATION_STAGE:?export QUALIFICATION_STAGE}"
SOURCE="${QUALIFICATION_SOURCE_ROOT:-$REPO}"
source "$SOURCE/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
source "$SOURCE/deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
export PYTHONPATH="$SOURCE/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
if [[ "$STAGE" == "mace" ]]; then
  PYTHON="$REPO/.venv-mace/bin/python"
elif [[ "$STAGE" == "hydragnn" ]]; then
  PYTHON="$REPO/.venv/bin/python"
  export HYDRAGNN_ROOT="$REPO/HydraGNN"
  export PYTHONPATH="$HYDRAGNN_ROOT/examples/multidataset_hpo_sc26:$HYDRAGNN_ROOT:$PYTHONPATH"
else
  configure_uma_model_artifacts "$REPO"
  PYTHON="$REPO/.venv-uma/bin/python"
fi
mkdir -p "$ROOT"
if [[ -f "$SOURCE/qualification-source-revision.txt" ]]; then
  cat "$SOURCE/qualification-source-revision.txt" > "$ROOT/$STAGE-source-revision.txt"
else
  git -C "$REPO" rev-parse HEAD > "$ROOT/$STAGE-source-revision.txt"
fi
echo "source_snapshot=$SOURCE"
"$PYTHON" "$SOURCE/deployments/perlmutter/jobs/qualify_real_models.py" \
  --repo "$REPO" --root "$ROOT" --stage "$STAGE" \
  --dft-backend "${QUALIFICATION_DFT_BACKEND:-qe}" \
  2>&1 | tee "$ROOT/$STAGE.log"
