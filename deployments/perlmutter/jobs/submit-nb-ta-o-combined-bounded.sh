#!/bin/bash
# Submit the seven-LLM Nb-Ta-O campaign with AL training and independent DFT hulls.
# MLIP_CONFIG (optional) is a YAML whose mlip block replaces the AL config's model.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PROJECT_ROOT="$REPO"
export MATSIM_CAMPAIGN_MODE=dft
export MATSIM_CAMPAIGN_AL_CONFIG="$REPO/deployments/perlmutter/jobs/config/campaign-nb-ta-o-uma-qe-bounded.yaml"
if [[ -n "${MLIP_CONFIG:-}" ]]; then
  MLIP_CONFIG="$(cd "$(dirname "$MLIP_CONFIG")" && pwd)/$(basename "$MLIP_CONFIG")"
  export MLIP_CONFIG
fi
export MATSIM_CAMPAIGN_MAX_ITERATIONS=3 MATSIM_CAMPAIGN_MAX_CANDIDATES=3
export MATSIM_CAMPAIGN_MAX_DFT=64 MATSIM_CAMPAIGN_MAX_AL_ITERATIONS=3
export MATSIM_CAMPAIGN_MAX_NODE_HOURS=5.5
export MATSIM_CAMPAIGN_RETRAIN=1 MATSIM_CAMPAIGN_TRAIN_EPOCHS=5
export MATSIM_CAMPAIGN_PROMOTION_VALIDATION_REFERENCE_SET="${MATSIM_CAMPAIGN_PROMOTION_VALIDATION_REFERENCE_SET:?provide a DFT-labelled elemental reference JSON manifest}"
export MATSIM_CAMPAIGN_PROMOTE_MODEL=1
export MATSIM_CAMPAIGN_PROMOTION_VALIDATION_FRACTION=0.2
export MATSIM_CAMPAIGN_PROMOTION_MIN_EVALUATED_FRAMES=2
export MATSIM_CAMPAIGN_CONTINUE_ON_PROMOTION_REJECTION=1
export MATSIM_CAMPAIGN_DFT_REFINE=1 MATSIM_CAMPAIGN_DFT_REFINE_CANDIDATES=2
export MATSIM_CAMPAIGN_DFT_RELAX_STEPS=100 MATSIM_CAMPAIGN_DFT_FORCE_TOLERANCE=0.01
export MATSIM_CAMPAIGN_DFT_METHOD_SIGNATURE=qe-pbe-pslibrary-80-640-k4-o2-triplet-gamma-v1
export MATSIM_CAMPAIGN_RANDOM_SEEDS=20 MATSIM_CAMPAIGN_UNARY_RANDOM=10
export MATSIM_CAMPAIGN_REFERENCE_PROTOTYPES=1 MATSIM_CAMPAIGN_SURROGATE_HULL=1
export MATSIM_CAMPAIGN_PERTURBATION_TRIALS=3
source "$REPO/deployments/perlmutter/setup/campaign-naming.sh"
unset MATSIM_CAMPAIGN_SOURCE_REVISION MATSIM_CAMPAIGN_SOURCE_DIRTY
PYTHON="${MATSIM_PERLMUTTER_VENV:-$REPO/.venv}/bin/python3"
MLIP="$(campaign_mlip_backend "$REPO" "$PYTHON" "${MLIP_CONFIG:-$MATSIM_CAMPAIGN_AL_CONFIG}")"
source "$REPO/deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
case "$MLIP" in
  uma)
    UMA_MODEL="$(PYTHONPATH="$REPO/src" mlip_uma_model_name "$PYTHON" "${MLIP_CONFIG:-$MATSIM_CAMPAIGN_AL_CONFIG}")"
    configure_uma_model_artifacts "$REPO" "$UMA_MODEL"
    ;;
  mace) configure_mace_model_artifacts "$REPO" ;;
esac
configure_campaign_run_name "$REPO" "$MATSIM_CAMPAIGN_MODE" qe "$MLIP"
cd "$REPO"
sbatch -A m5216_g -q premium -N 16 -t 06:00:00 \
  --export=ALL --job-name=nb-ta-o-combined-bounded \
  "$@" \
  "$REPO/deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-perlmutter.sh"
