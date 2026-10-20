#!/bin/bash
#SBATCH -J campaign-formula-e2e-all
#SBATCH -N 16
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH --gpus-per-node=4
#SBATCH -c 64
#SBATCH -t 24:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# ---------------------------------------------------------------------------
# Full local-model-panel variant of job-campaign-formula-discovery-perlmutter.sh:
# serves the same ~7-model local catalog panel as
# job-all-local-model-debate-perlmutter.sh (kimi-k2.5, glm-4.7, glm-4.7-flash,
# qwen3-235b-a22b-instruct-2507, qwen3-235b-a22b-thinking-2507, gemma-4-31b-it,
# gemma-4-26b-a4b-it -- deepseek-v3.2 and devstral-2 excluded), then runs either
# formula-discovery debate alone or a bounded, resumable campaign on the 16th
# node. MATSIM_CAMPAIGN_MODE selects debate-only, uma-only MLIP polymorph
# labelling, or the existing DFT workflow. The panel and its placement come from
# deployments/perlmutter/resources/campaign-panel{,-debate,-single}.yaml
# (override with MATSIM_RESOURCES; MODEL_ROOT defaults to $PROJ/models), served
# by `matsim-agents resources serve`.
#
# Required at submission (Slurm spools this script, so it cannot self-locate
# the checkout):
#   PROJECT_ROOT   matsim-agents checkout
#   MATSIM_CAMPAIGN_MODE=single-llm-once|debate-only|uma-only|dft (default: dft)
# Optional retraining controls:
#   MATSIM_CAMPAIGN_RETRAIN=1              train a candidate UMA checkpoint; needs
#                                          elemental DFT references to train against:
#   MATSIM_CAMPAIGN_PROMOTION_VALIDATION_REFERENCE_SET=/path/to/elemental-references.json
#                                          (or trainer.training_references in the AL config)
#   MATSIM_CAMPAIGN_TRAIN_EPOCHS=5         fine-tuning epochs per formula
#   MATSIM_CAMPAIGN_PROMOTE_MODEL=1        approve promotion and reevaluation
#   MATSIM_CAMPAIGN_PROMOTION_VALIDATION_SET=/path/to/held-out.extxyz
#   MATSIM_CAMPAIGN_PROMOTION_VALIDATION_FRACTION=0.2  alternative fresh-label split
#   MATSIM_CAMPAIGN_CONTINUE_ON_PROMOTION_REJECTION=1  retain incumbent if rejected
# Optional no-DFT validation controls:
#   MATSIM_CAMPAIGN_STATE_SOURCE=/path/to/campaign_state.json  reuse a prior registry
#   MATSIM_CAMPAIGN_RANDOM_SEEDS=50         pyXtal structures per formula
#   MATSIM_CAMPAIGN_PERTURBATION_TRIALS=5  robustness rerelaxations
#   MATSIM_CAMPAIGN_SURROGATE_HULL=1        model-specific MLIP proxy hulls
# Optional DFT hull controls:
#   MATSIM_CAMPAIGN_DFT_REFINE=0            disable DFT relaxation/hull ranking
#   MATSIM_CAMPAIGN_DFT_REFINE_CANDIDATES=1 refined phases per formula
#   MATSIM_CAMPAIGN_REFERENCE_PROTOTYPES=1   AFLOW polymorphs per reference formula
#   MATSIM_CAMPAIGN_CURATED_REFERENCES=...   optional curated manifest override
#   MATSIM_CAMPAIGN_OXYGEN_CORRECTION=0.0    O2 correction in eV/O atom
#   MATSIM_CAMPAIGN_UNARY_RANDOM=50           pyXtal unary candidates per element
#   MATSIM_CAMPAIGN_UNARY_RELAX_STEPS=200     MLIP unary relaxation steps
#
# Submit with:
#   PROJECT_ROOT=$PWD MATSIM_CAMPAIGN_MODE=uma-only sbatch -A m5216_g -q premium \
#     deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-perlmutter.sh
# Submit single-llm-once with -N 1 and debate-only with -N 15.
# ---------------------------------------------------------------------------

set -euo pipefail
export MATSIM_CAMPAIGN_MAX_DFT="${MATSIM_CAMPAIGN_MAX_DFT:-32}"
CAMPAIGN_MODE="${MATSIM_CAMPAIGN_MODE:-dft}"
[[ "$CAMPAIGN_MODE" == "single-llm-once" || "$CAMPAIGN_MODE" == "debate-only" || "$CAMPAIGN_MODE" == "uma-only" || "$CAMPAIGN_MODE" == "dft" ]] || {
  echo "ERROR: MATSIM_CAMPAIGN_MODE must be single-llm-once, debate-only, uma-only, or dft" >&2
  exit 2
}
if [[ "$CAMPAIGN_MODE" == "uma-only" ]]; then
  [[ "${MATSIM_CAMPAIGN_RETRAIN:-0}" == "0" && "${MATSIM_CAMPAIGN_PROMOTE_MODEL:-0}" == "0" ]] || {
    echo "ERROR: uma-only mode forbids retraining and model promotion" >&2
    exit 2
  }
  [[ "${MATSIM_CAMPAIGN_DFT_REFINE:-0}" == "0" ]] || {
    echo "ERROR: uma-only mode forbids DFT refinement" >&2
    exit 2
  }
fi

REPO="${PROJECT_ROOT:?export PROJECT_ROOT to the matsim-agents checkout}"
PYTHON="${MATSIM_PERLMUTTER_VENV:-$REPO/.venv}/bin/python3"
AL_CONFIG="${MATSIM_CAMPAIGN_AL_CONFIG:-$REPO/deployments/perlmutter/jobs/config/campaign-nb-ta-o-uma-qe.yaml}"
if [[ "$CAMPAIGN_MODE" == "dft" && "${MATSIM_CAMPAIGN_RETRAIN:-0}" == "1" ]]; then
  "$PYTHON" - "$AL_CONFIG" "${MATSIM_CAMPAIGN_PROMOTION_VALIDATION_REFERENCE_SET:-}" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
trainer = config.get("trainer") or {}
has_references = (
    trainer.get("training_references") is not None
    or trainer.get("validation_reference_set") is not None
)
if not has_references and not Path(sys.argv[2]).is_file():
    print(
        "ERROR: retraining requires elemental DFT references to train against: "
        "MATSIM_CAMPAIGN_PROMOTION_VALIDATION_REFERENCE_SET "
        "(DFT-labelled elemental JSON manifest), or trainer.training_references or "
        "trainer.validation_reference_set in the AL config",
        file=sys.stderr,
    )
    sys.exit(2)
PY
fi
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
PROJ="$(dirname "$REPO")"
MODELS_ROOT="${MODEL_ROOT:-$PROJ/models}"
source "$REPO/deployments/perlmutter/setup/campaign-naming.sh"
configure_campaign_run_name "$REPO" "$CAMPAIGN_MODE" "${MATSIM_CAMPAIGN_REFERENCE_BACKEND:-qe}"
CAMPAIGN_RUN_TAG="$MATSIM_CAMPAIGN_RUN_TAG"
RUN_NAME="${CAMPAIGN_RUN_TAG}--j${SLURM_JOB_ID}"
OUTPUT="${RUNS_ROOT:-$PROJ/runs}/portability/${RUN_NAME}"
MACE_MPA_CONFIG="${MATSIM_CAMPAIGN_MACE_MPA_CONFIG:-$REPO/deployments/perlmutter/jobs/config/campaign-nb-ta-o-mace-mpa-qe.yaml}"
MACE_OMAT_CONFIG="${MATSIM_CAMPAIGN_MACE_OMAT_CONFIG:-$REPO/deployments/perlmutter/jobs/config/campaign-nb-ta-o-mace-omat-qe.yaml}"
MACE_MATPES_CONFIG="${MATSIM_CAMPAIGN_MACE_MATPES_CONFIG:-$REPO/deployments/perlmutter/jobs/config/campaign-nb-ta-o-mace-matpes-qe.yaml}"
MACE_PYTHON="${MATSIM_MACE_PYTHON:-$REPO/.venv-mace/bin/python}"
HYDRAGNN_CONFIG="${MATSIM_CAMPAIGN_HYDRAGNN_CONFIG:-$REPO/deployments/perlmutter/jobs/config/campaign-nb-ta-o-hydragnn-qe.yaml}"
HYDRAGNN_PYTHON="${MATSIM_HYDRAGNN_PYTHON:-$REPO/.venv/bin/python}"
REFERENCE_BACKEND="${MATSIM_CAMPAIGN_REFERENCE_BACKEND:-qe}"
DFT_METHOD_SIGNATURE="${MATSIM_CAMPAIGN_DFT_METHOD_SIGNATURE:-qe-pbe-pslibrary-k4-o2-triplet-gamma-v1}"
if [[ "$CAMPAIGN_MODE" != "debate-only" && "$CAMPAIGN_MODE" != "single-llm-once" ]]; then
  [[ -f "$AL_CONFIG" ]] || { echo "ERROR: campaign AL config not found: $AL_CONFIG" >&2; exit 2; }
fi
[[ "$CAMPAIGN_MODE" != "dft" || "$REFERENCE_BACKEND" == "qe" || "$REFERENCE_BACKEND" == "vasp" ]] || {
  echo "ERROR: MATSIM_CAMPAIGN_REFERENCE_BACKEND must be qe or vasp" >&2
  exit 2
}

mkdir -p "$OUTPUT/servers"
RUNTIME_REVISION="$(git -C "$REPO" rev-parse HEAD)"
RUNTIME_STATUS="$(git -C "$REPO" status --porcelain)"
RUNTIME_DIRTY=0
[[ -z "$RUNTIME_STATUS" ]] || RUNTIME_DIRTY=1
printf 'source_revision=%s\nsource_dirty=%s\nruntime_revision=%s\nruntime_dirty=%s\nrun_name=%s\n' \
  "$MATSIM_CAMPAIGN_SOURCE_REVISION" \
  "${MATSIM_CAMPAIGN_SOURCE_DIRTY:?source dirty status must accompany source revision}" \
  "$RUNTIME_REVISION" "$RUNTIME_DIRTY" "$RUN_NAME" > "$OUTPUT/run_identity.txt"
mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
EXPECTED_NODES=16
[[ "$CAMPAIGN_MODE" == "debate-only" ]] && EXPECTED_NODES=15
[[ "$CAMPAIGN_MODE" == "single-llm-once" ]] && EXPECTED_NODES=1
(( ${#ALL_NODES[@]} == EXPECTED_NODES )) || {
  echo "ERROR: expected $EXPECTED_NODES nodes for $CAMPAIGN_MODE mode" >&2
  exit 2
}
CAMPAIGN_NODE=""
if [[ "$CAMPAIGN_MODE" == "uma-only" || "$CAMPAIGN_MODE" == "dft" ]]; then
  CAMPAIGN_NODE="${ALL_NODES[15]}"
fi

# ── model panel ──────────────────────────────────────────────────────────────
# The panel is a resource layout (kimi-k2.5, glm-4.7, glm-4.7-flash, both
# qwen3-235b-a22b-2507 models, gemma-4-31b-it, gemma-4-26b-a4b-it; deepseek-v3.2
# and devstral-2 are excluded because they crash vLLM on A100). `resources serve`
# starts every server on its nodes (one Ray cluster per multi-node model), reruns
# this script for the campaign stage, and stops the servers when it returns.
CAMPAIGN_SCRIPT="$REPO/deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-perlmutter.sh"
if [[ "${MATSIM_CAMPAIGN_STAGE:-serve}" == "serve" ]]; then
  case "$CAMPAIGN_MODE" in
    single-llm-once) LAYOUT=campaign-panel-single.yaml ;;
    debate-only) LAYOUT=campaign-panel-debate.yaml ;;
    *) LAYOUT=campaign-panel.yaml ;;
  esac
  RESOURCES="${MATSIM_RESOURCES:-$REPO/deployments/perlmutter/resources/$LAYOUT}"
  export MODEL_ROOT="$MODELS_ROOT" PROJECT_ROOT="$REPO"
  echo "[$(date)] Serving the model panel from $RESOURCES"
  "$PYTHON" -m matsim_agents.cli resources plan "$RESOURCES"
  exec env MATSIM_CAMPAIGN_STAGE=campaign \
    "$PYTHON" -m matsim_agents.cli resources serve "$RESOURCES" --output "$OUTPUT/servers" -- \
    bash "$CAMPAIGN_SCRIPT"
fi

# ── campaign stage: endpoints published by `resources serve` ─────────────────
: "${MATSIM_LLM_ENDPOINTS:?the campaign stage runs under matsim-agents resources serve}"
mapfile -t MODEL_ARGS < <("$PYTHON" -c '
import json, os
for name, endpoint in json.load(open(os.environ["MATSIM_LLM_ENDPOINTS"])).items():
    print("--model", name, endpoint["provider"], endpoint["model"], endpoint["base_url"], sep="\n")
')
EXPECTED_MODELS=$(( ${#MODEL_ARGS[@]} / 5 ))
echo "[$(date)] $EXPECTED_MODELS model endpoint(s) ready"

if [[ -n "${MATSIM_CAMPAIGN_STATE_SOURCE:-}" ]]; then
  [[ -f "$MATSIM_CAMPAIGN_STATE_SOURCE" ]] || {
    echo "ERROR: campaign state source not found: $MATSIM_CAMPAIGN_STATE_SOURCE" >&2
    exit 2
  }
  mkdir -p "$OUTPUT/campaign"
  cp "$MATSIM_CAMPAIGN_STATE_SOURCE" "$OUTPUT/campaign/campaign_state.json"
  echo "[$(date)] Reusing campaign registry from $MATSIM_CAMPAIGN_STATE_SOURCE"
else
  echo "[$(date)] Running campaign formula-discovery driver against $EXPECTED_MODELS local models ..."
  DISCOVERY_ROUNDS="${MATSIM_CAMPAIGN_ROUNDS:-2}"
  DISCOVERY_ARGS=()
  if [[ "$CAMPAIGN_MODE" == "single-llm-once" ]]; then
    DISCOVERY_ROUNDS=1
    DISCOVERY_ARGS+=(--single-call)
  fi
  "$PYTHON" "$REPO/deployments/perlmutter/jobs/campaign_formula_discovery.py" \
    --elements Nb Ta O \
    --oxidation-state Nb:3,4,5 --oxidation-state Ta:3,4,5 --oxidation-state O:-2 \
    --max-coefficient 6 --max-atoms 12 \
    --rounds "$DISCOVERY_ROUNDS" \
    --campaign-id "nb-ta-o-e2e-all-${SLURM_JOB_ID}" \
    --output-dir "$OUTPUT/campaign" --output-root "$OUTPUT" \
    "${DISCOVERY_ARGS[@]}" \
    "${MODEL_ARGS[@]}"
fi

if [[ "$CAMPAIGN_MODE" == "debate-only" || "$CAMPAIGN_MODE" == "single-llm-once" ]]; then
  echo "[$(date)] $CAMPAIGN_MODE campaign complete. Artifacts in $OUTPUT/campaign"
  exit 0
fi

printf -v MODEL_ARGS_QUOTED '%q ' "${MODEL_ARGS[@]}"
TRAIN_ARGS=()
if [[ "$CAMPAIGN_MODE" == "dft" && "${MATSIM_CAMPAIGN_RETRAIN:-0}" == "1" ]]; then
  TRAIN_ARGS=(
    --retrain
    --approve-retraining
    --train-script "$REPO/src/matsim_agents/training/finetune_uma.py"
    --train-epochs "${MATSIM_CAMPAIGN_TRAIN_EPOCHS:-5}"
  )
  if [[ -n "${MATSIM_CAMPAIGN_PROMOTION_VALIDATION_REFERENCE_SET:-}" ]]; then
    TRAIN_ARGS+=(--promotion-validation-reference-set "$MATSIM_CAMPAIGN_PROMOTION_VALIDATION_REFERENCE_SET")
  fi
  if [[ -n "${MATSIM_CAMPAIGN_TRAIN_LAUNCHER:-}" ]]; then
    TRAIN_ARGS+=(--train-launcher "$MATSIM_CAMPAIGN_TRAIN_LAUNCHER")
  fi
  if [[ "${MATSIM_CAMPAIGN_PROMOTE_MODEL:-0}" == "1" ]]; then
    TRAIN_ARGS+=(
      --promote-model
      --approve-model-promotion
      --promotion-max-energy-mae "${MATSIM_CAMPAIGN_PROMOTION_MAX_ENERGY_MAE:-0.1}"
      --promotion-max-force-mae "${MATSIM_CAMPAIGN_PROMOTION_MAX_FORCE_MAE:-0.2}"
      --promotion-min-relative-improvement "${MATSIM_CAMPAIGN_PROMOTION_MIN_RELATIVE_IMPROVEMENT:-0.05}"
      --promotion-min-evaluated-frames "${MATSIM_CAMPAIGN_PROMOTION_MIN_EVALUATED_FRAMES:-1}"
    )
    if [[ -n "${MATSIM_CAMPAIGN_PROMOTION_VALIDATION_SET:-}" ]]; then
      TRAIN_ARGS+=(--promotion-validation-set "$MATSIM_CAMPAIGN_PROMOTION_VALIDATION_SET")
    fi
    TRAIN_ARGS+=(
      --promotion-validation-fraction "${MATSIM_CAMPAIGN_PROMOTION_VALIDATION_FRACTION:-0}"
    )
    if [[ "${MATSIM_CAMPAIGN_CONTINUE_ON_PROMOTION_REJECTION:-0}" == "1" ]]; then
      TRAIN_ARGS+=(--continue-on-promotion-rejection)
    fi
  fi
fi
TRAIN_ARGS_QUOTED=""
if (( ${#TRAIN_ARGS[@]} > 0 )); then
  printf -v TRAIN_ARGS_QUOTED '%q ' "${TRAIN_ARGS[@]}"
fi
DFT_REFINEMENT_ARGS=()
if [[ "$CAMPAIGN_MODE" == "dft" && "${MATSIM_CAMPAIGN_DFT_REFINE:-1}" == "1" ]]; then
  REFERENCE_DIR="$OUTPUT/campaign/dft-references"
  REFERENCE_PREP_ARGS=(
    --output-dir "$REFERENCE_DIR"
    --backend "$REFERENCE_BACKEND"
    --max-prototypes-per-formula "${MATSIM_CAMPAIGN_REFERENCE_PROTOTYPES:-1}"
    --oxygen-correction-eV-per-atom "${MATSIM_CAMPAIGN_OXYGEN_CORRECTION:-0.0}"
  )
  if [[ -n "${MATSIM_CAMPAIGN_CURATED_REFERENCES:-}" ]]; then
    REFERENCE_PREP_ARGS+=(--curated-manifest "$MATSIM_CAMPAIGN_CURATED_REFERENCES")
  fi
  if [[ -n "${MATSIM_CAMPAIGN_REFERENCE_FORMULAS:-}" ]]; then
    read -r -a REFERENCE_FORMULAS <<<"$MATSIM_CAMPAIGN_REFERENCE_FORMULAS"
    REFERENCE_PREP_ARGS+=(--competing-formulas "${REFERENCE_FORMULAS[@]}")
  fi
  "$PYTHON" "$REPO/deployments/perlmutter/jobs/prepare_nb_ta_o_references.py" \
    "${REFERENCE_PREP_ARGS[@]}"
  DFT_REFINEMENT_ARGS=(
    --dft-reference-structures "$REFERENCE_DIR/reference_structures.json"
    --dft-method-signature "$DFT_METHOD_SIGNATURE"
    --dft-refine-candidates "${MATSIM_CAMPAIGN_DFT_REFINE_CANDIDATES:-1}"
    --dft-relax-max-steps "${MATSIM_CAMPAIGN_DFT_RELAX_STEPS:-100}"
    --dft-force-tolerance "${MATSIM_CAMPAIGN_DFT_FORCE_TOLERANCE:-0.02}"
  )
fi
DFT_REFINEMENT_ARGS_QUOTED=""
if (( ${#DFT_REFINEMENT_ARGS[@]} > 0 )); then
  printf -v DFT_REFINEMENT_ARGS_QUOTED '%q ' "${DFT_REFINEMENT_ARGS[@]}"
fi
TRAIN_ARGS_QUOTED="$DFT_REFINEMENT_ARGS_QUOTED$TRAIN_ARGS_QUOTED"
EXECUTION_ARGS=(--execution-mode "$CAMPAIGN_MODE")
[[ "$CAMPAIGN_MODE" == "dft" ]] && EXECUTION_ARGS+=(--approve-dft)
printf -v EXECUTION_ARGS_QUOTED '%q ' "${EXECUTION_ARGS[@]}"
VALIDATION_ARGS=(
  --validation-config "$MACE_MPA_CONFIG"
  --validation-python "$MACE_PYTHON"
  --validation-config "$MACE_OMAT_CONFIG"
  --validation-python "$MACE_PYTHON"
  --validation-config "$MACE_MATPES_CONFIG"
  --validation-python "$MACE_PYTHON"
  --validation-config "$HYDRAGNN_CONFIG"
  --validation-python "$HYDRAGNN_PYTHON"
  --perturbation-trials "${MATSIM_CAMPAIGN_PERTURBATION_TRIALS:-5}"
  --perturbation-scale-A "${MATSIM_CAMPAIGN_PERTURBATION_SCALE_A:-0.05}"
  --perturbation-seed "${MATSIM_CAMPAIGN_PERTURBATION_SEED:-20261001}"
  --surrogate-unary-max-steps "${MATSIM_CAMPAIGN_UNARY_RELAX_STEPS:-200}"
  --surrogate-unary-fmax "${MATSIM_CAMPAIGN_UNARY_FMAX:-0.02}"
  --surrogate-unary-maxstep "${MATSIM_CAMPAIGN_UNARY_MAXSTEP:-0.01}"
  --surrogate-minimum-unique-unary "${MATSIM_CAMPAIGN_MIN_UNIQUE_UNARY:-2}"
)
if [[ ( "$CAMPAIGN_MODE" == "uma-only" || "$CAMPAIGN_MODE" == "dft" ) && "${MATSIM_CAMPAIGN_SURROGATE_HULL:-1}" == "1" ]]; then
  SURROGATE_REFERENCE_DIR="$OUTPUT/campaign/surrogate-references"
  "$PYTHON" "$REPO/deployments/perlmutter/jobs/prepare_nb_ta_o_references.py" \
    --output-dir "$SURROGATE_REFERENCE_DIR" \
    --backend qe \
    --max-prototypes-per-formula "${MATSIM_CAMPAIGN_REFERENCE_PROTOTYPES:-1}" \
    --expand-unary-polymorphs \
    --unary-random "${MATSIM_CAMPAIGN_UNARY_RANDOM:-50}" \
    --unary-random-seed "${MATSIM_CAMPAIGN_UNARY_SEED:-20261001}"
  VALIDATION_ARGS+=(
    --surrogate-reference-structures "$SURROGATE_REFERENCE_DIR/reference_structures.json"
  )
fi
printf -v VALIDATION_ARGS_QUOTED '%q ' "${VALIDATION_ARGS[@]}"
echo "[$(date)] Running bounded $CAMPAIGN_MODE campaign on $CAMPAIGN_NODE ..."
srun --nodes=1 --ntasks=1 --nodelist="$CAMPAIGN_NODE" --overlap \
  --gpus-per-node=1 --cpus-per-task=16 \
  bash -c "
    set -euo pipefail
    source '$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh'
    load_perlmutter_modules_gpu
    source '$REPO/deployments/perlmutter/setup/model-artifacts-perlmutter.sh'
    configure_mace_model_artifacts '$REPO'
    source '$REPO/.venv-uma/bin/activate'
    export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
    export HF_HUB_OFFLINE=\"\${HF_HUB_OFFLINE:-1}\" TRANSFORMERS_OFFLINE=\"\${TRANSFORMERS_OFFLINE:-1}\"
    export PROJECT_ROOT='$REPO'
    # Present the campaign node as a one-node, four-GPU allocation to the resource plan.
    export SLURM_JOB_NODELIST='$CAMPAIGN_NODE' SLURM_JOB_NUM_NODES=1 SLURM_GPUS_ON_NODE=4
    cd '$REPO'
    python3 deployments/perlmutter/jobs/campaign_execute.py \
      --campaign-state '$OUTPUT/campaign/campaign_state.json' \
      --al-config '$AL_CONFIG' \
      --review-rounds \"\${MATSIM_CAMPAIGN_REVIEW_ROUNDS:-2}\" \
      --minimum-review-agreement \"\${MATSIM_CAMPAIGN_REVIEW_AGREEMENT:-1.0}\" \
      \${MATSIM_CAMPAIGN_FINAL_REVIEW_ARGS:-} \
      --formulas-per-iteration \"\${MATSIM_CAMPAIGN_FORMULAS_PER_ITERATION:-1}\" \
      --acquisition-mode \"\${MATSIM_CAMPAIGN_ACQUISITION_MODE:-insertion-order}\" \
      --acquisition-seed \"\${MATSIM_CAMPAIGN_ACQUISITION_SEED:-0}\" \
      --lambda-initial \"\${MATSIM_CAMPAIGN_LAMBDA_INITIAL:-0.5}\" \
      --lambda-maximum \"\${MATSIM_CAMPAIGN_LAMBDA_MAXIMUM:-0.8}\" \
      --lambda-update-rate \"\${MATSIM_CAMPAIGN_LAMBDA_UPDATE_RATE:-0.15}\" \
      --minimum-exploitation-fraction \"\${MATSIM_CAMPAIGN_MIN_EXPLOITATION:-0.2}\" \
      --minimum-exploration-fraction \"\${MATSIM_CAMPAIGN_MIN_EXPLORATION:-0.2}\" \
      --maximum-per-relaxed-family \"\${MATSIM_CAMPAIGN_MAX_PER_FAMILY:-1}\" \
      --reserved-dft-per-formula \"\${MATSIM_CAMPAIGN_RESERVED_DFT_PER_FORMULA:-0}\" \
      --reserved-node-hours-per-formula \"\${MATSIM_CAMPAIGN_RESERVED_NODE_HOURS_PER_FORMULA:-0}\" \
      \${MATSIM_CAMPAIGN_STOPPING_ARGS:-} \
      --max-iterations \"\${MATSIM_CAMPAIGN_MAX_ITERATIONS:-22}\" \
      --max-candidates \"\${MATSIM_CAMPAIGN_MAX_CANDIDATES:-22}\" \
      --max-dft-calculations \"\${MATSIM_CAMPAIGN_MAX_DFT:-6}\" \
      --max-al-iterations \"\${MATSIM_CAMPAIGN_MAX_AL_ITERATIONS:-3}\" \
      --max-node-hours \"\${MATSIM_CAMPAIGN_MAX_NODE_HOURS:-8}\" \
      --n-random \"\${MATSIM_CAMPAIGN_RANDOM_SEEDS:-50}\" \
      --relax-maxiter \"\${MATSIM_CAMPAIGN_RELAX_MAXITER:-200}\" \
      --degeneracy-tolerance-ev-per-atom \"\${MATSIM_CAMPAIGN_DEGENERACY_TOLERANCE_EV_PER_ATOM:-0.01}\" \
      --retry-failed --retry-inconclusive \
      $EXECUTION_ARGS_QUOTED \
      $VALIDATION_ARGS_QUOTED \
      $TRAIN_ARGS_QUOTED \
      $MODEL_ARGS_QUOTED
  " 2>&1 | tee "$OUTPUT/campaign-execute.log"

echo "[$(date)] campaign complete. Artifacts in $OUTPUT/campaign"
