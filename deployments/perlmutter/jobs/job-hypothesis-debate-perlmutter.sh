#!/bin/bash
#SBATCH -J hyp-debate
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH --gpus-per-node=4
#SBATCH -c 64
# ---------------------------------------------------------------------------
# matsim-agents: LLM hypothesis + MULTI-LLM debate demonstration, showing how
# the atomistic MLIP influences the hypothesis BEFORE vs AFTER active-learning
# fine-tuning.
#
# A panel of *different* models from the local model zoo is named in a resource
# layout; the driver starts each vLLM server on its GPU, runs the real
# matsim-agents debate machinery (_debate_hypothesis_response) twice -- once
# with the pretrained-MLIP evidence, once with the fine-tuned-MLIP evidence --
# records the two transcripts, and stops the servers.
#
# Layout (deployments/perlmutter/resources/hypothesis-debate.yaml):
#   GPU 0  proposer         Qwen2.5-14B-Instruct
#   GPU 1  critic-deepseek  DeepSeek-R1-Distill-Qwen-32B
#   GPU 2  critic-gemma     gemma-4-31B-it
#   GPU 3  (spare)
#
# Smoke test (one small model as proposer+critic, ~10 min; hypothesis-debate-smoke.yaml):
#   MATSIM_DEBATE_SMOKE=1 sbatch deployments/perlmutter/jobs/job-hypothesis-debate-perlmutter.sh
#
# Full run:
#   sbatch deployments/perlmutter/jobs/job-hypothesis-debate-perlmutter.sh
# ---------------------------------------------------------------------------

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
RUNTIME_ENV="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}/deployments/common/runtime-env.sh"
[[ -f "${RUNTIME_ENV}" ]] || { echo "ERROR: export PROJECT_ROOT before submission" >&2; exit 2; }
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}")"
PROJ="$(dirname "${REPO}")"
VENV=$REPO/.venv          # debate client (matsim-agents + openai)
init_run_dirs "$PROJ" "hyp-debate" "${SLURM_JOB_ID:-$$}"
export PROJECT_ROOT="$REPO" MODEL_ROOT="${MODEL_ROOT:-$PROJ/models}"

CASES=${MATSIM_DEBATE_CASES:-"cantor-fcc-al-001 hea-bcc-al-001"}
CASE_PRETTY=${MATSIM_DEBATE_CASE_PRETTY:-}   # empty => driver's registry name
RUNS_ROOT=${MATSIM_RUNS_ROOT:-$PROJ/runs/finetune-eval}
if [[ "${MATSIM_DEBATE_SMOKE:-0}" == "1" ]]; then
  DEFAULT_RESOURCES=$REPO/deployments/perlmutter/resources/hypothesis-debate-smoke.yaml
  CRITICS=(proposer)
else
  DEFAULT_RESOURCES=$REPO/deployments/perlmutter/resources/hypothesis-debate.yaml
  CRITICS=(critic-deepseek critic-gemma)
fi
RESOURCES=${MATSIM_RESOURCES:-$DEFAULT_RESOURCES}

source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VENV"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
matsim-agents resources plan "$RESOURCES"

CRITIC_ARGS=()
for critic in "${CRITICS[@]}"; do
  CRITIC_ARGS+=( --critic-server "$critic" )
done
STATUS=0
for CASE in $CASES; do
  CASE_ARGS=( --case "$CASE" )
  [[ -n "$CASE_PRETTY" ]] && CASE_ARGS+=( --case-pretty "$CASE_PRETTY" )
  CASE_OUT="$OUTPUT_DIR/$CASE"
  mkdir -p "$CASE_OUT"
  echo "[$(date)] running hypothesis-debate driver (case=$CASE) ..."
  python "$REPO/deployments/perlmutter/jobs/hypothesis_debate_beforeafter.py" \
      "${CASE_ARGS[@]}" \
      --runs-root "$RUNS_ROOT" \
      --output-dir "$CASE_OUT" \
      --resources "$RESOURCES" \
      --proposer-server proposer \
      "${CRITIC_ARGS[@]}" \
      --debate-rounds "${MATSIM_DEBATE_ROUNDS:-2}" \
      2>&1 | tee "$RUN_DIR/hypothesis-debate-${CASE}.log"
  [[ "${PIPESTATUS[0]}" -eq 0 ]] || STATUS=1
done
echo "[$(date)] done. Artifacts in $OUTPUT_DIR"
exit "$STATUS"
