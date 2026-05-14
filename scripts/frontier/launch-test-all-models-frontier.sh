#!/bin/bash
# ---------------------------------------------------------------------------
# launch-test-all-models-frontier.sh
#
# Submits one Slurm job per supported vLLM model (skipping known-broken
# DeepSeek-V4-Pro FP8). Jobs are chained via --dependency=afterany so they
# run sequentially regardless of debug-QOS one-job-at-a-time limits.
#
# Each job uses the existing smoke-vllm-{single,multi}node script with
# SMOKE_MODEL_PATH/SMOKE_MODEL_NAME overrides via sbatch --export.
#
# Usage:
#   bash scripts/frontier/launch-test-all-models-frontier.sh
#
# The summary table of (jobid, model, nodes) is written to:
#   $PROJ/runs/test-all-models-<timestamp>.tsv
# ---------------------------------------------------------------------------
set -euo pipefail

PROJ=/lustre/orion/mat746/proj-shared
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE="$SCRIPT_DIR/smoke-vllm-singlenode-frontier.sh"
MULTI="$SCRIPT_DIR/smoke-vllm-multinode-frontier.sh"

TS=$(date +%Y%m%d-%H%M%S)
SUMMARY="$PROJ/runs/test-all-models-$TS.tsv"
mkdir -p "$PROJ/runs"
printf "jobid\tmodel\tnodes\trun_dir\n" > "$SUMMARY"

# Format: <node_count> <local_dir_name> <served_model_name>
# Single-node (≤ ~35B): TP=8 on 1 node
# Multi-node  (≥ 70B):  TP=32 on 4 nodes
MODELS=(
  # --- single node ---
  "1 SmolLM3-3B                       HuggingFaceTB/SmolLM3-3B"
  "1 Llama-3.1-8B-Instruct            meta-llama/Llama-3.1-8B-Instruct"
  "1 Qwen2.5-14B-Instruct             Qwen/Qwen2.5-14B-Instruct"
  "1 gemma-4-26B-A4B-it               google/gemma-4-26B-A4B-it"
  "1 Qwen3.6-27B                      Qwen/Qwen3.6-27B"
  "1 gemma-4-31B-it                   google/gemma-4-31B-it"
  "1 DeepSeek-R1-Distill-Qwen-32B     deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  "1 Qwen3-32B                        Qwen/Qwen3-32B"
  "1 Qwen3.6-35B-A3B                  Qwen/Qwen3.6-35B-A3B"
  # --- multi node (4 × 8 GCDs = TP32) ---
  "4 Llama-3.1-70B-Instruct           meta-llama/Llama-3.1-70B-Instruct"
  "4 Llama-3.3-70B-Instruct           meta-llama/Llama-3.3-70B-Instruct"
  "4 Qwen2.5-72B-Instruct             Qwen/Qwen2.5-72B-Instruct"
  "4 Mixtral-8x22B-Instruct-v0.1      mistralai/Mixtral-8x22B-Instruct-v0.1"
)

PREV_JOB=""
echo "Submitting ${#MODELS[@]} chained jobs (debug QOS, --dependency=afterany)"
echo "Summary file: $SUMMARY"
echo ""

for entry in "${MODELS[@]}"; do
  read -r NODES LOCAL_DIR SERVED_NAME <<<"$entry"
  MODEL_PATH="$PROJ/models/$LOCAL_DIR"

  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[SKIP] $SERVED_NAME — not in $MODEL_PATH"
    printf "SKIP\t%s\t%s\tmissing local dir\n" "$SERVED_NAME" "$NODES" >> "$SUMMARY"
    continue
  fi

  if [[ "$NODES" == "1" ]]; then
    SCRIPT="$SINGLE"
    JOBNAME="t1-$LOCAL_DIR"
  else
    SCRIPT="$MULTI"
    JOBNAME="t${NODES}-$LOCAL_DIR"
  fi

  DEP=""
  [[ -n "$PREV_JOB" ]] && DEP="--dependency=afterany:$PREV_JOB"

  EXPORT="ALL,SMOKE_MODEL_PATH=$MODEL_PATH,SMOKE_MODEL_NAME=$SERVED_NAME"

  JID=$(sbatch --parsable --nodes="$NODES" -J "$JOBNAME" \
    --export="$EXPORT" $DEP "$SCRIPT")

  if [[ -z "$JID" ]]; then
    echo "[FAIL] sbatch returned empty for $SERVED_NAME"
    printf "FAIL\t%s\t%s\tsbatch error\n" "$SERVED_NAME" "$NODES" >> "$SUMMARY"
    continue
  fi

  if [[ "$NODES" == "1" ]]; then
    RUN_DIR="$PROJ/runs/smoke-singlenode-$JID"
  else
    RUN_DIR="$PROJ/runs/smoke-multinode-$JID"
  fi

  echo "[$JID] $JOBNAME  nodes=$NODES  $SERVED_NAME"
  printf "%s\t%s\t%s\t%s\n" "$JID" "$SERVED_NAME" "$NODES" "$RUN_DIR" >> "$SUMMARY"
  PREV_JOB="$JID"
done

echo ""
echo "All submitted. Watch with:"
echo "  squeue -u \$USER"
echo "  cat $SUMMARY"
