#!/bin/bash
# Multi-node test launcher: 4 large dense models (excludes DeepSeek-V4-Pro FP8).
# Submits one job at a time (debug QOS MaxSubmitPU=1), waits for each to finish.
#
# CONSTRAINT: vLLM requires tensor-parallel-size to divide evenly into the model's
# number of attention heads. E.g., Mixtral-8x22B has 48 heads, so TP must be a
# divisor of 48 (valid: 1,2,3,4,6,8,12,16,24,48; invalid: 5,7,10,14,32,etc).
# Frontier has 8 GPUs/node, so:
#   - 1 node  = TP=8  (divisors of 80, 64, 48: ✓all)
#   - 2 nodes = TP=16 (divisors of 80, 64, 48: ✓all)
#   - 3 nodes = TP=24 (divisors of 80? no; 64? no; 48? ✓yes)
#   - 4 nodes = TP=32 (divisors of 80? no; 64? ✓yes; 48? no)
# This script validates num_heads % TP == 0 before submitting each job.
set -euo pipefail

PROJ=/lustre/orion/mat746/proj-shared
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTI="$SCRIPT_DIR/smoke-vllm-multinode-frontier.sh"

TS=$(date +%Y%m%d-%H%M%S)
SUMMARY="$PROJ/runs/test-multinode-$TS.tsv"
mkdir -p "$PROJ/runs"
printf "jobid\tstate\tmodel\tnodes\trun_dir\n" > "$SUMMARY"

# Model metadata: <local_dir> <served_name> <nodes> <num_attn_heads>
# num_attn_heads is used to validate TP divisibility (TP must divide num_heads evenly)
declare -A MODEL_HEADS=(
  ["Llama-3.1-70B-Instruct"]=80
  ["Llama-3.3-70B-Instruct"]=80
  ["Qwen2.5-72B-Instruct"]=64
  ["Mixtral-8x22B-Instruct-v0.1"]=48
)

# Format: "<local_dir> <served_name> <nodes>"
# 3 nodes for Mixtral because 48 heads: TP=24 divides 48 evenly (24 divides 48).
MODELS=(
  "Llama-3.1-70B-Instruct        meta-llama/Llama-3.1-70B-Instruct        2"
  "Llama-3.3-70B-Instruct        meta-llama/Llama-3.3-70B-Instruct        2"
  "Qwen2.5-72B-Instruct          Qwen/Qwen2.5-72B-Instruct                2"
  "Mixtral-8x22B-Instruct-v0.1   mistralai/Mixtral-8x22B-Instruct-v0.1    3"
)

wait_for_job() {
  local jid=$1
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 30; done
}

for entry in "${MODELS[@]}"; do
  read -r LOCAL_DIR SERVED_NAME NODES <<<"$entry"
  MODEL_PATH="$PROJ/models/$LOCAL_DIR"
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[SKIP] $SERVED_NAME (missing $MODEL_PATH)"
    continue
  fi

  # Validate TP divisibility: tensor_parallel_size = nodes * 8 must divide num_attn_heads
  NUM_HEADS=${MODEL_HEADS["$LOCAL_DIR"]-0}
  TP_SIZE=$((NODES * 8))
  if [[ $NUM_HEADS -gt 0 ]] && [[ $((NUM_HEADS % TP_SIZE)) -ne 0 ]]; then
    echo "[FAIL] $SERVED_NAME: TP=$TP_SIZE does not divide evenly into $NUM_HEADS attention heads."
    echo "       Error: vLLM requires tensor-parallel-size to divide num_attn_heads."
    echo "       Fix: Use a node count N such that (N * 8) divides $NUM_HEADS."
    printf "FAIL\t-\t%s\t%s\tTP divisibility check failed: $TP_SIZE does not divide $NUM_HEADS heads\n" "$SERVED_NAME" "$NODES" >> "$SUMMARY"
    continue
  fi

  EXPORT="ALL,SMOKE_MODEL_PATH=$MODEL_PATH,SMOKE_MODEL_NAME=$SERVED_NAME"
  while :; do
    if JID=$(sbatch --parsable --nodes="$NODES" -J "tN-$LOCAL_DIR" \
                    --export="$EXPORT" "$MULTI" 2>/tmp/sbatch.err); then
      [[ -n "$JID" ]] && break
    fi
    grep -q QOSMaxSubmitJobPerUserLimit /tmp/sbatch.err && { sleep 30; continue; }
    echo "[FAIL] $SERVED_NAME"; cat /tmp/sbatch.err
    printf "FAIL\t-\t%s\t%s\tsbatch error\n" "$SERVED_NAME" "$NODES" >> "$SUMMARY"
    JID=""; break
  done
  [[ -z "$JID" ]] && continue

  RUN_DIR="$PROJ/runs/smoke-multinode-$JID"
  echo "[$JID submitted] $SERVED_NAME  nodes=$NODES"
  wait_for_job "$JID"
  STATE=$(sacct -X -j "$JID" --format=State -P --noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "[$JID done] $STATE"
  printf "%s\t%s\t%s\t%s\t%s\n" "$JID" "$STATE" "$SERVED_NAME" "$NODES" "$RUN_DIR" >> "$SUMMARY"
done

echo "Done. Summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY"
