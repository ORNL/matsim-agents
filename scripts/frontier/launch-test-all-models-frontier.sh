#!/bin/bash
# ---------------------------------------------------------------------------
# launch-test-all-models-frontier.sh
#
# Submits one Slurm job per supported vLLM model (skipping known-broken
# DeepSeek-V4-Pro FP8). Because debug QOS allows only MaxSubmitPU=1, this
# script polls Slurm and submits the next job only when the previous one
# clears the queue. Run in background or in a tmux/screen session.
#
# Each job uses the existing smoke-vllm-{single,multi}node script with
# SMOKE_MODEL_PATH/SMOKE_MODEL_NAME overrides via sbatch --export.
#
# Usage:
#   nohup bash scripts/frontier/launch-test-all-models-frontier.sh \
#     > $PROJ/runs/launch-test-all.log 2>&1 &
#
# Summary table written to:
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
printf "jobid\tstate\tmodel\tnodes\trun_dir\n" > "$SUMMARY"

# Format: <node_count> <local_dir_name> <served_model_name>
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

wait_for_job() {
  local jid=$1
  local timeout=${2:-7200}   # 2 hr cap
  local start=$SECONDS
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do
    if (( SECONDS - start > timeout )); then
      echo "[WARN] timeout waiting for $jid" >&2
      return 1
    fi
    sleep 30
  done
}

echo "Launcher PID $$  log file: tail -f \$0.log"
echo "Will submit ${#MODELS[@]} jobs sequentially (debug QOS = 1 at a time)"
echo "Summary: $SUMMARY"
echo ""

for entry in "${MODELS[@]}"; do
  read -r NODES LOCAL_DIR SERVED_NAME <<<"$entry"
  MODEL_PATH="$PROJ/models/$LOCAL_DIR"

  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[SKIP] $SERVED_NAME — not in $MODEL_PATH"
    printf "SKIP\t-\t%s\t%s\tmissing local dir\n" "$SERVED_NAME" "$NODES" >> "$SUMMARY"
    continue
  fi

  if [[ "$NODES" == "1" ]]; then
    SCRIPT="$SINGLE"
    JOBNAME="t1-$LOCAL_DIR"
  else
    SCRIPT="$MULTI"
    JOBNAME="t${NODES}-$LOCAL_DIR"
  fi

  EXPORT="ALL,SMOKE_MODEL_PATH=$MODEL_PATH,SMOKE_MODEL_NAME=$SERVED_NAME"

  # Retry submit if queue is full
  while :; do
    if JID=$(sbatch --parsable --nodes="$NODES" -J "$JOBNAME" \
                    --export="$EXPORT" "$SCRIPT" 2>/tmp/sbatch.err); then
      [[ -n "$JID" ]] && break
    fi
    if grep -q "QOSMaxSubmitJobPerUserLimit" /tmp/sbatch.err; then
      sleep 30
      continue
    fi
    echo "[FAIL] sbatch error for $SERVED_NAME:"
    cat /tmp/sbatch.err
    printf "FAIL\t-\t%s\t%s\tsbatch error\n" "$SERVED_NAME" "$NODES" >> "$SUMMARY"
    break
  done

  if [[ -z "${JID:-}" ]]; then
    continue
  fi

  if [[ "$NODES" == "1" ]]; then
    RUN_DIR="$PROJ/runs/smoke-singlenode-$JID"
  else
    RUN_DIR="$PROJ/runs/smoke-multinode-$JID"
  fi

  echo "[$JID submitted] $JOBNAME  nodes=$NODES  $SERVED_NAME"

  wait_for_job "$JID" || true

  STATE=$(sacct -X -j "$JID" --format=State -P --noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "[$JID done] state=$STATE"
  printf "%s\t%s\t%s\t%s\t%s\n" "$JID" "$STATE" "$SERVED_NAME" "$NODES" "$RUN_DIR" >> "$SUMMARY"
done

echo ""
echo "All done. Summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY"
