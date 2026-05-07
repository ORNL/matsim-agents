#!/bin/bash
# Continuation launcher: remaining single-node models only (no multi-node).
# Resumes after gemma-4-26B-A4B-it which is already running as 4535764.
set -euo pipefail

PROJ=/lustre/orion/mat746/proj-shared
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE="$SCRIPT_DIR/smoke-vllm-singlenode-frontier.sh"

TS=$(date +%Y%m%d-%H%M%S)
SUMMARY="$PROJ/runs/test-singlenode-resume-$TS.tsv"
mkdir -p "$PROJ/runs"
printf "jobid\tstate\tmodel\tnodes\trun_dir\n" > "$SUMMARY"

# Remaining single-node models (gemma-4-26B already running as 4535764)
MODELS=(
  "Qwen3.6-27B                      Qwen/Qwen3.6-27B"
  "gemma-4-31B-it                   google/gemma-4-31B-it"
  "DeepSeek-R1-Distill-Qwen-32B     deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  "Qwen3-32B                        Qwen/Qwen3-32B"
  "Qwen3.6-35B-A3B                  Qwen/Qwen3.6-35B-A3B"
)

wait_for_job() {
  local jid=$1
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 30; done
}

# First wait for the in-flight 4535764 to clear
echo "Waiting for in-flight job 4535764 to finish ..."
wait_for_job 4535764 || true

for entry in "${MODELS[@]}"; do
  read -r LOCAL_DIR SERVED_NAME <<<"$entry"
  MODEL_PATH="$PROJ/models/$LOCAL_DIR"
  [[ ! -d "$MODEL_PATH" ]] && { echo "[SKIP] $SERVED_NAME"; continue; }

  EXPORT="ALL,SMOKE_MODEL_PATH=$MODEL_PATH,SMOKE_MODEL_NAME=$SERVED_NAME"
  while :; do
    if JID=$(sbatch --parsable --nodes=1 -J "t1-$LOCAL_DIR" \
                    --export="$EXPORT" "$SINGLE" 2>/tmp/sbatch.err); then
      [[ -n "$JID" ]] && break
    fi
    grep -q QOSMaxSubmitJobPerUserLimit /tmp/sbatch.err && { sleep 30; continue; }
    echo "[FAIL] $SERVED_NAME"; cat /tmp/sbatch.err
    printf "FAIL\t-\t%s\t1\tsbatch error\n" "$SERVED_NAME" >> "$SUMMARY"
    JID=""; break
  done
  [[ -z "$JID" ]] && continue

  RUN_DIR="$PROJ/runs/smoke-singlenode-$JID"
  echo "[$JID submitted] $SERVED_NAME"
  wait_for_job "$JID"
  STATE=$(sacct -X -j "$JID" --format=State -P --noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "[$JID done] $STATE"
  printf "%s\t%s\t%s\t1\t%s\n" "$JID" "$STATE" "$SERVED_NAME" "$RUN_DIR" >> "$SUMMARY"
done

echo "Done. Summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY"
