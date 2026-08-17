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
#   nohup bash deployments/frontier/launchers/launch-test-all-models-frontier.sh \
#     > $PROJ/runs/launch-test-all.log 2>&1 &
#
# Summary table written to:
#   $PROJ/runs/test-all-models-<timestamp>.tsv
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=/lustre/orion/mat746/proj-shared/matsim-agents
PROJ="$(dirname "${REPO}")"
SMOKE_DIR="$SCRIPT_DIR/../../smoke-tests/frontier"
SINGLE="$SMOKE_DIR/smoke-vllm-singlenode-frontier.sh"
MULTI="$SMOKE_DIR/smoke-vllm-multinode-frontier.sh"

TS=$(date +%Y%m%d-%H%M%S)
SUMMARY="$PROJ/runs/test-all-models-$TS.tsv"
mkdir -p "$PROJ/runs"
printf "jobid\tstate\tmodel\tnodes\trun_dir\n" > "$SUMMARY"

# Format: <node_count> <local_dir_name> <served_model_name> <num_attn_heads>
# num_attn_heads is validated: TP (= nodes*8) must divide evenly into num_heads.
# Use 0 to skip TP validation (e.g. MoE models with non-standard head counts).
MODELS=(
  # --- single node (TP=8) ---
  "1 SmolLM3-3B                       HuggingFaceTB/SmolLM3-3B           0"
  "1 Llama-3.1-8B-Instruct            meta-llama/Llama-3.1-8B-Instruct   32"
  "1 Qwen2.5-14B-Instruct             Qwen/Qwen2.5-14B-Instruct          40"
  "1 gemma-4-26B-A4B-it               google/gemma-4-26B-A4B-it          0"
  "1 Qwen3.6-27B                      Qwen/Qwen3.6-27B                   0"
  "1 gemma-4-31B-it                   google/gemma-4-31B-it              0"
  "1 DeepSeek-R1-Distill-Qwen-32B     deepseek-ai/DeepSeek-R1-Distill-Qwen-32B 64"
  "1 Qwen3-32B                        Qwen/Qwen3-32B                     0"
  "1 Qwen3.6-35B-A3B                  Qwen/Qwen3.6-35B-A3B               0"
  # --- multi node ---
  # Llama-3.x-70B: 80 heads. 80%16=0 ✓ (2N=TP16), 80%32≠0 ✗ (4N=TP32)
  "2 Llama-3.1-70B-Instruct           meta-llama/Llama-3.1-70B-Instruct  80"
  "2 Llama-3.3-70B-Instruct           meta-llama/Llama-3.3-70B-Instruct  80"
  # Qwen2.5-72B: 64 heads. 64%16=0 ✓ (2N=TP16)
  "2 Qwen2.5-72B-Instruct             Qwen/Qwen2.5-72B-Instruct          64"
  # Mixtral-8x22B: 48 heads. 48%24=0 ✓ (3N=TP24), 48%32≠0 ✗ (4N=TP32)
  "3 Mixtral-8x22B-Instruct-v0.1      mistralai/Mixtral-8x22B-Instruct-v0.1 48"
)

# Model head counts for TP validation (declared but lookup done inline above)
# TP = NODES * 8 (Frontier has 8 GCDs/node)
validate_tp() {
  local nodes=$1 num_heads=$2 served_name=$3
  local tp=$(( nodes * 8 ))
  if [[ $num_heads -gt 0 ]] && [[ $(( num_heads % tp )) -ne 0 ]]; then
    echo "[ERROR] $served_name: TP=$tp does not divide evenly into $num_heads attention heads." >&2
    echo "        vLLM requires tensor-parallel-size to divide num_attn_heads." >&2
    echo "        Fix: use a node count N such that (N*8) divides $num_heads." >&2
    return 1
  fi
  return 0
}

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
  read -r NODES LOCAL_DIR SERVED_NAME NUM_HEADS <<<"$entry"
  MODEL_PATH="$PROJ/models/$LOCAL_DIR"

  if ! validate_tp "$NODES" "$NUM_HEADS" "$SERVED_NAME"; then
    printf "SKIP\t-\t%s\t%s\tTP validation failed\n" "$SERVED_NAME" "$NODES" >> "$SUMMARY"
    continue
  fi

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
