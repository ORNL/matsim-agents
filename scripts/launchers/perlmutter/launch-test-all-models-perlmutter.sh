#!/bin/bash
# ---------------------------------------------------------------------------
# launch-test-all-models-perlmutter.sh
#
# Submits one Slurm job per local HuggingFace model, sequentially. Each job
# runs scripts/smoke-tests/perlmutter/smoke-transformers-perlmutter.sh against
# a different model directory using sbatch --export overrides.
#
# Why transformers (not vLLM)?  vLLM is not installed in the Perlmutter
# hydragnn_venv. The HuggingFace `transformers + Accelerate` backend (with
# device_map="auto") is what `matsim_agents.llm.get_chat_model` already uses
# when MATSIM_LLM_PROVIDER=huggingface, and it runs natively on cu129.
# Models distribute themselves across the 4 A100s (320 GB total on 80 GB nodes)
# with no tensor-parallel divisibility constraints.
#
# Usage:
#   nohup bash scripts/launchers/perlmutter/launch-test-all-models-perlmutter.sh \
#     > $PROJ/runs/launch-test-all-pm.log 2>&1 &
#
# Override (e.g. only run a subset by editing the MODELS array, or run in
# parallel by setting MAX_CONCURRENT > 1; default = 1, sequential).
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/global/cfs/projectdirs/amsc001/cm2us/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"
SMOKE="$REPO/scripts/smoke-tests/perlmutter/smoke-transformers-perlmutter.sh"

if [[ ! -x "$SMOKE" ]]; then
  echo "ERROR: smoke script missing or not executable: $SMOKE" >&2
  exit 2
fi

TS=$(date +%Y%m%d-%H%M%S)
SUMMARY="$PROJ/runs/test-all-models-pm-$TS.tsv"
mkdir -p "$PROJ/runs"
printf "jobid\tstate\tmodel\tnodes\trun_dir\n" > "$SUMMARY"

# Format: "<nodes> <local_dir> <served_name>"
# All these models fit comfortably on 1 Perlmutter 80GB GPU node (4×80=320GB).
# Memory estimates (BF16 weights):
#   3B≈6, 14B≈28, 26B-A4B(MoE)≈52, 27B≈54, 31B≈62, 32B≈64, 35B-A3B(MoE)≈70
#   72B≈144, 8x22B(MoE)≈280  (all <320GB available on 80GB nodes).
MODELS=(
  "1 SmolLM3-3B                       HuggingFaceTB/SmolLM3-3B"
  "1 Qwen2.5-14B-Instruct             Qwen/Qwen2.5-14B-Instruct"
  "1 gemma-4-26B-A4B-it               google/gemma-4-26B-A4B-it"
  "1 Qwen3.6-27B                      Qwen/Qwen3.6-27B"
  "1 gemma-4-31B-it                   google/gemma-4-31B-it"
  "1 DeepSeek-R1-Distill-Qwen-32B     deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  "1 Qwen3-32B                        Qwen/Qwen3-32B"
  "1 Qwen3.6-35B-A3B                  Qwen/Qwen3.6-35B-A3B"
  "1 Qwen2.5-72B-Instruct             Qwen/Qwen2.5-72B-Instruct"
  "1 Mixtral-8x22B-Instruct-v0.1      mistralai/Mixtral-8x22B-Instruct-v0.1"
)

MAX_CONCURRENT=${MAX_CONCURRENT:-1}

wait_for_job() {
  local jid=$1
  local timeout=${2:-7200}
  local start=$SECONDS
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do
    if (( SECONDS - start > timeout )); then
      echo "[WARN] timeout waiting for $jid" >&2
      return 1
    fi
    sleep 30
  done
}

echo "Launcher PID $$"
echo "Will submit ${#MODELS[@]} jobs (max concurrent: $MAX_CONCURRENT)"
echo "Summary: $SUMMARY"
echo ""

declare -a SUBMITTED=()

for entry in "${MODELS[@]}"; do
  read -r NODES LOCAL_DIR SERVED_NAME <<<"$entry"
  MODEL_PATH="$PROJ/models/$LOCAL_DIR"

  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[SKIP] $SERVED_NAME — not in $MODEL_PATH"
    printf "SKIP\t-\t%s\t%s\tmissing local dir\n" "$SERVED_NAME" "$NODES" >> "$SUMMARY"
    continue
  fi

  EXPORT="ALL,MATSIM_MODEL_DIR=$MODEL_PATH,MATSIM_MODEL_NAME=$SERVED_NAME"
  JOBNAME="t${NODES}-$LOCAL_DIR"

  while :; do
    if JID=$(sbatch --parsable --nodes="$NODES" -J "$JOBNAME" \
                    --export="$EXPORT" "$SMOKE" 2>/tmp/sbatch.err); then
      [[ -n "$JID" ]] && break
    fi
    if grep -qE "QOSMaxSubmitJobPerUserLimit|QOSMaxJobsPerUserLimit|AssocMax" /tmp/sbatch.err; then
      sleep 30
      continue
    fi
    echo "[FAIL] sbatch error for $SERVED_NAME:"
    cat /tmp/sbatch.err
    printf "FAIL\t-\t%s\t%s\tsbatch error\n" "$SERVED_NAME" "$NODES" >> "$SUMMARY"
    JID=""
    break
  done
  [[ -z "${JID:-}" ]] && continue

  RUN_DIR="$PROJ/runs/smoke-transformers-pm-$JID"
  echo "[$JID submitted] $JOBNAME  nodes=$NODES  $SERVED_NAME"
  SUBMITTED+=("$JID|$SERVED_NAME|$NODES|$RUN_DIR")

  if (( MAX_CONCURRENT == 1 )); then
    wait_for_job "$JID" || true
    STATE=$(sacct -X -j "$JID" --format=State -P --noheader 2>/dev/null | head -1 | tr -d ' ')
    echo "[$JID done] state=$STATE"
    printf "%s\t%s\t%s\t%s\t%s\n" "$JID" "$STATE" "$SERVED_NAME" "$NODES" "$RUN_DIR" >> "$SUMMARY"
  fi
done

# Concurrent mode: wait for all and collect at the end
if (( MAX_CONCURRENT > 1 )); then
  for record in "${SUBMITTED[@]}"; do
    IFS='|' read -r JID SERVED_NAME NODES RUN_DIR <<<"$record"
    wait_for_job "$JID" || true
    STATE=$(sacct -X -j "$JID" --format=State -P --noheader 2>/dev/null | head -1 | tr -d ' ')
    echo "[$JID done] $SERVED_NAME state=$STATE"
    printf "%s\t%s\t%s\t%s\t%s\n" "$JID" "$STATE" "$SERVED_NAME" "$NODES" "$RUN_DIR" >> "$SUMMARY"
  done
fi

echo ""
echo "All done. Summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY"
