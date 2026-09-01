#!/bin/bash
# ---------------------------------------------------------------------------
# launch-test-multinode-perlmutter.sh
#
# Multi-node HuggingFace test launcher. Submits the torchrun-aware smoke job
# (smoke-transformers-multinode-perlmutter.sh) which runs one rank per GPU
# (4 ranks/node × N nodes) and shards the model with transformers'
# tp_plan="auto" tensor-parallel planner over NCCL.
#
# No DeepSpeed / no vLLM dependency; pure torch.distributed + transformers.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
SMOKE="$REPO/deployments/perlmutter/smoke-tests/smoke-transformers-multinode-perlmutter.sh"

TS=$(date +%Y%m%d-%H%M%S)
SUMMARY="$PROJ/runs/test-multinode-pm-$TS.tsv"
mkdir -p "$PROJ/runs"
printf "jobid\tstate\tmodel\tnodes\trun_dir\n" > "$SUMMARY"

# Format: "<local_dir> <served_name> <nodes>"
# Only the largest models are listed (the smaller ones live in the all-models
# launcher with N=1).
MODELS=(
  "Qwen2.5-72B-Instruct          Qwen/Qwen2.5-72B-Instruct                2"
  "Mixtral-8x22B-Instruct-v0.1   mistralai/Mixtral-8x22B-Instruct-v0.1    2"
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

  EXPORT="ALL,MATSIM_MODEL_DIR=$MODEL_PATH,MATSIM_MODEL_NAME=$SERVED_NAME"
  while :; do
    if JID=$(sbatch --parsable --nodes="$NODES" -J "tN-$LOCAL_DIR" \
                    --export="$EXPORT" "$SMOKE" 2>/tmp/sbatch.err); then
      [[ -n "$JID" ]] && break
    fi
    grep -qE "QOSMaxSubmitJobPerUserLimit|QOSMaxJobsPerUserLimit|AssocMax" /tmp/sbatch.err \
      && { sleep 30; continue; }
    echo "[FAIL] $SERVED_NAME"; cat /tmp/sbatch.err
    printf "FAIL\t-\t%s\t%s\tsbatch error\n" "$SERVED_NAME" "$NODES" >> "$SUMMARY"
    JID=""; break
  done
  [[ -z "$JID" ]] && continue

  RUN_DIR="$PROJ/runs/smoke-transformers-pm-$JID"
  echo "[$JID submitted] $SERVED_NAME  nodes=$NODES"
  wait_for_job "$JID"
  STATE=$(sacct -X -j "$JID" --format=State -P --noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "[$JID done] $STATE"
  printf "%s\t%s\t%s\t%s\t%s\n" "$JID" "$STATE" "$SERVED_NAME" "$NODES" "$RUN_DIR" >> "$SUMMARY"
done

echo "Done. Summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY"
