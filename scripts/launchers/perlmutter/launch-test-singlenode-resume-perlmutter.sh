#!/bin/bash
# ---------------------------------------------------------------------------
# launch-test-singlenode-resume-perlmutter.sh
#
# Continuation launcher: submits the remaining single-node models, optionally
# blocking until a previously-submitted in-flight job (RESUME_AFTER_JOBID)
# finishes first.
#
# Usage:
#   # plain resume of remaining models:
#   bash scripts/launchers/perlmutter/launch-test-singlenode-resume-perlmutter.sh
#
#   # wait for an in-flight job before starting:
#   RESUME_AFTER_JOBID=12345678 \
#     bash scripts/launchers/perlmutter/launch-test-singlenode-resume-perlmutter.sh
#
#   # restrict the model list:
#   RESUME_MODELS="Qwen3-32B,Qwen3.6-35B-A3B" \
#     bash scripts/launchers/perlmutter/launch-test-singlenode-resume-perlmutter.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/global/cfs/projectdirs/amsc001/cm2us/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"
SMOKE="$REPO/scripts/smoke-tests/perlmutter/smoke-transformers-perlmutter.sh"

TS=$(date +%Y%m%d-%H%M%S)
SUMMARY="$PROJ/runs/test-singlenode-resume-pm-$TS.tsv"
mkdir -p "$PROJ/runs"
printf "jobid\tstate\tmodel\tnodes\trun_dir\n" > "$SUMMARY"

ALL_MODELS=(
  "Qwen3.6-27B                      Qwen/Qwen3.6-27B"
  "gemma-4-31B-it                   google/gemma-4-31B-it"
  "DeepSeek-R1-Distill-Qwen-32B     deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  "Qwen3-32B                        Qwen/Qwen3-32B"
  "Qwen3.6-35B-A3B                  Qwen/Qwen3.6-35B-A3B"
)

# Optional whitelist (comma-separated local-dir names)
declare -a MODELS=()
if [[ -n "${RESUME_MODELS:-}" ]]; then
  IFS=',' read -ra WHITE <<<"$RESUME_MODELS"
  for entry in "${ALL_MODELS[@]}"; do
    name="${entry%% *}"
    for w in "${WHITE[@]}"; do
      [[ "$name" == "$w" ]] && MODELS+=("$entry")
    done
  done
else
  MODELS=("${ALL_MODELS[@]}")
fi

wait_for_job() {
  local jid=$1
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 30; done
}

if [[ -n "${RESUME_AFTER_JOBID:-}" ]]; then
  echo "Waiting for in-flight job $RESUME_AFTER_JOBID to finish ..."
  wait_for_job "$RESUME_AFTER_JOBID" || true
fi

for entry in "${MODELS[@]}"; do
  read -r LOCAL_DIR SERVED_NAME <<<"$entry"
  MODEL_PATH="$PROJ/models/$LOCAL_DIR"
  [[ ! -d "$MODEL_PATH" ]] && { echo "[SKIP] $SERVED_NAME"; continue; }

  EXPORT="ALL,MATSIM_MODEL_DIR=$MODEL_PATH,MATSIM_MODEL_NAME=$SERVED_NAME"
  while :; do
    if JID=$(sbatch --parsable --nodes=1 -J "t1-$LOCAL_DIR" \
                    --export="$EXPORT" "$SMOKE" 2>/tmp/sbatch.err); then
      [[ -n "$JID" ]] && break
    fi
    grep -qE "QOSMaxSubmitJobPerUserLimit|QOSMaxJobsPerUserLimit|AssocMax" /tmp/sbatch.err \
      && { sleep 30; continue; }
    echo "[FAIL] $SERVED_NAME"; cat /tmp/sbatch.err
    printf "FAIL\t-\t%s\t1\tsbatch error\n" "$SERVED_NAME" >> "$SUMMARY"
    JID=""; break
  done
  [[ -z "$JID" ]] && continue

  RUN_DIR="$PROJ/runs/smoke-transformers-pm-$JID"
  echo "[$JID submitted] $SERVED_NAME"
  wait_for_job "$JID"
  STATE=$(sacct -X -j "$JID" --format=State -P --noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "[$JID done] $STATE"
  printf "%s\t%s\t%s\t1\t%s\n" "$JID" "$STATE" "$SERVED_NAME" "$RUN_DIR" >> "$SUMMARY"
done

echo "Done. Summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY"
