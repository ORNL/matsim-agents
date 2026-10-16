#!/bin/bash
# ---------------------------------------------------------------------------
# launch-test-singlenode-resume-aurora.sh
#
# Continuation launcher: submits the remaining single-node models to
# deployments/aurora/smoke-tests/smoke-vllm-singlenode-aurora.sh, optionally
# waiting for a previously submitted job (RESUME_AFTER_JOBID) first. Mirrors
# the Frontier and Perlmutter resume launchers.
#
# Usage:
#   bash deployments/aurora/launchers/launch-test-singlenode-resume-aurora.sh
#   RESUME_AFTER_JOBID=12345.aurora-pbs-0001 bash ...
#   RESUME_MODELS="Qwen3.6-27B,gemma-4-31B-it" bash ...
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ -f "${REPO}/pyproject.toml" ]] || REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
MODEL_ROOT="${MODEL_ROOT:-${PROJ}/models}"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
SMOKE="${REPO}/deployments/aurora/smoke-tests/smoke-vllm-singlenode-aurora.sh"

TS=$(date +%Y%m%d-%H%M%S)
SUMMARY="${RUNS_ROOT}/test-singlenode-resume-aurora-${TS}.tsv"
mkdir -p "${RUNS_ROOT}"
printf "jobid\texit_status\tmodel\ttp\trun_dir\n" >"${SUMMARY}"

# Format: "<local_dir> <served_name> <tensor_parallel_tiles>"
ALL_MODELS=(
  "Qwen3.6-27B                   Qwen/Qwen3.6-27B                          4"
  "gemma-4-31B-it                google/gemma-4-31B-it                     4"
  "DeepSeek-R1-Distill-Qwen-32B  deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  4"
  "Qwen3.6-35B-A3B               Qwen/Qwen3.6-35B-A3B                      4"
)

declare -a MODELS=()
if [[ -n "${RESUME_MODELS:-}" ]]; then
  IFS=',' read -ra WHITE <<<"${RESUME_MODELS}"
  for entry in "${ALL_MODELS[@]}"; do
    name="${entry%% *}"
    for w in "${WHITE[@]}"; do
      [[ "${name}" == "${w}" ]] && MODELS+=("${entry}")
    done
  done
else
  MODELS=("${ALL_MODELS[@]}")
fi

wait_for_job() {
  while qstat "$1" >/dev/null 2>&1; do sleep 30; done
}

if [[ -n "${RESUME_AFTER_JOBID:-}" ]]; then
  echo "Waiting for in-flight job ${RESUME_AFTER_JOBID} to finish ..."
  wait_for_job "${RESUME_AFTER_JOBID}"
fi

for entry in "${MODELS[@]}"; do
  read -r LOCAL_DIR SERVED_NAME TP <<<"${entry}"
  MODEL_PATH="${MODEL_ROOT}/${LOCAL_DIR}"
  [[ -d "${MODEL_PATH}" ]] || { echo "[SKIP] ${SERVED_NAME}"; continue; }
  if ! JID=$(qsub -N "t1-${LOCAL_DIR:0:12}" \
      -v "SMOKE_MODEL_PATH=${MODEL_PATH},SMOKE_MODEL_NAME=${SERVED_NAME},TP_SIZE=${TP}" \
      "${SMOKE}"); then
    echo "[FAIL] ${SERVED_NAME}"
    printf "FAIL\t-\t%s\t%s\tqsub error\n" "${SERVED_NAME}" "${TP}" >>"${SUMMARY}"
    continue
  fi
  RUN_DIR="${RUNS_ROOT}/smoke-vllm-singlenode-${JID}"
  echo "[${JID} submitted] ${SERVED_NAME}"
  wait_for_job "${JID}"
  STATUS=$(qstat -xf "${JID}" 2>/dev/null | awk '/Exit_status/ {print $3; exit}' || true)
  echo "[${JID} done] exit_status=${STATUS}"
  printf "%s\t%s\t%s\t%s\t%s\n" "${JID}" "${STATUS}" "${SERVED_NAME}" "${TP}" "${RUN_DIR}" >>"${SUMMARY}"
done

echo "Done. Summary: ${SUMMARY}"
column -t -s $'\t' "${SUMMARY}"
