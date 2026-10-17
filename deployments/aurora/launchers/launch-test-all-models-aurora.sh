#!/bin/bash
# ---------------------------------------------------------------------------
# launch-test-all-models-aurora.sh
#
# Submits one PBS job per local model, sequentially by default. Each job runs
# deployments/aurora/smoke-tests/smoke-vllm-singlenode-aurora.sh (vLLM on XPU
# tiles) against a different model via `qsub -v` overrides. Mirrors the
# Frontier and Perlmutter all-models launchers.
#
# Usage (from a login node):
#   nohup bash deployments/aurora/launchers/launch-test-all-models-aurora.sh \
#     > $PROJ/runs/launch-test-all-aurora.log 2>&1 &
#
# MAX_CONCURRENT > 1 submits every job first and collects states at the end.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ -f "${REPO}/pyproject.toml" ]] || REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
MODEL_ROOT="${MODEL_ROOT:-${PROJ}/models}"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
SMOKE="${REPO}/deployments/aurora/smoke-tests/smoke-vllm-singlenode-aurora.sh"
[[ -f "${SMOKE}" ]] || { echo "ERROR: smoke script missing: ${SMOKE}" >&2; exit 2; }

TS=$(date +%Y%m%d-%H%M%S)
SUMMARY="${RUNS_ROOT}/test-all-models-aurora-${TS}.tsv"
mkdir -p "${RUNS_ROOT}"
printf "jobid\texit_status\tmodel\ttp\trun_dir\n" >"${SUMMARY}"

# Format: "<local_dir> <served_name> <tensor_parallel_tiles>"
# One Aurora node = 12 PVC tiles x 64 GB. TP must divide the attention heads.
MODELS=(
  "SmolLM3-3B                    HuggingFaceTB/SmolLM3-3B                  2"
  "Llama-3.1-8B-Instruct         meta-llama/Llama-3.1-8B-Instruct          2"
  "Qwen2.5-14B-Instruct          Qwen/Qwen2.5-14B-Instruct                 2"
  "gemma-4-26B-A4B-it            google/gemma-4-26B-A4B-it                 4"
  "Qwen3.6-27B                   Qwen/Qwen3.6-27B                          4"
  "gemma-4-31B-it                google/gemma-4-31B-it                     4"
  "DeepSeek-R1-Distill-Qwen-32B  deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  4"
  "Qwen3.6-35B-A3B               Qwen/Qwen3.6-35B-A3B                      4"
  "Llama-3.1-70B-Instruct        meta-llama/Llama-3.1-70B-Instruct         8"
  "Qwen2.5-72B-Instruct          Qwen/Qwen2.5-72B-Instruct                 8"
  "Mixtral-8x22B-Instruct-v0.1   mistralai/Mixtral-8x22B-Instruct-v0.1     8"
)
MAX_CONCURRENT=${MAX_CONCURRENT:-1}

wait_for_job() {
  while qstat "$1" >/dev/null 2>&1; do sleep 30; done
}

exit_status() {
  qstat -xf "$1" 2>/dev/null | awk '/Exit_status/ {print $3; exit}' || true
}

declare -a SUBMITTED=()
for entry in "${MODELS[@]}"; do
  read -r LOCAL_DIR SERVED_NAME TP <<<"${entry}"
  MODEL_PATH="${MODEL_ROOT}/${LOCAL_DIR}"
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[SKIP] ${SERVED_NAME} — not in ${MODEL_PATH}"
    printf "SKIP\t-\t%s\t%s\tmissing local dir\n" "${SERVED_NAME}" "${TP}" >>"${SUMMARY}"
    continue
  fi
  if ! JID=$(qsub -N "t1-${LOCAL_DIR:0:12}" \
      -v "SMOKE_MODEL_PATH=${MODEL_PATH},SMOKE_MODEL_NAME=${SERVED_NAME},TP_SIZE=${TP}" \
      "${SMOKE}"); then
    echo "[FAIL] qsub error for ${SERVED_NAME}"
    printf "FAIL\t-\t%s\t%s\tqsub error\n" "${SERVED_NAME}" "${TP}" >>"${SUMMARY}"
    continue
  fi
  RUN_DIR="${RUNS_ROOT}/smoke-vllm-singlenode-${JID}"
  echo "[${JID} submitted] ${SERVED_NAME}  tp=${TP}"
  SUBMITTED+=("${JID}|${SERVED_NAME}|${TP}|${RUN_DIR}")
  if ((MAX_CONCURRENT == 1)); then
    wait_for_job "${JID}"
    STATUS=$(exit_status "${JID}")
    echo "[${JID} done] exit_status=${STATUS}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${JID}" "${STATUS}" "${SERVED_NAME}" "${TP}" "${RUN_DIR}" >>"${SUMMARY}"
  fi
done

if ((MAX_CONCURRENT > 1)); then
  for record in "${SUBMITTED[@]}"; do
    IFS='|' read -r JID SERVED_NAME TP RUN_DIR <<<"${record}"
    wait_for_job "${JID}"
    STATUS=$(exit_status "${JID}")
    echo "[${JID} done] ${SERVED_NAME} exit_status=${STATUS}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${JID}" "${STATUS}" "${SERVED_NAME}" "${TP}" "${RUN_DIR}" >>"${SUMMARY}"
  done
fi

echo "All done. Summary: ${SUMMARY}"
column -t -s $'\t' "${SUMMARY}"
