#!/bin/bash
# ---------------------------------------------------------------------------
# launch-test-multinode-aurora.sh
#
# Multi-node vLLM test launcher. Submits
# deployments/aurora/jobs/job-serve-multinode-vllm-aurora.sh with SERVE_SMOKE=1
# for the models that need more than one Aurora node: the job bootstraps Ray
# across the allocation, serves the model with tensor parallelism over the PVC
# tiles, sends one chat completion, and exits. Mirrors the Frontier and
# Perlmutter multi-node launchers.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ -f "${REPO}/pyproject.toml" ]] || REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
MODEL_ROOT="${MODEL_ROOT:-${PROJ}/models}"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
SERVE="${REPO}/deployments/aurora/jobs/job-serve-multinode-vllm-aurora.sh"

TS=$(date +%Y%m%d-%H%M%S)
SUMMARY="${RUNS_ROOT}/test-multinode-aurora-${TS}.tsv"
mkdir -p "${RUNS_ROOT}"
printf "jobid\texit_status\tmodel\tnodes\ttp\trun_dir\n" >"${SUMMARY}"

# Format: "<local_dir> <served_name> <nodes> <tensor_parallel_tiles>"
# TP must divide the attention heads (64 for the 70B-class models, 48 for Mixtral).
MODELS=(
  "Llama-3.1-70B-Instruct        meta-llama/Llama-3.1-70B-Instruct        2  16"
  "Llama-3.3-70B-Instruct        meta-llama/Llama-3.3-70B-Instruct        2  16"
  "Qwen2.5-72B-Instruct          Qwen/Qwen2.5-72B-Instruct                2  16"
  "Mixtral-8x22B-Instruct-v0.1   mistralai/Mixtral-8x22B-Instruct-v0.1    2  24"
)

wait_for_job() {
  while qstat "$1" >/dev/null 2>&1; do sleep 30; done
}

for entry in "${MODELS[@]}"; do
  read -r LOCAL_DIR SERVED_NAME NODES TP <<<"${entry}"
  MODEL_PATH="${MODEL_ROOT}/${LOCAL_DIR}"
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[SKIP] ${SERVED_NAME} (missing ${MODEL_PATH})"
    continue
  fi
  if ! JID=$(qsub -N "tN-${LOCAL_DIR:0:12}" -l "select=${NODES}" \
      -v "SERVE_MODEL_PATH=${MODEL_PATH},SERVE_MODEL_NAME=${SERVED_NAME},SERVE_TP_SIZE=${TP},SERVE_SMOKE=1" \
      "${SERVE}"); then
    echo "[FAIL] ${SERVED_NAME}"
    printf "FAIL\t-\t%s\t%s\t%s\tqsub error\n" "${SERVED_NAME}" "${NODES}" "${TP}" >>"${SUMMARY}"
    continue
  fi
  echo "[${JID} submitted] ${SERVED_NAME}  nodes=${NODES} tp=${TP}"
  wait_for_job "${JID}"
  STATUS=$(qstat -xf "${JID}" 2>/dev/null | awk '/Exit_status/ {print $3; exit}' || true)
  echo "[${JID} done] exit_status=${STATUS}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${JID}" "${STATUS}" "${SERVED_NAME}" "${NODES}" "${TP}" \
    "${RUNS_ROOT}/vllm-multinode-aurora-${JID}" >>"${SUMMARY}"
done

echo "Done. Summary: ${SUMMARY}"
column -t -s $'\t' "${SUMMARY}"
