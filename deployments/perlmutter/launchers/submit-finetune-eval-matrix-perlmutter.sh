#!/bin/bash
# ---------------------------------------------------------------------------
# submit-finetune-eval-matrix-perlmutter.sh
#
# Submits the fine-tune + evaluation matrix (config/finetune-eval-matrix.yaml:
# variants = foundation model + fine-tune recipe, x the AL-corrected paper
# datasets) as independent single-GPU jobs of job-finetune-eval-perlmutter.sh.
# finetune_eval_matrix.py validates every run and writes its mlip.yaml and
# evaluation.yaml to $RUNS_ROOT/finetune-eval/<variant>/<case>/, so runs never
# collide and can be resubmitted individually.
#
# Usage (EVALUATION_CONFIG is required; templates in examples/evaluation/):
#   export EVALUATION_CONFIG=examples/evaluation/random-frame-interpolation.yaml
#   bash deployments/perlmutter/launchers/submit-finetune-eval-matrix-perlmutter.sh
#
#   # restrict variants and/or cases (comma-separated names from the matrix):
#   VARIANTS=uma,uma-lora CASES=lifepo4-al-001,hea-bcc-al-001 \
#     bash deployments/perlmutter/launchers/submit-finetune-eval-matrix-perlmutter.sh
#
#   # print the sbatch commands without submitting:
#   DRY_RUN=1 bash deployments/perlmutter/launchers/submit-finetune-eval-matrix-perlmutter.sh
#
# Optional env: MATRIX (default config/finetune-eval-matrix.yaml), VARIANTS
# (default hydragnn,uma), CASES, SEED, RESCORE=1, QOS, WALLTIME, plus the path
# overrides of the per-job script (PROJECT_ROOT, RUNS_ROOT, ...). Change models
# or recipes by editing the matrix, not with environment variables.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
export PROJECT_ROOT="${REPO}"
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
JOB="${SCRIPT_DIR}/job-finetune-eval-perlmutter.sh"
[[ ! -f "${JOB}" ]] && { echo "ERROR: missing per-job script ${JOB}" >&2; exit 2; }
MATRIX="${MATRIX:-${SCRIPT_DIR}/config/finetune-eval-matrix.yaml}"
EVALUATION_CONFIG="${EVALUATION_CONFIG:?set EVALUATION_CONFIG=<evaluation YAML>, see examples/evaluation/}"
[[ ! -r "${EVALUATION_CONFIG}" ]] && { echo "ERROR: evaluation config not readable: ${EVALUATION_CONFIG}" >&2; exit 2; }
PYTHON="${MATSIM_HYDRAGNN_VENV:-${REPO}/.venv}/bin/python3"
FTEVAL_ROOT="${FTEVAL_ROOT:-${RUNS_ROOT}/finetune-eval}"

RUNS="$("${PYTHON}" "${SCRIPT_DIR}/finetune_eval_matrix.py" \
  --matrix "${MATRIX}" \
  --evaluation-config "${EVALUATION_CONFIG}" \
  --out-root "${FTEVAL_ROOT}" \
  --variants "${VARIANTS:-hydragnn,uma}" \
  ${CASES:+--cases "${CASES}"})"

TS=$(date +%Y%m%d-%H%M%S)
SUMMARY="${RUNS_ROOT}/finetune-eval-submit-${TS}.tsv"
mkdir -p "${RUNS_ROOT}"
printf "jobid\tvariant\tcase\tbackend\tout_dir\n" > "${SUMMARY}"

SBATCH_OPTS=()
[[ -n "${QOS:-}" ]]      && SBATCH_OPTS+=(-q "${QOS}")
[[ -n "${WALLTIME:-}" ]] && SBATCH_OPTS+=(-t "${WALLTIME}")

nsub=0
while IFS=$'\t' read -r variant case backend out_dir; do
  [[ -n "${variant}" ]] || continue
  dataset="${RUNS_ROOT}/${case}/dataset.extxyz"
  if [[ ! -f "${dataset}" ]]; then
    echo "[SKIP] ${variant}/${case}: dataset not found (${dataset})" >&2
    continue
  fi
  reference_manifest="${ELEMENTAL_REFERENCE_MANIFEST:-${RUNS_ROOT}/${case}/elemental_references.json}"
  if [[ ! -f "${reference_manifest}" ]]; then
    echo "[ERROR] ${case}: DFT-labelled elemental reference manifest not found (${reference_manifest})" >&2
    exit 2
  fi
  exports="ALL,CASE=${case},OUT_DIR=${out_dir},MLIP_CONFIG=${out_dir}/mlip.yaml"
  exports+=",EVALUATION_CONFIG=${out_dir}/evaluation.yaml"
  exports+=",ELEMENTAL_REFERENCE_MANIFEST=${reference_manifest}"
  [[ -n "${SEED:-}" ]] && exports+=",SEED=${SEED}"
  [[ "${RESCORE:-0}" == "1" ]] && exports+=",RESCORE=1"
  jobname="fte-${variant}-${case}"
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "sbatch -J ${jobname} ${SBATCH_OPTS[*]:-} --export=${exports} ${JOB}"
    continue
  fi
  if JID=$(sbatch --parsable -J "${jobname}" ${SBATCH_OPTS[@]+"${SBATCH_OPTS[@]}"} \
              --export="${exports}" "${JOB}"); then
    echo "[submitted] ${JID}  ${variant}/${case} (${backend})"
    printf "%s\t%s\t%s\t%s\t%s\n" "${JID}" "${variant}" "${case}" "${backend}" "${out_dir}" >> "${SUMMARY}"
    nsub=$((nsub + 1))
  else
    echo "[ERROR] sbatch failed for ${variant}/${case}" >&2
  fi
done <<<"${RUNS}"

echo "Submitted ${nsub} job(s). Summary: ${SUMMARY}"
