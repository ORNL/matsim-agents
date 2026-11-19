#!/bin/bash
# ---------------------------------------------------------------------------
# submit-finetune-eval-pack4-perlmutter.sh
#
# Run up to four fine-tune + evaluation matrix variants per case as one packed
# full-node job (one variant per A100) via job-finetune-eval-pack4-perlmutter.sh.
# The default packs the four HydraGNN transfer-learning strategies -- routed,
# unfrozen, frozen, scratch -- so the 20 single-GPU `-q shared` runs become five
# full-node `-q premium` jobs (all 4 GPUs used; 2x charge but top scheduling
# priority). premium/regular QOS require a full node, so they are only valid on
# these packed jobs.
#
# finetune_eval_matrix.py validates each run and writes its mlip.yaml and
# evaluation.yaml to $RUNS_ROOT/finetune-eval/<variant>/<case>/. SPECS is passed
# through the inherited environment (NOT --export) so the ';'-separated,
# space-containing string survives sbatch's comma-delimited --export parser.
#
# Usage (EVALUATION_CONFIG is required; templates in examples/evaluation/):
#   export EVALUATION_CONFIG=examples/evaluation/random-frame-interpolation.yaml
#   bash deployments/perlmutter/launchers/submit-finetune-eval-pack4-perlmutter.sh
#   VARIANTS=mace-small,mace,mace-large,mace-lora bash .../submit-...  # any <= 4
#   QOS=regular  bash .../submit-finetune-eval-pack4-perlmutter.sh
#   CASES=lifepo4-al-001,hea-bcc-al-001  bash .../submit-...   # subset
#   DRY_RUN=1    bash .../submit-...                            # print only
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
export PROJECT_ROOT="${REPO}"
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
JOB="${SCRIPT_DIR}/job-finetune-eval-pack4-perlmutter.sh"
[[ ! -f "${JOB}" ]] && { echo "ERROR: missing pack4 job script ${JOB}" >&2; exit 2; }
MATRIX="${MATRIX:-${SCRIPT_DIR}/config/finetune-eval-matrix.yaml}"
EVALUATION_CONFIG="${EVALUATION_CONFIG:?set EVALUATION_CONFIG=<evaluation YAML>, see examples/evaluation/}"
[[ ! -r "${EVALUATION_CONFIG}" ]] && { echo "ERROR: evaluation config not readable: ${EVALUATION_CONFIG}" >&2; exit 2; }
PYTHON="${MATSIM_HYDRAGNN_VENV:-${REPO}/.venv}/bin/python3"
FTEVAL_ROOT="${FTEVAL_ROOT:-${RUNS_ROOT}/finetune-eval}"
QOS="${QOS:-premium}"
VARIANTS="${VARIANTS:-hydragnn,hydragnn-unfrozen,hydragnn-frozen,hydragnn-scratch}"
IFS=',' read -ra VARIANT_LIST <<<"${VARIANTS}"
(( ${#VARIANT_LIST[@]} <= 4 )) || { echo "ERROR: at most 4 variants per node (got ${VARIANTS})" >&2; exit 2; }

RUNS="$("${PYTHON}" "${SCRIPT_DIR}/finetune_eval_matrix.py" \
  --matrix "${MATRIX}" \
  --evaluation-config "${EVALUATION_CONFIG}" \
  --out-root "${FTEVAL_ROOT}" \
  --variants "${VARIANTS}" \
  ${CASES:+--cases "${CASES}"})"
CASE_LIST=()
while IFS=$'\t' read -r _variant case _backend _out_dir; do
  [[ -n "${case}" && " ${CASE_LIST[*]-} " != *" ${case} "* ]] && CASE_LIST+=("${case}")
done <<<"${RUNS}"

nsub=0
for case in ${CASE_LIST[@]+"${CASE_LIST[@]}"}; do
  dataset="${RUNS_ROOT}/${case}/dataset.extxyz"
  if [[ ! -f "${dataset}" ]]; then
    echo "[SKIP] ${case}: dataset not found (${dataset})" >&2
    continue
  fi
  reference_manifest="${ELEMENTAL_REFERENCE_MANIFEST:-${RUNS_ROOT}/${case}/elemental_references.json}"
  if [[ ! -f "${reference_manifest}" ]]; then
    echo "[ERROR] ${case}: DFT-labelled elemental reference manifest not found (${reference_manifest})" >&2
    exit 2
  fi
  # One campaign per variant of this case, each with its own matrix inputs.
  specs=""
  while IFS=$'\t' read -r variant run_case _backend out_dir; do
    [[ "${run_case}" == "${case}" ]] || continue
    [[ -n "${specs}" ]] && specs+="; "
    specs+="CASE=${case} OUT_DIR=${out_dir} MLIP_CONFIG=${out_dir}/mlip.yaml"
    specs+=" EVALUATION_CONFIG=${out_dir}/evaluation.yaml"
  done <<<"${RUNS}"

  jobname="fte-pack4-${case}"
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "ELEMENTAL_REFERENCE_MANIFEST='${reference_manifest}' SPECS='${specs}' sbatch -q ${QOS} -J ${jobname} ${JOB}"
    continue
  fi
  if JID=$(ELEMENTAL_REFERENCE_MANIFEST="${reference_manifest}" SPECS="${specs}" \
             sbatch --parsable -q "${QOS}" -J "${jobname}" "${JOB}"); then
    echo "[submitted] ${JID}  ${jobname}  (${QOS}; variants: ${VARIANTS})"
    nsub=$((nsub + 1))
  else
    echo "[ERROR] sbatch failed for ${case}" >&2
  fi
done

echo "---------------------------------------------------------------"
if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: no jobs submitted."
else
  echo "Submitted ${nsub} packed job(s) on QOS=${QOS}."
  echo "Track with: squeue --me    |    outputs under ${FTEVAL_ROOT}/"
fi
