#!/bin/bash
# ---------------------------------------------------------------------------
# submit-finetune-eval-pack4-hydragnn-perlmutter.sh
#
# Re-run the HydraGNN fine-tune+eval block of the paper as FIVE packed
# full-node jobs (one per AL-corrected case), each running the four
# transfer-learning strategies -- routed, unfrozen, frozen, scratch -- on the
# node's four A100 GPUs concurrently via job-finetune-eval-pack4-perlmutter.sh.
#
# 20 single-GPU `-q shared` campaigns  ->  5 full-node `-q premium` jobs
# (all 4 GPUs used; 2x charge but top scheduling priority). premium/regular
# QOS require a full node, so they are only valid on these packed jobs.
#
# BACKEND and SPECS are passed through the inherited environment (NOT
# --export) so the ';'-separated, space-containing SPECS string survives
# sbatch's comma-delimited --export parser intact.
#
# Usage:
#   bash deployments/perlmutter/launchers/submit-finetune-eval-pack4-hydragnn-perlmutter.sh
#   QOS=regular  bash .../submit-finetune-eval-pack4-hydragnn-perlmutter.sh
#   CASES=lifepo4-al-001,hea-bcc-al-001  bash .../submit-...   # subset
#   DRY_RUN=1    bash .../submit-...                            # print only
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
JOB="${SCRIPT_DIR}/job-finetune-eval-pack4-perlmutter.sh"
[[ ! -f "${JOB}" ]] && { echo "ERROR: missing pack4 job script ${JOB}" >&2; exit 2; }

QOS="${QOS:-premium}"
STRATEGIES="${HYDRAGNN_STRATEGIES:-routed,unfrozen,frozen,scratch}"

ALL_CASES=(
  zn-formate-mof-uma-al-001
  cantor-fcc-al-001
  phosphorene-2d-al-001
  lifepo4-al-001
  hea-bcc-al-001
)
if [[ -n "${CASES:-}" ]]; then
  IFS=',' read -ra CASE_LIST <<<"${CASES}"
else
  CASE_LIST=("${ALL_CASES[@]}")
fi

nsub=0
for case in "${CASE_LIST[@]}"; do
  dataset="${RUNS_ROOT}/${case}/dataset.extxyz"
  if [[ ! -f "${dataset}" ]]; then
    echo "[SKIP] ${case}: dataset not found (${dataset})" >&2
    continue
  fi
  # Build the ';'-separated SPECS: one campaign per strategy, all same case.
  specs=""
  IFS=',' read -ra STRAT_LIST <<<"${STRATEGIES}"
  for strat in "${STRAT_LIST[@]}"; do
    [[ -n "${specs}" ]] && specs+="; "
    specs+="CASE=${case} HYDRAGNN_STRATEGY=${strat}"
  done

  jobname="fte-pack4-hg-${case}"
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "BACKEND=hydragnn SPECS='${specs}' sbatch -q ${QOS} -J ${jobname} ${JOB}"
    continue
  fi
  if JID=$(BACKEND=hydragnn SPECS="${specs}" \
             sbatch --parsable -q "${QOS}" -J "${jobname}" "${JOB}"); then
    echo "[submitted] ${JID}  ${jobname}  (${QOS}; strategies: ${STRATEGIES})"
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
  echo "Track with: squeue --me    |    outputs under ${RUNS_ROOT}/finetune-eval/"
fi
