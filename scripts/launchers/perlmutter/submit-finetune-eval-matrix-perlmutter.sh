#!/bin/bash
# ---------------------------------------------------------------------------
# submit-finetune-eval-matrix-perlmutter.sh
#
# Submits the full 2 x 5 fine-tune + eval matrix -- {hydragnn, uma} x the five
# AL-corrected paper datasets -- as ten independent single-GPU Perlmutter jobs
# via job-finetune-eval-perlmutter.sh. Each (backend, case) writes to its own
# output dir, so jobs never collide and can be resubmitted individually.
#
# Usage:
#   bash scripts/launchers/perlmutter/submit-finetune-eval-matrix-perlmutter.sh
#
#   # restrict backends and/or cases (comma-separated):
#   BACKENDS=uma CASES=lifepo4-al-001,hea-bcc-al-001 \
#     bash scripts/launchers/perlmutter/submit-finetune-eval-matrix-perlmutter.sh
#
#   # submit the new-head HydraGNN strategies (5 datasets x 3 strategies):
#   BACKENDS=hydragnn HYDRAGNN_STRATEGIES=unfrozen,frozen,scratch \
#     bash scripts/launchers/perlmutter/submit-finetune-eval-matrix-perlmutter.sh
#
#   # print the sbatch commands without submitting:
#   DRY_RUN=1 bash scripts/launchers/perlmutter/submit-finetune-eval-matrix-perlmutter.sh
#
# Optional env: EPOCHS, BATCH_SIZE, DEVICE, QOS, WALLTIME, plus any path
# overrides understood by the per-job script (PROJECT_ROOT, RUNS_ROOT, ...).
#
# HYDRAGNN_STRATEGIES (comma-separated; default 'routed') selects which
# HydraGNN fine-tune heads to run. 'routed' = branch-MLP reuse; 'unfrozen',
# 'frozen', 'scratch' each drop all 16 heads and grow one fresh head. UMA jobs
# ignore this knob. FT_REPO points at the ORNL fine-tune utils checkout.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/global/cfs/projectdirs/m5216/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
JOB="${SCRIPT_DIR}/job-finetune-eval-perlmutter.sh"
[[ ! -f "${JOB}" ]] && { echo "ERROR: missing per-job script ${JOB}" >&2; exit 2; }

# ── the 5 datasets and their UMA task (HydraGNN ignores UMA_TASK) ─────────────
#   case                         uma_task
MATRIX=(
  "zn-formate-mof-uma-al-001     omol"
  "cantor-fcc-al-001             omat"
  "phosphorene-2d-al-001         omat"
  "lifepo4-al-001                omat"
  "hea-bcc-al-001                omat"
)

# ── optional whitelists ───────────────────────────────────────────────────────
IFS=',' read -ra BACKEND_LIST <<<"${BACKENDS:-hydragnn,uma}"
IFS=',' read -ra HG_STRATEGIES <<<"${HYDRAGNN_STRATEGIES:-routed}"
declare -a CASE_WHITE=()
[[ -n "${CASES:-}" ]] && IFS=',' read -ra CASE_WHITE <<<"${CASES}"

in_whitelist() {  # $1=case
  [[ ${#CASE_WHITE[@]} -eq 0 ]] && return 0
  local c
  for c in "${CASE_WHITE[@]}"; do [[ "$1" == "$c" ]] && return 0; done
  return 1
}

TS=$(date +%Y%m%d-%H%M%S)
SUMMARY="${RUNS_ROOT}/finetune-eval-submit-${TS}.tsv"
mkdir -p "${RUNS_ROOT}"
printf "jobid\tbackend\tstrategy\tcase\tuma_task\tout_dir\n" > "${SUMMARY}"

# ── sbatch overrides ──────────────────────────────────────────────────────────
declare -a SBATCH_OPTS=()
[[ -n "${QOS:-}" ]]      && SBATCH_OPTS+=(-q "${QOS}")
[[ -n "${WALLTIME:-}" ]] && SBATCH_OPTS+=(-t "${WALLTIME}")

nsub=0
for backend in "${BACKEND_LIST[@]}"; do
  # UMA has no head-strategy concept; run it exactly once per case.
  if [[ "${backend}" == "hydragnn" ]]; then
    strategies=("${HG_STRATEGIES[@]}")
  else
    strategies=("routed")
  fi
  for strategy in "${strategies[@]}"; do
    for entry in "${MATRIX[@]}"; do
      read -r case uma_task <<<"${entry}"
      in_whitelist "${case}" || continue
      dataset="${RUNS_ROOT}/${case}/dataset.extxyz"
      if [[ ! -f "${dataset}" ]]; then
        echo "[SKIP] ${backend}/${case}: dataset not found (${dataset})" >&2
        continue
      fi
      if [[ "${backend}" == "hydragnn" && "${strategy}" != "routed" ]]; then
        out_dir="${RUNS_ROOT}/finetune-eval/${backend}-${strategy}/${case}"
        jobname="fte-${backend}-${strategy}-${case}"
      else
        out_dir="${RUNS_ROOT}/finetune-eval/${backend}/${case}"
        jobname="fte-${backend}-${case}"
      fi

      # env passed through to the per-job script
      exports="ALL,BACKEND=${backend},CASE=${case},UMA_TASK=${uma_task}"
      [[ "${backend}" == "hydragnn" ]] && exports+=",HYDRAGNN_STRATEGY=${strategy}"
      [[ -n "${FT_REPO:-}" ]]    && exports+=",FT_REPO=${FT_REPO}"
      [[ -n "${EPOCHS:-}" ]]     && exports+=",EPOCHS=${EPOCHS}"
      [[ -n "${BATCH_SIZE:-}" ]] && exports+=",BATCH_SIZE=${BATCH_SIZE}"
      [[ -n "${UMA_EPOCHS:-}" ]]       && exports+=",UMA_EPOCHS=${UMA_EPOCHS}"
      [[ -n "${UMA_LR:-}" ]]           && exports+=",UMA_LR=${UMA_LR}"
      [[ -n "${UMA_FORCE_WEIGHT:-}" ]] && exports+=",UMA_FORCE_WEIGHT=${UMA_FORCE_WEIGHT}"
      [[ -n "${UMA_WEIGHT_DECAY:-}" ]] && exports+=",UMA_WEIGHT_DECAY=${UMA_WEIGHT_DECAY}"
      [[ -n "${UMA_FREEZE_BACKBONE:-}" ]] && exports+=",UMA_FREEZE_BACKBONE=${UMA_FREEZE_BACKBONE}"
      [[ -n "${DEVICE:-}" ]]     && exports+=",DEVICE=${DEVICE}"

      if [[ -n "${DRY_RUN:-}" ]]; then
        echo "sbatch -J ${jobname} ${SBATCH_OPTS[*]:-} --export=${exports} ${JOB}"
        continue
      fi

      if JID=$(sbatch --parsable -J "${jobname}" "${SBATCH_OPTS[@]}" \
                  --export="${exports}" "${JOB}"); then
        echo "[submitted] ${JID}  ${backend}/${strategy}/${case} (${uma_task})"
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${JID}" "${backend}" "${strategy}" "${case}" "${uma_task}" "${out_dir}" >> "${SUMMARY}"
        nsub=$((nsub + 1))
      else
        echo "[ERROR] sbatch failed for ${backend}/${strategy}/${case}" >&2
      fi
    done
  done
done

echo "---------------------------------------------------------------"
if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: no jobs submitted."
else
  echo "Submitted ${nsub} job(s). Summary: ${SUMMARY}"
  echo "Track with: squeue --me    |    outputs under ${RUNS_ROOT}/finetune-eval/"
fi
