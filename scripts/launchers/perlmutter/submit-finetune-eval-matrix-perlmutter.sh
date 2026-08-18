#!/bin/bash
# ---------------------------------------------------------------------------
# submit-finetune-eval-matrix-perlmutter.sh
#
# Submits the fine-tune + eval matrix -- {hydragnn, uma, mace} x the five
# AL-corrected paper datasets -- as independent single-GPU Perlmutter jobs via
# job-finetune-eval-perlmutter.sh. Each (backend, variant, case) writes to its
# own output dir, so jobs never collide and can be resubmitted individually.
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
#   # submit ALL MACE sizes (5 datasets x 3 model variants):
#   BACKENDS=mace MACE_MODELS=small,medium,large \
#     bash scripts/launchers/perlmutter/submit-finetune-eval-matrix-perlmutter.sh
#
#   # submit MACE naive + LoRA per size (MACE_LORA=both), or LoRA only (=1):
#   BACKENDS=mace MACE_MODELS=small,medium,large MACE_LORA=both \
#     bash scripts/launchers/perlmutter/submit-finetune-eval-matrix-perlmutter.sh
#
#   # submit UMA full + frozen + LoRA variants (5 datasets x 3 variants):
#   BACKENDS=uma UMA_VARIANTS=full,frozen,lora \
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
# 'frozen', 'scratch' each drop all 16 heads and grow one fresh head. UMA/MACE
# jobs ignore this knob. FT_REPO points at the ORNL fine-tune utils checkout.
#
# MACE_MODELS (comma-separated; default 'medium') selects which MACE size(s)/
# variant(s) to fine-tune -- e.g. small,medium,large, a release tag/URL, or a
# local .model path. MACE_FAMILY (mace_mp|mace_off; default mace_mp) picks the
# foundation family. MACE_LORA (0|1|both; default 0) adds the native LoRA
# fine-tune variant (both = naive + LoRA per size). Extra MACE knobs
# (MACE_PRECISION, MACE_EPOCHS, MACE_LR, MACE_FORCE_WEIGHT, MACE_WEIGHT_DECAY,
# MACE_FREEZE_BACKBONE, MACE_DISPERSION, MACE_LORA_RANK, MACE_LORA_ALPHA) are
# forwarded when set.
#
# UMA_VARIANTS (comma-separated; default 'full') selects UMA fine-tune modes:
# 'full' (whole model), 'frozen' (head-only), 'lora' (backbone LoRA adapters).
# UMA_LORA_R / UMA_LORA_ALPHA are forwarded when set.
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
IFS=',' read -ra MACE_MODEL_LIST <<<"${MACE_MODELS:-medium}"
MACE_FAMILY="${MACE_FAMILY:-mace_mp}"
IFS=',' read -ra UMA_VARIANT_LIST <<<"${UMA_VARIANTS:-full}"
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
  # Middle dimension per backend: HydraGNN -> head strategies; MACE -> model
  # variants; UMA -> a single run per case.
  case "${backend}" in
    hydragnn) variants=("${HG_STRATEGIES[@]}") ;;
    mace)     variants=("${MACE_MODEL_LIST[@]}") ;;
    *)        variants=("_single") ;;
  esac
  # Secondary "mode" dimension (LoRA / freeze variants).
  case "${backend}" in
    mace)
      case "${MACE_LORA:-0}" in
        both) modes=("naive" "lora") ;;
        1)    modes=("lora") ;;
        *)    modes=("naive") ;;
      esac ;;
    uma)  modes=("${UMA_VARIANT_LIST[@]}") ;;
    *)    modes=("_single") ;;
  esac
  for variant in "${variants[@]}"; do
    strategy="routed"
    mace_model="medium"
    [[ "${backend}" == "hydragnn" ]] && strategy="${variant}"
    [[ "${backend}" == "mace" ]]     && mace_model="${variant}"
    for mode in "${modes[@]}"; do
      for entry in "${MATRIX[@]}"; do
        read -r case uma_task <<<"${entry}"
        in_whitelist "${case}" || continue
        dataset="${RUNS_ROOT}/${case}/dataset.extxyz"
        if [[ ! -f "${dataset}" ]]; then
          echo "[SKIP] ${backend}/${case}: dataset not found (${dataset})" >&2
          continue
        fi

        # Compose a variant tag from the head strategy / MACE size / LoRA/freeze
        # mode so distinct runs never collide in the output tree or job names.
        label_parts=()
        if [[ "${backend}" == "hydragnn" && "${strategy}" != "routed" ]]; then
          label_parts+=("${strategy}")
        fi
        if [[ "${backend}" == "mace" ]]; then
          [[ "${mace_model}" != "medium" ]] && label_parts+=("$(basename "${mace_model}")")
          [[ "${mode}" == "lora" ]] && label_parts+=("lora")
        fi
        if [[ "${backend}" == "uma" ]]; then
          [[ "${mode}" == "frozen" ]] && label_parts+=("frozen")
          [[ "${mode}" == "lora" ]] && label_parts+=("lora")
        fi
        if ((${#label_parts[@]})); then
          suffix="$(IFS=-; echo "${label_parts[*]}")"
          out_dir="${RUNS_ROOT}/finetune-eval/${backend}-${suffix}/${case}"
          jobname="fte-${backend}-${suffix}-${case}"
          variant_label="${suffix}"
        else
          out_dir="${RUNS_ROOT}/finetune-eval/${backend}/${case}"
          jobname="fte-${backend}-${case}"
          variant_label="routed"
          [[ "${backend}" == "mace" ]] && variant_label="${mace_model}"
          [[ "${backend}" == "uma" ]] && variant_label="full"
        fi

        # env passed through to the per-job script
        exports="ALL,BACKEND=${backend},CASE=${case},UMA_TASK=${uma_task}"
        [[ "${backend}" == "hydragnn" ]] && exports+=",HYDRAGNN_STRATEGY=${strategy}"
        if [[ "${backend}" == "mace" ]]; then
          exports+=",MACE_FAMILY=${MACE_FAMILY},MACE_MODEL=${mace_model}"
          [[ "${mode}" == "lora" ]]           && exports+=",MACE_LORA=1"
          [[ -n "${MACE_PRECISION:-}" ]]      && exports+=",MACE_PRECISION=${MACE_PRECISION}"
          [[ -n "${MACE_EPOCHS:-}" ]]         && exports+=",MACE_EPOCHS=${MACE_EPOCHS}"
          [[ -n "${MACE_LR:-}" ]]             && exports+=",MACE_LR=${MACE_LR}"
          [[ -n "${MACE_FORCE_WEIGHT:-}" ]]   && exports+=",MACE_FORCE_WEIGHT=${MACE_FORCE_WEIGHT}"
          [[ -n "${MACE_WEIGHT_DECAY:-}" ]]   && exports+=",MACE_WEIGHT_DECAY=${MACE_WEIGHT_DECAY}"
          [[ -n "${MACE_FREEZE_BACKBONE:-}" ]] && exports+=",MACE_FREEZE_BACKBONE=${MACE_FREEZE_BACKBONE}"
          [[ -n "${MACE_DISPERSION:-}" ]]     && exports+=",MACE_DISPERSION=${MACE_DISPERSION}"
          [[ -n "${MACE_LORA_RANK:-}" ]]      && exports+=",MACE_LORA_RANK=${MACE_LORA_RANK}"
          [[ -n "${MACE_LORA_ALPHA:-}" ]]     && exports+=",MACE_LORA_ALPHA=${MACE_LORA_ALPHA}"
        fi
        if [[ "${backend}" == "uma" ]]; then
          [[ "${mode}" == "frozen" ]]      && exports+=",UMA_FREEZE_BACKBONE=1"
          [[ "${mode}" == "lora" ]]        && exports+=",UMA_LORA=1"
          [[ -n "${UMA_LORA_R:-}" ]]       && exports+=",UMA_LORA_R=${UMA_LORA_R}"
          [[ -n "${UMA_LORA_ALPHA:-}" ]]   && exports+=",UMA_LORA_ALPHA=${UMA_LORA_ALPHA}"
        fi
        [[ -n "${FT_REPO:-}" ]]    && exports+=",FT_REPO=${FT_REPO}"
        [[ -n "${EPOCHS:-}" ]]     && exports+=",EPOCHS=${EPOCHS}"
        [[ -n "${BATCH_SIZE:-}" ]] && exports+=",BATCH_SIZE=${BATCH_SIZE}"
        [[ -n "${UMA_EPOCHS:-}" ]]       && exports+=",UMA_EPOCHS=${UMA_EPOCHS}"
        [[ -n "${UMA_LR:-}" ]]           && exports+=",UMA_LR=${UMA_LR}"
        [[ -n "${UMA_FORCE_WEIGHT:-}" ]] && exports+=",UMA_FORCE_WEIGHT=${UMA_FORCE_WEIGHT}"
        [[ -n "${UMA_WEIGHT_DECAY:-}" ]] && exports+=",UMA_WEIGHT_DECAY=${UMA_WEIGHT_DECAY}"
        [[ -n "${DEVICE:-}" ]]     && exports+=",DEVICE=${DEVICE}"
        [[ "${RESCORE:-0}" == "1" ]] && exports+=",RESCORE=1"

        if [[ -n "${DRY_RUN:-}" ]]; then
          echo "sbatch -J ${jobname} ${SBATCH_OPTS[*]:-} --export=${exports} ${JOB}"
          continue
        fi

        if JID=$(sbatch --parsable -J "${jobname}" "${SBATCH_OPTS[@]}" \
                    --export="${exports}" "${JOB}"); then
          echo "[submitted] ${JID}  ${backend}/${variant_label}/${case} (${uma_task})"
          printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${JID}" "${backend}" "${variant_label}" "${case}" "${uma_task}" "${out_dir}" >> "${SUMMARY}"
          nsub=$((nsub + 1))
        else
          echo "[ERROR] sbatch failed for ${backend}/${variant_label}/${case}" >&2
        fi
      done
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
