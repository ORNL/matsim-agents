#!/bin/bash
# Print (default) or submit (--submit) the Aurora E4 science campaign for one DFT code.
# Usage: submit-e4-science-campaign.sh vasp|qe [--submit] [--validate] [--experiments "weak strong fixed"]
#   --validate  only the 2-node debug weak run (repetition 0); qualify this before production.
# Queue: <=16 nodes -> capacity; otherwise walltime <= 1 h -> debug-scaling.
# Resubmitting a printed command resumes that run.
set -euo pipefail

CODE=${1:?usage: $0 vasp|qe [--submit] [--validate] [--experiments LIST]}
shift
SUBMIT=0
VALIDATE=0
EXPERIMENTS="weak strong fixed"
while (($#)); do
  case $1 in
    --submit) SUBMIT=1 ;;
    --validate) VALIDATE=1 ;;
    --experiments) EXPERIMENTS=$2; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
M=${E4_PROJECT_DIR:-$(dirname "$REPO")}
VENV=${VENV_PATH:-$M/matsim-agents/.venv}
OUT=${E4_OUTPUT_ROOT:-$M/runs/e4-framework}
ACCOUNT=${E4_ACCOUNT:-HydraGNN}
CONFIGS=$REPO/benchmarks/e4/framework/configs
HYDRAGNN=${E4_HYDRAGNN_LOGDIR:-$M/HydraGNN-e4b-frozen-81d5b883/examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64}

case $CODE in
  vasp)
    SUFFIX=""
    DFT_ENV="E4_VASP_BIN=${E4_VASP_BIN:-$M/matsim-agents/external/vasp6/src/vasp.6.6.1/bin/vasp_std},E4_VASP_POTCAR_DIR=${E4_VASP_POTCAR_DIR:-$M/matsim-agents/external/vasp6/potcar}"
    ;;
  qe)
    SUFFIX="-qe"
    # Aurora QE 7.6 offloads only EPW to the GPU, so pw.x is the CPU MPI+OpenMP build.
    DFT_ENV="E4_QE_PW_BIN=${E4_QE_PW_BIN:-/soft/applications/quantum_espresso/7.6-oneapi2025.3.1/cpu-openmp/bin/pw.x},E4_QE_PSEUDO_DIR=${E4_QE_PSEUDO_DIR:-$M/matsim-agents/external/qe-pseudo}"
    ;;
  *) echo "DFT code must be vasp or qe" >&2; exit 2 ;;
esac

# cli.py jobs expands ${VAR} in the config, so the DFT and HydraGNN variables must be set.
export E4_HYDRAGNN_LOGDIR=$HYDRAGNN E4_MLP_DEVICE=${E4_MLP_DEVICE:-xpu}
IFS=, read -r -a pairs <<<"$DFT_ENV"
for pair in "${pairs[@]}"; do
  export "${pair?}"
  [[ -e ${pair#*=} ]] || { echo "missing: ${pair#*=}" >&2; exit 1; }
done
[[ -d $E4_HYDRAGNN_LOGDIR ]] || { echo "missing: $E4_HYDRAGNN_LOGDIR" >&2; exit 1; }
mkdir -p "$OUT/pbs"

plan() {  # config experiment [cli.py jobs options] -> "nodes walltime repetition" lines
  local config=$1 experiment=$2
  shift 2
  PYTHONPATH="$REPO:$REPO/src" "$VENV/bin/python" -W ignore "$REPO/benchmarks/e4/framework/cli.py" \
    jobs "$config" --facility aurora --experiment "$experiment" "$@" 2>/dev/null |
    python3 -c 'import json,sys;[print(r["nodes"],r["walltime"],r["repetition"]) for r in json.load(sys.stdin)["runs"]]'
}

queue_for() {  # nodes walltime
  local minutes=$((10#${2:0:2} * 60 + 10#${2:3:2}))
  if (($1 <= 2)) && ((minutes <= 60)) && ((VALIDATE)); then echo debug
  elif (($1 <= 16)); then echo capacity
  elif ((minutes <= 60)); then echo debug-scaling
  else echo "no queue for $1 nodes x $2" >&2; return 1
  fi
}

emit() {  # config experiment nodes walltime repetition name
  local queue env
  queue=$(queue_for "$3" "$4")
  env="PROJECT_ROOT=$REPO,VENV_PATH=$VENV,E4_CONFIG=${1#"$REPO"/},E4_EXPERIMENT=$2,E4_REPETITION=$5"
  env+=",E4_OUTPUT_ROOT=$OUT,E4_POLICY=concurrent,E4_HYDRAGNN_LOGDIR=$E4_HYDRAGNN_LOGDIR,$DFT_ENV"
  local cmd=(qsub -A "$ACCOUNT" -q "$queue" -l "select=$3" -l "walltime=$4" -l filesystems=flare:home
    -N "$6" -o "$OUT/pbs/" -v "$env" "$REPO/deployments/aurora/jobs/job-e4-framework-aurora.pbs")
  if ((SUBMIT)); then
    local out
    out=$("${cmd[@]}" 2>&1) || { echo "$6: qsub failed: $out" >&2; exit 1; }
    echo "$6: $out"
  else
    printf '%q ' "${cmd[@]}"
    echo
  fi
}

tag=${CODE:0:1}
if ((VALIDATE)); then
  emit "$CONFIGS/e4.1-science$SUFFIX.json" weak 2 00:55:00 0 "e4${tag}-val"
  exit 0
fi
for experiment in $EXPERIMENTS; do
  if [[ $experiment == fixed ]]; then
    config=$CONFIGS/e4.3-heterogeneous$SUFFIX.json
    # The E4.3 allocation is fixed at 8 nodes (7 DFT node groups + 1 MLIP node).
    while read -r nodes wall rep; do
      emit "$config" fixed "$nodes" "$wall" "$rep" "e4${tag}-h-r$rep"
    done < <(plan "$config" fixed --nodes 8)
  else
    config=$CONFIGS/e4.1-science$SUFFIX.json
    while read -r nodes wall rep; do
      emit "$config" "$experiment" "$nodes" "$wall" "$rep" "e4${tag}-${experiment:0:1}$nodes-r$rep"
    done < <(plan "$config" "$experiment")
  fi
done
