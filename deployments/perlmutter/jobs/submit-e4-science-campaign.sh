#!/bin/bash
# Print (default) or submit (--submit) the Perlmutter E4 science campaign for one DFT code.
# Usage: E4_ACCOUNT=<proj> submit-e4-science-campaign.sh vasp|qe [--submit] [--validate] [--experiments "weak strong fixed"]
#   --validate  only the 2-node debug weak run (repetition 0); qualify this before production.
# Perlmutter GPU nodes: 4 A100 (perlmutter-gpu preset). QOS: regular (debug for --validate).
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
PROJ=${E4_PROJECT_DIR:-$(dirname "$REPO")}
VENV=${MATSIM_PERLMUTTER_VENV:-$REPO/.venv}
OUT=${E4_OUTPUT_ROOT:-${SCRATCH:-$PROJ}/runs/e4-framework}
ACCOUNT=${E4_ACCOUNT:?set E4_ACCOUNT to your NERSC project}
QOS=${E4_QOS:-regular}
CONFIGS=$REPO/benchmarks/e4/framework/configs
source "$REPO/deployments/common/hydragnn-env.sh"
HYDRAGNN=${E4_HYDRAGNN_LOGDIR:-$HYDRAGNN_BEST6_LOGDIR}

case $CODE in
  vasp)
    SUFFIX=""
    DFT_ENV="E4_VASP_BIN=${E4_VASP_BIN:-$REPO/external/vasp6/src/vasp.6.6.1/bin/vasp_std},E4_VASP_POTCAR_DIR=${E4_VASP_POTCAR_DIR:-$REPO/external/vasp6/potcar/potpaw_PBE.64}"
    ;;
  qe)
    SUFFIX="-qe"
    DFT_ENV="E4_QE_PW_BIN=${E4_QE_PW_BIN:-$REPO/external/quantum-espresso/install-gpu/bin/pw.x},E4_QE_PSEUDO_DIR=${E4_QE_PSEUDO_DIR:-$REPO/external/quantum-espresso/src/pseudo}"
    ;;
  *) echo "DFT code must be vasp or qe" >&2; exit 2 ;;
esac

# cli.py jobs expands ${VAR} in the config, so the DFT and HydraGNN variables must be set.
export E4_HYDRAGNN_LOGDIR=$HYDRAGNN E4_MLP_DEVICE=${E4_MLP_DEVICE:-cuda}
IFS=, read -r -a pairs <<<"$DFT_ENV"
for pair in "${pairs[@]}"; do
  export "${pair?}"
  [[ -e ${pair#*=} ]] || { echo "missing: ${pair#*=}" >&2; exit 1; }
done
[[ -d $E4_HYDRAGNN_LOGDIR ]] || { echo "missing: $E4_HYDRAGNN_LOGDIR" >&2; exit 1; }
mkdir -p "$OUT/slurm"

plan() {  # config experiment [cli.py jobs options] -> "nodes walltime repetition" lines
  local config=$1 experiment=$2
  shift 2
  PYTHONPATH="$REPO:$REPO/src" "$VENV/bin/python3" -W ignore "$REPO/benchmarks/e4/framework/cli.py" \
    jobs "$config" --facility perlmutter --experiment "$experiment" "$@" 2>/dev/null |
    python3 -c 'import json,sys;[print(r["nodes"],r["walltime"],r["repetition"]) for r in json.load(sys.stdin)["runs"]]'
}

emit() {  # config experiment nodes walltime repetition name
  local env qos=$QOS
  ((VALIDATE)) && qos=debug
  env="ALL,PROJECT_ROOT=$REPO,MATSIM_PERLMUTTER_VENV=$VENV,HYDRAGNN_ROOT=$HYDRAGNN_ROOT"
  env+=",E4_CONFIG=${1#"$REPO"/},E4_EXPERIMENT=$2,E4_REPETITION=$5,E4_OUTPUT_ROOT=$OUT"
  env+=",E4_POLICY=concurrent,E4_HYDRAGNN_LOGDIR=$E4_HYDRAGNN_LOGDIR,$DFT_ENV"
  local cmd=(sbatch -A "$ACCOUNT" -C gpu -q "$qos" --gpus-per-node=4 -N "$3" -t "$4" -J "$6"
    -o "$OUT/slurm/%x-%j.out" -e "$OUT/slurm/%x-%j.err" --export="$env"
    "$REPO/deployments/perlmutter/jobs/job-e4-framework-perlmutter.sh")
  if ((SUBMIT)); then
    local out
    out=$("${cmd[@]}" 2>&1) || { echo "$6: sbatch failed: $out" >&2; exit 1; }
    echo "$6: $out"
  else
    printf '%q ' "${cmd[@]}"
    echo
  fi
}

tag=${CODE:0:1}
if ((VALIDATE)); then
  # Perlmutter debug QOS caps walltime at 30 min.
  emit "$CONFIGS/e4.1-science$SUFFIX.json" weak 2 00:30:00 0 "e4${tag}-val"
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
