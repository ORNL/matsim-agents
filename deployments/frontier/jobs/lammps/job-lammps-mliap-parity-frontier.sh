#!/bin/bash
#SBATCH -J lammps-mliap-parity
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -N 1
#SBATCH -p batch
# ---------------------------------------------------------------------------
# ASE vs LAMMPS ML-IAP energy/force parity and NVE stability on a compute node (single rank, CPU).
# The multi-rank check is job-lammps-mliap-mpi-frontier.sh.
# Needs external/lammps/activate-mliap.sh from
# deployments/frontier/setup/lammps/build-lammps-mliap-frontier.sh. MACE uses the
# cached MACE-MP-0 small weights. HydraGNN needs the frozen source (HYDRAGNN_DIR) and the
# BEST6 weights tree (HYDRAGNN_WEIGHTS); UMA needs the bundle from
# deployments/frontier/download/download-uma-frontier.sh and is skipped without it.
#
# Submit: MLIAP_OUT=<dir> MATSIM_MLIAP_SEED=<root seed> sbatch -A <ACCOUNT> -t 00:30:00 -q debug \
#   deployments/frontier/jobs/lammps/job-lammps-mliap-parity-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO="${PROJECT_ROOT:-$(d="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." 2>/dev/null && pwd)"; [[ -f "$d/pyproject.toml" ]] && echo "$d" || echo "${SLURM_SUBMIT_DIR:?}")}"
OUT="${MLIAP_OUT:?set MLIAP_OUT}"
SEED="${MATSIM_MLIAP_SEED:?set MATSIM_MLIAP_SEED (root seed of the run)}"

# shellcheck disable=SC1091
source "${LAMMPS_ROOT:-$REPO/external/lammps}/activate-mliap.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 XALT_EXECUTABLE_TRACKING=no OMP_NUM_THREADS=1
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cd "$OUT"

status=0
run() { # name python args...
    local name="$1" py="$2"
    shift 2
    "$py" "$REPO/scripts/lammps/mliap_parity.py" "$@" --seed "$SEED" --out "$OUT/$name.json" || status=1
}

MD_STEPS="${MATSIM_MLIAP_MD_STEPS:-20}"  # NVE stability stage on the unstrained-lattice runs
run emt "$REPO/.venv/bin/python" --calc emt --md-steps "$MD_STEPS"
run emt-pka "$REPO/.venv/bin/python" --calc emt --md-steps 200 --md-timestep-fs 0.2 --pka-eV 200
run emt-zbl "$REPO/.venv/bin/python" --calc emt --a 2.3 --rattle 0.02 --zbl
run mace "$REPO/.venv-mace/bin/python" --calc mace --model small --md-steps "$MD_STEPS"
run mace-zbl "$REPO/.venv-mace/bin/python" --calc mace --model small --a 2.3 --rattle 0.02 --zbl
PROJ="$(dirname "$REPO")"
if source "$REPO/deployments/common/hydragnn-env.sh" && [[ -d "$HYDRAGNN_BEST6_LOGDIR" ]]; then
    HG_PP="$HYDRAGNN_EXAMPLE:$HYDRAGNN_ROOT:$PYTHONPATH"
    PYTHONPATH="$HG_PP" run hydragnn "$REPO/.venv/bin/python" --calc hydragnn \
        --md-steps "$MD_STEPS"
    PYTHONPATH="$HG_PP" run hydragnn-zbl "$REPO/.venv/bin/python" --calc hydragnn \
        --a 2.3 --rattle 0.02 --zbl
else
    echo "hydragnn: skipped (pinned source unavailable or BEST6 weights missing)" | tee "$OUT/hydragnn.skipped"
fi

export MATSIM_UMA_ARTIFACT_DIR="${MATSIM_UMA_ARTIFACT_DIR:-$PROJ/models/artifacts/uma}"
UMA_MODEL="${UMA_MODEL:-uma-s-1p1}"
if [[ -f "$MATSIM_UMA_ARTIFACT_DIR/$UMA_MODEL/checkpoint.pt" && -x "$REPO/.venv-uma/bin/python" ]]; then
    export UMA_CHECKPOINT="$MATSIM_UMA_ARTIFACT_DIR/$UMA_MODEL/checkpoint.pt"
    run uma "$REPO/.venv-uma/bin/python" --calc uma --model "$UMA_MODEL" \
        --md-steps "$MD_STEPS"
    run uma-zbl "$REPO/.venv-uma/bin/python" --calc uma --model "$UMA_MODEL" \
        --a 2.3 --rattle 0.02 --zbl
else
    echo "uma: skipped (no bundle in $MATSIM_UMA_ARTIFACT_DIR)" | tee "$OUT/uma.skipped"
fi
exit "$status"
