#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the DFT-concurrency strong-scaling sweep on ALCF Aurora: one AL
# iteration with a fixed 32-job DFT workload at N = 1, 2, 4, 8 nodes, repeated
# REPS times. Each repeat draws an independent random workload (via AL_SEED)
# held fixed across node counts within the repeat. Mirrors the Frontier and
# Perlmutter sweep wrappers.
#
#   deployments/aurora/jobs/submit-al-dft-scaling-sweep.sh
#   NODES="1 2 4" REPS=5 QUEUE=prod deployments/aurora/jobs/submit-al-dft-scaling-sweep.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="${SCRIPT_DIR}/job-al-dft-scaling-aurora.sh"
[[ -f "${JOB}" ]] || { echo "ERROR: missing ${JOB}" >&2; exit 2; }

NODES="${NODES:-1 2 4 8}"
REPS="${REPS:-5}"
REP_START="${REP_START:-1}"
QUEUE="${QUEUE:-prod}"
WALL="${WALL:-}"

echo "Submitting DFT strong-scaling sweep (queue=${QUEUE}): N in [${NODES}] x REPS=${REP_START}..${REPS}"
for r in $(seq "${REP_START}" "${REPS}"); do
  for n in ${NODES}; do
    jid=$(qsub -l "select=${n}" -q "${QUEUE}" ${WALL:+-l "walltime=${WALL}"} -v "REP=${r}" "${JOB}")
    echo "  N=${n}  rep=${r}  ->  job ${jid}"
  done
done
echo "Done. Track with: qstat -u \$USER"
