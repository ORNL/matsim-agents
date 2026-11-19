#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the Aurora active-learning loop for every paper case.
#
# Usage:
#   deployments/aurora/jobs/submit-active-learning-paper-cases.sh
#   CASES="hea_bcc lifepo4" deployments/aurora/jobs/submit-active-learning-paper-cases.sh
#   MLIP_CONFIG=examples/mlip/uma-s-1p1-omat.yaml deployments/aurora/jobs/submit-active-learning-paper-cases.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
JOB="${SCRIPT_DIR}/job-active-learning-paper-cases-aurora.sh"

CASES="${CASES:-lifepo4 hea_bcc phosphorene phosphorene_qe}"
if [[ -n "${MLIP_CONFIG:-}" ]]; then
  MLIP_CONFIG="$(cd "$(dirname "${MLIP_CONFIG}")" && pwd)/$(basename "${MLIP_CONFIG}")"
fi

echo "Submitting Aurora AL paper-case jobs: ${CASES}"
for c in ${CASES}; do
  VARS="CASE=${c}"
  [[ -n "${MLIP_CONFIG:-}" ]] && VARS="${VARS},MLIP_CONFIG=${MLIP_CONFIG}"
  [[ -n "${DFT_BACKEND:-}" ]] && VARS="${VARS},DFT_BACKEND=${DFT_BACKEND}"
  jid=$(qsub -v "${VARS}" "${JOB}")
  echo "  ${c} -> job ${jid}"
done
echo "Done. Track with: qstat -u \$USER"
