#!/usr/bin/env bash
#PBS -N matsim-mlip-xpu
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
set -eo pipefail

REPO="${PROJECT_ROOT:-${PBS_O_WORKDIR:?Set PROJECT_ROOT to the Matsim checkout}}"
[[ -f "${REPO}/pyproject.toml" ]] || exit 2
RUN_DIR="${QUALIFICATION_RUN_DIR:-${REPO}/.hpc-build/aurora/qualification/${PBS_JOBID:-local}}"
mkdir -p "${RUN_DIR}"
finish() {
    result=$?
    printf 'FINISHED exit_code=%s finished=%s\n' "${result}" "$(date -Is)" >"${RUN_DIR}/status.txt"
}
trap finish EXIT
printf 'RUNNING started=%s host=%s job=%s\n' "$(date -Is)" "$(hostname)" "${PBS_JOBID:-local}" \
    >"${RUN_DIR}/status.txt"
source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_xpu_modules
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"
source "${REPO}/deployments/common/hydragnn-env.sh"
hydragnn_export_pythonpath
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
ulimit -c 0
cd "${REPO}"
# MLIP_CONFIG selects one model (a YAML file with an mlip: block, device: xpu);
# unset, the job qualifies the HydraGNN, UMA, and MACE selections in config/.
if [[ -n "${MLIP_CONFIG:-}" ]]; then
    configs=("${MLIP_CONFIG}")
else
    configs=("${REPO}"/deployments/aurora/jobs/config/mlip-xpu-{hydragnn,uma,mace}.yaml)
fi
failed=0
for config in "${configs[@]}"; do
    if ! backend="$("${REPO}/.venv/bin/python" -I - "${config}" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    print(((yaml.safe_load(handle) or {}).get("mlip") or {}).get("backend", ""))
PY
)"; then
        printf 'Cannot read the mlip block of %s\n' "${config}" >&2
        failed=1
        continue
    fi
    case "${backend}" in
        hydragnn) python="${REPO}/.venv/bin/python" ;;
        uma) python="${REPO}/.venv-uma/bin/python" ;;
        mace) python="${REPO}/.venv-mace/bin/python" ;;
        *)
            printf '%s: mlip.backend must be hydragnn, uma, or mace\n' "${config}" >&2
            failed=1
            continue
            ;;
    esac
    if timeout --kill-after=30 480 mpiexec -n 1 --ppn 1 "${python}" \
        "${REPO}/deployments/aurora/jobs/qualify_mlip_xpu.py" \
        --mlip-config "${config}" --output "${RUN_DIR}/${backend}.json" \
        >"${RUN_DIR}/${backend}.log" 2>&1; then
        printf '%s PASSED\n' "${backend}"
    else
        launcher_status=$?
        "${python}" -I -S - "${RUN_DIR}/${backend}.json" "${backend}" "${launcher_status}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
result = json.loads(path.read_text()) if path.exists() else {"backend": sys.argv[2]}
progress_path = path.with_suffix(".progress.json")
if progress_path.exists():
    result["progress"] = json.loads(progress_path.read_text())
result["launcher_exit_code"] = int(sys.argv[3])
if result["launcher_exit_code"] in (124, 137):
    result["inference_status"] = result.get("status", "INCOMPLETE")
    result["status"] = "TIMED_OUT"
else:
    result["status"] = "FAILED"
path.write_text(json.dumps(result, indent=2) + "\n")
PY
        printf '%s FAILED; inspect %s\n' "${backend}" "${RUN_DIR}/${backend}.log"
        failed=1
    fi
done
exit "${failed}"