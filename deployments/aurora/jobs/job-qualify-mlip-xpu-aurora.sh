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
module load pti-gpu oneapi/release/2026.1.0 intel_gpu_umd_aicoe hdf5
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"
export HYDRAGNN_ROOT="${HYDRAGNN_ROOT:-$(dirname "${REPO}")/HydraGNN}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
ulimit -c 0
cd "${REPO}"
HYDRA_EXAMPLE="${HYDRAGNN_ROOT}/examples/multidataset_hpo_sc26"
failed=0
for backend in hydragnn uma mace; do
    case "${backend}" in
        hydragnn) python="${REPO}/.venv/bin/python" ;;
        uma) python="${REPO}/.venv-uma/bin/python" ;;
        mace) python="${REPO}/.venv-mace/bin/python" ;;
    esac
    if timeout 480 mpiexec -n 1 --ppn 1 "${python}" \
        "${REPO}/deployments/aurora/jobs/qualify_mlip_xpu.py" \
        --backend "${backend}" --output "${RUN_DIR}/${backend}.json" \
        --hydragnn-logdir "${MATSIM_HYDRAGNN_LOGDIR:-${HYDRA_EXAMPLE}/multidataset_hpo-BEST6-fp64}" \
        --hydragnn-mlp "${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-${HYDRA_EXAMPLE}/mlp_branch_weights.pt}" \
        --uma-model "${MATSIM_UMA_MODEL_NAME:-uma-s-1p1}" \
        --mace-model "${MATSIM_MACE_MODEL:-small}" >"${RUN_DIR}/${backend}.log" 2>&1; then
        printf '%s PASSED\n' "${backend}"
    else
        printf '%s FAILED; inspect %s\n' "${backend}" "${RUN_DIR}/${backend}.log"
        failed=1
    fi
done
exit "${failed}"