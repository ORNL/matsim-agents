#!/bin/bash
# ---------------------------------------------------------------------------
# prebuild-tvm-ffi-frontier.sh
#
# One-time SLURM job that compiles the tvm-ffi torch<->DLPack ROCm shared
# library into the project-shared GPFS cache. Subsequent matsim-agents /
# vLLM jobs will find the prebuilt .so and skip a ~5 min cold-start compile.
#
# Why a SLURM job (not the install script)?
#   The build needs the real ROCm install at /opt/rocm-7.1.1, which on
#   Frontier is a 60-byte stub on login nodes — only populated on compute
#   nodes via bind-mount. So the build MUST run on a compute node.
#
# Why is this needed at all?
#   tvm_ffi (used by vLLM) JIT-compiles this addon on first import. Without
#   prebuilding, every job pays the cost; worse, a job killed mid-build
#   leaves a stale lockfile in the cache that deadlocks every subsequent job
#   sharing the same cache directory.
#
# Submit:
#   sbatch scripts/prebuild-tvm-ffi-frontier.sh
# ---------------------------------------------------------------------------
#SBATCH -A mat746
#SBATCH -J prebuild-tvm-ffi
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/prebuild-tvm-ffi-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/prebuild-tvm-ffi-%j.out
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug

set -euo pipefail

PROJ=/lustre/orion/mat746/proj-shared
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv
CACHE_DIR=$PROJ/cache/tvm-ffi

mkdir -p "$CACHE_DIR" "$PROJ/runs"

# ── conda + modules (must mirror install-time exactly) ──────────────────────
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
module reset
ml cpe/24.07 rocm/7.1.1 amd-mixed/7.1.1 PrgEnv-gnu miniforge3/23.11.0-0
module unload darshan-runtime
source activate "$VENV"

# torch's cpp_extension.include_paths() demands CUDA_HOME, even for ROCm.
export CUDA_HOME="${ROCM_PATH:-/opt/rocm-7.1.1}"

echo "[$(date)] === Prebuild diagnostics ==="
echo "Hostname:      $(hostname)"
echo "ROCM_PATH:     ${ROCM_PATH:-unset}"
echo "CUDA_HOME:     $CUDA_HOME"
echo "rocm dir size: $(du -sh ${CUDA_HOME} 2>/dev/null || echo 'unreadable')"
echo "Python:        $(which python)  ($(python --version 2>&1))"
python -c 'import torch; print("torch:", torch.__version__, "hip:", torch.version.hip)'
echo "Cache dir:     $CACHE_DIR"
echo "============================================="

# Clear any stale locks from previously killed jobs.
rm -f "$CACHE_DIR"/*.lock "$HOME/.cache/tvm-ffi"/*.lock 2>/dev/null || true

TORCH_MM=$(python -c 'import torch,sys; v=torch.__version__.split(".")[:2]; sys.stdout.write("".join(v))')
TVM_LIB="libtorch_c_dlpack_addon_torch${TORCH_MM}-rocm.so"

if [[ -f "$CACHE_DIR/$TVM_LIB" ]]; then
    echo "[$(date)] $TVM_LIB already present in $CACHE_DIR — nothing to do."
    ls -la "$CACHE_DIR/$TVM_LIB"
    exit 0
fi

echo "[$(date)] Building $TVM_LIB into $CACHE_DIR ..."
time python -m tvm_ffi.utils._build_optional_torch_c_dlpack \
    --output-dir "$CACHE_DIR" \
    --build-with-rocm

echo ""
echo "[$(date)] === Result ==="
ls -la "$CACHE_DIR"
if [[ -f "$CACHE_DIR/$TVM_LIB" ]]; then
    echo "OK: $CACHE_DIR/$TVM_LIB"
    echo ""
    echo "Future jobs need:  export TVM_FFI_CACHE_DIR=$CACHE_DIR"
    exit 0
else
    echo "FAIL: $TVM_LIB not produced."
    exit 1
fi
