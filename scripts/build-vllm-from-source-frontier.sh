#!/bin/bash
#SBATCH -A mat746
#SBATCH -J build-vllm-src
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/build-vllm-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/build-vllm-%j/job-%j.out
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug

set -euo pipefail

PROJ=/lustre/orion/mat746/proj-shared
RUN_DIR=$PROJ/runs/build-vllm-${SLURM_JOB_ID:-local}
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv
SRC_DIR=$PROJ/cache/vllm-src
VLLM_REF=${VLLM_REF:-v0.20.0}

mkdir -p "$RUN_DIR" "$SRC_DIR"

source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$PROJ/matsim-agents/scripts/frontier-module-stack.sh"
load_frontier_rocm_modules
source activate "$VENV"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export ROCM_HOME=/opt/rocm-7.1.1
export HIP_PATH=/opt/rocm-7.1.1
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx90a
export MAX_JOBS=${MAX_JOBS:-56}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-56}

# Point CMake to a pre-cloned triton repo so it doesn't try to fetch from GitHub.
# Pre-clone was done on the login node:
#   git clone --depth=1 --branch v3.6.0 --filter=blob:none --sparse \
#     https://github.com/triton-lang/triton.git $PROJ/cache/triton-src/triton
#   cd $PROJ/cache/triton-src/triton && git sparse-checkout set python/triton_kernels
export TRITON_KERNELS_SRC_DIR="$PROJ/cache/triton-src/triton/python/triton_kernels"

# CMake picks up /usr/bin/c++ (GCC 7.5) by default; force it to the module-loaded GCC 13.
export CXX=/opt/cray/pe/gcc-native/13/bin/g++
export CC=/opt/cray/pe/gcc-native/13/bin/gcc

# Ensure Cray runtime libs and ROCm are visible during build and import.
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD=/opt/rocm-7.1.1/lib/libamdhip64.so${LD_PRELOAD:+:${LD_PRELOAD}}

echo "[$(date)] Python: $(which python) ($(python --version 2>&1))"
echo "[$(date)] Torch before build:"
python - <<'PY'
import torch
print(torch.__version__)
print('cuda_available=', torch.cuda.is_available(), 'device_count=', torch.cuda.device_count())
PY

echo "[$(date)] Refreshing vLLM source in $SRC_DIR"
if [[ ! -d "$SRC_DIR/vllm/.git" ]]; then
    # Compute nodes have no internet; clone must be done on the login node first.
    echo "ERROR: vLLM source not found at $SRC_DIR/vllm. Pre-clone from a login node:" >&2
    echo "  git clone --depth=1 --branch $VLLM_REF https://github.com/vllm-project/vllm.git $SRC_DIR/vllm" >&2
    exit 1
fi

# Skip fetch — compute nodes have no outbound internet access.
# The source was pre-cloned at the correct ref from the login node.
echo "[$(date)] Using pre-cloned source; current HEAD: $(git -C "$SRC_DIR/vllm" rev-parse --short HEAD)"
git -C "$SRC_DIR/vllm" checkout "$VLLM_REF" 2>/dev/null || true

echo "[$(date)] Removing previously installed vLLM artifacts from env"
python -m pip uninstall -y vllm vllm-flash-attn flash-attn flashinfer-python flashinfer || true

# Build tools must be pre-installed on the login node.
# Compute nodes have no PyPI access, so we skip pip install here.
echo "[$(date)] Using pre-installed build tooling (cmake, ninja, setuptools_scm, packaging, pybind11)"
which cmake ninja >/dev/null || {
    echo "ERROR: cmake or ninja not found. Pre-install via login node:" >&2
    echo "  pip install cmake ninja packaging pybind11 setuptools_scm setuptools wheel" >&2
    exit 1
}

echo "[$(date)] Building and installing vLLM from source ($VLLM_REF)"
cd "$SRC_DIR/vllm"
# Clear stale CMake FetchContent cache and build directory so cmake re-configures
# from scratch, picking up the correct CXX/CC and TRITON_KERNELS_SRC_DIR.
rm -rf "$SRC_DIR/vllm/.deps" "$SRC_DIR/vllm/build"
# Dependencies are pre-installed from requirements/rocm.txt on the login node.
# Compute nodes have no external network, so avoid any dependency resolution.
python -m pip install -v --no-build-isolation --no-deps .

echo "[$(date)] Validating install"
python - <<'PY'
import vllm
print('vllm', vllm.__version__)
from vllm import _C
print('vllm._C import OK')
PY

echo "[$(date)] Build finished successfully"
