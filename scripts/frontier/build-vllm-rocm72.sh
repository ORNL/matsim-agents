#!/bin/bash
#SBATCH -A mat746
#SBATCH -J build-vllm-rocm72
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/build-vllm-rocm72-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/build-vllm-rocm72-%j.out
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
#
# Compute-node job: build vLLM from source against ROCm 7.2.
#
# Prerequisites (must be done on the login node first via install-rocm72.sh):
#   - HydraGNN ROCm 7.2 venv already exists at $VENV
#   - vLLM source already cloned at $VLLM_SRC
#   - vLLM ROCm build deps already pip-installed into the venv

set -euo pipefail

PROJ=/lustre/orion/mat746/proj-shared
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
VLLM_SRC=$PROJ/cache/vllm-src/vllm

# Load ROCm 7.2 modules (compute node has /opt/rocm-7.2.0 bind-mounted)
source "$PROJ/matsim-agents/scripts/frontier/frontier-module-stack.sh"
load_frontier_rocm72_modules
module load miniforge3/23.11.0-0 2>/dev/null || true
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
conda activate "$VENV"

[[ -d "$VLLM_SRC/.git" ]] || { echo "ERROR: vLLM source not found at $VLLM_SRC"; exit 1; }
[[ -d "$VENV/bin" ]]       || { echo "ERROR: venv not found at $VENV"; exit 1; }

echo "=== Building vLLM from source (ROCm 7.2, gfx90a) ==="
echo "Source: $VLLM_SRC  ($(cd $VLLM_SRC && git describe --tags --always))"
echo "venv:   $VENV"
echo "Node:   $(hostname)"

export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_HOME=/opt/rocm-7.2.0
export HIP_HOME=/opt/rocm-7.2.0
export CMAKE_PREFIX_PATH="/opt/rocm-7.2.0:${CMAKE_PREFIX_PATH:-}"

# Force GCC 13 from gcc-native module (system /usr/bin/c++ is GCC 7.5 which is too old for PyTorch headers)
GCC13=/opt/cray/pe/gcc-native/13/bin
export CC=$GCC13/gcc
export CXX=$GCC13/g++
export PATH="$GCC13:$PATH"

# Point cmake at pre-cloned triton v3.6.0 to avoid git clone on compute node
# (compute nodes have no outbound internet access)
export TRITON_KERNELS_SRC_DIR=$PROJ/cache/triton-src/triton/python/triton_kernels/triton_kernels

cd "$VLLM_SRC"
pip install --no-deps --no-build-isolation . --verbose

python -c "import vllm; print('vllm.__version__ =', vllm.__version__)"
echo "=== vLLM build complete ==="
