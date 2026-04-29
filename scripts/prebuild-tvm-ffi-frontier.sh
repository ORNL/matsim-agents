#!/bin/bash
# ---------------------------------------------------------------------------
# prebuild-tvm-ffi-frontier.sh
#
# One-time SLURM job that compiles the tvm-ffi torch<->DLPack ROCm shared
# library into the project-shared GPFS cache. Subsequent matsim-agents /
# vLLM jobs will find the prebuilt .so and skip a ~5 min cold-start compile.
#
# Why a SLURM job (not the install script)?
# ------------------------------------------
# The rest of the HydraGNN/matsim-agents install (torch+rocm wheel,
# torch_geometric, torch_scatter/sparse/cluster, vllm, etc.) succeeds on the
# Frontier login node because every one of those packages comes as a
# PREBUILT BINARY WHEEL — pip just unpacks them, never invoking hipcc or
# touching /opt/rocm-7.1.1.
#
# tvm-ffi is the one exception. It uses torch.utils.cpp_extension.load() to
# JIT-compile a C++ addon against PyTorch headers AND the system ROCm
# toolchain (hipcc, /opt/rocm-7.1.1/include/*, libamdhip64). On Frontier:
#
#   • Login nodes:    /opt/rocm-7.1.1 is a 60-byte stub. hipcc/headers/libs
#                     are NOT present. The compile cannot even start —
#                     torch._find_cuda_home() returns None and aborts.
#   • Compute nodes:  /opt/rocm-7.1.1 is bind-mounted with the real install.
#                     The compile succeeds.
#
# So the build MUST execute on a compute node — but the OUTPUT .so is
# written to a project-shared GPFS cache (Lustre), where every future job
# on any compute node reads it back. Compute-node-vs-login is about toolchain
# AVAILABILITY (read /opt/rocm-7.1.1); Lustre is about output SHARING.
# These are orthogonal — we need both, for different reasons.
#
# Why is this needed at all?
# --------------------------
# Without this prebuild, every vLLM job pays the ~5 min JIT cost on first
# tvm_ffi import — and during that compile the job appears silently hung.
# Worse: tvm-ffi guards the build with a FileLock; a job killed mid-compile
# leaves a stale lock that deadlocks every subsequent job sharing the cache.
# Doing the compile once, here, in a job we control, eliminates both issues.
#
# Cross-node safety: Frontier compute nodes are homogeneous (MI250X, same
# OS image, same ROCm bind-mount). The .so is keyed by torch major.minor +
# the --build-with-rocm flag, so any compute node consuming it from the
# shared cache gets an ABI-compatible artifact as long as the conda env and
# the rocm/X.Y.Z module pin match what produced it.
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
# Cray's PrgEnv-gnu loads gcc-native/13.2 but ONLY exposes Cray's `cc`/`CC`
# wrappers. The actual `gcc`/`g++` 13.3 binaries live in the gcc-native
# module's bin dir, which is NOT in PATH after PrgEnv-gnu alone. Loading
# `gcc-native` explicitly puts /opt/cray/pe/gcc-native/13/bin first in PATH,
# overriding the SLES system /usr/bin/gcc (GCC 7.5) which PyTorch rejects.
ml gcc-native
module unload darshan-runtime
source activate "$VENV"

# torch's cpp_extension.include_paths() demands CUDA_HOME, even for ROCm.
export CUDA_HOME="${ROCM_PATH:-/opt/rocm-7.1.1}"

# Force the build to use Cray's gcc-native (GCC 13.3, satisfies PyTorch's
# >=9 requirement). After `ml gcc-native`, `which g++` returns the Cray path.
export CC="$(which gcc)"
export CXX="$(which g++)"
echo "Using CC=$CC"
echo "Using CXX=$CXX"
"$CXX" --version | head -1

# CRITICAL: prevent tvm_ffi from auto-spawning its OWN build at import time.
# `python -m tvm_ffi.utils._build_optional_torch_c_dlpack` triggers
# `import tvm_ffi`, whose _optional_torch_c_dlpack module checks for the addon
# .so and, if absent, fires off its own subprocess to build it into
# $TVM_FFI_CACHE_DIR (default: ~/.cache/tvm-ffi) BEFORE our main() runs.
# Without this guard, two concurrent builds race on the same lockfile, the
# auto-spawned one writes to ~/.cache (NOT our shared GPFS cache), and the
# whole thing wedges. Also export TVM_FFI_CACHE_DIR so any tvm_ffi process
# (auto-spawned or not) writes to the project-shared cache.
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export TVM_FFI_CACHE_DIR="$CACHE_DIR"

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
