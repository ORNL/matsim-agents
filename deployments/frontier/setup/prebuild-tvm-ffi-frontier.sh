#!/bin/bash
# ---------------------------------------------------------------------------
# prebuild-tvm-ffi-frontier.sh
#
# One-time SLURM job that compiles the tvm-ffi torch<->DLPack ROCm shared
# library into the repository-owned external/tvm-ffi install tree on Lustre.
# Subsequent matsim-agents / vLLM jobs load this durable artifact directly.
#
# Why a SLURM job (not the install script)?
# ------------------------------------------
# The rest of the HydraGNN/matsim-agents install (torch+rocm wheel,
# torch_geometric, torch_scatter/sparse/cluster, vllm, etc.) succeeds on the
# Frontier login node because every one of those packages comes as a
# PREBUILT BINARY WHEEL — pip just unpacks them, never invoking hipcc or
# touching /opt/rocm-7.1.1.
#
# tvm-ffi is the one exception. It uses torch's extension APIs to
# JIT-compile a C++ addon against PyTorch headers AND the system ROCm
# toolchain (hipcc, /opt/rocm-7.1.1/include/*, libamdhip64). On Frontier:
#
#   • Login nodes:    /opt/rocm-7.2.0 is a 60-byte stub. hipcc/headers/libs
#                     are NOT present. The compile cannot even start —
#                     torch._find_cuda_home() returns None and aborts.
#   • Compute nodes:  /opt/rocm-7.2.0 is bind-mounted with the real install.
#                     The compile succeeds.
#
# So the build MUST execute on a compute node, while the OUTPUT .so is written
# to project-shared Lustre, where every future job on any compute node reads it
# back. Compute-node-vs-login is about toolchain
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
# Cross-node safety: Frontier compute nodes are homogeneous (MI250X, same OS
# image and ROCm bind-mount). The filename captures the full Torch version,
# backend and C++ ABI hash expected by tvm-ffi's loader.
#
# Module pin: must match the venv's installed torch build (torch+rocm7.2) and
# every other Frontier job script (load_frontier_rocm72_modules). Loading
# rocm/7.1.1 instead pulls in an incompatible /opt/rocm-7.1.1/lib/libamd_smi.so
# (missing symbols the venv's amdsmi wheel expects, since amdsmi was installed
# from the ROCm 7.2 bundle), which crashes `import torch` before the build
# can even start.
#
# Submit:
#   sbatch scripts/prebuild-tvm-ffi-frontier.sh
# ---------------------------------------------------------------------------
#SBATCH -J prebuild-tvm-ffi
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
if [[ ! -f "${REPO}/pyproject.toml" ]]; then
    REPO="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
fi
[[ -f "${REPO}/pyproject.toml" ]] || {
    echo "ERROR: submit from the repository root or export PROJECT_ROOT" >&2
    exit 2
}
SCRIPT_DIR="${REPO}/deployments/frontier/setup"
PROJ="$(dirname "${REPO}")"
VENV=$REPO/.venv
source "${SCRIPT_DIR}/tvm-ffi-artifact.sh"
ARTIFACT_DIR="${TVM_FFI_ARTIFACT_DIR:-${REPO}/external/tvm-ffi/lib}"

mkdir -p "$ARTIFACT_DIR" "$PROJ/runs"

# ── conda + modules (must mirror install-time exactly) ──────────────────────
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$REPO/deployments/frontier/setup/frontier-module-stack.sh"
load_frontier_rocm72_modules
# Cray's PrgEnv-gnu loads gcc-native/13.2 but ONLY exposes Cray's `cc`/`CC`
# wrappers. The actual `gcc`/`g++` 13.3 binaries live in the gcc-native
# module's bin dir, which is NOT in PATH after PrgEnv-gnu alone. Loading
# `gcc-native` explicitly puts /opt/cray/pe/gcc-native/13/bin first in PATH,
# overriding the SLES system /usr/bin/gcc (GCC 7.5) which PyTorch rejects.
ml gcc-native
source activate "$VENV"

# torch's cpp_extension.include_paths() demands CUDA_HOME, even for ROCm.
export CUDA_HOME="${ROCM_PATH:-/opt/rocm-7.2.0}"

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
# whole thing wedges. tvm-ffi requires the variable name TVM_FFI_CACHE_DIR,
# even when it points to this durable install directory.
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export TVM_FFI_ARTIFACT_DIR="$ARTIFACT_DIR"
export TVM_FFI_CACHE_DIR="$ARTIFACT_DIR"

echo "[$(date)] === Prebuild diagnostics ==="
echo "Hostname:      $(hostname)"
echo "ROCM_PATH:     ${ROCM_PATH:-unset}"
echo "CUDA_HOME:     $CUDA_HOME"
echo "rocm dir size: $(du -sh ${CUDA_HOME} 2>/dev/null || echo 'unreadable')"
echo "Python:        $(which python)  ($(python --version 2>&1))"
python -c 'import torch; print("torch:", torch.__version__, "hip:", torch.version.hip)'
echo "Artifact dir:  $ARTIFACT_DIR"
echo "============================================="

# Clear any stale locks from previously killed jobs.
rm -f "$ARTIFACT_DIR"/*.lock "$HOME/.cache/tvm-ffi"/*.lock 2>/dev/null || true

TVM_LIB="$(tvm_ffi_artifact_name "$VENV")"

if [[ -s "$ARTIFACT_DIR/$TVM_LIB" ]]; then
    echo "[$(date)] $TVM_LIB already installed in $ARTIFACT_DIR — nothing to do."
    ls -la "$ARTIFACT_DIR/$TVM_LIB"
    exit 0
fi

echo "[$(date)] Building $TVM_LIB into $ARTIFACT_DIR ..."
time python -m tvm_ffi.utils._build_optional_torch_c_dlpack \
    --output-dir "$ARTIFACT_DIR" \
    --libname "$TVM_LIB" \
    --build-with-rocm

echo ""
echo "[$(date)] === Result ==="
ls -la "$ARTIFACT_DIR"
if [[ -s "$ARTIFACT_DIR/$TVM_LIB" ]]; then
    echo "OK: $ARTIFACT_DIR/$TVM_LIB"
    echo ""
    echo "Installed artifact is ready for Frontier vLLM jobs."
    exit 0
else
    echo "FAIL: $TVM_LIB not produced."
    exit 1
fi
