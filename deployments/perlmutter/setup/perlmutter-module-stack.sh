#!/bin/bash

# Perlmutter module stack for CUDA/A100 jobs.
# Source this file and call the appropriate function:
#   load_perlmutter_modules      — Load standard Perlmutter CUDA stack
#   load_perlmutter_modules_gpu  — Load Perlmutter modules for GPU (A100) jobs

load_perlmutter_modules() {
    # Initialize modules if not already available
    if ! command -v module >/dev/null 2>&1; then
        if [[ -f /etc/profile.d/modules.sh ]]; then
            source /etc/profile.d/modules.sh
        elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
            source /usr/share/lmod/lmod/init/bash
        elif [[ -f /usr/share/Modules/init/bash ]]; then
            source /usr/share/Modules/init/bash
        fi
    fi

    # Perform Cray "hard reset"
    if [[ -f /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh ]]; then
        source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh || true
    fi

    module reset
    ml nersc-default/1.0 || true

    # Cray programming environment + MPI
    ml cpe/24.07
    ml PrgEnv-gnu/8.5.0
    ml cray-mpich/8.1.30

    # CUDA toolkit
    ml cudatoolkit/12.9

    # Modern compiler toolchain for C++ extensions
    ml gcc-native/13.2

    # Build helpers
    ml cmake/3.30.2 || ml cmake/3.24.3 || true

    # Conda
    ml conda/Miniforge3-24.11.3-0 || ml conda/Miniforge3-24.7.1-0 || true

    echo "✓ Perlmutter modules loaded"
}

load_perlmutter_modules_gpu() {
    # Initialize modules if not already available
    if ! command -v module >/dev/null 2>&1; then
        if [[ -f /etc/profile.d/modules.sh ]]; then
            source /etc/profile.d/modules.sh
        elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
            source /usr/share/lmod/lmod/init/bash
        elif [[ -f /usr/share/Modules/init/bash ]]; then
            source /usr/share/Modules/init/bash
        fi
    fi

    # Perform Cray "hard reset"
    if [[ -f /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh ]]; then
        source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh || true
    fi

    module reset
    ml nersc-default/1.0 || true

    # Cray programming environment + MPI
    ml cpe/24.07
    ml PrgEnv-gnu/8.5.0
    ml cray-mpich/8.1.30

    # A100 target (SM80) for GPU compute
    ml craype-accel-nvidia80

    # CUDA toolkit (match PyTorch GPU requirements)
    ml cudatoolkit/12.9

    # Modern compiler toolchain for C++ extensions
    ml gcc-native/13.2

    # Build helpers
    ml cmake/3.30.2 || ml cmake/3.24.3 || true

    # Conda
    ml conda/Miniforge3-24.11.3-0 || ml conda/Miniforge3-24.7.1-0 || true

    echo "✓ Perlmutter GPU modules loaded"
}

load_perlmutter_modules_nvidia() {
    # Load NVIDIA/CUDA tools for GPU-accelerated builds (e.g., QE with CUDA)
    # Simplified approach: manually expose nvfortran binary and set CUDA paths
    
    # Initialize modules if not already available
    if ! command -v module >/dev/null 2>&1; then
        if [[ -f /etc/profile.d/modules.sh ]]; then
            source /etc/profile.d/modules.sh
        elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
            source /usr/share/lmod/lmod/init/bash
        elif [[ -f /usr/share/Modules/init/bash ]]; then
            source /usr/share/Modules/init/bash
        fi
    fi

    # Load basic modules (versions match HydraGNN Perlmutter installer)
    ml cpe/24.07 2>/dev/null || true
    ml PrgEnv-gnu/8.5.0 2>/dev/null || ml PrgEnv-gnu 2>/dev/null || true
    ml cray-mpich/8.1.30 2>/dev/null || true
    ml gcc-native/13.2 2>/dev/null || true
    ml cudatoolkit/12.9 2>/dev/null || ml cudatoolkit 2>/dev/null || true
    ml cray-fftw 2>/dev/null || true
    ml cmake 2>/dev/null || true
    ml git 2>/dev/null || true

    # Manually add nvfortran to PATH (located in NVIDIA HPC SDK)
    # Pin to NVHPC 25.5: it is the most recent bundle that ships CUDA 12.9,
    # matching HydraGNN's cudatoolkit/12.9 + PyTorch cu129 wheels. NVHPC 25.9
    # jumped to CUDA 13.0 and would create a CUDA-major mismatch with PyTorch.
    export NVIDIA_HPC_SDK_DIR="${NVIDIA_HPC_SDK_DIR:-/opt/nvidia/hpc_sdk/Linux_x86_64}"
    export NVIDIA_COMPILER_VERSION="${NVIDIA_COMPILER_VERSION:-25.5}"
    NVFORTRAN_BIN="${NVIDIA_HPC_SDK_DIR}/${NVIDIA_COMPILER_VERSION}/compilers/bin"
    if [[ -d "${NVFORTRAN_BIN}" ]]; then
        export PATH="${NVFORTRAN_BIN}:${PATH}"
    elif [[ -d "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/bin" ]]; then
        export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/bin:${PATH}"
    elif [[ -d "/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/bin" ]]; then
        export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/bin:${PATH}"
    else
        # Fallback: try 23.1
        export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin:${PATH}"
    fi

    # Set CUDA paths for CMake (NVHPC 25.5 bundles CUDA 12.9)
    if [[ -z "${CUDA_HOME:-}" ]]; then
        export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9"
        if [[ ! -d "${CUDA_HOME}" ]]; then
            export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4"
        fi
        if [[ ! -d "${CUDA_HOME}" ]]; then
            export CUDA_HOME="/usr/local/cuda-12.9"
        fi
    fi

    echo "✓ Perlmutter NVIDIA/CUDA modules configured"
    echo "  nvfortran: $(which nvfortran 2>/dev/null || echo 'not found')"
    echo "  CUDA_HOME: ${CUDA_HOME}"
}
