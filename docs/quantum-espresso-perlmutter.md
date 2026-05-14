# Quantum ESPRESSO on Perlmutter (NVIDIA A100 GPUs)

This guide covers building and running Quantum ESPRESSO (QE) on NERSC Perlmutter with NVIDIA A100 GPU support.

## Overview

Perlmutter provides:
- **GPU partition**: 1,792 nodes with 8× NVIDIA A100 GPUs per node (sm_80 architecture)
- **CPU partition**: 384 nodes with 128 cores each (AMD EPYC 7763)
- **Interconnect**: Slingshot 11 (200 Gbps)
- **CUDA toolkit**: 12.4 (cu124) with cuFFT, cuBLAS, cuSolver
- **Compilers**: PrgEnv-nvidia (nvfortran 24.5+), PrgEnv-gnu (gfortran 13)

## Build Scripts

Two build scripts are provided:

### 1. GPU Build: `build-qe-gpu-perlmutter.sh`
- **Compiler**: PrgEnv-nvidia (nvfortran, nvc++, nvc)
- **CUDA**: 12.4 with GPU offload to A100 (sm_80)
- **Libraries**: cuFFT, cuBLAS, cuSolver, cray-mpich, cray-fftw
- **Execution time**: ~20-30 minutes on login node or compute node
- **Produces**: pw.x, cp.x, ph.x, pp.x, neb.x, hp.x, ld1.x, epw.x, kcw.x, and 15+ more

### 2. CPU Build: `build-qe-cpu-perlmutter.sh`
- **Compiler**: PrgEnv-gnu (gfortran 13, gcc 13)
- **CPU-only**: No GPU support
- **Execution time**: ~10-15 minutes
- **Useful for**: Testing, CPU nodes, development

## Building QE on Perlmutter

### Prerequisites

Ensure your matsim-agents environment is set up:
```bash
cd /global/cfs/projectdirs/amsc001/cm2us/mlupopa/matsim-agents
source scripts/setup/perlmutter/perlmutter-module-stack.sh
```

### Option A: GPU Build (Recommended)

**Method 1: On a login node (survives disconnect)**
```bash
mkdir -p runs/build-qe-gpu-login
nohup bash scripts/setup/perlmutter/build-qe-gpu-perlmutter.sh \
      > runs/build-qe-gpu-login/build-$(date +%Y%m%d-%H%M%S).log 2>&1 &
```

**Method 2: As a batch job**
```bash
sbatch scripts/setup/perlmutter/build-qe-gpu-perlmutter.sh
```

**Method 3: Custom build prefix**
```bash
QE_PREFIX=/scratch/user/qe bash scripts/setup/perlmutter/build-qe-gpu-perlmutter.sh
```

### Option B: CPU Build

```bash
sbatch scripts/setup/perlmutter/build-qe-cpu-perlmutter.sh
```

## Build Output

Successful builds create:
```
external/quantum-espresso/
├── src/                          # QE source code
│   └── .git/                     # full git repo
├── build-gpu/ (or build-cpu/)    # CMake build artifacts
│   ├── cmake.log                 # CMake configuration log
│   ├── build.log                 # make output
│   └── bin/                      # compiled executables
└── install-gpu/ (or install-cpu/)
    └── bin/                      # pw.x, cp.x, ph.x, etc. (~92 executables)
```

## Running QE on Perlmutter

### Setup Environment

Load the same modules and environment as used in the build:
```bash
source scripts/setup/perlmutter/perlmutter-module-stack.sh
load_perlmutter_modules_gpu      # for GPU jobs
# or
# load_perlmutter_modules        # for CPU-only jobs
conda activate scripts/perlmutter_venv
```

### GPU Execution

For GPU-enabled pw.x, set up your job script:
```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -A amsc001
#SBATCH --job-name=qe-gpu-job

# Load environment
source scripts/setup/perlmutter/perlmutter-module-stack.sh
load_perlmutter_modules_gpu
conda activate scripts/perlmutter_venv

# Set GPU offload controls (optional but recommended)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run pw.x with 8 MPI ranks × 14 OMP threads = 112 cores
# (A100 has 108 cores, so 8 GPU-bound ranks fit on 1 node)
srun -N1 -n8 -c14 --gpus-per-node=8 --gpu-bind=closest \
     external/quantum-espresso/install-gpu/bin/pw.x -in test.in
```

### CPU Execution

For CPU-only pw.x:
```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -A amsc001

source scripts/setup/perlmutter/perlmutter-module-stack.sh
load_perlmutter_modules
conda activate scripts/perlmutter_venv

# Run with full 128 cores per node
srun -N1 -n64 \
     external/quantum-espresso/install-cpu/bin/pw.x -in test.in
```

## Comparison: Perlmutter vs Frontier

| Aspect | Perlmutter (NVIDIA A100) | Frontier (AMD MI250X) |
|--------|--------------------------|----------------------|
| **GPU/node** | 8× A100 | 8× MI250X |
| **Compiler** | nvfortran/PrgEnv-nvidia | cce/PrgEnv-cray |
| **Offload method** | CUDA (native) | OpenMP target + ROCm |
| **Build workarounds** | None | Complex ICE retry loop |
| **Compile time** | ~20-30 min | ~45-60 min (with ICE workarounds) |
| **SM/GCD mapping** | sm_80 (A100) | gfx90a (MI250X) |
| **Complexity** | Lower (stable NVIDIA toolchain) | Higher (Cray compiler ICE bugs) |

## CMake Configuration Flags

The GPU builds use these key CMake flags:

```bash
cmake \
  -DQE_GPU_ARCHS=sm_80              # A100 GPU architecture
  -DQE_ENABLE_OFFLOAD=ON            # Enable GPU offload
  -DQE_ENABLE_MPI=ON                # MPI support
  -DQE_ENABLE_OPENMP=ON             # OpenMP for threading
  -DQE_ENABLE_SCALAPACK=OFF         # Not needed for GPU
  -DQE_ENABLE_HDF5=OFF              # Optional; skip for first build
  -DQE_ENABLE_LIBXC=OFF             # Optional; use QE internal XC
  -DQE_FFTW_VENDOR=FFTW3            # Use CPU FFTW3
  -DFFTW3_ROOT=$CRAY_FFTW_PREFIX    # Cray FFTW3 location
  -DCMAKE_BUILD_TYPE=Release        # Optimization level
  ...
```

The NVIDIA C/Fortran compilers (`cc`, `ftn`, `CC`) are Cray wrappers that:
- Inject NVIDIA compiler flags automatically under PrgEnv-nvidia
- Link NVIDIA/Cray MPI libraries (cray-mpich GPU-aware)
- Resolve CUDA paths from `cudatoolkit` module

## Troubleshooting

### Build hangs or times out

If the build exceeds 2-3 hours, check:
1. Disk I/O performance (network filesystem can be slow)
2. Build logs: `external/quantum-espresso/build-gpu/build.log`
3. Retry with `CLEAN_BUILD=0 bash ...` to resume from cached objects

### CUDA libraries not linked

Verify GPU build created CUDA-linked executables:
```bash
ldd external/quantum-espresso/install-gpu/bin/pw.x | grep -i cuda
```

Should show: `libcufft.so`, `libcublas.so`, etc.

### "nvfortran not found" error

Ensure you're using the correct environment:
```bash
module load PrgEnv-nvidia
nvfortran --version
```

### CMake configuration fails

Check that all required modules are loaded:
```bash
module list | grep -E "cmake|cuda|fftw|cray-mpich"
```

## Environment Variables for Execution

When running QE with GPU offload:

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | Restrict GPUs visible to MPI ranks |
| `OPAL_DEVICE_MANAGER` | `none` (optional) | Disable OpenPAL device binding if conflicts arise |
| `MPICH_GPU_SUPPORT_ENABLED` | `1` | Enable GPU-aware MPI (usually auto-detected) |

## External Resources

- [Quantum ESPRESSO Documentation](https://www.quantum-espresso.org/)
- [NERSC Perlmutter User Guide](https://docs.nersc.gov/systems/perlmutter/)
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [PrgEnv-nvidia on Perlmutter](https://docs.nersc.gov/development/programming-languages/fortran/)

## Quick Start: Build & Test

```bash
# 1. Set up environment
cd /global/cfs/projectdirs/amsc001/cm2us/mlupopa/matsim-agents
source scripts/setup/perlmutter/perlmutter-module-stack.sh
load_perlmutter_modules_gpu
conda activate scripts/perlmutter_venv

# 2. Build QE GPU version (on login node, background)
mkdir -p runs/build-qe-gpu-login
nohup bash scripts/setup/perlmutter/build-qe-gpu-perlmutter.sh \
      > runs/build-qe-gpu-login/build.log 2>&1 &
tail -f runs/build-qe-gpu-login/build.log

# 3. Once build completes, test the executable
ldd external/quantum-espresso/install-gpu/bin/pw.x

# 4. Create a simple test input
cat > test-qe.in << 'EOF'
&control
  calculation = 'scf'
  prefix = 'test'
/
&system
  ibrav = 1
  celldm(1) = 10.0
  nat = 2
  ntyp = 1
  ecutwfc = 50
/
&electrons
/
ATOMIC_SPECIES
H 1.0 H.pz-vbc.UPF
ATOMIC_POSITIONS
H 0.0 0.0 0.0
H 0.5 0.5 0.5
K_POINTS {gamma}
EOF

# 5. Create and submit a test job
cat > job-qe-test.sh << 'EOF'
#!/bin/bash
#SBATCH -N 1 -C gpu -q regular -t 00:10:00 -A amsc001

source scripts/setup/perlmutter/perlmutter-module-stack.sh
load_perlmutter_modules_gpu
conda activate scripts/perlmutter_venv

srun -N1 -n1 -c14 --gpus-per-node=1 --gpu-bind=closest \
     external/quantum-espresso/install-gpu/bin/pw.x -in test-qe.in
EOF

sbatch job-qe-test.sh
```

---

**Last updated**: May 12, 2026  
**Maintainer**: matsim-agents team
