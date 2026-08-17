# Quantum ESPRESSO Build Scripts for Perlmutter

## Quick Reference

### Build Scripts Available

| Script | Target | Best For | Time |
|--------|--------|----------|------|
| `build-qe-gpu-perlmutter.sh` | NVIDIA A100 GPU | Production runs, fast code | ~20-30 min |
| `build-qe-cpu-perlmutter.sh` | CPU-only | Testing, dev, CPU nodes | ~10-15 min |

### Decision Tree

```
Do you want GPU support?
├─ YES → Use build-qe-gpu-perlmutter.sh (NVHPC 25.5 + CUDA 12.9)
│        ✓ Faster execution
│        ✓ Required for GPU partition jobs
│        ✓ Recommended for production
│
└─ NO → Use build-qe-cpu-perlmutter.sh (PrgEnv-gnu + gfortran)
         ✓ For CPU-only nodes
         ✓ For quick testing/development
         ✓ Simpler build process
```

## Getting Started

### 1. Build on Login Node (Recommended - survives disconnect)

**GPU build:**
```bash
mkdir -p runs/build-qe-gpu-login
nohup bash deployments/perlmutter/setup/build-qe-gpu-perlmutter.sh \
      > runs/build-qe-gpu-login/build-$(date +%Y%m%d-%H%M%S).log 2>&1 &
```

Monitor progress:
```bash
tail -f runs/build-qe-gpu-login/build-<timestamp>.log
```

### 2. Build as Batch Job

**GPU build:**
```bash
sbatch deployments/perlmutter/setup/build-qe-gpu-perlmutter.sh
squeue -u $USER
```

**CPU build:**
```bash
sbatch deployments/perlmutter/setup/build-qe-cpu-perlmutter.sh
```

### 3. Check Build Success

```bash
ls -lh external/quantum-espresso/install-gpu/bin/
```

Should show: `pw.x`, `cp.x`, `ph.x`, `pp.x`, `neb.x`, `hp.x`, `ld1.x`, etc.

## Running QE

### Before Any QE Run

Always load the environment:
```bash
source deployments/perlmutter/setup/perlmutter-module-stack.sh
load_perlmutter_modules_gpu     # for GPU runs
# or: load_perlmutter_modules   # for CPU-only
conda activate scripts/perlmutter_venv
```

### GPU Run (on GPU partition)

```bash
srun -N1 -n8 -c14 --gpus-per-node=8 --gpu-bind=closest \
     external/quantum-espresso/install-gpu/bin/pw.x -in input.in
```

### CPU Run (on CPU partition)

```bash
srun -N1 -n64 \
     external/quantum-espresso/install-cpu/bin/pw.x -in input.in
```

## File Locations

```
matsim-agents/
├── deployments/perlmutter/setup/
│   ├── build-qe-gpu-perlmutter.sh    ← GPU build script
│   ├── build-qe-cpu-perlmutter.sh    ← CPU build script
│   ├── perlmutter-module-stack.sh    ← Environment setup
│   └── README.md                     ← Setup documentation
│
├── external/quantum-espresso/        ← Output location (gitignored)
│   ├── src/                          ← QE source code
│   ├── build-gpu/                    ← GPU build directory
│   ├── install-gpu/
│   │   └── bin/                      ← Compiled executables
│   ├── build-cpu/
│   └── install-cpu/
│
└── docs/
    ├── quantum-espresso-frontier.md  ← Frontier (AMD GPU) guide
    └── quantum-espresso-perlmutter.md ← Perlmutter (NVIDIA GPU) guide
```

## Customization

### Override Install Location

```bash
QE_PREFIX=/scratch/user/qe bash deployments/perlmutter/setup/build-qe-gpu-perlmutter.sh
```

### Use Different QE Version

```bash
QE_VERSION=7.4 bash deployments/perlmutter/setup/build-qe-gpu-perlmutter.sh
```

### Clean Rebuild (Remove Old Build)

```bash
CLEAN_BUILD=1 bash deployments/perlmutter/setup/build-qe-gpu-perlmutter.sh
```

(Default: `CLEAN_BUILD=1`, i.e., always clean)

### Reuse Build Cache (Faster Rebuild)

```bash
CLEAN_BUILD=0 bash deployments/perlmutter/setup/build-qe-gpu-perlmutter.sh
```

### Increase Parallelism (on compute node)

```bash
NCORES=128 bash deployments/perlmutter/setup/build-qe-gpu-perlmutter.sh
```

## Build Workflow Comparison

### Frontier (AMD MI250X) vs Perlmutter (NVIDIA A100)

| Step | Frontier | Perlmutter |
|------|----------|-----------|
| **Compiler** | PrgEnv-cray (cce/18.0.1) | PrgEnv-nvidia (nvfortran) |
| **Module load** | PrgEnv-cray, rocm/6.2.4 | PrgEnv-gnu/8.5.0, cudatoolkit/12.9, NVHPC 25.5 |
| **GPU arch flag** | `-DQE_GPU_ARCHS=gfx90a` | `-DQE_GPU_ARCHS=sm_80` |
| **Compiler issues** | ICE retry loop needed | None — stable toolchain |
| **Build complexity** | High (300+ lines for workarounds) | Low (straightforward) |
| **Expected time** | 45-60 min | 20-30 min |

## Troubleshooting

### "Module not found" errors

Ensure you're in the correct directory and have loaded the modules:
```bash
cd /global/cfs/projectdirs/amsc001/cm2us/mlupopa/matsim-agents
source deployments/perlmutter/setup/perlmutter-module-stack.sh
```

### Build hangs or times out

Check the build log:
```bash
tail -100 external/quantum-espresso/build-gpu/build.log
```

If out of disk space:
```bash
df -h /global/cfs/projectdirs/amsc001/
```

### GPU libraries not linked

Verify GPU support:
```bash
ldd external/quantum-espresso/install-gpu/bin/pw.x | grep cuda
```

### Executable not found

Check if build completed successfully:
```bash
ls -lh external/quantum-espresso/install-gpu/bin/ | head
```

## Advanced: Building Multiple Versions

To maintain multiple QE versions:

```bash
# GPU version 7.4
QE_PREFIX=/global/cfs/projectdirs/amsc001/cm2us/mlupopa/qe-7.4 \
QE_VERSION=7.4 \
bash deployments/perlmutter/setup/build-qe-gpu-perlmutter.sh

# GPU develop version
QE_PREFIX=/global/cfs/projectdirs/amsc001/cm2us/mlupopa/qe-develop \
QE_VERSION=develop \
bash deployments/perlmutter/setup/build-qe-gpu-perlmutter.sh

# CPU version
QE_PREFIX=/global/cfs/projectdirs/amsc001/cm2us/mlupopa/qe-cpu \
bash deployments/perlmutter/setup/build-qe-cpu-perlmutter.sh
```

Then set `PATH` to use the desired version:
```bash
export PATH=/global/cfs/projectdirs/amsc001/cm2us/mlupopa/qe-7.4/install-gpu/bin:$PATH
```

## Further Reading

- Full documentation: [docs/quantum-espresso-perlmutter.md](../../docs/quantum-espresso-perlmutter.md)
- Frontier build guide: [docs/quantum-espresso-frontier.md](../../docs/quantum-espresso-frontier.md)
- QE homepage: https://www.quantum-espresso.org/
- NERSC Perlmutter docs: https://docs.nersc.gov/systems/perlmutter/

---

**Last updated**: May 12, 2026
