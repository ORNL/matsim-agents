# Quantum ESPRESSO on Aurora (Intel Data Center GPU Max / PVC)

This page documents how matsim-agents builds and runs Quantum ESPRESSO with
Intel GPU offload on ALCF Aurora.

The recipe scripts are committed in:

- `deployments/aurora/setup/build-qe-gpu-aurora.sh`
- `deployments/aurora/launchers/run-pw-gpu-aurora.sh`

Source code, build artifacts, and installation are placed under
`external/quantum-espresso/` (gitignored).

## Status

The Intel GPU build and runtime are not yet validated. The previous Aurora
build job exited before CMake because PBS staged the script outside the
repository and the builder could not locate the checkout. The builder now
falls back to `PROJECT_ROOT` or `PBS_O_WORKDIR` and verifies the requested QE
GPU features and architecture in the CMake cache. A successful build still
needs a compute-node smoke test to confirm actual offload.

Aurora's system QE 7.6 CPU/OpenMP binary is not evidence of GPU support; use
the repository-built `install-gpu/bin/pw.x` for the GPU smoke test.

In the QE source revision used by this build, the Intel oneAPI OpenMP target
regions are in EPW, not the PWSCF, FFTXlib, or eigensolver paths. A successful
`pw.x` SCF therefore does not demonstrate Intel GPU execution. The Aurora GPU
smoke job runs the `epw_mob_polar` test and requires every EPW stage's
`LIBOMPTARGET_INFO` log to record executed GPU kernels; EPW kernels are too short
to show up reliably in the `xpu-smi` telemetry it also records.

Unpatched QE `develop` (71e4e9b) aborts `epw_mob_polar` stages `epw2`/`epw3`
under `OMP_TARGET_OFFLOAD=MANDATORY` with a PVC page fault
(`Segmentation fault from GPU ... (CCS) type: 0 (NotPresent)`), while the same
inputs pass with offload disabled. In the oneMKL branch of
`vmewan2bloch_batch` (`EPW/src/wannier2bloch.f90`), `ir_cfac` is filled on the
host and passed to a `!$omp dispatch` `ZGEMV_BATCH_STRIDED` without ever being
mapped, so its `target update to` is a no-op and oneMKL receives a host pointer.
The builder applies
`deployments/aurora/setup/patches/qe-epw-onemkl-ir-cfac/epw-map-ir-cfac.patch`,
which adds the missing `!$omp target enter data map(alloc:ir_cfac)`.

## What gets built

- QE source from `QEF/q-e` (default branch: `develop`, pinned to
  `QE_COMMIT=71e4e9b25b35`; set `QE_COMMIT=''` to follow the branch tip).
- GPU build via QE CMake flags:
  - `-DQE_GPU="openmp;oneapi"`
  - `-DQE_GPU_ARCHS=intel_gpu_pvc`
- MPI-enabled executables and common target groups, including:
  - `pw`, `cp`, `ph`, `pp`, `neb`, `hp`, `ld1`, `pwall`, `pwcond`,
    `tddfpt`, `upf`, `xspectra`, `epw`

## Toolchain and environment

The Aurora script uses the site module stack and MPI wrappers that resolve to
Intel oneAPI compilers:

- `module load frameworks`
- compilers via wrappers: `mpicc`, `mpicxx`, `mpifort`
- build system: CMake

This keeps the setup aligned with Aurora-provided MPI/compiler integration.

## Build

Run from the repository root:

```bash
bash deployments/aurora/setup/build-qe-gpu-aurora.sh
```

Useful overrides:

```bash
QE_VERSION=7.4 bash deployments/aurora/setup/build-qe-gpu-aurora.sh
NCORES=32 bash deployments/aurora/setup/build-qe-gpu-aurora.sh
QE_GPU_ARCHS=intel_gpu_pvc QE_GPU="openmp;oneapi" \
  bash deployments/aurora/setup/build-qe-gpu-aurora.sh
EXTRA_CMAKE_ARGS="-DVAR=VALUE" bash deployments/aurora/setup/build-qe-gpu-aurora.sh
```

Default layout:

```text
external/quantum-espresso/
├── src/
├── build-gpu/
└── install-gpu/
    └── bin/
```

## Run `pw.x` on Aurora

Use the launcher:

```bash
bash deployments/aurora/launchers/run-pw-gpu-aurora.sh path/to/pw.in
```

Launcher behavior:

- loads `frameworks`
- sets GPU-relevant runtime defaults:
  - `MPICH_GPU_SUPPORT_ENABLED=1`
  - `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`
- launches with `srun` when available (falls back to `mpiexec`)

Common runtime overrides:

```bash
NRANKS=6 OMP_NUM_THREADS=4 GPU_BIND=closest \
  bash deployments/aurora/launchers/run-pw-gpu-aurora.sh path/to/pw.in
```

## Verify installation quickly

```bash
du -sh external/quantum-espresso/{build-gpu,install-gpu}
ls -lh external/quantum-espresso/install-gpu/bin/{pw.x,cp.x,ph.x,pp.x,epw.x}
ls external/quantum-espresso/install-gpu/bin/*.x | wc -l
```

Expected characteristics after a successful build:

- core binaries in the multi-megabyte range
- `pw.x` exists and the CMake cache records `QE_GPU=openmp;oneapi` and
  `QE_GPU_ARCHS=intel_gpu_pvc`

## Notes

- Intel GPU support in QE is actively evolving; the recommended approach is to
  keep script defaults and only override flags when debugging branch-specific
  CMake behavior.
- CPU and GPU builds can coexist by using different build/install directories
  (`build-cpu`/`install-cpu` vs `build-gpu`/`install-gpu`).

## Troubleshooting

### `module: command not found`

The scripts already attempt to source common module-init paths. If your site
shell differs, source your site module init first, then re-run:

```bash
source /etc/profile.d/modules.sh
bash deployments/aurora/setup/build-qe-gpu-aurora.sh
```

### MPI wrapper compilers missing (`mpicc`, `mpicxx`, `mpifort`)

Confirm the Aurora frameworks stack is loaded and wrappers resolve:

```bash
module reset
module load frameworks
which mpicc mpicxx mpifort
```

If your site prefers different wrappers, override script variables:

```bash
C_COMPILER=<your-mpicc> CXX_COMPILER=<your-mpicxx> FORTRAN_COMPILER=<your-mpifort> \
  bash deployments/aurora/setup/build-qe-gpu-aurora.sh
```

### CMake configure fails on GPU flags

Start with script defaults first. If testing another QE branch/revision,
override GPU settings explicitly:

```bash
QE_GPU="openmp;oneapi" QE_GPU_ARCHS=intel_gpu_pvc \
  bash deployments/aurora/setup/build-qe-gpu-aurora.sh
```

Pass additional CMake debug flags when needed:

```bash
EXTRA_CMAKE_ARGS="-DCMAKE_VERBOSE_MAKEFILE=ON" \
  bash deployments/aurora/setup/build-qe-gpu-aurora.sh
```

### `pw.x` launcher cannot find executable

Confirm install path and binary:

```bash
ls -lh external/quantum-espresso/install-gpu/bin/pw.x
```

If you installed elsewhere, set `QE_PREFIX` or `PW_BIN` when launching.

### GPU not selected at runtime

Check runtime variables and launcher environment:

```bash
echo "$MPICH_GPU_SUPPORT_ENABLED"
echo "$ONEAPI_DEVICE_SELECTOR"
```

Expected defaults from launcher:

- `MPICH_GPU_SUPPORT_ENABLED=1`
- `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`

### Sanity check after any rebuild

```bash
du -sh external/quantum-espresso/{build-gpu,install-gpu}
ls external/quantum-espresso/install-gpu/bin/*.x | wc -l
```

These checks quickly catch partial installs and path mismatches.