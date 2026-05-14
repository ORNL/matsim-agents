# Quantum ESPRESSO on Frontier (AMD MI250X, gfx90a)

This page documents how matsim-agents builds and runs Quantum ESPRESSO with
AMD GPU offload on OLCF Frontier. Source code, build tree, and install
prefix live under `external/quantum-espresso/` (gitignored). The recipe
itself is committed under `scripts/setup/frontier/` and
`scripts/launchers/frontier/`.

## What gets built

- QE `develop` branch (full clone — shallow clones break `git describe` and
  `Modules/environment.f90`).
- GPU offload via OpenMP target (`-DQE_GPU_ARCHS=gfx90a`, which auto-derives
  `QE_GPU="openmp;rocm"`).
- All user-facing executables: `pw.x`, `cp.x`, `ph.x`, `pp.x`, `neb.x`,
  `hp.x`, `ld1.x`, `pioud.x`, `all_currents.x`, `epw.x`, `kcw.x`, full
  `tddfpt`/`turbo_*` suite, `pwcond.x`, `xspectra` tools, etc. (~92 binaries).

## Toolchain (pinned)

| Component | Version | Why pinned |
|---|---|---|
| `PrgEnv-cray` | site default | only PrgEnv with working OpenMP target offload to gfx90a |
| `cce` | 18.0.1 (Frontier default) | cce/19/20/21 ftn wrapper requires `libopenacc.pc` not shipped on Frontier |
| `craype-accel-amd-gfx90a` | site default | enables `-target-accel=amd_gfx90a -h omp` in `ftn` |
| `rocm` | 6.2.4 | cray-mpich/8.1.31 GTL is hard-linked against `libamdhip64.so.6`; rocm/7.x ships `.so.7` and breaks the MPI Fortran link probe |
| `cray-fftw` | 3.3.10.9 | CPU FFTW3 (offload kernels still call CPU FFTW for non-annotated paths) |
| `cmake` | 3.30.5 | QE develop CMake API |

## Build

```bash
# From the matsim-agents repo root, on a Frontier login node:
nohup bash scripts/setup/frontier/build-qe-gpu-frontier.sh \
      > runs/build-qe-gpu-login/build-$(date +%Y%m%d-%H%M%S).log 2>&1 &
```

Cross-compile only — no GPU node required for the build. Wall time is
typically 30–60 min including the ICE retry loop (see below).

### Layout (defaults)

```
matsim-agents/external/quantum-espresso/
├── src/           # full git clone of QEF/q-e (develop)
├── build-gpu/     # CMake build tree
└── install-gpu/
    └── bin/       # pw.x, cp.x, ph.x, ... (~92 executables)
```

Override with `QE_PREFIX=/path/to/elsewhere bash scripts/setup/...` to
relocate everything as a unit, or override `SRC_DIR`/`BUILD_DIR`/
`INSTALL_DIR` individually.

### Workarounds baked into the build script

1. **cce/18.0.1 ftn-7991 ICE loop.** Cray Fortran 18.0.1 hits an internal
   compiler error on ~1 dozen QE sources whenever
   `-target-accel=amd_gfx90a -h omp` is in effect. The script runs `make`
   in a loop; on each ICE failure it parses the failed source paths and
   recompiles them by hand with `-target-accel=...` removed and
   `-h omp → -h noomp`. Safe because none of the affected files contain
   `!$omp target` directives (QE's only such directive lives in
   `UtilXlib/tests/`). Per-iter logs land in
   `external/quantum-espresso/build-gpu/ice-workaround-logs/`.

2. **PIOUD `etime()` link error.** `PIOUD/src/pimd_subrout.f90`'s `scnd()`
   timing helper calls the GNU Fortran extension `etime()`, which Cray
   Fortran does not provide. The script idempotently rewrites `scnd()` to
   use the F95-standard `cpu_time` intrinsic (semantically equivalent for
   QE's wall-time stamps).

3. **No `make all` / `make install`.** Both pull in dev/test targets that
   reference uninstalled tools. The script builds an explicit list of
   user-facing target groups and copies `bin/*.x` to `install-gpu/bin/`.

## Run

```bash
# Submit a 30-min batch job on 1 node, 8 GCDs:
sbatch scripts/launchers/frontier/run-pw-gpu-frontier.sh path/to/your.in

# Or, inside a salloc'd compute node:
bash scripts/launchers/frontier/run-pw-gpu-frontier.sh path/to/your.in
```

The launcher loads the same module stack as the build, sets
`MPICH_GPU_SUPPORT_ENABLED=1` and `OMP_TARGET_OFFLOAD=DEFAULT`, and runs
`pw.x` with `-n8 -c7 --gpus-per-node=8 --gpu-bind=closest` (one MPI rank
per GCD, 7 cores/rank).

Override defaults with env vars: `PW_BIN`, `QE_PREFIX`, `NRANKS`,
`OMP_NUM_THREADS`, `OMP_TARGET_OFFLOAD`.

## Verifying GPU linkage

```bash
ldd external/quantum-espresso/install-gpu/bin/pw.x \
  | grep -iE 'amdhip|hsa|rocm|amd_comgr|mpi_gtl'
```

Expected:
- `libamdhip64.so.6` (HIP runtime)
- `libamd_comgr.so.2` (AMD code-object manager)
- `libhsa-runtime64.so.1` (HSA kernel dispatch)
- `librocprofiler-register.so.0`
- `libmpi_gtl_hsa.so.0` (Cray MPICH GPU-aware transport)

## Note on GPU coverage

A "GPU-enabled build" means QE's OpenMP-target-annotated routines *can*
offload to the MI250X. Actual coverage at runtime depends on which code
paths your input exercises (e.g. plain SCF on a medium cell hits
GPU-annotated FFT/`vloc_psi`/`h_psi` hotspots; some niche functionals and
post-processing tools have no offload annotations and run on CPU).

## Module-stack isolation from matsim-agents Python

The QE module stack (PrgEnv-cray + cce + rocm + cray-fftw) is incompatible
with the Python/PyTorch ROCm stack used by HydraGNN and the vLLM model
servers. The two are kept isolated by **never co-loading their modules in
the same shell**. Coupling between QE and matsim-agents-driven workflows
runs through Slurm + the filesystem (input/output files), not in-process.
