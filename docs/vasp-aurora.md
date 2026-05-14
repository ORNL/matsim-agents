# VASP on Aurora (Intel Data Center GPU Max / PVC)

This page records which VASP 6.6.0 makefile was used for the Aurora build in
this repository without committing the proprietary VASP source tree.

## Repository policy

- The VASP source tree under `external/vasp6/src/vasp.6.6.0/` is gitignored.
- The original archive `external/vasp6/src/vasp.6.6.0.tgz` is gitignored.
- Only build notes, provenance, and non-proprietary configuration summaries are
  committed.

## Makefile provenance

- VASP version: `6.6.0`
- Upstream template selected: `arch/makefile.include.oneapi_omp_off`
- Local working file used for the Aurora build:
  `external/vasp6/src/vasp.6.6.0/makefile.include`
- Aurora build script:
  `scripts/setup/aurora/build-vasp-gpu-aurora.sh`

The Aurora build started from the VASP-provided Intel oneAPI OpenMP offload
template and was then adjusted locally for Aurora's module stack and Intel PVC
GPUs.

The script builds the three standard executables in one invocation by running
the VASP targets `std`, `gam`, and `ncl` sequentially.

## Aurora-specific changes applied

The committed record here is intentionally descriptive rather than a verbatim
copy of the proprietary makefile.

- Switched the Fortran and C MPI entry points to Aurora's wrappers:
  `mpifort` and `mpicc`
- Kept Intel compilers for non-wrapper C/C++ compilation where needed:
  `icx` and `icpx`
- Retained OpenMP offload for Intel GPUs with PVC device targeting
- Added Level Zero linkage needed by the oneAPI offload stack on Aurora
- Kept Intel MKL for BLAS, LAPACK, FFT, and ScaLAPACK linkage
- Used host-specific CPU tuning for the build host via `-xHOST`
- Preserved the `OMP_OFFLOAD` preprocessor path required by the GPU build

## Build command

Run from the repository root:

```bash
bash scripts/setup/aurora/build-vasp-gpu-aurora.sh
```

Useful overrides:

```bash
NCORES=32 bash scripts/setup/aurora/build-vasp-gpu-aurora.sh
CLEAN_BUILD=1 bash scripts/setup/aurora/build-vasp-gpu-aurora.sh
VASP_TARGET=gam bash scripts/setup/aurora/build-vasp-gpu-aurora.sh
PREFIX=build-aurora bash scripts/setup/aurora/build-vasp-gpu-aurora.sh
```

## Validated build state

The Aurora tuning above was used for the successful GPU-enabled build of:

- `bin/vasp_std`

The scripted default build target produces all three executables:

- `bin/vasp_std`
- `bin/vasp_gam`
- `bin/vasp_ncl`

If additional VASP targets are built later, extend this page with the date,
target name, and any makefile deltas beyond the base Aurora configuration.

## Rebuild checklist

When reproducing the Aurora build, confirm these points before compiling:

1. Start from `arch/makefile.include.oneapi_omp_off` in the VASP 6.6.0 source.
2. Re-apply the Aurora wrapper/compiler adjustments summarized above.
3. Keep the edited file only as the local
   `external/vasp6/src/vasp.6.6.0/makefile.include` working copy.
4. Do not commit the VASP source tree, generated objects, or the vendor tarball.

## Related repository docs

- Aurora setup overview: `scripts/setup/aurora/README.md`
- HPC index: `docs/hpc-platforms.md`
- Top-level project guide: `README.md`