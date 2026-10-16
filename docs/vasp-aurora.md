# VASP on Aurora (Intel Data Center GPU Max / PVC)

This page records the VASP 6.6.1 build configuration and helper for Aurora.

## Repository policy

- The VASP source tree under `external/vasp6/src/vasp.6.6.1/` is gitignored.
- The original archive `external/vasp6/src/vasp.6.6.1.tgz` is gitignored.
- The build helper and `deployments/aurora/setup/makefile.include.aurora-gpu`
  are tracked alongside the other platform build configurations.
- Source distributions, pseudopotentials, and generated binaries are not committed.

## Makefile provenance

- VASP version: `6.6.1`
- Upstream template selected: `arch/makefile.include.oneapi_omp_off`
- Tracked build configuration:
  `deployments/aurora/setup/makefile.include.aurora-gpu`
- Local working file used for the Aurora build:
  `external/vasp6/src/vasp.6.6.1/makefile.include`
- Aurora build script:
  `deployments/aurora/setup/build-vasp-gpu-aurora.sh`

The Aurora build started from the VASP-provided Intel oneAPI OpenMP offload
template and was then adjusted locally for Aurora's module stack and Intel PVC
GPUs.

The script builds the three standard executables in one invocation by running
the VASP targets `std`, `gam`, and `ncl` sequentially. It installs the tracked
configuration as the working `makefile.include` before invoking make, replacing
any existing working configuration. Set `MAKEFILE_INCLUDE` to use another recipe.

## Aurora build configuration

- Uses Aurora's `mpifort` MPI wrapper for Fortran compilation and linkage
- Kept Intel compilers for non-wrapper C/C++ compilation where needed:
  `icx` and `icpx`
- Retained OpenMP offload for Intel GPUs with PVC device targeting
- Added Level Zero linkage needed by the oneAPI offload stack on Aurora
- Kept Intel MKL for BLAS, LAPACK, FFT, and ScaLAPACK linkage
- Used host-specific CPU tuning for the build host via `-xHOST`
- Preserved the `OMP_OFFLOAD` preprocessor path required by the GPU build

The helper resets the module environment and loads
`oneapi/release/2026.1.0` and `intel_gpu_umd_aicoe`, rather than the Python
frameworks module. Override these with `ONEAPI_MODULE` and `GPU_UMD_MODULE`.
The module-provided `MKLROOT` takes precedence over the recipe's default,
`/opt/aurora/26.181.0/oneapi/mkl/latest`, which matches the current toolchain.

## Build command

Run from the repository root:

```bash
bash deployments/aurora/setup/build-vasp-gpu-aurora.sh
```

Useful overrides:

```bash
NCORES=32 bash deployments/aurora/setup/build-vasp-gpu-aurora.sh
CLEAN_BUILD=1 bash deployments/aurora/setup/build-vasp-gpu-aurora.sh
VASP_TARGET=gam bash deployments/aurora/setup/build-vasp-gpu-aurora.sh
PREFIX=build-aurora bash deployments/aurora/setup/build-vasp-gpu-aurora.sh
MAKEFILE_INCLUDE=/path/to/makefile.include bash deployments/aurora/setup/build-vasp-gpu-aurora.sh
```

## Build targets

The default build requests all three executables:

- `bin/vasp_std`
- `bin/vasp_gam`
- `bin/vasp_ncl`

Use `VASP_TARGET=std`, `gam`, or `ncl` to select a single target. Compilation
success does not by itself validate a VASP calculation or GPU runtime behavior.

## Rebuild checklist

When reproducing the Aurora build, confirm these points before compiling:

1. Provide your licensed VASP 6.6.1 source tree, or override `VASP_ROOT`.
2. Review the tracked Aurora recipe or select a custom `MAKEFILE_INCLUDE`.
3. Run the helper with the desired target and parallelism; it installs the recipe
  as `external/vasp6/src/vasp.6.6.1/makefile.include` by default.
4. Do not commit the VASP source tree, generated objects, or the vendor tarball.

## Related repository docs

- Aurora setup overview: `deployments/aurora/setup/README.md`
- HPC index: `docs/hpc-platforms.md`
- Top-level project guide: `README.md`

## Driving VASP from Python

Two paths are exposed:

- **Active-learning labelling (single-point SCF, no ionic motion):**
  driven by [`src/matsim_agents/backends/dft/vasp.py`](../src/matsim_agents/backends/dft/vasp.py)
  using a user-supplied `INCAR.template`. This is what `matsim-agents al run`
  invokes when `dft.backend: vasp` is set in the AL config.
- **Standalone DFT relaxations** (`scf` / `relax` / `vc-relax` /
  `vc-relax-shape`): driven by [`src/matsim_agents/backends/dft/vasp_relax.py`](../src/matsim_agents/backends/dft/vasp_relax.py),
  the symmetric pair to `backends/dft/qe_relax.py`. It writes a self-contained
  INCAR (no template required for the common case), composition-aware
  `ENCUT/ISMEAR/SIGMA/KSPACING`, and parses `vasprun.xml`+`OUTCAR` for the
  full per-ionic-step trajectory. Set `MATSIM_VASP_LAUNCHER` to point at a
  wrapper script (analogue of the QE launcher pattern).