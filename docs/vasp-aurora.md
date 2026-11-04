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

## Aurora runtime requirements

A successful build is not enough: the first VASP calculations on Aurora failed
three different ways before any SCF step completed. All three fixes are runtime
settings; none changes the physics or the staged scientific inputs. Validated
with `vasp_std` 6.6.1 on one node, 12 MPI ranks x 8 OpenMP threads, one PVC
tile per rank, Nb4Ta4O20 (28 atoms, `ENCUT=600`, `PREC=Accurate`, `ALGO=Normal`,
4x8x9 Gamma mesh, 146 k-points, 132 bands).

| Symptom | Cause | Fix | Where |
|---|---|---|---|
| `launch failed: Invalid GPU visibility keyword: closest`, exit 127, VASP never starts (job 9038831) | PALS `mpiexec` 1.8 accepts only `none`, `per_task:N`, `single:N`, `list:...` for `--gpu-bind`; `closest` is Slurm syntax | Always pass an explicit one-tile-per-rank list, e.g. `--gpu-bind=list:0:1:...:11` | `deployments/aurora/launchers/_vasp-step-aurora.sh` (also `_qe-step-aurora.sh`) |
| All ranks `SIGSEGV` before the first SCF step, in an OpenMP worker thread (`setdij` -> `setylm_aug` -> `asa_mp_setylm`), exit 139 (job 9038870) | OpenMP worker-thread stack overflow with the Intel runtime's small default stack | `export OMP_STACKSIZE=512m` and `ulimit -s unlimited` before `mpiexec` | `_vasp-step-aurora.sh` |
| Enters SCF iteration 1 and never returns; at walltime every rank is idle inside MPI/libfabric (job 9039042) | Davidson (`ALGO=Normal`) hangs when a k-point's bands are distributed over several ranks. Not GPU-aware MPI: it also hangs with `MPICH_GPU_SUPPORT_ENABLED=0` | `KPAR` = number of MPI ranks (12 per node), so each rank owns whole k-points | Runtime INCAR written by `run-nbtao5-case-aurora.sh`; set it in any other Aurora INCAR that uses Davidson |

Diagnostic evidence (debug jobs 9039247 and 9039368, same structure and
POTCAR, at most 4 SCF steps):

| Ranks | `ALGO` | `KPAR` | Mesh | Result |
|---|---|---|---|---|
| 12 | Normal | 1 | 2x2x2 | hang, 0 SCF steps in 240 s (also with GPU-aware MPI off) |
| 1 | Normal | 1 | 2x2x2 | 4 steps in 28 s |
| 12 | VeryFast | 1 | 2x2x2 | 4 steps in 77 s |
| 12 | Normal | 12 | 2x2x2 | 4 steps in 14 s |
| 12 | Normal | 12 | 4x8x9 | 4 steps in 57 s including setup |

With all three fixes, qualification job 9039526 ran 18 ionic relaxation steps in
about 24 minutes (roughly 80 s per ionic step for this system) with monotonically
decreasing energy, before its soft walltime stop. `KPAR` affects only how work
is distributed; results are not expected to depend on it, but this has not been
compared numerically against a `KPAR=1` run on another platform.

Caveats:

- `KPAR` must divide the rank count, and with `KPAR` equal to the rank count each
  rank needs at least one k-point. Gamma-only or very small meshes with fewer
  k-points than ranks need fewer ranks instead.
- `ALGO=VeryFast` (RMM-DIIS) avoided the hang without `KPAR` but changes the
  electronic minimizer; prefer `KPAR` to keep the protocol's `ALGO`.
- Long qualification relaxations exceed 30 minutes on one node for this system
  size; size walltime caps from the measured per-ionic-step cost above.

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