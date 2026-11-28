# Quantum ESPRESSO on Frontier (AMD MI250X, gfx90a)

This page documents how matsim-agents builds and runs Quantum ESPRESSO with
AMD GPU offload on OLCF Frontier. Source code, build tree, and install
prefix live under `external/quantum-espresso/` (gitignored). The recipe
itself is committed under `deployments/frontier/setup/` and
`deployments/frontier/launchers/`.

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
| `PrgEnv-cray` | 8.7.0 (site default) | only PrgEnv with working OpenMP target offload to gfx90a |
| `cpe` / `cce` | 26.03 / 21.0.4 (site default) | the build and smoke jobs below ran on this stack |
| `craype-accel-amd-gfx90a` | site default | enables `-target-accel=amd_gfx90a -h omp` in `ftn` |
| `rocm` | 7.2.0 | `pw.x` and `neb.x` link `libamdhip64.so.7`; override with `ROCM_MODULE` |
| `cray-mpich` | 9.1.0 (from cpe/26.03) | |
| `cray-fftw` | site default | CPU FFTW3 (offload kernels still call CPU FFTW for non-annotated paths) |
| `cmake` | 3.31.11 | QE develop CMake API; override with `CMAKE_MODULE` / `GIT_MODULE` |

## Build

```bash
# From the matsim-agents repo root, on a Frontier login node:
nohup bash deployments/frontier/setup/build-qe-gpu-frontier.sh \
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

4. **cce uninitialized-descriptor segfault in `pw_init_qexsd_input`** (found on
   cce/18.0.1; the patch was carried into the cce/21.0.4 build, where the
   `ALLOCATE`s are guarded with `IF (.NOT. ALLOCATED(...))` because `neb.x`
   calls `iosys` once per image and an unconditional `ALLOCATE` aborts with
   "already allocated").

   On cce/18.0.1, `pw.x` segfaults deterministically on every run: `hybrid_type`/`dftU_type`/
   `vdW_type` (`Modules/qes_types_module.f90`) carry several `ALLOCATABLE`
   array components, and cce/18.0.1 does not reliably initialize the
   allocation-status descriptor of those components when the containing type
   is a plain automatic local variable. The fix (developed upstream on
   [`fix/frontier-cce-uninitialized-descriptor-segfault`](https://gitlab.com/max.lupopasini/q-e/-/tree/fix/frontier-cce-uninitialized-descriptor-segfault))
   has two parts:
   - `SAVE` + explicit reset/allocate of the affected locals in
     `PW/src/pw_init_qexsd_input.f90` (and matching hardening in
     `PW/src/pw_restart_new.f90`), so cce always sees a validly-initialized
     descriptor.
   - a descriptor-safe manual deep copy (`qes_safe_copy_dftU`/
     `qes_safe_copy_vdW` in `Modules/qes_init_module.f90`) that replaces
     whole-structure derived-type assignment (`obj%dftU = dftU`) with
     explicit `ALLOCATE` + element-by-element field copies for every
     Hubbard/vdW array component (including nested allocatable
     sub-components), so DFT+U/vdW-correction data is actually preserved in
     the XML restart file rather than silently dropped.

   The script applies this as a single unified diff,
   [`deployments/frontier/setup/patches/qe-cce-segfault-fix/cce-uninitialized-descriptor-segfault.patch`](../deployments/frontier/setup/patches/qe-cce-segfault-fix/cce-uninitialized-descriptor-segfault.patch),
   right after the `pimd_subrout.f90` patch above, guarded by a
   `grep -q 'Frontier/cce-18.0.1 workaround'` marker check so re-running the
   build script against an already-patched source tree is a no-op. Verified
   with a bulk BCC-Fe DFT+U SCF run: `pw.x` converges (`JOB DONE`) and the
   written `data-file-schema.xml` contains real `Hubbard_Occ`/`Hubbard_ns`
   occupation data (cce/18.0.1 build).

   On the cce/21.0.4 + rocm/7.2.0 build, verified by smoke jobs only: the
   `pw.x` smoke converged (job 5998158), and the `neb.x` smoke (job 5998160,
   `nimage=1`, 4 images, tiny Li cell, low cutoffs) ran to "convergence
   achieved in 3 iterations" and its output was ingested. That checks launch
   and ingest only, not a physical or numerical validation. `nimage` must not
   exceed the number of path images (`NEB_SMOKE_NIMAGE`, default 1).

## Run

```bash
# Submit a 30-min batch job on 1 node, 8 GCDs:
sbatch deployments/frontier/launchers/run-pw-gpu-frontier.sh path/to/your.in

# Or, inside a salloc'd compute node:
bash deployments/frontier/launchers/run-pw-gpu-frontier.sh path/to/your.in
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

## Pseudopotential libraries

The two libraries live side by side under
`/lustre/orion/lrn110/proj-shared/cache/pseudopotentials/`. Each directory holds a
`SHA256SUMS` file. Check a directory with `sha256sum -c SHA256SUMS` before you use it.

| Directory | Role | Types |
|---|---|---|
| `pslibrary-1.0.0-pbe-paw` | **Production protocol** (`deployments/frontier/campaigns/qe-1024`) | PAW |
| `sssp-1.3.0-pbe-precision` (read-only) | **Qualification candidate only** | GBRV US (W Ta Cr V Re Os), SG15 NC (Mo H He), Dojo NC (Hf) |

### SSSP candidate

- **Source and checks.** SSSP 1.3.0 PBE Precision comes from
  <https://archive.materialscloud.org/records/gvag2-y0636> (DOI 10.24435/materialscloud:eg-28).
  - The archive and JSON MD5s match the record.
  - Each UPF MD5 matches the official SSSP JSON.
  - The original archive is kept in `_source/`.
  - `SOURCE.txt` records the provenance.
- **Manifest.** The per-element SHA-256, UPF header data and SSSP cutoffs are in
  [`deployments/common/pseudopotentials/sssp-1.3.0-pbe-precision.yaml`](../deployments/common/pseudopotentials/sssp-1.3.0-pbe-precision.yaml).
  It is generated by `scripts/utils/build_pseudo_manifest.py`, which exits non-zero on any
  MD5, element or functional mismatch.
- **Header data.** All files are PBE and scalar-relativistic with no spin-orbit.
  - The GBRV files are UPF v1. The relativistic treatment comes from their `PP_INFO` text.
    QE's `upfconv.x` writes `relativistic="no"` for them anyway.
  - The SG15 files (Mo, H, He) have no nonlinear core correction. Magnetic Mo environments
    need an explicit check.
- **Cutoffs.** The SSSP minimum over the 10 elements is 80 Ry wfc (set by H) and 400 Ry rho
  (set by W and Ta). The candidate config uses 80/640. These are SSSP recommendations, not
  convergence tests for this alloy.
- **QE compatibility (reader level only).** `upfconv.x` from `install-gpu` reads all 10
  files. No pw.x calculation has been run.
- **Config.** `examples/fusion_bcc_1024/campaign-sssp-candidate.yaml`.
  - It uses its own `MATSIM_SSSP_PSEUDO_DIR` and campaign id.
  - It has an explicit file map, because prefix auto-resolution is ambiguous in the full SSSP
    directory (for example, `H` vs `Hf-sp...`).
  - It keeps the same 1,024-atom parent cell.

**Keeping labels apart.** `dft_protocol.scientific_dft_payload` includes the SHA-256 of each
UPF. PSlibrary and SSSP labels and elemental references therefore get different
`dft_method_signature` values. Never pool them.

**Qualification plan.** This waits until the pw.x `wfcinit` segfault is fixed. Do not switch
pseudopotentials to work around that crash.
1. Run PSlibrary and SSSP on the same small set of structures.
2. Compare energy differences, forces, magnetic states, defect energies and cost.
3. Recommend a production set only after that comparison. "Precision" in the name is not a
   reason to prefer SSSP.
4. If SSSP is adopted, create a new versioned protocol. Update `campaign.yaml`, the QE
   configs, the elemental references and the fine-tuning metadata together.

## Numerical protocol and convergence runs

- **Cutoff check.** `backends/dft/pseudo_validation.py` runs inside `write_pw_input` and
  `write_neb_input`. It rejects a cutoff below the UPF-recommended minimum, a functional
  mismatch with the manifest and a missing or mismatched file. A deliberate low value needs
  `cutoff_override_reason` and is then diagnostic only.
- **Campaign values.** `examples/fusion_bcc_1024/campaign.yaml` uses the provisional
  recommendation from the 54-site scans: 100/1200 Ry, 3x3x3 (k-spacing 0.10), MV 0.010 Ry,
  `conv_thr` 1e-7. That clears the PSlibrary maxima (wfc 53.8 Ry for Os, rho 645 Ry for V) but
  is provisional, not a qualified protocol; see `deployments/common/qe-protocols.yaml`.
- **Convergence runs.** They run on OLCF Frontier. `examples/qe_convergence/frontier_point_defects.yaml`
  drives `workflows/numerical_convergence.py` through `deployments/frontier/launchers/run-pw-gpu-frontier.sh`,
  one node and one GCD per run (the `execution/slurm_jobs` limit). Cells come from
  `workflows/qualification_cells.py`. A run counts only with a `JOB DONE` line and a
  converged SCF.
- **Status.** The 64-run matrix and two 16-run follow-ups (dense k-meshes, higher cutoffs) have run.
  Only `conv_thr` (1e-7 Ry) qualified. The provisional values (7x7x7 on the 54-site cell, 100/1200 Ry,
  MV 0.010 Ry) are in [dft-parameterization.md](../examples/fusion_bcc_1024/dft-parameterization.md).
  All seven protocols in `deployments/common/qe-protocols.yaml` are `unqualified`. Larger cells need
  a multi-node sbatch like `deployments/frontier/campaigns/qe-1024/qe_kconv_job.sbatch`.
