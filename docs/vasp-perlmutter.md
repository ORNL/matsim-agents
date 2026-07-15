# VASP on Perlmutter (NVIDIA A100, sm_80)

This page documents how VASP 6.6.0 (OpenACC GPU port) is built, launched, and
driven from the matsim-agents workflows on NERSC Perlmutter, and records every
runtime problem that had to be solved to get GPU VASP running **both** as a
standalone relaxation and as the DFT-labelling step inside the active-learning
(AL) loop.

If VASP "runs" but the AL loop produces **0 labelled frames**, or a job
finishes suspiciously fast, jump straight to [Troubleshooting](#troubleshooting)
— every failure mode observed in practice is catalogued there with its symptom,
root cause, and fix.

## Repository policy

Same as Aurora (see [`docs/vasp-aurora.md`](vasp-aurora.md)): the VASP source
tree and vendor tarball under `external/vasp6/` are gitignored. Only build
notes, launchers, and non-proprietary configuration summaries are committed.

## Build

- VASP version: `6.6.0`, OpenACC GPU offload port.
- Toolchain: **NVHPC 25.5** (`nvfortran`/`nvc`/`nvc++`, bundled **CUDA 12.9**),
  `PrgEnv-gnu` + `cpe/24.07` + `cray-mpich/8.1.30` + `cray-libsci` (BLAS /
  LAPACK / scaLAPACK) + `cray-fftw`, plus netlib **scaLAPACK** (required for
  multi-node parallel diagonalization).
- GPU arch: A100 = `cc80`.
- Build script:
  [`scripts/setup/perlmutter/build-vasp-gpu-perlmutter.sh`](../scripts/setup/perlmutter/build-vasp-gpu-perlmutter.sh)
- Module helper (single source of truth for the toolchain):
  [`scripts/setup/perlmutter/perlmutter-module-stack.sh`](../scripts/setup/perlmutter/perlmutter-module-stack.sh)
  → `load_perlmutter_modules_nvidia`.
- Resulting binary: `external/vasp6/src/vasp.6.6.0/bin/vasp_std`.

> **Why not `PrgEnv-nvhpc`?** `PrgEnv-nvhpc/8.5.0` only pairs with `cpe/25.09`
> (`cray-mpich/9.0.1`), which breaks ABI alignment with the matsim/fairchem
> Python venvs. The build and launchers deliberately pin `cpe/24.07` /
> `cray-mpich/8.1.30`.

```bash
bash scripts/setup/perlmutter/build-vasp-gpu-perlmutter.sh
# useful overrides: CLEAN_BUILD=1  VASP_TARGET=std  GPU_ARCH=cc80  CUDA_VER=12.9
```

## The two launchers

There are **two** VASP GPU launchers on Perlmutter. Keep their runtime
environment (module stack, `LD_LIBRARY_PATH`, and the CUDA-aware-MPI / NCCL env)
**in sync** — historically a fix applied to one was missing from the other.

| Launcher | Used by | Invocation |
|----------|---------|------------|
| [`scripts/launchers/perlmutter/run-vasp-gpu-perlmutter.sh`](../scripts/launchers/perlmutter/run-vasp-gpu-perlmutter.sh) | `matsim_agents.tools.vasp_relax` and the VASP warm-start benchmark, via `MATSIM_VASP_LAUNCHER` | run *inside* a work dir already holding `INCAR/POSCAR/KPOINTS/POTCAR`; **no argv** |
| [`scripts/launchers/perlmutter/_vasp-step-perlmutter.sh`](../scripts/launchers/perlmutter/_vasp-step-perlmutter.sh) | the AL loop's DFT step (`dft.vasp.vasp_wrapper` in the AL YAML), called from `active_learning/backends/vasp.py` | `bash _vasp-step-perlmutter.sh <work_dir> <vasp_bin> <nodes> <ranks_per_node> <threads_per_rank>` |

Both do a full `module reset` to the VASP build toolchain, unset venv-injected
variables that perturb the launch (`LD_PRELOAD`, `VLLM_CUDART_SO_PATH`,
`PYTHONPATH`), and issue an `srun` step with `--gpus-per-node=4
--gpu-bind=closest` (4 MPI ranks = one per A100, 16 OMP threads/rank).

## Runtime environment requirements

These are set by both launchers and are **mandatory** for the GPU build to even
start:

1. **NVHPC runtime libs on `LD_LIBRARY_PATH`.** The binary links
   `libqdmod.so.0` / `libqd.so.0` (NVHPC quad-double lib). Without them VASP
   dies at startup with `error while loading shared libraries: libqdmod.so.0`.
   Fix: prepend the NVHPC `cuda/12.9/lib64`, `math_libs`, `compilers/lib`, and
   crucially `compilers/extras/qd/lib` to `LD_LIBRARY_PATH`, and set
   `CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`.
2. **CUDA-aware MPI.** VASP 6.6.0 was built with GPU-aware MPI and refuses to
   run without it: `export MPICH_GPU_SUPPORT_ENABLED=1`.
3. **NCCL transport override** (see Troubleshooting #1):
   `export NCCL_P2P_DISABLE=1` and `export NCCL_SHM_DISABLE=1`.

## Troubleshooting

### 1. `NCCL WARN Cuda failure 101 'invalid device ordinal'` / `CUDA_ERROR_ILLEGAL_ADDRESS`

**Symptom** — every VASP single-point crashes; `vasp.out` shows:

```
NCCL WARN Cuda failure 101 'invalid device ordinal'
Accelerator Fatal Error: cuStreamSynchronize returned error 700
(CUDA_ERROR_ILLEGAL_ADDRESS) ... wave_mpi.f90  redis_pw
```

In the AL loop this manifests as `VASP converged=0 failed=N` and **0 labelled
frames every iteration**, even though the job "completes".

**Root cause** — on single-node Perlmutter GPU jobs, NCCL's intra-node P2P/SHM
device enumeration conflicts with the Slurm GPU cgroup isolation (each rank sees
a remapped device ordinal), so NCCL addresses a device that isn't in its
cgroup.

**Fix** — force NCCL onto socket (loopback TCP) transport:

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
```

Both launchers now set these. This was the single most impactful fix — the
AL launcher `_vasp-step-perlmutter.sh` was missing it while the warm-start
launcher already had it, which is why warm-start VASP worked but the AL loop
silently produced no data.

### 2. Concurrent multi-node DFT: `srun: Job step creation temporarily disabled`

**Symptom** — with more than one DFT job dispatched at once, the 2nd+ `srun`
step blocks/loops on `Job step creation temporarily disabled, retrying`.

**Root cause** — the AL driver launches up to
`SLURM_JOB_NUM_NODES / nodes_per_job` VASP steps concurrently from a
`ThreadPoolExecutor`. Without `--exclusive`, each `srun` step defaults to
requesting the allocation's *full* resource set, so only one can run at a time.

**Fix** — the inner `srun` in `_vasp-step-perlmutter.sh` uses `--exclusive`, so
each `-N1` step claims one whole, distinct node and they run in parallel. This
is a no-op for single-node allocations.

### 3. `srun: error: Only allocated 1 nodes asked for 2`

**Symptom** — every DFT step of a case fails instantly with this message;
`vasp.out` never gets past the `[vasp-step] srun ... -N 2 ...` banner.

**Root cause** — the AL config's `dft.vasp.nodes_per_job` exceeds the job's
node count. **`al_zn_formate*.yaml` uses `nodes_per_job: 2`** (dense framework
cell, ~20+ atoms); all other paper cases use `nodes_per_job: 1`.

**Fix** — submit with `-N >= nodes_per_job`. For `zn_formate` use `-N 2`
(serial) or `-N 4` (2-way concurrent). See
[Running the AL loop](#running-the-al-loop).

### 4. Job "completes" in ~1–2 minutes doing nothing

**Symptom** — a resubmitted AL job finishes almost instantly; the log shows
`Resuming AL loop at iteration N` then `Active-learning loop complete` one
second later, and no new `iteration_*` dirs appear.

**Root cause** — `loop.resume: true` counts existing `iteration_*` directories
under `runs/${RUN_TAG}` and resumes past them. If a *previous* run recorded
those iterations while VASP was crashing (0 real frames), the resume just skips
all of them → the poisoned state makes the rerun a no-op.

**Fix** — after repairing the DFT problem, archive the stale run-state dir so
the rerun starts fresh at iteration 0 (rename, don't delete — it's reversible):

```bash
mv runs/<RUN_TAG> runs/<RUN_TAG>.broken-$(date +%Y%m%d-%H%M)
```

RUN_TAG per case (`vars.RUN_TAG` in each `examples/paper_cases/al_*.yaml`):

| Case | RUN_TAG |
|------|---------|
| lifepo4 | `lifepo4-al-001` |
| phosphorene | `phosphorene-2d-al-001` |
| zn_formate (uma) | `zn-formate-mof-uma-al-001` |
| hea_fcc | `cantor-fcc-al-001` |
| hea_bcc | `hea-bcc-al-001` |

> `zn_formate` defaults to `al_zn_formate_uma.yaml` when `MLIP_BACKEND=uma`
> (the job script overrides the CASE→YAML map), so its RUN_TAG is the `-uma-`
> one. A *legitimate* resume (prior iterations hold real energies) needs **no**
> archiving — only archive when the prior state is known-bad.

### 5. UMA weights must be prefetched (AL loop prerequisite)

The AL loop that drives VASP uses the UMA MLFF, and compute nodes have no
internet. `fairchem-core` ignores `HF_HOME` and uses `FAIRCHEM_CACHE_DIR`
(default `~/.cache/fairchem` on CFS, which cannot `fcntl.flock` over DVS →
`OSError [Errno 524]`). Prefetch once and point the cache at `$SCRATCH`:

```bash
sbatch scripts/download/perlmutter/download-uma-perlmutter.sh
# the job scripts export FAIRCHEM_CACHE_DIR=$SCRATCH/matsim-agents/fairchem_cache
```

See [`docs/model-download.md`](model-download.md) ("UMA MLIP weights on
Perlmutter").

## Running the AL loop

Submit a paper case with
[`scripts/advanced/perlmutter/job-active-learning-paper-cases-perlmutter.sh`](../scripts/advanced/perlmutter/job-active-learning-paper-cases-perlmutter.sh)
(`CASE=<name>`). The script defaults to `-N 1` (serial DFT).

**DFT concurrency = `SLURM_JOB_NUM_NODES / nodes_per_job`.** Request more nodes
to parallelize the VASP single-points:

```bash
# nodes_per_job=1 cases (lifepo4, hea_bcc, hea_fcc, phosphorene):
#   -N 4  -> 4 VASP jobs in parallel
CASE=hea_bcc sbatch -q premium -N 4 -t 12:00:00 \
  scripts/advanced/perlmutter/job-active-learning-paper-cases-perlmutter.sh

# zn_formate has nodes_per_job=2 -> MUST use -N >= 2 (use -N 4 for 2-way):
CASE=zn_formate sbatch -q premium -N 4 -t 12:00:00 \
  scripts/advanced/perlmutter/job-active-learning-paper-cases-perlmutter.sh
```

The premium QOS (`sbatch -q premium`) allows `MaxWall 2-00:00:00` but
`MaxSubmitJobsPerUser 5`. Long cases (large `acquisition.n_select`, e.g.
`hea_bcc` n_select=200) rely on `loop.resume: true` to accumulate iterations
across multiple wall-clock chunks; a timeout is not a failure as long as the
completed iterations hold real energies.

## Verifying a VASP single-point actually succeeded

The AL DFT step is a **single-point SCF** (`NSW=0`). It does **not** print
`reached required accuracy` — that phrase only appears for *ionic relaxation*,
so grepping for it gives false "0 converged" results. Use these signals in the
work dir instead:

| File | Success marker |
|------|----------------|
| `OUTCAR` | `free  energy   TOTEN  = ... eV` and `Total CPU time used` |
| `OSZICAR` | final `F= ... E0= ...` line |
| `vasprun.xml` | contains `</modeling>` |

Quick audit of a run:

```bash
# real converged energies produced?
grep -rl "free  energy   TOTEN" runs/<RUN_TAG>/iteration_*/dft/*/OUTCAR | wc -l
# any NCCL/CUDA crashes left? (should be 0)
grep -rlE "NCCL WARN|invalid device ordinal|ILLEGAL_ADDRESS" \
  runs/<RUN_TAG>/iteration_*/dft/*/vasp.out | wc -l
```

## Related repository docs

- Aurora VASP build/run: [`docs/vasp-aurora.md`](vasp-aurora.md)
- Perlmutter setup overview: [`scripts/setup/perlmutter/README.md`](../scripts/setup/perlmutter/README.md)
- Paper AL cases (submission guide): [`examples/paper_cases/README.md`](../examples/paper_cases/README.md)
- UMA weight prefetch: [`docs/model-download.md`](model-download.md)
- HPC index: [`docs/hpc-platforms.md`](hpc-platforms.md)
