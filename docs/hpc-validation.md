# HPC validation matrix

Machine support is a tested toolchain claim, not a synonym for having a job
script. Record a row here only after running the checks on the named machine.

| Site | Scheduler | Accelerator | Python/ML stack | DFT stack | Status |
|---|---|---|---|---|---|
| OLCF Frontier | Slurm | MI250X (`gfx90a`, 8 GCD/node) | ROCm 7.2 recipe | QE ROCm 6.2.4 / VASP ROCm 6.2 | recipe present; on-system revalidation required |
| ALCF Aurora | PBS Pro | 6 PVC GPUs / 12 tiles | `frameworks` + Intel PyTorch recipe | oneAPI QE/VASP recipe | recipe present; on-system revalidation required |
| NERSC Perlmutter | Slurm | 4 A100 (`sm_80`) | CUDA 12.9 recipe | NVHPC 25.5 QE/VASP recipe | recipe present; on-system revalidation required |

Module collections change over time. Every setup script must check its required
module or binary and exit instead of silently selecting another user's install.
Updating a version pin requires a new smoke record.

## Required smoke record

Save the following with the job output:

1. `module -t list`, Python version, package lock/freeze, and git commit.
2. `torch.__version__`, compiled accelerator version, and device enumeration.
3. Per-rank hostname, local rank, CPU affinity, and visible accelerator ID.
4. One MLIP energy/force evaluation and a short geometry relaxation.
5. One small QE or VASP calculation using a separate module-clean launcher.
6. For vLLM, server health, model identity, tensor-parallel size, and a short request.
7. For active learning, restart a run and show that completed labels are not duplicated.

The tests under `deployments/<site>/smoke-tests/` are the starting point. Logs
may contain site paths or allocation IDs; reusable scripts may not.

## Resource and data separation

The LangGraph controller is lightweight. Run large vLLM servers in a separate
allocation and address them over HTTP. Request one accelerator for a
single-process MLIP smoke test; request a full node only when the numerical
backend explicitly starts one worker per accelerator.

VASP and QE totals use different energy references. The dataset writer rejects
a backend change for an existing dataset. A justified reference transformation
must write a new dataset and preserve the original backend metadata.
