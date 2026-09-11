# Distributed DFT dispatch

The DFT dispatcher operates inside an allocation that the user has already
obtained. It does not submit nested scheduler jobs. The same planning contract
is used on Frontier, Aurora, and Perlmutter.

## Allocation discovery

- Frontier and Perlmutter use Slurm variables and expand
  `SLURM_JOB_NODELIST` with `scontrol`.
- Aurora reads `PBS_NODEFILE`, removes repeated CPU-slot entries while
  preserving node order, and uses PBS/PALS launchers.
- A non-scheduler process receives a one-node local allocation for testing.

The allocation is partitioned into stable, disjoint groups of
`nodes_per_job`. Each group owns a serial queue of DFT jobs. Different groups
run concurrently; a group is not reused until its preceding calculation has
finished. Consequently, `max_concurrent_jobs` is a cap on independent groups,
not permission to oversubscribe a node.

## Resource keys

| Setting | Meaning |
| --- | --- |
| `nodes_per_job` | Whole nodes assigned to one VASP or QE calculation |
| `ranks_per_node` | Accelerator-using MPI ranks launched on each assigned node |
| `threads_per_rank` | CPU threads reserved for each rank |
| `dft.max_concurrent_jobs` | Optional cap below the number of complete node groups |
| `MATSIM_GPUS_PER_NODE` | Explicit device count when the scheduler environment does not report it |
| `MATSIM_DFT_ASSIGNED_NODES` | Host list passed by the planner to a facility step launcher |

When the GPU count is known, `ranks_per_node` must equal it. This fail-fast
rule prevents an apparently successful benchmark from using only part of a
node. Aurora jobs normally set `MATSIM_GPUS_PER_NODE=12` because the supported
configuration treats the two tiles on each of six PVC GPUs as devices.

## Facility launchers

| Facility | Scheduler | Step mechanism |
| --- | --- | --- |
| Frontier | Slurm | `srun --exclusive` over the assigned MI250X nodes |
| Perlmutter | Slurm | `srun --exclusive` over the assigned A100 nodes |
| Aurora | PBS Pro/PALS | `mpiexec --hosts ... --ppn ...` with CPU/GPU binding |

The launcher is a deployment adapter. It may select modules, binding, and a
binary path, but it must not modify k-points, pseudopotentials, convergence
thresholds, or other scientific settings.

## Operational requirements

- Supply allocation IDs at `sbatch` or `qsub` time; never commit them.
- Keep licensed VASP binaries and POTCARs outside the repository.
- Record the exact QE pseudopotential set or VASP POTCAR family.
- Do not mix VASP and QE total energies in one dataset without a documented
  reference transformation.
- A timeout, nonzero return code, unconverged calculation, or malformed result
  must produce a persisted failure record.

The shared validation target is described in
[HPC validation](hpc-validation.md) and
[the portability benchmark](../benchmarks/portability/README.md).

