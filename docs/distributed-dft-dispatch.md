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

The `dft` pool of the config's `resources:` plan
([resource dispatch](resource-dispatch.md)) selects nodes and devices from the
allocation. Its nodes are partitioned into stable, disjoint groups of
`nodes_per_job`. Each group owns a serial queue of DFT jobs. Different groups
run concurrently; a group is not reused until its preceding calculation has
finished. Consequently, the pool's `max_concurrent` is a cap on independent
groups, not permission to oversubscribe a node.

## Resource keys

| Setting | Meaning |
| --- | --- |
| `resources.pools.dft.nodes` / `devices` | Nodes and per-node devices of the DFT pool |
| `resources.pools.dft.nodes_per_job` | Whole nodes assigned to one VASP or QE calculation; must divide the pool's nodes |
| `resources.pools.dft.ranks_per_device` | MPI ranks per device; ranks per node = devices per node x this value |
| `resources.pools.dft.threads_per_rank` | CPU threads per rank; defaults to cores per node / ranks per node |
| `resources.pools.dft.max_concurrent` | Optional cap below the number of job groups |
| `MATSIM_ASSIGNED_NODES` | Host list of one job group, passed to the facility step launcher |
| `MATSIM_DEVICES_PER_NODE` | Devices per node of the job group; Slurm launchers request this many GPUs |

Step launchers keep the positional contract
`<nodes> <ranks_per_node> <threads_per_rank>` after their input arguments. The
device-mask variable of the vendor (`CUDA_VISIBLE_DEVICES`,
`ROCR_VISIBLE_DEVICES`, or `ZE_AFFINITY_MASK`) is also set for each group.
Device counts come from the scheduler when it reports them and otherwise from
the plan's preset, so Aurora uses the `aurora` preset's 12 tiles per node.

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

