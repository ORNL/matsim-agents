# LAMMPS (external classical MD)

LAMMPS (GPL-2.0) is treated as an external executable. matsim-agents never links to it: it writes
an input deck, runs `lmp` as a subprocess (or submits it to Slurm), and parses the log and a final
dump. No potential is built in; `pair_style`, `pair_coeff` and the potential files are explicit and
the files are hashed into the run record.

Two ways to get the executable are supported. On Frontier both are available and checked; on any
other machine build it from source with the common script.

| Option | When to use it | Executable |
|---|---|---|
| A. Site module | A module exists with the packages you need | `module load ...`; `lmp` on `PATH` |
| B. Build from source | No module, a missing package (for example ML-IAP), a newer version, or a portable, reproducible build | `.hpc-build/<machine>/lammps/install/bin/lmp` |

The adapter finds the executable in this order: an explicit path, `$LAMMPS_EXECUTABLE`, then `lmp`,
`lmp_mpi`, `lmp_serial` on `PATH`. `$LAMMPS_POTENTIALS` points at a potential directory (the
LAMMPS `potentials/` directory for both options).

## Status

| Part | State |
|---|---|
| Adapter (`backends/external/lammps/`: availability, inputs, parser, runner) | `tests/test_lammps_adapter.py`, mocked executable only |
| Option A, Frontier `lammps/20240829.1-mpi-omp` (29 Aug 2024 Update 1, GCC 14.2, Cray MPICH 8.1.32; packages KSPACE MANYBODY MOLECULE RIGID) | job 6002098 on one compute node: W EAM single point matched ASE to 2.4e-10 eV/atom; NVE drift reported (see section 4) |
| Option B, `build-lammps-frontier.sh` (`stable_22Jul2025_update6`, commit `9c5ab448`, GCC 14.2, Cray MPICH 8.1.32, CPU MPI + OpenMP) | built and installed on a Frontier login node from a fresh clone; `lmp -h` probe passed; job 6002498 on one compute node gave the same single point and NVE numbers as option A (section 4) |
| Option B on other machines (`deployments/common/setup/build-lammps.sh`) | not run anywhere except through the Frontier wrapper |
| GPU (KOKKOS/HIP or GPU package) and ML-IAP builds | not built; `LAMMPS_CMAKE_ARGS` passes the flags, but no such build has been made or tested |

Neither option has produced a cascade or any physical result. A passing check means the binary
runs and agrees with ASE for one EAM potential.

## 1. Option A: site module (Frontier)

```bash
module load gcc-native/14.2 cray-mpich/8.1.32 lammps/20240829.1-mpi-omp
export LAMMPS_POTENTIALS=$(dirname $(dirname $(which lmp)))/share/lammps/potentials
export XALT_EXECUTABLE_TRACKING=no
```

`module spider lammps` lists the versions; each needs its compiler and MPI loaded first
(`module spider lammps/<version>` shows the combinations). `-gpu-` modules exist but have not been
used here. The CPU module lacks ML-IAP, PYTHON, EXTRA-FIX and EXTRA-PAIR, so it cannot run MLIPs and
has no `fix electron/stopping`.

## 2. Option B: clone and build

Run on a login node (the clone needs internet). No allocation is needed; a build takes a few
minutes with 16 jobs.

### Frontier

```bash
bash deployments/frontier/setup/build-lammps-frontier.sh
```

The wrapper sources `deployments/frontier/setup/lammps-toolchain-frontier.sh` (PrgEnv-gnu,
`gcc-native/14.2`, `cray-mpich/8.1.32`, `cmake/3.31.11`; override with `LAMMPS_GCC_MODULE`,
`LAMMPS_MPICH_MODULE`, `LAMMPS_CMAKE_MODULE`), uses the Cray wrappers `CC`/`cc`, and calls the
common script with `LAMMPS_ROOT=.hpc-build/frontier/lammps`. Everything lands there (Git-ignored):

| Path | Content |
|---|---|
| `src/` | shallow clone of `lammps/lammps` at the pinned tag |
| `build/` | CMake build tree (recreated on every run) |
| `install/bin/lmp` | executable |
| `install/share/lammps/potentials/` | potential files shipped with LAMMPS |
| `activate.sh` | sets `LAMMPS_EXECUTABLE`, `LAMMPS_POTENTIALS`, `PATH` and loads the toolchain |
| `lammps-build-record.txt` | tag, commit, compiler, packages, extra CMake flags, SHA-256 of `lmp` |
| `lmp-help.txt` | output of the `lmp -h` probe (version, packages, styles) |

### Other systems

Load a C++17 compiler, MPI and CMake >= 3.20, then:

```bash
bash deployments/common/setup/build-lammps.sh
source .hpc-build/lammps/activate.sh
```

The script:

1. clones `https://github.com/lammps/lammps.git` at `LAMMPS_TAG` (shallow) and refuses a checkout
   at another tag;
2. configures `cmake/` with `BUILD_MPI`, `BUILD_OMP=on`, `LAMMPS_EXCEPTIONS=on`, a static
   library, and one `PKG_<name>=on` per package;
3. builds, installs to `LAMMPS_PREFIX`, and checks that `bin/lmp` and the potentials exist;
4. writes `activate.sh` and the build record;
5. runs `lmp -h` and fails without the LAMMPS banner.

| Variable | Default |
|---|---|
| `LAMMPS_ROOT` | `.hpc-build/lammps` (Frontier wrapper: `.hpc-build/frontier/lammps`) |
| `LAMMPS_SRC_DIR`, `LAMMPS_PREFIX` | `$LAMMPS_ROOT/src`, `$LAMMPS_ROOT/install` |
| `LAMMPS_TAG` | `stable_22Jul2025_update6` |
| `LAMMPS_PACKAGES` | `MANYBODY EXTRA-PAIR EXTRA-FIX KSPACE MOLECULE RIGID OPENMP` |
| `LAMMPS_MPI` | `on` (`off` builds a serial `lmp` with `g++`) |
| `LAMMPS_CXX`, `LAMMPS_CC` | `mpicxx`/`mpicc` (Frontier: `CC`/`cc`) |
| `LAMMPS_CMAKE` | `cmake` |
| `LAMMPS_CMAKE_ARGS` | empty; extra flags such as `-D PKG_ML-IAP=on -D PKG_ML-SNAP=on` or a KOKKOS backend |
| `LAMMPS_JOBS` | 16 |
| `LAMMPS_PROBE_CMD` | empty; a launcher prefix if an MPI binary cannot start on the login node |

MANYBODY gives EAM, Tersoff and Stillinger-Weber; EXTRA-FIX gives `fix electron/stopping` for
cascades; EXTRA-PAIR adds further pair styles. Adding a package needs a rebuild, and the build
record shows which flags produced the binary.

## 3. Check and run

Check the executable (no input, no dynamics):

```bash
source .hpc-build/frontier/lammps/activate.sh     # or Option A's module load
PYTHONPATH=src python -m matsim_agents.cli external lammps check
```

`READY` means `lmp -h` ran and printed a version; it reports packages, compiler and MPI. It is tool
readiness only.

Run from Python:

```python
from pathlib import Path
from ase.build import bulk
from matsim_agents.backends.external.lammps.availability import check_availability
from matsim_agents.backends.external.lammps.inputs import LammpsTask, potential_from_file, write_inputs
from matsim_agents.backends.external.lammps.runner import run_local

av = check_availability()
pot = potential_from_file("eam/alloy", Path(av.potentials_dir) / "W_zhou.eam.alloy", ["W"])
atoms = bulk("W", "bcc", a=3.165, cubic=True).repeat(4)
write_inputs(atoms, Path("runs/w_sp"), pot, LammpsTask())
outcome = run_local(Path(av.executable), Path("runs/w_sp"), 600, launcher=("srun", "-n", "8"))
```

Tasks are `single_point`, `minimize`, `nvt` and `nve`; a temperature needs an explicit seed. A run
is a process success only with exit code 0, `Total wall time` in the log and the script's end
marker. `runner.submit_slurm` submits the same command through `execution.slurm_jobs`.

## 4. Compute-node check (Frontier)

```bash
# Option A (default)
LAMMPS_OUT=$PWD/runs/lammps-validation sbatch -A <ACCOUNT> -t 00:20:00 -q debug \
    deployments/frontier/jobs/job-lammps-validation-frontier.sh
# Option B
LAMMPS_SOURCE=build LAMMPS_OUT=$PWD/runs/lammps-validation-build sbatch -A <ACCOUNT> \
    -t 00:20:00 -q debug deployments/frontier/jobs/job-lammps-validation-frontier.sh
```

The job runs 56 MPI ranks on one node: a 432-atom W `W_zhou.eam.alloy` single point compared with
ASE's EAM calculator, and a 16000-atom NVE run (2000 steps, 1 fs, 600 K start, seed 11) whose
total-energy drift it reports in `lammps_validation.json`.

| Option | Job | Energy diff (eV/atom) | Max force diff (eV/A) | NVE drift (eV/atom) | NVE wall (s) |
|---|---|---|---|---|---|
| A, module 29 Aug 2024 Update 1 | 6002098 | 2.4e-10 | 5.0e-6 | 9.6e-6 | 3.56 |
| B, build `stable_22Jul2025_update6` | 6002498 | 2.4e-10 | 5.0e-6 | 9.6e-6 | 3.53 |

The self-built binary links `libmpi_gnu_123.so.12` by soname. Without `LD_LIBRARY_PATH` set from
`CRAY_LD_LIBRARY_PATH`, the loader picks the default MPICH from `/opt/cray/pe/lib64` (9.0.1)
instead of the 8.1.32 used for the build. `lammps-toolchain-frontier.sh` sets it, and
`activate.sh` sources that file. An earlier job (6002469) ran before this fix with MPICH 9.0.1 and
gave the same numbers.

This tests the binary and the adapter for one EAM potential. It is not a cascade and says nothing
about the potential's physics.

## Limits

- No MLIP runs inside LAMMPS: that needs ML-IAP with Python coupling (or a model-specific pair
  style) built against the same Python and Torch as the MLIP. Not attempted.
- Only CPU builds exist. GPU builds (KOKKOS with HIP on Frontier) need their own toolchain and are
  not covered by the scripts' defaults.
- The common script is validated only through the Frontier wrapper; other machines need their
  compiler, MPI and CMake loaded first and are untested.
