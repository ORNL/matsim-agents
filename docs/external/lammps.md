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
| B. Build from source | No module, a missing package (for example ML-IAP), a newer version, or a portable, reproducible build | `external/lammps/install/bin/lmp` |

The adapter finds the executable in this order: an explicit path, `$LAMMPS_EXECUTABLE`, then `lmp`,
`lmp_mpi`, `lmp_serial` on `PATH`. `$LAMMPS_POTENTIALS` points at a potential directory (the
LAMMPS `potentials/` directory for both options).

## Status

| Part | State |
|---|---|
| Adapter (`backends/external/lammps/`: availability, inputs, parser, runner) | `tests/test_lammps_adapter.py`, mocked executable only |
| Option A, Frontier `lammps/20240829.1-mpi-omp` (29 Aug 2024 Update 1, GCC 14.2, Cray MPICH 8.1.32; packages KSPACE MANYBODY MOLECULE RIGID) | job 6002098 on one compute node: W EAM single point matched ASE to 2.4e-10 eV/atom; NVE drift reported (see section 4) |
| Option B, `build-lammps-frontier.sh` (`stable_22Jul2025_update6`, commit `9c5ab448`, GCC 14.2, Cray MPICH 8.1.32, CPU MPI + OpenMP) | built and installed on a Frontier login node from a fresh clone; `lmp -h` probe passed; job 6002498 on one compute node gave the same single point and NVE numbers as option A (section 4) |
| Option B on other machines (`deployments/common/setup/lammps/build-lammps.sh`) | not run anywhere except through the Frontier wrapper |
| ML-IAP + PYTHON, `setup/lammps/build-lammps-mliap-frontier.sh` (same tag and commit, embedded `.venv` Python 3.11.18, CPU) | built on a Frontier login node into `external/lammps/install-mliap`; the upstream unified-LJ example reproduced its reference thermo at steps 0, 50 and 100. matsim-agents MLIPs run through it with the bridge in section 5 (login-node parity: EMT, MACE-MP-0 small, HydraGNN BEST6, UMA uma-s-1p1; compute-node job not yet run) |
| GPU (KOKKOS/HIP or GPU package) builds | not built; `LAMMPS_KOKKOS=hip` and `LAMMPS_CMAKE_ARGS` pass the flags, but no such build has been made or tested |

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
bash deployments/frontier/setup/lammps/build-lammps-frontier.sh
```

The wrapper sources `deployments/frontier/setup/lammps/lammps-toolchain-frontier.sh` (PrgEnv-gnu,
`gcc-native/14.2`, `cray-mpich/8.1.32`, `cmake/3.31.11`; override with `LAMMPS_GCC_MODULE`,
`LAMMPS_MPICH_MODULE`, `LAMMPS_CMAKE_MODULE`; the resolved names are written into
`activate.sh`, so a job loads the modules the binary was linked against), uses the Cray wrappers `CC`/`cc`, and calls the
common script. Everything lands in `external/lammps/` (Git-ignored), like the other external
codes (`external/quantum-espresso`, `external/SPECTRA-PKA`):

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
bash deployments/common/setup/lammps/build-lammps.sh
source external/lammps/activate.sh
```

The script:

1. clones `https://github.com/lammps/lammps.git` at `LAMMPS_TAG` (shallow), refuses a checkout
   at another tag, and checks that the tag still resolves to the validated commit (`LAMMPS_COMMIT`);
2. configures `cmake/` with `BUILD_MPI`, `BUILD_OMP=on`, `LAMMPS_EXCEPTIONS=on`, a static
   library, and one `PKG_<name>=on` per package;
3. builds, installs to `LAMMPS_PREFIX`, and checks that `bin/lmp` and the potentials exist;
4. writes `activate.sh` and the build record;
5. runs `lmp -h` and fails without the LAMMPS banner.

| Variable | Default |
|---|---|
| `LAMMPS_ROOT` | `external/lammps` |
| `LAMMPS_SRC_DIR`, `LAMMPS_PREFIX` | `$LAMMPS_ROOT/src`, `$LAMMPS_ROOT/install` |
| `LAMMPS_TAG` | `stable_22Jul2025_update6` |
| `LAMMPS_COMMIT` | `9c5ab448c78a14fd534619622162ba418d6a1fb1` for the default tag; empty (no check) for other tags |
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
source external/lammps/activate.sh     # or Option A's module load
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

Tests:

- `tests/test_lammps_adapter.py`: input writing, parsing, runner and availability with a mocked
  executable; runs everywhere, including CI.
- `tests/integration/test_lammps_eam.py`: a real 54-atom W EAM single point compared with ASE and a
  500-step NVE drift check. It skips unless LAMMPS is `READY` and `$LAMMPS_POTENTIALS` has
  `W_zhou.eam.alloy`; `LAMMPS_TEST_LAUNCHER` (e.g. `srun -n 8`) sets the launcher.
  `job-integration-tests-frontier.sh` runs it against `external/lammps` when that build exists.

## 4. Compute-node check (Frontier)

```bash
# Option A (default)
LAMMPS_OUT=$PWD/runs/lammps-validation sbatch -A <ACCOUNT> -t 00:20:00 -q debug \
    deployments/frontier/jobs/lammps/job-lammps-validation-frontier.sh
# Option B
LAMMPS_SOURCE=build LAMMPS_OUT=$PWD/runs/lammps-validation-build sbatch -A <ACCOUNT> \
    -t 00:20:00 -q debug deployments/frontier/jobs/lammps/job-lammps-validation-frontier.sh
```

The job runs 56 MPI ranks on one node: a 432-atom W `W_zhou.eam.alloy` single point compared with
ASE's EAM calculator, and a 16000-atom NVE run (2000 steps, 1 fs, 600 K start, seed 11) whose
total-energy drift it reports in `lammps_validation.json`.

| Option | Job | Energy diff (eV/atom) | Max force diff (eV/A) | NVE drift (eV/atom) | NVE wall (s) |
|---|---|---|---|---|---|
| A, module 29 Aug 2024 Update 1 | 6002098 | 2.4e-10 | 5.0e-6 | 9.6e-6 | 3.56 |
| B, build `stable_22Jul2025_update6` | 6002498 | 2.4e-10 | 5.0e-6 | 9.6e-6 | 3.53 |
| B, same tag, rebuilt from a fresh clone in `external/lammps` | 6002722 | 2.4e-10 | 5.0e-6 | 9.6e-6 | 3.92 |

The self-built binary links `libmpi_gnu_123.so.12` by soname. Without `LD_LIBRARY_PATH` set from
`CRAY_LD_LIBRARY_PATH`, the loader picks the default MPICH from `/opt/cray/pe/lib64` (9.0.1)
instead of the 8.1.32 used for the build. `lammps-toolchain-frontier.sh` sets it, and
`activate.sh` sources that file. An earlier job (6002469) ran before this fix with MPICH 9.0.1 and
gave the same numbers.

This tests the binary and the adapter for one EAM potential. It is not a cascade and says nothing
about the potential's physics.

## 5. MLIPs inside LAMMPS (ML-IAP bridge)

`backends/external/lammps/mliap/bridge.py` (`CalculatorMLIAP`) is an ML-IAP unified object that
hands the local atoms to any matsim-agents ASE calculator (MACE, UMA, HydraGNN, or a
`SumCalculator` with additive ZBL) each step and adds the returned forces and energy. LAMMPS owns
the dynamics and the calculator owns the physics, so ZBL goes through the same
`build_short_range_calculator` path as in ASE. `parity.py` compares one ASE evaluation against one
LAMMPS evaluation and `write_export_record` stores the checkpoint sha256, species and short-range
choice next to the result. `UncertaintyGate` scores force disagreement across an ensemble every
`every` force calls; `run_with_gate` stops the run when it fires so the caller can fine-tune,
re-export and rerun the parity check (the fine-tune and re-export steps are not automated here).

```bash
source external/lammps/activate-mliap.sh
PYTHONPATH=$PWD/src .venv-mace/bin/python scripts/lammps/mliap_parity.py \
    --calc mace --model small --a 2.3 --rattle 0.02 --zbl --out outputs/mliap-parity/mace.json
```

Login-node results (108-atom rattled Cu, CPU, tolerance 1e-4 eV and 1e-4 eV/A):

| Calculator | Case | Energy diff (eV) | Max force diff (eV/A) |
|---|---|---|---|
| EMT | a=3.6 | 1.2e-14 | 3.9e-14 |
| EMT + ZBL | a=2.3 (ZBL active) | 2.7e-12 | 8.0e-13 |
| EMT, then EMT + ZBL | a=2.0 (3659 eV, then 5407 eV) | 0.0, then 9.1e-13 | 2.3e-12, then 4.5e-12 |
| MACE-MP-0 small | a=3.6 | 0.0 | 1.7e-14 |
| MACE-MP-0 small + ZBL | a=2.3 (ZBL active) | 3.5e-10 | 9.2e-11 |
| HydraGNN BEST6 (fp64, frozen source `81d5b883`) | a=3.6 | 2.8e-14 | 3.6e-15 |
| HydraGNN BEST6 + ZBL | a=2.3 (ZBL active) | 1.8e-12 | 4.2e-12 |
| UMA uma-s-1p1 (fp32) | a=3.6 | 1.3e-6 | 3.6e-7 |
| UMA uma-s-1p1 + ZBL | a=2.3 (ZBL active) | 2.2e-5 | 5.4e-5 |

UMA runs in fp32, so its differences are orders of magnitude above the fp64 models and the ZBL
case is within a factor of two of the 1e-4 tolerance. The HydraGNN checkpoint only loads with
the frozen source (`docs/hydragnn-frozen-source.md`); the shared `HydraGNN` checkout fails on
unexpected state-dict keys. UMA weights come from
`deployments/frontier/download/download-uma-frontier.sh` (needs `HF_TOKEN`).

Compute-node job: `deployments/frontier/jobs/lammps/job-lammps-mliap-parity-frontier.sh`.

## Limits

- The bridge is single-rank and CPU: forces are written to local atoms only, so use NVE/NVT
  (pressure and virial are not reported) and one MPI rank. Multi-rank, GPU/KOKKOS and per-pair
  ML-IAP coupling are not implemented. The parity check covers a single point, not MD stability.
- Parity for UMA and HydraGNN was measured on a login node only; the compute-node job has not
  been run with them. Aurora and Perlmutter are untested.
- Only CPU builds exist. GPU builds (KOKKOS with HIP on Frontier) need their own toolchain and are
  not covered by the scripts' defaults.
- The common script is validated only through the Frontier wrapper; other machines need their
  compiler, MPI and CMake loaded first and are untested.
