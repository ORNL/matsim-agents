# VASP installations and HPC execution

MATSIM-AGENTS accepts facility-provided and custom VASP binaries through the
same interface. Building VASP is optional; a licensed, executable installation
and its compatible runtime libraries are required.

Keep three choices separate:

- **Installation:** executable path, runtime setup script, optional SHA-256 pin.
- **Facility placement:** the `vasp_wrapper` and the existing `resources` plan
  determine MPI launch, nodes, CPU affinity, and GPU binding.
- **Calculation:** INCAR, PAWs, staging, parsing, and recovery do not depend on
  whether the installation was supplied by a facility or built locally.

## Python DFT configuration

The existing `dft.vasp` block remains compatible. Optional fields select the
installation environment without changing the wrapper contract:

```yaml
dft:
  backend: vasp
  vasp:
    vasp_bin: /soft/applications/vasp/vasp.6.6.1/bin/vasp_std
    vasp_wrapper: /path/to/matsim-agents/deployments/aurora/launchers/_vasp-step-aurora.sh
    vasp_env_setup: /path/to/matsim-agents/deployments/aurora/setup/vasp-runtime-aurora.sh
    incar_template: /path/to/INCAR.template
    potcar_dir: /path/to/element-potcars
```

`vasp_expected_sha256` optionally pins the executable's lowercase SHA-256.
Switching to a custom installation means changing `vasp_bin` and, if its
runtime differs, `vasp_env_setup`. The wrapper and workflow remain the same.
The runtime setup file contributes to execution provenance/cache identity.

## Shell and relaxation entry points

The shipped Aurora, Frontier, and Perlmutter VASP launchers accept:

```bash
export VASP_BIN=/path/to/vasp_std
export VASP_ENV_SETUP=/absolute/path/to/vasp-environment.sh
# Optional:
export VASP_EXPECTED_SHA256=<64-character-lowercase-sha256>
```

Step wrappers still receive the binary as their second argument. The
no-argument `run-vasp-gpu-*.sh` launchers use `VASP_BIN`. The relaxation
configuration also accepts an `environment` mapping for these variables.

The setup script is **trusted executable Bash configuration**, sourced inside
each VASP launcher before placement. It must establish compatible modules and
library paths, return successfully, and must not launch VASP itself. A setup
failure, missing executable, or checksum mismatch aborts the launch. Runtime
logs record the resolved executable, checksum, setup path and module list.

An explicit setup script overrides the launcher's default module selection,
regardless of executable location. With no override, existing Frontier and
Perlmutter facility-path/custom-toolchain selection is preserved. Existing
custom executable defaults are retained; no installation is built or replaced
by selecting a runtime.

## Aurora / ALCF

The default Aurora setup loads oneAPI/GPU UMD and then the tested
`mpich/prd/5.0.0.aurora_test.51a9474` module, for either the ALCF executable or
the tested custom oneAPI build. Override `VASP_MPICH_MODULE`, `ONEAPI_MODULE`,
and `GPU_UMD_MODULE` only for a compatible installation, or supply a different
`VASP_ENV_SETUP` for a different toolchain.

Example facility selection:

```bash
export VASP_BIN=/soft/applications/vasp/vasp.6.6.1/bin/vasp_std
export MATSIM_VASP_LAUNCHER=/path/to/matsim-agents/deployments/aurora/launchers/run-vasp-gpu-aurora.sh
```

Custom selection uses the same launcher with a different `VASP_BIN`.
No path-based guess selects the Aurora MPI version. An executable path alone
does not establish runtime compatibility or scientific convergence.
