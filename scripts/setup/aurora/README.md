# Aurora Setup Scripts for matsim-agents

This directory provides an Aurora-focused setup flow that mirrors the phased
install style used by the Frontier and Perlmutter scripts.

## Model Download Safety Policy

See the canonical cross-platform policy in `docs/model-download-safety.md`.

Download entry points:

- `scripts/download/aurora/download-models-aurora.sh` (full set, gated models optional via token)
- `scripts/download/aurora/download-open-models-aurora.sh` (open-access set)

## vLLM-XPU

### install-vllm-xpu-aurora.sh

Verifies that the `frameworks` module ships a working vLLM-XPU stack (no
source build required as of `frameworks/2025.3.1`).  Run on a login node; XPU
will report unavailable there — that is expected.

```bash
bash scripts/setup/aurora/install-vllm-xpu-aurora.sh
```

After verification, submit the smoke test:

```bash
qsub scripts/smoke-tests/aurora/smoke-vllm-singlenode-aurora.sh
```

**Important Aurora-specific pitfalls** — see the full guide at
`docs/vllm-aurora.md` for details on every challenge and fix, including:

- SIGSEGV in vLLM's model-registry subprocess (node-specific, fixed via
  `aurora_vllm_entrypoint.py`)
- `ONEAPI_DEVICE_SELECTOR` syntax differences between node driver versions
  (`opencl:cpu` is the canonical cross-node form)
- Must launch the server via `mpiexec -n 1 --ppn 1` for PALS permissions
- numpy must **not** be pinned to 1.x on Aurora (frameworks ships 2.2.6, 2.x ABI)

---


Runs a two-stage install:

1. Run HydraGNN's Aurora installer from installation_DOE_supercomputers to
	create/configure the environment and install HydraGNN dependencies.
2. Install matsim-agents and additional runtime dependencies.

Usage:

```bash
bash scripts/setup/aurora/install_matsim_aurora.sh
```

Common overrides:

```bash
# Choose env location
VENV_PATH=/path/to/aurora_venv bash scripts/setup/aurora/install_matsim_aurora.sh

# Choose Python interpreter
PYTHON_BIN=python3.11 bash scripts/setup/aurora/install_matsim_aurora.sh

# Install matsim-agents with extra backends
LLM_BACKENDS="huggingface,dev" bash scripts/setup/aurora/install_matsim_aurora.sh

# Also install vLLM
INSTALL_VLLM_SERVER=1 bash scripts/setup/aurora/install_matsim_aurora.sh

# Install UMA backend (creates a separate fairchem_venv — see note below)
INSTALL_UMA=1 bash scripts/setup/aurora/install_matsim_aurora.sh
```

> **UMA / fairchem-core note:** `fairchem-core >= 2.0` requires `numpy >= 2.0`,
> which conflicts with HydraGNN's `numpy == 1.26.4` pin. `INSTALL_UMA=1`
> therefore creates a **separate** `fairchem_venv` alongside `hydragnn_venv`
> rather than installing into the shared environment. UMA jobs must activate
> `fairchem_venv` instead of `hydragnn_venv`. This is a known incompatibility
> until HydraGNN relaxes its numpy pin. Note also that Aurora's `frameworks`
> module ships numpy 2.x, so the fairchem_venv should not have the overlay
> cleanup issue that affects CUDA-torch; the install script handles this.

Default environment path created by this flow:

```bash
$HYDRAGNN_ROOT/installation_DOE_supercomputers/HydraGNN-Installation-Aurora/hydragnn_venv
```

### setup_matsim_aurora.sh
Quick setup script for daily use after installation.

Usage:

```bash
source scripts/setup/aurora/setup_matsim_aurora.sh
```

Override environment path:

```bash
MATSIM_AURORA_VENV=/path/to/venv source scripts/setup/aurora/setup_matsim_aurora.sh
```

### build-qe-cpu-aurora.sh
Builds Quantum ESPRESSO on Aurora (CPU-focused), mirroring the workflow used by
the Frontier/Perlmutter QE build scripts.

Usage:

```bash
bash scripts/setup/aurora/build-qe-cpu-aurora.sh
```

Common overrides:

```bash
QE_VERSION=develop bash scripts/setup/aurora/build-qe-cpu-aurora.sh
QE_PREFIX=/path/to/quantum-espresso bash scripts/setup/aurora/build-qe-cpu-aurora.sh
NCORES=32 bash scripts/setup/aurora/build-qe-cpu-aurora.sh
```

### build-qe-gpu-aurora.sh
Builds Quantum ESPRESSO on Aurora with Intel GPU offload intent, using the same
clone/configure/build/install flow as the other machine scripts.

Usage:

```bash
bash scripts/setup/aurora/build-qe-gpu-aurora.sh
```

Common overrides:

```bash
QE_GPU_ARCHS=intel_gpu_pvc bash scripts/setup/aurora/build-qe-gpu-aurora.sh
QE_GPU="openmp;oneapi" bash scripts/setup/aurora/build-qe-gpu-aurora.sh
EXTRA_CMAKE_ARGS="-DVAR=VALUE" bash scripts/setup/aurora/build-qe-gpu-aurora.sh
```

Validated results from the current repository run:

- build/install completed successfully (exit code 0)
- install prefix populated under `external/quantum-espresso/install-gpu/`
- 106 executables installed in `install-gpu/bin/`
- core binaries present: `pw.x`, `cp.x`, `ph.x`, `pp.x`, `epw.x`

See full Aurora QE documentation:

- `docs/quantum-espresso-aurora.md`

Run `pw.x` with the Aurora launcher after building:

```bash
bash scripts/launchers/aurora/run-pw-gpu-aurora.sh path/to/pw.in
```

### VASP 6.6.0 on Aurora

The repository does not commit VASP source because it is proprietary, but it
does track the build provenance for the Aurora GPU build.

- Provenance note: `docs/vasp-aurora.md`
- Build script: `scripts/setup/aurora/build-vasp-gpu-aurora.sh`
- Upstream template used: `arch/makefile.include.oneapi_omp_off`
- Local working file during the build: `external/vasp6/src/vasp.6.6.0/makefile.include`

This keeps the exact makefile lineage discoverable without checking any VASP
source or vendor archive into git. The default script target builds
`vasp_std`, `vasp_gam`, and `vasp_ncl` in one invocation.
