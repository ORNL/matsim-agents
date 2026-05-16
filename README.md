# matsim-agents

**Multi-agent AI framework for atomistic materials simulation and discovery.**

`matsim-agents` orchestrates large language models, machine-learned
interatomic potentials, and ASE-based atomistic workflows into a single
agentic loop. The user states a research objective in natural language;
agents plan, run HydraGNN-driven simulations, score chemical and
dynamical stability, and report the findings — with optional human
review at every gate.

The framework is **backend-agnostic**: HydraGNN is the default MLFF
backend, but the relaxation tool, phase explorer, and stability scorer
are written so other potentials (MACE, NequIP, Orb, ...) can be plugged
in via the same interfaces.

---

## Table of contents

1. [Architecture](#architecture)
2. [Portability across DOE supercomputers](#portability-across-doe-supercomputers)
3. [Running on Frontier (OLCF)](#running-on-frontier-olcf)
4. [Running on Aurora (ALCF)](#running-on-aurora-alcf)
5. [Running on Perlmutter (NERSC)](#running-on-perlmutter-nersc)
6. [HPC Documentation Index](#hpc-documentation-index)
7. [Installation](#installation)
8. [LLM backends](#llm-backends)
9. [Downloading models for vLLM](#downloading-models-for-vllm)
10. [Quick start](#quick-start)
11. [The agent graph](#the-agent-graph)
12. [Hypothesis-driven discovery chat](#hypothesis-driven-discovery-chat)
13. [Programmatic API](#programmatic-api)
14. [CLI reference](#cli-reference)
15. [Active-learning loop (HydraGNN ↔ DFT)](#active-learning-loop-hydragnn--dft)
16. [Codabench Competition](#codabench-competition)
17. [Project layout](#project-layout)
18. [Configuration reference](#configuration-reference)
19. [Current capabilities and planned work](#current-capabilities-and-planned-work)
20. [Contributing](#contributing)
21. [License & citation](#license--citation)

---

## Architecture

```
                ┌──────────────────────────────────────────────┐
                │                  USER                        │
                │  natural-language objective / chat dialogue  │
                └───────────────────────┬──────────────────────┘
                                        │
                ┌───────────────────────▼──────────────────────┐
                │              LangGraph workflow              │
                │                                              │
                │   planner ───► executor ──┐                  │
                │                  ▲        │                  │
                │                  └────────┤  while pending   │
                │                           ▼                  │
                │                        analyst ──► END       │
                └───────────────────────┬──────────────────────┘
                                        │  tool calls
                ┌───────────────────────▼──────────────────────┐
                │              Discovery wrapper               │
                │   composition parsing → phase enumeration    │
                │   → relaxation (HydraGNN+ASE) → stability    │
                └───────────────────────┬──────────────────────┘
                                        │
                ┌───────────────────────▼──────────────────────┐
                │             Atomistic backends               │
                │   HydraGNN (fused MLFF + BranchWeightMLP)    │
                │   ASE (FIRE / BFGS / BFGSLineSearch)         │
                │   pymatgen (optional prototypes)             │
                └──────────────────────────────────────────────┘
```

### Capabilities

- **Multi-agent orchestration** with LangGraph: typed shared state, checkpointed steps, conditional routing, human-in-the-loop gates.
- **Hypothesis-generation chat** with any local LLM (Qwen 2.5 via Ollama by default).
- **Automatic composition detection** in user/LLM messages — when a new chemical formula is proposed, the system offers to run a substantial atomistic exploration.
- **HydraGNN-powered structure relaxation** using the fused MLFF + branch-weight MLP stack from `examples/multidataset_hpo_sc26/structure_optimization_ASE.py`.
- **3-D crystal-phase enumeration** across common prototypes:
  - elemental: fcc, bcc, hcp, sc, diamond
  - binary: rocksalt, CsCl, zincblende, wurtzite, fluorite, rutile
  - ternary: cubic perovskite (ABX₃), normal spinel (AB₂X₄)
  - quaternary: rocksalt-ordered double perovskite (A₂BB'X₆, Fm-3̄m)
- **2-D phase enumeration** (opt-in via `--include-2d`):
  - graphene-like (1 element honeycomb)
  - h-BN-like (binary 1:1 honeycomb)
  - MoS₂-family monolayers in trigonal-prismatic 2H and octahedral 1T (binary 1:2)
  - configurable **multilayer stacking** with adjustable interlayer separation and vacuum gap
- **Supercell control**: explicit `NxNxN` tiling or auto-tile each prototype to a minimum atom count so dopants, AFM ordering, and symmetry-breaking distortions can develop.
- **Stability scoring**: relative chemical stability (ΔE/atom rankings) and a dynamical-stability proxy (residual force tolerance).
- **Local & HPC ready, portable across diverse DOE accelerators**: same
  Python entry points run on **Frontier (OLCF, AMD MI250X)**, **Aurora
  (ALCF, Intel PVC)**, and **Perlmutter (NERSC, NVIDIA A100)**, plus
  Andes and laptops. The setup script delegates to HydraGNN's
  installers and auto-relaxes HydraGNN's overly-tight `click==8.0.0` /
  `tqdm==4.67.1` pins so the env is conflict-free on every site.
- **First-class DFT labellers built per platform**: validated
  build/run recipes for both **VASP 6.6** (Frontier MI250X, Aurora PVC)
  and **Quantum ESPRESSO `pw.x` GPU** (Frontier MI250X via OpenMP
  target offload, Aurora PVC via `oneapi`/`openmp`, Perlmutter A100 via
  CUDA). Build scripts and SLURM/PBS launchers are checked in for each
  site — see [Portability across DOE supercomputers](#portability-across-doe-supercomputers).
- **Pluggable LLMs**: Ollama, vLLM, OpenAI, Anthropic via a single factory.
- **Active-learning loop** (`matsim-agents al run`): HydraGNN-driven MD
  generates candidates → ensemble / MC-dropout uncertainty selects the
  most informative → a DFT backend (**VASP 6.6** *or* **Quantum
  ESPRESSO `pw.x`**) labels them in parallel inside one SLURM
  allocation → dataset is grown and HydraGNN is retrained → repeat. The
  DFT backend is a single YAML toggle (`dft.backend: vasp | qe`); both
  share an INCAR-style template path (`INCAR.template` / `pw.template`).
- **LLM-generated MD seeds** (`md.seed_source.kind: prompt`): the LLM
  proposes plausible chemical compositions for a target objective and
  the loop materialises seed structures from common crystal prototypes,
  no curated POSCAR collection required.
- **Templated YAML configs**: `${VAR}`, `${VAR:-default}`, `${VAR:?msg}`
  shell-style substitution with optional in-file `vars:` block, so the
  same config can be re-targeted across users / scratch dirs / runs
  without editing it.

---

## Portability across DOE supercomputers

`matsim-agents` is designed to run **the same Python code path on three
DOE leadership-class systems with three very different accelerators**.
All heavy backends (HydraGNN MLFF inference/training, vLLM model
serving, VASP, and Quantum ESPRESSO) have validated build + launcher
recipes per site, with all platform-specific gotchas (toolchains, MPI
GTL pins, ROCm/Cray cross-builds, CUDA-aware MPI) baked in.

| Capability                          | Frontier (OLCF) MI250X            | Aurora (ALCF) PVC                       | Perlmutter (NERSC) A100         |
|-------------------------------------|-----------------------------------|-----------------------------------------|---------------------------------|
| **Hardware**                        | AMD MI250X (gfx90a), 64-core EPYC | Intel Data Center GPU Max 1550 (PVC)    | NVIDIA A100 (40/80 GB), AMD EPYC|
| **HydraGNN venv**                   | ROCm 7.2.0 + PyTorch              | oneAPI + Intel Extension for PyTorch    | CUDA 12 + PyTorch               |
| **vLLM model server**               | ROCm 7.2.0, source build          | oneAPI                                  | CUDA                            |
| **VASP 6.6**                        | `build-vasp-gpu-frontier.sh`      | `build-vasp-gpu-aurora.sh` (`vasp_std`/`vasp_gam`/`vasp_ncl`) | (use site module if available)  |
| **Quantum ESPRESSO `pw.x` (GPU)**   | OpenMP target offload to gfx90a   | `QE_GPU="openmp;oneapi"`, PVC arch      | CUDA build                      |
| **Setup entry point**               | `scripts/setup/frontier/install-rocm72.sh` | `scripts/setup/aurora/install_matsim_aurora.sh` | `scripts/setup/perlmutter/install_matsim_perlmutter.sh` |
| **Active-learning launcher**        | `scripts/launchers/frontier/run-active-learning-frontier.sh` | (file-coupled via SLURM)                | (file-coupled via SLURM)        |
| **Per-platform docs**               | [docs/quantum-espresso-frontier.md](docs/quantum-espresso-frontier.md) | [docs/quantum-espresso-aurora.md](docs/quantum-espresso-aurora.md), [docs/vasp-aurora.md](docs/vasp-aurora.md) | [docs/quantum-espresso-perlmutter.md](docs/quantum-espresso-perlmutter.md) |

Single entry-point index covering all three systems:
[`docs/hpc-platforms.md`](docs/hpc-platforms.md).

Design principles that keep the code portable:

- **DFT and Python/ML stacks are never co-loaded** in the same shell
  on any platform — each uses its own module set, and the active-
  learning loop couples them through SLURM steps + the filesystem.
  This avoids the pervasive ABI/toolchain conflicts (Cray MPI GTL
  SONAMEs on Frontier, oneAPI vs PyTorch CUDA stack on Perlmutter,
  etc.) that otherwise break shared builds.
- **Backend-agnostic active learning** — the same `matsim-agents al
  run` driver works whether the labeller is VASP or QE, and on any of
  the three platforms, because the DFT backend is selected by a single
  YAML field (`dft.backend: vasp | qe`).
- **Templated YAML configs** — `${VAR}` / `${VAR:-default}` /
  `${VAR:?msg}` substitution lets one config file follow you between
  Frontier scratch, Aurora flare, and Perlmutter pscratch without
  edits.

---

## Running on Frontier (OLCF)

> **⚠️ Frontier users — read this first:**
> See [`scripts/docs/frontier/README-frontier.md`](scripts/docs/frontier/README-frontier.md)
> for required setup and known issues. **Critically: a prebuilt `tvm_ffi`
> shared library must exist at
> `$PROJ/cache/tvm-ffi/libtorch_c_dlpack_addon_torch211-rocm.so`
> (where `$PROJ` is your project's proj-shared directory) or every vLLM job
> will silently hang forever** (the script preflight check will fail-fast in
> 2 seconds with a clear error message). If missing, rebuild with
> `sbatch scripts/setup/frontier/prebuild-tvm-ffi-frontier.sh`.

### Quantum ESPRESSO (DFT) backend on Frontier

The repo also ships a fully reproducible recipe for building Quantum
ESPRESSO `develop` with **AMD MI250X (gfx90a) OpenMP target offload**:

- Build script: [`scripts/setup/frontier/build-qe-gpu-frontier.sh`](scripts/setup/frontier/build-qe-gpu-frontier.sh)
- Run launcher: [`scripts/launchers/frontier/run-pw-gpu-frontier.sh`](scripts/launchers/frontier/run-pw-gpu-frontier.sh)
- Full docs: [`docs/quantum-espresso-frontier.md`](docs/quantum-espresso-frontier.md)
- Platform index: [`docs/hpc-platforms.md`](docs/hpc-platforms.md)

The build is cross-compiled on a login node and produces ~92 binaries
(`pw.x`, `cp.x`, `ph.x`, `pp.x`, `neb.x`, `epw.x`, `kcw.x`, `tddfpt`/
`turbo_*` suite, `pioud.x`, `all_currents.x`, …) under
`external/quantum-espresso/install-gpu/bin/` (gitignored). The recipe
includes baked-in workarounds for the cce/18.0.1 ftn-7991 ICE, the
PIOUD `etime()` link error (rewritten to F95 `cpu_time`), and the
rocm/7.x cray-mpich SONAME mismatch.

QE uses a different module stack than matsim-agents' Python; the two
are deliberately kept isolated and coupled only through Slurm + files.

### VASP (DFT) backend on Frontier

VASP 6.6 is also wired up on Frontier MI250X for the active-learning
labeller path:

- Build script: [`scripts/setup/frontier/build-vasp-gpu-frontier.sh`](scripts/setup/frontier/build-vasp-gpu-frontier.sh)
- In-allocation step launcher (called by the AL loop):
  [`scripts/launchers/frontier/_vasp-step-frontier.sh`](scripts/launchers/frontier/_vasp-step-frontier.sh)

As with QE, the proprietary VASP source itself is **not** committed;
only the build recipe is. The repository assumes you have a licensed
VASP source tree under `external/vasp6/`.

---

## Running on Aurora (ALCF)

The repository also includes a validated build/run path for Quantum
ESPRESSO with Intel GPU offload on Aurora.

- Build script: [`scripts/setup/aurora/build-qe-gpu-aurora.sh`](scripts/setup/aurora/build-qe-gpu-aurora.sh)
- Run launcher: [`scripts/launchers/aurora/run-pw-gpu-aurora.sh`](scripts/launchers/aurora/run-pw-gpu-aurora.sh)
- Full docs: [`docs/quantum-espresso-aurora.md`](docs/quantum-espresso-aurora.md)
- Platform index: [`docs/hpc-platforms.md`](docs/hpc-platforms.md)

Validated outcome in this repo:

- successful CMake build + install (exit code 0)
- 106 installed executables in `external/quantum-espresso/install-gpu/bin/`
- core binaries verified: `pw.x`, `cp.x`, `ph.x`, `pp.x`, `epw.x`

Quick run pattern:

```bash
bash scripts/launchers/aurora/run-pw-gpu-aurora.sh path/to/pw.in
```

Aurora QE and the Python/ML environment are intentionally isolated and
typically coupled only via files and scheduler jobs.

For VASP on Aurora, the repository keeps only build provenance, not the vendor
source itself. The recorded makefile lineage is documented in
[`docs/vasp-aurora.md`](docs/vasp-aurora.md), including the upstream template
used (`arch/makefile.include.oneapi_omp_off`) and the local working makefile
path under `external/vasp6/`. The Aurora build entry point is
[`scripts/setup/aurora/build-vasp-gpu-aurora.sh`](scripts/setup/aurora/build-vasp-gpu-aurora.sh),
which defaults to building `vasp_std`, `vasp_gam`, and `vasp_ncl` in one run.

### vLLM on Aurora (Intel PVC)

Aurora supports vLLM-XPU serving and inference via the official ALCF `frameworks` module stack (Python 3.12, torch-xpu, ipex, vllm, ray, triton). The repo provides:

- **Single-node smoke test:** [`scripts/smoke-tests/aurora/smoke-vllm-singlenode-aurora.sh`](scripts/smoke-tests/aurora/smoke-vllm-singlenode-aurora.sh)
- **Advanced launchers:** [`scripts/advanced/aurora/job-serve-multinode-vllm-aurora.sh`](scripts/advanced/aurora/job-serve-multinode-vllm-aurora.sh) (multi-node Ray serve), plus single-relax, active-learning, and QE warmstart launchers

**Key requirements and gotchas:**

- **PVC visibility:** On Aurora compute nodes, bare `python` does NOT see the GPUs. Always wrap Python in `mpiexec -n 1 --ppn 1` (as in the smoke script) to expose XPUs via PALS.
- **Device mask:** Use `ZE_FLAT_DEVICE_HIERARCHY=FLAT` and a non-dotted `ZE_AFFINITY_MASK` (e.g., `0,1` for TP=2). In FLAT, each tile is a root device; dotted notation (`0.0,0.1`) is only valid in COMPOSITE and will result in `device_count()=0` in FLAT.
- **TMPDIR:** PBS sets `$TMPDIR` to a long path that exceeds the Unix socket limit for ZMQ IPC. Always set `export TMPDIR=/tmp` before launching vLLM.
- **oneCCL KVS:** Do NOT set `CCL_KVS_MODE=mpi` or `CCL_PROCESS_LAUNCHER=pmix` for vLLM. vLLM's multiproc_executor uses forked workers, not MPI ranks; oneCCL must use its default internal KVS over TCP.
- **Debug queue:** The default `debug` queue has a per-user limit of 1 queued job and short walltime. For parallel jobs, use `workq` or `prod`.
- **Model download:** Place models in `$PROJ/models/` (e.g., Mistral-Small-24B-Instruct-2501). Use the provided `hf_download.py` script if needed.

**To run the smoke test:**

1. Build the vLLM XPU venv (if not already):
  ```bash
  bash scripts/setup/aurora/install-vllm-xpu-aurora.sh
  ```
2. Download a supported model (e.g., Mistral-Small-24B):
  ```bash
  source /path/to/hydragnn_venv/bin/activate
  python scripts/setup/aurora/hf_download.py mistralai/Mistral-Small-24B-Instruct-2501
  ```
3. Submit the smoke test:
  ```bash
  qsub scripts/smoke-tests/aurora/smoke-vllm-singlenode-aurora.sh
  # or override model:
  qsub -v SMOKE_MODEL_PATH=$PROJ/models/Qwen2.5-32B-Instruct scripts/smoke-tests/aurora/smoke-vllm-singlenode-aurora.sh
  ```
4. Inspect results in `runs/smoke-vllm-singlenode-<jobid>/`.

If the job fails, check `vllm.log` for device mask, TMPDIR, or oneCCL errors. Each error layer is documented in the script comments.

For multi-node serving, see the advanced launchers in `scripts/advanced/aurora/`.

## Running on Perlmutter (NERSC)

Perlmutter (NERSC, NVIDIA A100) is supported as a first-class target
for both the Python/ML stack and Quantum ESPRESSO GPU.

- Setup overview: [`scripts/setup/perlmutter/README.md`](scripts/setup/perlmutter/README.md)
- Matsim env install: [`scripts/setup/perlmutter/install_matsim_perlmutter.sh`](scripts/setup/perlmutter/install_matsim_perlmutter.sh)
- QE GPU build: [`scripts/setup/perlmutter/build-qe-gpu-perlmutter.sh`](scripts/setup/perlmutter/build-qe-gpu-perlmutter.sh)
  (CPU-only variant: [`build-qe-cpu-perlmutter.sh`](scripts/setup/perlmutter/build-qe-cpu-perlmutter.sh))
- QE detailed build guide: [`scripts/setup/perlmutter/QE-BUILD-GUIDE.md`](scripts/setup/perlmutter/QE-BUILD-GUIDE.md)
- Full QE docs: [`docs/quantum-espresso-perlmutter.md`](docs/quantum-espresso-perlmutter.md)
- Launchers:
  - QE `pw.x` GPU: [`scripts/launchers/perlmutter/run-pw-gpu-perlmutter.sh`](scripts/launchers/perlmutter/run-pw-gpu-perlmutter.sh)
  - QE warm-start benchmark: [`scripts/launchers/perlmutter/run-qe-warmstart-benchmark-perlmutter.sh`](scripts/launchers/perlmutter/run-qe-warmstart-benchmark-perlmutter.sh)
  - Single-node / multi-node / all-models LLM smoke tests:
    `launch-test-singlenode-resume-perlmutter.sh`,
    `launch-test-multinode-perlmutter.sh`,
    `launch-test-all-models-perlmutter.sh`

Quick run pattern:

```bash
./scripts/launchers/perlmutter/run-pw-gpu-perlmutter.sh path/to/pw.in
```

As on Frontier and Aurora, the DFT module stack and the Python/ML
environment are intentionally isolated and coupled only through Slurm
+ files.

---

## HPC Documentation Index

For a single entry point across Frontier, Aurora, Perlmutter, and model-serving
docs, see [`docs/hpc-platforms.md`](docs/hpc-platforms.md).

---

## Installation

`matsim-agents` depends on HydraGNN (which itself wraps PyTorch + PyTorch
Geometric). The provided installer delegates the heavy install to
HydraGNN's official scripts so the same code path works on a laptop and
on a DOE supercomputer.

```bash
git clone git@code.ornl.gov:multi-agentic-ai-materials/matsim-agents.git
cd matsim-agents

# Local workstation (CPU or single GPU)
./scripts/setup_env.sh

# Frontier (OLCF, ROCm 7.2 — current standard)
bash scripts/setup/frontier/install-rocm72.sh

# Perlmutter (NERSC)
PLATFORM=perlmutter ./scripts/setup_env.sh
```

Available `PLATFORM` values for the generic `setup_env.sh`:
`workstation` (default), `perlmutter`, `aurora`, `andes`,
`frontier-rocm71`, `frontier-rocm64` (legacy — the supported Frontier
path is `scripts/setup/frontier/install-rocm72.sh`).

### ROCm version matrix on Frontier

The three Frontier-targeted backends in this repo do **not** all use the
same ROCm version. The combinations below are what is actually wired up
in the scripts and what you should expect at runtime:

| Backend | Module | Why this version |
|---|---|---|
| **HydraGNN venv** (used by every Frontier launcher: vLLM, HF, downloaders, smoke tests, six-model bench) | `rocm/7.2.0` | Current Frontier-supported PyTorch + ROCm path; built once into `HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72/` |
| **vLLM model server** | `rocm/7.2.0` | Shares the HydraGNN ROCm 7.2 venv; built from source via [`scripts/setup/frontier/build-vllm-rocm72.sh`](scripts/setup/frontier/build-vllm-rocm72.sh) |
| **Quantum ESPRESSO GPU** | `rocm/6.2.4` *(forced)* | Frontier's `cray-mpich/8.1.31` GTL `libmpi_gtl_hsa.so` is hard-linked against `libamdhip64.so.6` (rocm 6.x SONAME). rocm/7.x ships `.so.7` and breaks the MPI Fortran link probe at CMake configure. Pin documented in [`docs/quantum-espresso-frontier.md`](docs/quantum-espresso-frontier.md). |

QE and the Python/ML stacks are deliberately never co-loaded in the same
shell; they couple through Slurm + the filesystem.

Environment overrides accepted by the installer:

| Variable | Purpose | Default |
|---|---|---|
| `PYTHON` | Python interpreter | `python3` |
| `HYDRAGNN_REPO` | HydraGNN git URL | `https://github.com/ORNL/HydraGNN.git` |
| `HYDRAGNN_REF` | Branch/tag/commit | `main` |
| `HYDRAGNN_DIR` | Reuse an existing HydraGNN checkout | `third_party/HydraGNN` |
| `HYDRAGNN_EXTRAS` | Args forwarded to `install_dependencies.sh` | `all dev` |
| `LLM_BACKENDS` | Subset of `ollama vllm openai anthropic huggingface` | `ollama vllm` |
| `BOOTSTRAP_OLLAMA` | Set to `1` to install the Ollama daemon, start it, and pull `OLLAMA_MODELS` (workstation only) | `0` |
| `OLLAMA_MODELS` | Space-separated list of models to pull when `BOOTSTRAP_OLLAMA=1` | `qwen2.5:14b` |

After the script finishes:

```bash
source .venv/bin/activate    # workstation case
matsim-agents --help
```

To bootstrap the local Ollama daemon and pull a model in one go:

```bash
BOOTSTRAP_OLLAMA=1 OLLAMA_MODELS="qwen2.5:14b llama3.1:8b" \
    ./scripts/setup_env.sh
```

---

## LLM backends

Set the provider at runtime via CLI flag, environment variable, or in
code. Local/open-source backends are the default.

For a detailed comparison of the two open-source local backends (vLLM vs
HuggingFace Transformers + Accelerate) — including pros, cons, and guidance
for Frontier (ROCm) — see [docs/llm-backends-comparison.md](docs/llm-backends-comparison.md).

| Provider | Install | Typical model | Notes |
|---|---|---|---|
| **`ollama`** *(default)* | `brew install ollama && ollama pull qwen2.5:14b` | `qwen2.5:14b`, `llama3.1:8b`, `deepseek-r1:14b` | Fully local, CPU/GPU/Metal. |
| **`vllm`** | Run a vLLM server (`vllm serve <model> --port 8000`) | `meta-llama/Llama-3.1-8B-Instruct` | OpenAI-compatible; great for HPC. |
| **`openai`** | `pip install matsim-agents[openai]` | `gpt-4o-mini` | Hosted. Set `OPENAI_API_KEY`. |
| **`anthropic`** | `pip install matsim-agents[anthropic]` | `claude-3-5-sonnet-latest` | Hosted. Set `ANTHROPIC_API_KEY`. |
| **`huggingface`** | `pip install matsim-agents[huggingface]` | `Qwen/Qwen2.5-72B-Instruct` | Direct HF Transformers + Accelerate; no server needed. Ideal as fallback on HPC when vLLM is unavailable. Set `MATSIM_HF_MODEL_PATH` to a local model directory. |

### Downloading models for vLLM

For the vLLM backend you need to download the model weights locally before
starting the server. The recommended model for matsim-agents on HPC is
`Qwen/Qwen2.5-72B-Instruct`. A quick one-liner using the `hf` CLI that ships
with `huggingface_hub>=1.12`:

```bash
hf download Qwen/Qwen2.5-72B-Instruct \
    --local-dir /path/to/models/Qwen2.5-72B-Instruct
```

For detailed instructions — including Frontier-specific steps, running the
download as a background job, and resuming interrupted downloads — see
[docs/model-download.md](docs/model-download.md).

Configuration knobs:

```bash
export MATSIM_LLM_PROVIDER=ollama          # or vllm | openai | anthropic | huggingface
export MATSIM_OLLAMA_BASE_URL=http://...    # optional
export MATSIM_VLLM_BASE_URL=http://node:8000/v1
export MATSIM_VLLM_API_KEY=EMPTY            # only if vLLM is auth-protected
export MATSIM_HF_MODEL_PATH=/path/to/model  # huggingface provider: local model dir
```

---

## Quick start

### 1. Run the agent graph end-to-end

```bash
matsim-agents run \
  "Relax structures/mos2-B_Defect-Free_PBE.vasp and report the final energy." \
  --logdir ./multidataset_hpo-BEST6-fp64 \
  --mlp-checkpoint ./mlp_branch_weights.pt \
  --llm-provider ollama --llm-model qwen2.5:14b
```

### 2. Hypothesis-generation chat with auto-triggered exploration

```bash
ollama pull qwen2.5:14b

matsim-agents chat \
  --logdir ./multidataset_hpo-BEST6-fp64 \
  --mlp-checkpoint ./mlp_branch_weights.pt \
  --min-atoms 64
```

A typical session:

```
you> I want a Pb-free halide double perovskite for photovoltaics with band gap near 1.5 eV.

assistant> A promising candidate is Cs2AgBiBr6 ...

Proposed composition detected: AgBiBr6Cs2. Run HydraGNN-based phase exploration? [y/N]: y

>>> Exploring composition AgBiBr6Cs2
  starting double_perovskite   .../AgBiBr6Cs2_double_perovskite.vasp
  done    double_perovskite   E=-365.4123 eV  |F|max=0.0118 eV/Å  steps=112

Stability report for AgBiBr6Cs2:
  Predicted ground state: AgBiBr6Cs2_double_perovskite_optimized_structure.vasp
  E/atom = -9.1353 eV   |F|max = 0.012 eV/Å   dynamically_stable_proxy = True
  Chemical-stability proxy: PASS

you> Now suggest a Sb-substituted variant.
```

### 3. 2-D and multilayer materials discovery

```bash
matsim-agents chat \
  --logdir ./multidataset_hpo-BEST6-fp64 \
  --mlp-checkpoint ./mlp_branch_weights.pt \
  --include-2d --num-layers 3 --vacuum 20.0 --min-atoms 36
```

When the conversation introduces a 1-element (graphene-like), 1:1 binary
(h-BN-like), or 1:2 binary (MoS₂-family) composition, the discovery
wrapper additionally enumerates 2-D monolayer / multilayer slabs
alongside the 3-D bulk prototypes.

---

## The agent graph

Three nodes share a typed `MatSimState`:

- **planner** — turns the objective into a list of `TaskSpec` items
  (kinds: `relax`, `analyze`, `report`). Uses the LLM with structured
  output; falls back to a deterministic plan when the LLM is unavailable.
- **executor** — pops the next task, dispatches the matching tool
  (currently `relax_structure`), appends a `RelaxationResult` to the
  state, increments `iteration`. Routed back to itself until the queue
  drains or `max_iterations` is reached.
- **analyst** — summarizes the accumulated results into a human-readable
  report (LLM-assisted when available, deterministic baseline otherwise).

State is checkpointed via LangGraph's `MemorySaver`, so every node
transition is replayable and inspectable.

---

## Hypothesis-driven discovery chat

The `chat` REPL is more than a wrapper around the LLM — it is a
**closed loop between dialogue and atomistic simulation**:

1. The user and the assistant exchange messages about a target property.
2. After each turn, [`extract_compositions`](src/matsim_agents/discovery/composition.py) scans both messages for chemical formulas (validates element symbols, reduces stoichiometry, ignores English words like "Carbon" or "Hello").
3. For every newly-seen formula the user is asked (or `--auto-confirm` is honored) whether to launch a substantial atomistic exploration.
4. The wrapper [`explore_composition`](src/matsim_agents/discovery/wrapper.py) then:
   - **enumerates** plausible crystal phases. The selection is
     stoichiometry-aware:
     - 1 element → fcc, bcc, hcp, sc, diamond (and graphene if `--include-2d`)
     - binary 1:1 → rocksalt, CsCl, zincblende, wurtzite, fluorite, rutile (and h-BN if 2-D enabled)
     - binary 1:2 → same bulk set + MoS₂-family 2H/1T monolayers if 2-D enabled
     - ternary 1:1:3 → cubic perovskite
     - ternary 1:2:4 → perovskite + normal spinel
     - quaternary 1:1:2:6 → rocksalt-ordered double perovskite (proper 2×2×2 Fm-3̄m cell)
   - **expands** every prototype into a supercell large enough for
     dopants, AFM ordering, and symmetry-breaking distortions to develop
     (`--min-atoms` auto-tile or explicit `--supercell NxNxN`).
   - **samples site decorations** within that supercell
     (`--n-orderings N`): for multi-species prototypes, generates up to
     `N` symmetrically-distinct cation/anion arrangements (random label
     shuffling, deduplicated with pymatgen's `StructureMatcher`).
     Captures normal vs. (partially) inverse spinel, ordered vs.
     antisite-disordered double perovskite, alloy / solid-solution
     decorations, and antisites in general. Single-element cells
     correctly collapse to one ordering.
   - **sweeps lattice constants** (`--lattice-scales 0.96,1.0,1.04`):
     each ordering is replicated at every isotropic cell-scale factor,
     bracketing the equilibrium volume so the relaxer starts from a
     reasonable basin even when the per-prototype default lattice
     parameter is off.
   - **stacks** 2-D prototypes into multilayers when `--num-layers > 1`,
     with a per-prototype default interlayer separation and a
     configurable vacuum gap.
   - **relaxes** each seed with HydraGNN + ASE (FIRE/BFGS).
   - **scores** chemical stability (ΔE/atom ranking, near-degeneracy
     warning) and a dynamical-stability proxy (max residual force).
5. The summary is fed back into the conversation as a system message so
   the LLM can refine its hypothesis on the next turn.

Output artifacts per composition (under `--output-dir`):

```
outputs/discovery/<formula>/
  seeds/    <formula>_<phase>[_L<n>][_sc<NxNxN>].vasp     # initial structures
  relaxed/  <formula>_<phase>..._optimized_structure.vasp
            <formula>_<phase>..._optimization.traj        # ASE trajectory
            <formula>_<phase>..._optimization.csv         # per-step E, |F|max, branch weights
```

File-name tags reflect the cell that was actually built:
`_L3` = 3 stacked layers (2-D), `_sc2x2x2` = 2×2×2 supercell.

> **Honest caveats.** Phase enumeration is intentionally seed-only (a
> handful of common prototypes) and the dynamical-stability check is a
> force-residual proxy — not a full phonon analysis. Plug in phonopy or
> a richer prototype generator (e.g. `pymatgen.Structure.from_prototype`,
> CALYPSO, USPEX, AIRSS) when the wrapper signature gives you the hook.

---

## Programmatic API

### Single relaxation

```python
from matsim_agents.tools.relaxation import RelaxStructureInput, _run

result = _run(RelaxStructureInput(
    structure_path="structures/mos2.vasp",
    logdir="./multidataset_hpo-BEST6-fp64",
    mlp_checkpoint="./mlp_branch_weights.pt",
    optimizer="FIRE",
    maxiter=200,
))
print(result.final_energy_eV, result.optimized_structure_path)
```

### Composition exploration

```python
from matsim_agents.discovery import explore_composition

# 3-D bulk discovery with a 40-atom minimum cell
result = explore_composition(
    "Cs2AgBiBr6",
    logdir="./multidataset_hpo-BEST6-fp64",
    mlp_checkpoint="./mlp_branch_weights.pt",
    output_dir="./outputs",
    min_atoms=40,
)
print(result.stability.summary)

# 2-D / multilayer discovery (graphene, h-BN, MoS2-family)
result = explore_composition(
    "MoS2",
    logdir="./multidataset_hpo-BEST6-fp64",
    mlp_checkpoint="./mlp_branch_weights.pt",
    output_dir="./outputs",
    include_2d=True,
    num_layers=3,
    vacuum=20.0,
    min_atoms=24,
)
```

### Run the LangGraph workflow

```python
import uuid
from matsim_agents.graph import build_graph
from matsim_agents.state import MatSimState

graph = build_graph()
final = graph.invoke(
    MatSimState(
        objective="Relax structures/foo.vasp and summarize.",
        llm_provider="ollama",
        llm_model="qwen2.5:14b",
    ),
    config={"configurable": {
        "thread_id": str(uuid.uuid4()),
        "logdir": "./multidataset_hpo-BEST6-fp64",
        "mlp_checkpoint": "./mlp_branch_weights.pt",
    }},
)
print(final["analysis"])
```

### Embed the chat loop in your own app

```python
from matsim_agents.chat import DiscoveryChatConfig, DiscoveryChatSession, chat_once

session = DiscoveryChatSession(config=DiscoveryChatConfig(
    logdir="./multidataset_hpo-BEST6-fp64",
    mlp_checkpoint="./mlp_branch_weights.pt",
    output_dir="./outputs",
    llm_model="qwen2.5:14b",
    auto_confirm=True,
))
reply = chat_once(session, "Propose a Pb-free perovskite for PV.")
```

---

## CLI reference

```text
matsim-agents run     OBJECTIVE [options]   # planner -> executor -> analyst
matsim-agents plan    OBJECTIVE             # show the planner's task list
matsim-agents chat    [options]             # interactive discovery REPL
matsim-agents al      run CONFIG.yaml       # active-learning loop (HydraGNN <-> DFT)
matsim-agents al      validate-config CONFIG.yaml   # parse + dump resolved config as JSON
```

Common options (all commands that touch HydraGNN):

| Flag | Description |
|---|---|
| `--logdir PATH` | HydraGNN logdir with `config.json` and checkpoint. |
| `--mlp-checkpoint PATH` | BranchWeightMLP `.pt` file. |
| `--checkpoint NAME` | HydraGNN checkpoint filename or absolute path. |
| `--mlp-device {cuda,cpu}` | Device for the auxiliary MLP. |
| `--precision {fp32,fp64,bf16}` | HydraGNN precision override. |
| `--mlp-precision {fp32,fp64,bf16}` | MLP precision override. |
| `--llm-provider {ollama,vllm,openai,anthropic,huggingface}` | Chat backend. |
| `--llm-model NAME` | Provider-specific model identifier. |
| `--llm-base-url URL` | Override server URL (Ollama / vLLM). |

`chat`-specific:

| Flag | Description |
|---|---|
| `--output-dir PATH` | Where discovery artifacts are written (default `./outputs`). |
| `--optimizer {FIRE,BFGS,BFGSLineSearch}` | ASE optimizer for relaxations. |
| `--maxiter INT` | Max relaxation steps per phase. |
| `--min-atoms INT` | Auto-tile every prototype to at least this many atoms (default `32`). |
| `--supercell NxNxN` | Explicit tiling for every prototype. Overrides `--min-atoms`. For 2-D slabs the z component is forced to 1. |
| `--include-2d / --no-include-2d` | Also enumerate 2-D prototypes (graphene, h-BN, MoS₂-family). Default off. |
| `--num-layers INT` | Number of monolayers stacked for every 2-D prototype (default `1`). |
| `--vacuum FLOAT` | Vacuum gap (Å) along z for 2-D prototypes (default `15.0`). |
| `--interlayer FLOAT` | Override the per-prototype default interlayer separation (Å). |
| `--n-orderings INT` | Sample up to N symmetrically-distinct site decorations per multi-species prototype (default `1`). |
| `--lattice-scales LIST` | Comma-separated isotropic cell-scale factors per ordering, e.g. `0.96,1.0,1.04`. |
| `--ordering-seed INT` | RNG seed for the ordering sampler (reproducibility). |
| `--auto-confirm / --ask` | Skip the y/N prompt for every detected composition. |

---

## Active-learning loop (HydraGNN ↔ DFT)

The `matsim-agents al` subcommand runs an end-to-end active-learning loop
that grows a HydraGNN training set from DFT labels of structures the
current model is most uncertain about. Both **VASP 6.6** and **Quantum
ESPRESSO `pw.x`** are supported as the labeller — the choice is a single
YAML field.

```
  HydraGNN MLFF ── MD ──► candidates ────────────────────────────────────┐
        ▲                       │                                            │
        │                       ▼                                            │
        │             ensemble / MC-dropout                                   │
        │             uncertainty + diversity                                 │
        │                       │                                            │
        │                       ▼                                            │
        │             top-K most informative                                  │
        │                       │                                            │
        │                       ▼                                            │
        │             DFT backend (parallel, in-allocation)                   │
        │             vasp_std  │  pw.x  (one toggle)                         │
        │                       │                                            │
        │                       ▼                                            │
        │             dataset.extxyz / dataset.db  (tagged with backend)      │
        │                       │                                            │
        │                       ▼                                            │
        └─ retrain HydraGNN ── next iteration ─────────────────────────────┘
```

### Quick start

```bash
# 1. Edit the templated example, or override via env vars at runtime
export PROJ_ROOT=$PWD
export RUNS_ROOT=/lustre/orion/<proj>/scratch/$USER/runs
export RUN_TAG=al-mptrj-001
export DFT_BACKEND=qe          # or: vasp

# 2. Validate the resolved config (no run)
matsim-agents al validate-config examples/active_learning/al_config.example.yaml

# 3. Submit on Frontier
sbatch --export=ALL,AL_CONFIG=$PWD/examples/active_learning/al_config.example.yaml \
    -N 64 -t 12:00:00 \
    scripts/launchers/frontier/run-active-learning-frontier.sh
```

### Backend toggle

The example YAML carries both backend sub-blocks; flip `dft.backend:` to
select one. The unused sub-block is ignored.

```yaml
dft:
  backend: ${DFT_BACKEND:-vasp}    # vasp | qe
  vasp:
    vasp_bin: ${VASP_BIN}
    potcar_dir: ${POTCAR_DIR}
    incar_template: ${PROJ_ROOT}/examples/active_learning/INCAR.template
  qe:
    pw_bin: ${PW_BIN}
    pseudo_dir: ${PSEUDO_DIR}
    pw_template: ${PROJ_ROOT}/examples/active_learning/pw.template
```

### Variable substitution in YAMLs

All AL example configs use shell-style placeholders that are expanded
at load time by `ALConfig.from_yaml`:

| Syntax                  | Meaning                                          |
| ----------------------- | ------------------------------------------------ |
| `${VAR}`                | required; raises if unset                        |
| `${VAR:-default}`       | falls back to `default` if unset                 |
| `${VAR:?error message}` | aborts with `error message`                      |

Resolution order: (1) `os.environ`, (2) optional top-level `vars:`
block in the YAML itself. Nested references inside `vars:` resolve
iteratively, so `VASP_BIN: ${PROJ_ROOT}/external/.../vasp_std` just
works. The `vars:` block is consumed before pydantic validation and
never appears in the parsed `ALConfig`.

### Seed sources for MD

`md.seed_source.kind` selects how initial MD structures are obtained:

- `paths` — a curated list of POSCAR / CIF / XYZ files on disk.
- `prompt` — the LLM proposes plausible compositions for a target
  objective (e.g. *“Pb-free halide perovskites for PV”*) and the loop
  materialises seed structures by running the same crystal-prototype
  enumerator used by the discovery wrapper. No curated structure
  collection is required.

### Energy-reference warning

VASP PAW totals and QE pseudopotential totals are **not** directly
comparable. Every frame written to the dataset is tagged with
`info["dft_backend"]`; never train one HydraGNN model on a mixed
VASP+QE dataset without an explicit per-backend energy offset.

Full walkthrough — including templated INCAR / `pw.in` files, in-allocation
launcher details, and per-backend ROCm/MPI gotchas — lives in
[`examples/active_learning/README.md`](examples/active_learning/README.md).

---

## Codabench Competition

The `codabench_competition/` directory contains a fully self-contained
[Codabench](https://www.codabench.org/) challenge called the
**Matsim-Agents Materials Discovery Challenge**.

### What is tested

159 atomistic test structures spanning 11 material classes — 2D monolayers,
intermetallics, BCC/FCC high-entropy alloys, catalysis slabs, critical
minerals, high-entropy ceramics, MAX phases, nuclear oxides, perovskites,
thermoelectrics — each available in ideal, vacancy, antisite, and interstitial
variants. Tasks cover:

| # | Task | Metric |
|---|------|--------|
| 1 | Formation energy prediction | MAE (eV/atom) ↓ |
| 2 | Atomic force prediction | MAE (eV/Å) ↓ |
| 3 | ML structure relaxation | RMSD vs DFT geometry (Å) ↓ |
| 4 | AI-accelerated DFT relaxation | RMSD + energy MAE ↓ |
| 5 | Phase stability ranking | Mean Spearman ρ ↑ |

The overall score is a weighted average mapped to [0, 1]; tasks with no
submission are excluded (not penalised).

### Leaderboard — public / private split

To prevent participants from reverse-engineering the reference labels by
repeatedly probing the leaderboard, the 159 test structures are split into
two partitions:

| Partition | Size | When visible |
|-----------|------|-------------|
| **Public** | 51 structures (~30 %) | Always — during the competition |
| **Private** | 108 structures (~70 %) | Only at competition close (final ranking) |

The split is deterministic and reproducible (SEED=42, stratified by chemical
formula so every formula has ≥ 1 structure in each partition). The
`reference_data/public_ids.txt` and `reference_data/private_ids.txt` files
record which structure IDs belong to each partition.

The scoring program (`scoring_program/score.py`) computes metrics for both
partitions and emits `public_*` and `private_*` keys to `scores.json`. The
Codabench leaderboard is configured to display only `public_*` columns during
the competition. To switch to final ranking, change the key prefix from
`public_` → `private_` in `competition.yaml`.

**Submission rate limit**: 3 submissions per day, enforced via
`max_submissions_per_day: 3` in `competition.yaml`.

### Baselines

Four baselines are provided in `codabench_competition/baselines/`:

| Baseline | Architecture | Source |
|----------|-------------|--------|
| **MACE-MP-0** | Equivariant GNN (MACE) | Universal MLIP (Cambridge) |
| **HydraGNN** | Multi-headed graph NN | This repo / ORNL |
| **UMA** (`uma-s-1p2`) | Transformer-based universal model | Meta / fairchem |
| **AllScAIP** (`allscaip-md-conserving-all-omol`) | Message-passing NN | Meta / OMol25 |

Run any or all baselines:

```bash
cd codabench_competition
python run_baselines.py --model mace        # MACE-MP-0
python run_baselines.py --model hydragnn    # HydraGNN
python run_baselines.py --model uma         # UMA (requires fairchem-core ≥2.20)
python run_baselines.py --model allscaip    # AllScAIP (requires fairchem-core ≥2.20)
python run_baselines.py --model all --relax # all baselines incl. relaxation (Tasks 3/4)
```

UMA and AllScAIP require the `fairchem-core` package and the model checkpoints
(downloaded on first use from HuggingFace — the relevant model cards must be
accepted before use at <https://huggingface.co/facebook/UMA> and
<https://huggingface.co/facebook/OMol25>).

### Directory layout

```
codabench_competition/
├── competition.yaml             # Codabench bundle manifest & leaderboard config
├── run_baselines.py             # entry point: --model mace/hydragnn/uma/allscaip/all
├── evaluate.py                  # local evaluation helper (mirrors the Codabench scorer)
├── requirements.txt             # Python deps for the competition bundle
├── install_mace_aurora.sh       # MACE-MP-0 install helper for Aurora (XPU)
├── fix_h5py_system_conflict_aurora.sh  # h5py/HDF5 conflict workaround (Aurora)
├── baselines/
│   ├── mace_mp0/model.py        # MACE-MP-0 baseline
│   ├── hydragnn/model.py        # HydraGNN baseline
│   ├── uma/model.py             # UMA (fairchem) baseline
│   └── allscaip/model.py        # AllScAIP (fairchem) baseline
├── scoring_program/
│   └── score.py                 # Codabench scorer (public + private partitions)
├── reference_data/
│   ├── public_ids.txt           # 51 structure IDs in the public partition
│   ├── private_ids.txt          # 108 structure IDs in the private partition
│   ├── create_split.py          # reproducible split generator (SEED=42)
│   ├── formation_energies.csv   # DFT reference energies (server-side, not public)
│   ├── elemental_energies.json  # elemental DFT references (published to participants)
│   └── forces/                  # per-structure force arrays (server-side, not public)
├── public_data/
│   ├── generate_structures.py   # generates the 159 test structures
│   ├── structures_metadata.csv  # anonymised MATS-XXXX → class / formula mapping
│   └── structures/              # XYZ files of all test structures
└── starting_kit/
    ├── README.md                # participant guide (tasks, formats, scoring)
    └── MODEL_INTERFACE.md       # how to write a custom MLIP adapter
```

See [`codabench_competition/starting_kit/README.md`](codabench_competition/starting_kit/README.md)
for the full participant guide including submission formats.

---

## Project layout

```
matsim-agents/
├── pyproject.toml
├── docs/
│   ├── hpc-platforms.md                     # single index across Frontier/Aurora/Perlmutter
│   ├── llm-backends-comparison.md           # vLLM vs HF Transformers on ROCm
│   ├── model-download.md                    # HF model download how-to
│   ├── quantum-espresso-frontier.md         # QE GPU build/run on Frontier (MI250X)
│   ├── quantum-espresso-aurora.md           # QE GPU build/run on Aurora (PVC)
│   ├── quantum-espresso-perlmutter.md       # QE GPU build/run on Perlmutter (A100)
│   └── vasp-aurora.md                       # VASP 6.6 makefile lineage on Aurora
├── scripts/
│   ├── setup_env.sh                         # workstation / legacy HPC env install
│   ├── setup/
│   │   ├── frontier/                        # Frontier (OLCF, MI250X) installers
│   │   │   ├── install-rocm72.sh            # vLLM ROCm 7.2 master install
│   │   │   ├── install_matsim_frontier.sh   # matsim-agents env on Frontier
│   │   │   ├── prebuild-tvm-ffi-frontier.sh
│   │   │   ├── build-vllm-rocm72.sh         # vLLM source build
│   │   │   ├── build-qe-cpu-frontier.sh     # Quantum ESPRESSO CPU build
│   │   │   ├── build-qe-gpu-frontier.sh     # Quantum ESPRESSO MI250X build
│   │   │   ├── build-vasp-gpu-frontier.sh   # VASP 6.6 MI250X build
│   │   │   └── frontier-module-stack.sh     # shared module-load helpers
│   │   ├── aurora/                          # Aurora (ALCF, Intel PVC) installers
│   │   │   ├── install_matsim_aurora.sh
│   │   │   ├── setup_matsim_aurora.sh
│   │   │   ├── build-qe-cpu-aurora.sh
│   │   │   ├── build-qe-gpu-aurora.sh       # QE PVC build (oneapi+openmp)
│   │   │   └── build-vasp-gpu-aurora.sh     # VASP 6.6 PVC build (vasp_std/_gam/_ncl)
│   │   └── perlmutter/                      # Perlmutter (NERSC, A100) installers
│   │       ├── install_matsim_perlmutter.sh
│   │       ├── setup_matsim_perlmutter.sh
│   │       ├── build-qe-cpu-perlmutter.sh
│   │       ├── build-qe-gpu-perlmutter.sh   # QE A100 CUDA build
│   │       ├── perlmutter-module-stack.sh
│   │       └── QE-BUILD-GUIDE.md
│   ├── launchers/
│   │   ├── frontier/                        # Frontier sbatch launchers
│   │   │   ├── run-active-learning-frontier.sh  # `matsim-agents al run` driver
│   │   │   ├── _vasp-step-frontier.sh       # in-allocation VASP step
│   │   │   ├── _qe-step-frontier.sh         # in-allocation QE step
│   │   │   ├── _hydragnn-train-step-frontier.sh
│   │   │   ├── run-pw-gpu-frontier.sh       # QE pw.x GPU launcher
│   │   │   ├── run-qe-warmstart-benchmark.sh
│   │   │   ├── launch-test-singlenode-resume-frontier.sh
│   │   │   ├── launch-test-multinode-frontier.sh
│   │   │   └── launch-test-all-models-frontier.sh
│   │   ├── aurora/
│   │   │   └── run-pw-gpu-aurora.sh         # QE pw.x GPU launcher
│   │   └── perlmutter/
│   │       ├── run-pw-gpu-perlmutter.sh
│   │       ├── run-vasp-gpu-perlmutter.sh
│   │       ├── run-qe-warmstart-benchmark-perlmutter.sh
│   │       ├── launch-test-singlenode-resume-perlmutter.sh
│   │       ├── launch-test-multinode-perlmutter.sh
│   │       └── launch-test-all-models-perlmutter.sh
│   ├── smoke-tests/
│   │   ├── frontier/
│   │   │   ├── smoke-vllm-singlenode-frontier.sh
│   │   │   ├── smoke-vllm-multinode-frontier.sh
│   │   │   └── smoke-transformers-frontier.sh
│   │   ├── aurora/
│   │   │   └── smoke-vllm-singlenode-aurora.sh   # vLLM-XPU single-node smoke (qsub)
│   │   └── perlmutter/
│   │       ├── smoke-transformers-perlmutter.sh
│   │       ├── smoke-transformers-multinode-perlmutter.sh
│   │       └── _torchrun_smoke_loader.py
│   ├── advanced/
│   │   ├── frontier/                        # Frontier multi-step sbatch job scripts
│   │   │   ├── job-serve-multinode-frontier.sh
│   │   │   ├── job-discovery-chat-frontier.sh
│   │   │   ├── job-discovery-chat-vllm-frontier.sh
│   │   │   ├── job-single-relaxation-frontier.sh
│   │   │   ├── job-active-learning-uq-frontier.sh
│   │   │   ├── job-qe-warmstart-frontier.sh
│   │   │   ├── job-sequential-benchmark-frontier.sh
│   │   │   └── job-six-model-benchmark-frontier.sh
│   │   ├── aurora/                          # Aurora multi-step qsub job scripts
│   │   │   ├── job-serve-multinode-aurora.sh
│   │   │   ├── job-serve-multinode-vllm-aurora.sh
│   │   │   ├── job-discovery-chat-aurora.sh
│   │   │   ├── job-single-relaxation-aurora.sh
│   │   │   ├── job-active-learning-uq-aurora.sh
│   │   │   ├── job-qe-warmstart-aurora.sh
│   │   │   └── _mpi_xpu_loader.py
│   │   └── perlmutter/                      # Perlmutter multi-step sbatch job scripts
│   │       ├── job-discovery-chat-perlmutter.sh
│   │       ├── job-single-relaxation-perlmutter.sh
│   │       ├── job-active-learning-uq-perlmutter.sh
│   │       └── job-qe-warmstart-perlmutter.sh
│   └── docs/
│       └── frontier/                        # Frontier-specific docs
│           ├── README-frontier.md
│           └── README-six-model-benchmark.md
├── src/matsim_agents/
│   ├── state.py                  # typed shared LangGraph state
│   ├── graph.py                  # planner -> executor -> analyst
│   ├── llm.py                    # Ollama | vLLM | OpenAI | Anthropic | HuggingFace
│   ├── cli.py                    # `matsim-agents run|plan|chat|al`
│   ├── chat.py                   # interactive discovery REPL
│   ├── agents/
│   │   ├── planner.py
│   │   ├── executor.py
│   │   └── analyst.py
│   ├── tools/
│   │   ├── relaxation.py         # HydraGNN + ASE relaxation tool
│   │   ├── qe_relax.py           # Quantum ESPRESSO pw.x relaxer (scf|relax|vc-relax)
│   │   ├── vasp_relax.py         # VASP relaxer (scf|relax|vc-relax|vc-relax-shape)
│   │   ├── warmstart_benchmark_qe.py   # HydraGNN warm-start vs cold-start QE benchmark
│   │   └── warmstart_benchmark_vasp.py # HydraGNN warm-start vs cold-start VASP benchmark
│   └── discovery/
│       ├── composition.py        # formula parsing
│       ├── phase_explorer.py     # crystal-phase seed enumeration
│       ├── stability.py          # ΔE/atom ranking & |F|max proxy
│       └── wrapper.py            # explore_composition()
│   └── active_learning/          # HydraGNN <-> DFT active-learning loop
│       ├── config.py             # pydantic schema + ${VAR} substitution
│       ├── loop.py               # top-level driver (matsim-agents al run)
│       ├── candidates.py         # MD sampling + per-step candidate capture
│       ├── uncertainty.py        # ensemble / MC-dropout scoring + diversity
│       ├── seeds.py              # paths or LLM-prompted seed materialisation
│       ├── trainer.py            # HydraGNN retraining wrapper
│       ├── dft_backend.py        # backend-agnostic Protocol
│       ├── dft_runner.py         # in-allocation parallel job dispatcher
│       ├── vasp_io.py            # POSCAR/INCAR/KPOINTS/POTCAR writers + parser
│       └── backends/
│           ├── vasp.py           # VASP 6.6 single-point labeller
│           └── qe.py             # Quantum ESPRESSO pw.x single-point labeller
├── examples/
│   ├── single_relaxation.py
│   ├── discovery_chat.py
│   └── active_learning/
│       ├── al_config.example.yaml          # unified VASP+QE templated config
│       ├── al_config.prompt.example.yaml   # LLM-seeded variant
│       ├── INCAR.template                  # VASP single-point template
│       ├── pw.template                     # QE pw.in namelist template
│       └── README.md
├── tests/
│   ├── test_state_and_graph.py
│   ├── test_discovery.py
│   ├── test_phase_explorer.py
│   ├── test_al_config.py         # AL config: ${VAR} substitution + validators + legacy shims
│   ├── test_al_uncertainty.py    # acquisition strategies (ensemble / random / FPS)
│   ├── test_al_seeds.py          # seed resolution: paths + LLM-prompted (stubbed)
│   ├── test_vasp_relax.py        # vasp_relax driver + parser
│   └── integration/
│       ├── test_al_loop_dryrun.py    # one full AL iteration, all heavy parts mocked
│       ├── test_qe_warmstart.py      # end-to-end QE warm-start (env-gated)
│       └── test_vasp_warmstart.py    # end-to-end VASP warm-start (env-gated)
├── external/                     # gitignored: large external builds
│   └── quantum-espresso/         # src/, build-gpu/, install-gpu/
└── third_party/HydraGNN/         # cloned by setup_env.sh
```

---

## Configuration reference

### `MatSimState`

| Field | Type | Purpose |
|---|---|---|
| `objective` | `str` | Free-form research goal. |
| `plan` | `list[TaskSpec]` | Tasks emitted by the planner. |
| `pending_tasks` | `list[TaskSpec]` | Queue consumed by the executor. |
| `results` | `list[RelaxationResult]` | Accumulated tool outputs. |
| `analysis` | `str \| None` | Final analyst summary. |
| `iteration` / `max_iterations` | `int` | Executor loop guard. |
| `llm_provider` / `llm_model` / `llm_base_url` | `str \| None` | LLM selection. |

### `TaskSpec`

```python
TaskSpec(
    kind="relax",                  # relax | analyze | report
    structure_path="foo.vasp",
    optimizer="FIRE",              # FIRE | BFGS | BFGSLineSearch
    maxiter=200,
    maxstep=1e-2,
    charge=0.0,
    spin=0.0,
    random_displacement=False,
)
```

### `RelaxStructureInput` / `RelaxationResult`

See [`src/matsim_agents/tools/relaxation.py`](src/matsim_agents/tools/relaxation.py) — fields mirror the
options of the upstream HydraGNN ASE script
(`structure_optimization_ASE.py`).

### Standalone DFT relaxers (outside the AL loop)

For cases where the user wants a *real* DFT relaxation rather than the
cheap HydraGNN one (e.g. validating a discovered structure, refining a
final candidate), two sibling drivers ship under `src/matsim_agents/tools/`
with matching APIs:

| Module | Backend | Calculation modes | Composition-aware defaults |
|---|---|---|---|
| [`qe_relax.py`](src/matsim_agents/tools/qe_relax.py) | Quantum ESPRESSO `pw.x` | `scf`, `relax`, `vc-relax` | `ecutwfc` (SSSP-PBE-eff-1.3 table), smearing, k-mesh |
| [`vasp_relax.py`](src/matsim_agents/tools/vasp_relax.py) | VASP `vasp_std` | `scf`, `relax`, `vc-relax`, `vc-relax-shape` | `ENCUT` = 1.3 × max(ENMAX) from POTCARs (else 520 eV); `ISMEAR/SIGMA/KSPACING` flip metallic vs insulator |

Both follow the same workflow:

```python
from ase.build import bulk
from matsim_agents.tools.vasp_relax import (
    recommend_settings, prepare_relax_workdir, run_vasp,
)

atoms    = bulk("Si")
settings = recommend_settings(atoms, potcar_dir="/path/to/potcars",
                              calculation="vc-relax")
workdir  = prepare_relax_workdir(atoms, "./Si_vcrelax", settings,
                                 potcar_dir="/path/to/potcars")
result   = run_vasp(workdir, launcher_cmd=["bash", "run-vasp-frontier.sh"])
print(result.final_energy_eV, result.n_ionic_steps, result.converged)
```

`qe_relax` has the same shape; both honour an env-overridable launcher
(`MATSIM_QE_LAUNCHER` / `MATSIM_VASP_LAUNCHER`) and parse the per-ionic-step
trajectory + walltime + convergence flag from the native output files
(`pw.out` for QE, `vasprun.xml` + `OUTCAR` for VASP).

> **Note:** the active-learning loop itself never calls these relaxers —
> AL labelling always uses the SCF-only backends under
> [`src/matsim_agents/active_learning/backends/`](src/matsim_agents/active_learning/backends/).
> A relaxation per AL candidate would defeat the point of uncertainty-driven
> sampling. The standalone relaxers are intended for one-off DFT validation
> work outside the AL pipeline.

### HydraGNN warm-start benchmarks

A second pair of sibling drivers wraps the standalone relaxers in a
"cold start vs HydraGNN-warm start" experiment and emits a JSON summary
that the integration tests consume:

| Module | Backend | CLI |
|---|---|---|
| [`warmstart_benchmark_qe.py`](src/matsim_agents/tools/warmstart_benchmark_qe.py)     | Quantum ESPRESSO `pw.x` | `python -m matsim_agents.tools.warmstart_benchmark_qe …`   |
| [`warmstart_benchmark_vasp.py`](src/matsim_agents/tools/warmstart_benchmark_vasp.py) | VASP `vasp_std`         | `python -m matsim_agents.tools.warmstart_benchmark_vasp …` |

Each driver runs (1) HydraGNN ASE relaxation, (2) DFT relaxation from the
*original* coordinates (cold), (3) DFT relaxation from the
HydraGNN-relaxed coordinates (warm), then reports `Δ ionic-steps`,
`Δ total-SCF-iterations`, `Δ energy`, and a `warm_helped` boolean. If
HydraGNN is unavailable (or `--skip-hydragnn` is passed) only the cold
DFT run is executed and the `warm` block is left `None`.

---

## Current capabilities and planned work

This section spells out what the framework does *today* and what is on
the roadmap but **not yet implemented**, so users know what to expect
before building a workflow on top of it.

### Available today

- **Single-point energies and forces** from a HydraGNN MLFF checkpoint
  through an ASE calculator interface.
- **Geometry relaxation** of atoms and (optionally) cell, driven by
  HydraGNN through the upstream `structure_optimization_ASE.py` wrapper.
- **Isotropic lattice scans** to locate equilibrium volume / lattice
  constant.
- **Random-shuffle ordering enumeration** for disordered sites,
  deduplicated with pymatgen's `StructureMatcher`.
- **AA-stacked 2-D multilayer** construction.
- **Relative chemical-stability scoring** (energy-above-hull style
  comparisons within the explored phase set).
- **LLM-driven planner / executor / reporter** agents (LangGraph) with
  optional human-in-the-loop gates.
- **Pluggable LLM backends**: vLLM (Frontier ROCm), Hugging Face
  Transformers, and OpenAI-compatible HTTP endpoints.
- **Active-learning loop** with HydraGNN as the surrogate and either
  VASP 6.6 or Quantum ESPRESSO `pw.x` as the DFT labeller, selectable
  via a single `dft.backend:` YAML field. Includes ensemble /
  MC-dropout uncertainty scoring, in-allocation parallel DFT dispatch,
  templated INCAR / `pw.in` inputs, and shell-style `${VAR}` /
  `${VAR:-default}` substitution in all YAML configs.
- **LLM-generated MD seeds** as a first-class seed source
  (`md.seed_source.kind: prompt`).

### Not yet implemented (roadmap)

- [ ] **Phonon-based dynamical stability** (phonopy / finite differences).
- [ ] **Formation-energy reference set** for *absolute* (not relative)
      chemical-stability scoring.
- [ ] **Richer phase enumeration** via pymatgen prototypes / CALYPSO /
      USPEX hooks.
- [ ] **Symmetry-aware ordering enumeration** via `enumlib` (today's
      enumerator is random-shuffle + `StructureMatcher` dedup).
- [ ] **Anisotropic / per-axis lattice scans** (today's scan is
      isotropic only).
- [ ] **AB / AA' stacking** for 2-D multilayers (today's builder is
      AA-stacked only).
- [ ] **2-D heterostructures** (e.g. graphene/h-BN, MoS₂/WSe₂) with
      lattice-mismatch search.
- [ ] **MD agent**: NVT/NPT runs with the same HydraGNN calculator.
- [ ] **MCP tool server** so external clients (Claude Desktop, IDE
      agents) can call the discovery wrapper directly.
- [ ] **Distributed executor** for parallel composition exploration on
      HPC.
- [ ] **Pluggable MLFF backends** (MACE, NequIP, Orb) behind the same
      calculator interface.

---

## Contributing

1. Fork and create a feature branch.
2. `pip install -e .[dev]`
3. `pytest` and `ruff check .` before pushing.
4. Open a merge request on
   [code.ornl.gov/multi-agentic-ai-materials/matsim-agents](https://code.ornl.gov/multi-agentic-ai-materials/matsim-agents).

---

## License & citation

Released under the **BSD 3-Clause License** (see [LICENSE](LICENSE)).

If you use `matsim-agents` in academic work, please cite both this
repository and HydraGNN:

> *HydraGNN: Distributed PyTorch implementation of multi-headed graph
> convolutional neural networks*, Copyright ID #81929619,
> <https://doi.org/10.11578/dc.20211019.2>

---

*Maintained by the ORNL Multi-Agentic AI for Materials team.*
