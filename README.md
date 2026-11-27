# matsim-agents

<img src="assets/matsim-agents-logo-wide.png" alt="MATSIM-AGENTS logo" width="800">

| Logo mark | Repository QR code |
|:--:|:--:|
| <img src="assets/matsim-agents-logo-mark.png" alt="MATSIM-AGENTS symbol" width="180"> | [<img src="matsim-agents-qr.png" alt="QR code linking to the matsim-agents GitHub repository" width="180">](matsim-agents-qr.png) |

**Multi-agent AI framework for atomistic materials simulation and discovery.**

`matsim-agents` orchestrates large language models, machine-learned
interatomic potentials, and ASE-based atomistic workflows into a single
agentic loop. The user states a research objective in natural language;
agents plan, run MLIP-driven simulations, score chemical and
dynamical stability, and report the findings. By default, chat asks before
each exploration (`--auto-confirm` skips the prompt). Explicit approvals gate
DFT through `matsim-agents run relax.yaml`, elemental-reference DFT,
and model promotion; active-learning DFT labeling runs without a separate
approval once `al run` is started.

The framework is **backend-agnostic**: HydraGNN, UMA, and the open MACE
foundation-model families are selectable MLIP backends. The relaxation tool,
crystal-phase generator, and stability scorer use shared interfaces so further
potentials can be added without changing orchestration logic.

---

## Start with a small phase search

For a bounded phase search with any supported MLIP, use one YAML file with
`matsim-agents explore check`, `explore run`, and `explore summary`.
Every run supports resume. Use `iterations: 3` for multiple rounds and `selection: diverse`
to skip equivalent structures and prioritize structural diversity.
See the [step-by-step guide](docs/workflows/simple-exploration.md) and
[Nb2O5 example](examples/workflows/explore-nb2o5.yaml).
This path needs no custom Python driver and never launches LLM, DFT, or
active-learning work. Multi-round LLM steering is not part of this first interface.

For a common interface across advanced workflows, start with
[run a workflow](docs/workflows/run-campaigns.md). Campaigns such as the fusion
BCC 1024-site study are applications in [examples/](examples/): each declares
an `application.yaml` loaded by the framework.
Then [interpret results](docs/workflows/interpret-results.md) or
[extend the library](docs/workflows/extend-library.md). The interface provides
check/run/status/summary and capability-based native resume; scheduler submission
and scientific approvals remain explicit.

## Documentation

Start with the [documentation index](docs/index.md), which is organized by task.
Reference pages:

- [Command-line guide](docs/guide/cli.md)
- [Architecture and capabilities](docs/guide/architecture.md)
- [Choosing a scientific workflow](docs/guide/workflows.md)
- [Running on DOE supercomputers](docs/guide/hpc.md)
- [LLM backends](docs/guide/llm-backends.md)
- [External MCP tools](docs/mcp.md)
- [Hypothesis-driven discovery chat](docs/guide/discovery-chat.md)
- [Python API](docs/guide/python-api.md)
- [Active learning (MLIP and DFT)](docs/guide/active-learning.md)
- [Codabench competition](docs/guide/codabench.md)
- [Configuration reference](docs/guide/configuration.md)

---

## Installation

matsim-agents requires Python 3.11–3.14.

```bash
git clone https://github.com/ORNL/matsim-agents.git
cd matsim-agents
```

### Lightweight install (one MLIP, no HydraGNN)

For a workstation or a first trial, install the package with one MLIP backend
and one LLM provider:

```bash
python -m venv .venv-mace
source .venv-mace/bin/activate
pip install -e ".[mace,ollama]"
matsim-agents doctor
```

| Extras | Purpose |
|---|---|
| `mace`, `uma`, `hydragnn` | MLIP backends. HydraGNN itself and its compiled PyTorch Geometric extensions are installed separately (below). |
| `ollama`, `vllm`, `openai`, `anthropic`, `huggingface`, `mag`, `copilot` | LLM providers; see [LLM backends](docs/guide/llm-backends.md). |
| `mcp` | Client for external MCP servers used by `matsim-agents objective --mcp-config`. |
| `dev` | Test, lint, and type-check tools. |

MACE, UMA, and HydraGNN require mutually incompatible `e3nn` versions, so
install each MLIP backend in its own environment. `matsim-agents doctor` reports
which backends the active environment can use without loading a model.

### HydraGNN and HPC installers

The following installers prepare HydraGNN and its PyTorch/PyTorch Geometric
stack through HydraGNN's official scripts:

```bash
# Local workstation (CPU or single GPU)
./scripts/setup_env.sh

# Frontier (OLCF), Aurora (ALCF), or Perlmutter (NERSC)
bash deployments/frontier/setup/install.sh
bash deployments/aurora/setup/install.sh
bash deployments/perlmutter/setup/install.sh
```

The generic `scripts/setup_env.sh` remains the workstation implementation. If
called with `PLATFORM=frontier`, `PLATFORM=aurora`, or
`PLATFORM=perlmutter`, it delegates immediately to the corresponding canonical
facility `install.sh` above. Each facility installer runs HydraGNN's current
official HPC installer first and leaves HydraGNN and matsim-agents in one
verified environment. Facility toolchains, including the Frontier ROCm version
matrix, are described in [Running on DOE supercomputers](docs/guide/hpc.md).

Environment overrides accepted by the installers:

| Variable | Purpose | Default |
|---|---|---|
| `PYTHON` | Python interpreter | `python3` |
| `HYDRAGNN_REPO` | HydraGNN git URL | `https://github.com/ORNL/HydraGNN.git` |
| `HYDRAGNN_REF` | Branch/tag/commit | `main` |
| `HYDRAGNN_DIR` | Reuse an existing HydraGNN checkout | sibling `HydraGNN` checkout |
| `INSTALL_ROOT` | Compiled HydraGNN dependency build root | `.hpc-build/<facility>` |
| `VENV_PATH` | Override the matsim-owned Python environment | `.venv` |
| `MATSIM_EXTRAS` | matsim-agents extras installed in the target environment | `hydragnn,dev,openai,ollama,anthropic,huggingface` |
| `INSTALL_UMA` | Create and verify the matsim-owned UMA compatibility environment (HPC installers; `0` skips) | `1` |
| `UMA_VENV_PATH` | Override the UMA compatibility environment | `.venv-uma` |
| `INSTALL_MACE` | Create and verify the matsim-owned MACE compatibility environment (HPC installers; `0` skips) | `1` |
| `MACE_VENV_PATH` | Override the MACE compatibility environment | `.venv-mace` |
| `BOOTSTRAP_OLLAMA` | Set to `1` to install the Ollama daemon, start it, and pull `OLLAMA_MODELS` (workstation only) | `0` |
| `OLLAMA_MODELS` | Space-separated list of models to pull when `BOOTSTRAP_OLLAMA=1` | `qwen2.5:14b` |

After the script finishes:

```bash
source .venv/bin/activate    # workstation case
matsim-agents --help
```

The HPC installers put HydraGNN in `.venv` and, by default, create separate
compatibility environments `.venv-uma` for FairChem/UMA and `.venv-mace` for
MACE. Their `e3nn` requirements conflict with HydraGNN's. Select the prepared
environment for the backend you run; `INSTALL_UMA=0` or `INSTALL_MACE=0` skips
the corresponding HPC environment. On Aurora, `RUN_MODEL_TESTS=1` enables
both environments regardless of those settings.

To bootstrap the local Ollama daemon and pull a model in one go:

```bash
BOOTSTRAP_OLLAMA=1 OLLAMA_MODELS="qwen2.5:14b llama3.1:8b" \
    ./scripts/setup_env.sh
```

---

## Quick start

Every command that runs an MLIP takes `--mlip-config FILE`, a YAML file whose
`mlip:` block selects the backend (`hydragnn`, `uma`, or `mace`) and its
settings. Start from a ready-made file in [examples/mlip/](examples/mlip/), for
example the MACE-MP-0 model installed by the lightweight install above:

```bash
cp examples/mlip/mace-mp-medium.yaml mlip.yaml
# Edit mlip.yaml: set device to cuda, xpu, or cpu for this machine.
```

`examples/mlip/hydragnn-gfm.yaml` selects the HydraGNN foundation model from a
HydraGNN checkout named by `HYDRAGNN_DIR`; `${VAR}` references are expanded
when the file is loaded. Each backend's fields and how its relative paths
resolve are listed in the [configuration reference](docs/guide/configuration.md#mlip-selection).

### 1. Run the agent graph end-to-end

```bash
matsim-agents objective \
  "Relax examples/characterization/structures/MgO-rocksalt.vasp and report the final energy." \
  --mlip-config mlip.yaml \
  --llm-provider ollama --llm-model qwen2.5:14b
```

### 2. Hypothesis-generation chat with auto-triggered exploration

```bash
ollama pull qwen2.5:14b

matsim-agents chat \
  --mlip-config mlip.yaml \
  --n-random 50 --random-seed 0
```

A typical session (abridged; energies and paths are illustrative):

```
you> I want a Pb-free halide double perovskite for photovoltaics with band gap near 1.5 eV.

assistant> A promising candidate is Cs2AgBiBr6 ...

Proposed composition detected: AgBiBr6Cs2. Run surrogate-based phase exploration? [y/N]: y

>>> Exploring composition AgBiBr6Cs2
  starting  <prototype id>            .../AgBiBr6Cs2_<prototype id>.vasp
  converged <prototype id>            E=-365.4123 eV  |F|max=0.0118 eV/Å  steps=112
  ...

Stability report for AgBiBr6Cs2:
Composition AgBiBr6Cs2: 52 candidate seed(s) relaxed (2 prototype + 50 random).
Predicted ground state: .../AgBiBr6Cs2_<prototype id>_optimized_structure.vasp (E/atom = -9.1353 eV, |F|max = 0.0118 eV/Å, dynamically_stable_proxy = True, source = prototype, ...).
Chemical-stability proxy: PASS.

you> Now suggest a Sb-substituted variant.
```

### 3. Discovery followed by active learning

```bash
matsim-agents objective "Find stable phases of Li2MnO3" --mlip-config mlip.yaml
matsim-agents al run examples/active_learning/al_config.example.yaml
```

In `matsim-agents chat`, the `/al [composition]` command starts the same loop
for a composition (`--al-config`, plus `--al-dry-run` or `--al-run`).

### 4. Random-enriched exploration of exotic compositions

To request more pyXtal structures, including for compositions that have no
AFLOW prototype match:

```bash
matsim-agents chat \
  --mlip-config mlip.yaml \
  --n-random 200 --random-seed 42
```

This chat command still generates every applicable AFLOW prototype. If none
matches, only the pyXtal pass contributes seeds; random candidates are flagged
`(novel)`. Generation and deduplication can yield fewer than requested. For a
strictly pyXtal-only search, use the [bounded exploration guide](docs/workflows/simple-exploration.md)
with `generation.max_aflow_candidates: 0` and a positive `pyxtal_count`.

---

## CLI reference

```text
matsim-agents doctor [--require BACKEND]   # Python/backend preflight without loading models
matsim-agents config list                  # configuration kinds, templates, and consuming commands
matsim-agents config init KIND OUTPUT.yaml # copy an annotated starter config; never overwrites
matsim-agents config schema KIND [-o FILE] # JSON Schema for editor validation and completion
matsim-agents tools list                   # training, dataset, and benchmark tools in the package
matsim-agents tools run NAME [ARGS...]     # run a tool; its own --help lists its options
matsim-agents validate FILE                # check any config without running it (exit 2 if invalid)
matsim-agents plan     FILE                # show what `run` would do and what blocks it
matsim-agents run      FILE [--runs-dir D] # execute any config; blocked runs exit 3 with reasons
matsim-agents resume   RUN_ID              # continue an interrupted run or retry a blocked one
matsim-agents status   [RUN_ID]            # one run's state, or list recorded runs
matsim-agents evidence RUN_ID              # a run's record, summary, blockers and native directory
matsim-agents objective "TEXT" [options]   # planner -> executor -> analyst (--plan-only)
matsim-agents chat    [options]             # interactive discovery REPL
matsim-agents explore check CONFIG.yaml    # static preflight for a bounded MLIP phase search
matsim-agents explore run CONFIG.yaml      # generate, select, relax, and rank MLIP-only batches
matsim-agents explore summary RUN_DIRECTORY # inspect counts, termination reasons, and errors
matsim-agents explore status RUN_DIRECTORY # last saved checkpoint progress; no scheduler query
matsim-agents explore resume RUN_DIRECTORY # resume checkpointed exploration (one or more rounds)
matsim-agents ledger  verify LEDGER.jsonl    # check the evidence ledger's hash chain and artifacts
matsim-agents ledger  show LEDGER.jsonl      # evidence counts and current hypothesis revisions
matsim-agents benchmark known-answers [--tier replay|live]  # grade the Si/SiC known-answer suite
matsim-agents benchmark agent-eval CONFIG.yaml [--case ID]  # compare LLM agent arms on graded claims
matsim-agents neb prepare|validate|inspect|resume ...  # QE NEB for defect migration; dry run unless --dispatch
matsim-agents external spectra-pka check|prepare|run|status|import ...  # optional interface to externally installed SPECTRA-PKA; dry run by default
matsim-agents external lammps check  # report the LAMMPS executable, version and packages (docs/external/lammps.md)
matsim-agents capabilities check [--kind K] [--require NAME]  # which backends can run here
matsim-agents capabilities matrix [-o FILE] # backend capability matrix (docs/capabilities.md)
matsim-agents coordination list [--repo PATH] [--remote NAME]  # read shared HPC task board
matsim-agents coordination init|add|claim|heartbeat|complete|fail ...  # manage tasks; see docs/hpc-task-coordination.md
matsim-agents inventory dft ROOT [--output FILE]  # inventory retained DFT run artifacts
matsim-agents al      run CONFIG.yaml [--mlip-config FILE]  # active-learning loop (MLIP <-> DFT)
matsim-agents al      validate-config CONFIG.yaml   # parse + dump resolved config as JSON
matsim-agents active-learning run|validate-config ...  # same commands as `al`
matsim-agents magnetic generate|policies|show ...  # collinear magnetic starting states for QE
matsim-agents resources plan CONFIG.yaml [--nodes N]   # preview LLM/MLIP/DFT placement
matsim-agents resources validate CONFIG.yaml           # exit non-zero on conflicts
matsim-agents resources serve CONFIG.yaml -- CMD ...    # run CMD while the plan's LLM servers are up
```

Use `matsim-agents --help` (or `-h`) for command details and examples, and
`matsim-agents --version` to report the installed package version. Put the
global `--debug` option before a command to show a traceback for diagnosis.
Run `matsim-agents --install-completion` once to enable tab-completion in your shell.

Exit codes are the same for every command: `0` success, `1` the command ran but
the scientific work, a preflight check or a required backend failed, and `2` a
usage or configuration error (bad option, unreadable or invalid file).

Options for `run` and `chat`, starter configurations,
diagnostics, and package tools are described in the
[command-line guide](docs/guide/cli.md). The fields of each backend's `mlip:`
block, including HydraGNN decoder-head selection, are listed in the
[configuration reference](docs/guide/configuration.md#mlip-selection).

---

## Project layout

The concise, responsibility-oriented package map is maintained in
[docs/architecture.md](docs/architecture.md). Machine assets live under
`deployments/<facility>/{setup,launchers,jobs,smoke-tests}`. The compact tree
below shows the current ownership model; use the per-facility READMEs for full
file inventories. The cross-facility release gate lives under
`benchmarks/portability/`.

```text
matsim-agents/
├── src/matsim_agents/     # library code and stable public interfaces
│   ├── orchestration/     # agent graphs, state, and routing policies
│   ├── workflows/         # composable scientific workflow contracts
│   ├── backends/          # LLM, MLIP, and DFT adapters
│   ├── execution/         # resources, allocation, provenance, and run directories
│   ├── discovery/         # composition parsing, seed generation, and stability
│   └── active_learning/   # acquisition, labeling, training, and promotion
├── tests/                 # unit, integration, smoke, and documentation contracts
├── examples/              # runnable configuration and workflow examples
├── deployments/           # facility-specific setup, launchers, jobs, and smoke tests
├── benchmarks/            # portability and Codabench workflows
├── docs/                  # architecture, scientific, and deployment guides
└── scripts/               # setup, diagnostics, validation, and reporting utilities
```

---

## Contributing

1. Fork the repository or create a feature branch.
2. Install the development tools: `pip install -e ".[dev,mcp]"`.
3. Run `bash scripts/check.sh` (lint, documentation and deployment checks, and
   non-GPU tests) before pushing.
4. Open a pull request against `main` on
   [GitHub](https://github.com/ORNL/matsim-agents).

The [contributing guide](.github/wiki/Contributing.md) covers code style, test
tiers, and review rules; [CI/CD](.github/wiki/CI-CD.md) describes the required
checks.

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
