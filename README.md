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
dynamical stability, and report the findings — with optional human
review at every gate.

The framework is **backend-agnostic**: HydraGNN, UMA, and the open MACE
foundation-model families are selectable MLIP backends. The relaxation tool,
crystal-phase generator, and stability scorer use shared interfaces so further
potentials can be added without changing orchestration logic.

---

## Start with a small phase search

For a bounded MACE-only phase search, use one YAML file with
`matsim-agents explore check`, `explore run`, and `explore summary`.
Every run supports resume. Use `iterations: 3` for multiple rounds and `selection: diverse`
to skip equivalent structures and prioritize structural diversity.
See the [step-by-step guide](docs/workflows/simple-exploration.md) and
[Nb2O5 example](examples/workflows/explore-nb2o5.yaml).
This path needs no custom Python driver and never launches LLM, DFT, or
active-learning work. Multi-round LLM steering is not part of this first interface.

For the frozen E3-MLIP/E3-DFT presets and a common interface across advanced workflows,
start with [run a campaign](docs/workflows/run-campaigns.md).
Then [interpret results](docs/workflows/interpret-results.md) or
[extend the library](docs/workflows/extend-library.md). The interface provides
check/run/status/summary and capability-based native resume; scheduler submission
and scientific approvals remain explicit.

## Documentation

Start with the [documentation index](docs/index.md), which is organized by task.
Reference pages:

- [Architecture and capabilities](docs/guide/architecture.md)
- [Choosing a scientific workflow](docs/guide/workflows.md)
- [Running on DOE supercomputers](docs/guide/hpc.md)
- [LLM backends](docs/guide/llm-backends.md)
- [Hypothesis-driven discovery chat](docs/guide/discovery-chat.md)
- [Python API](docs/guide/python-api.md)
- [Active learning (HydraGNN and DFT)](docs/guide/active-learning.md)
- [Codabench competition](docs/guide/codabench.md)
- [Configuration reference](docs/guide/configuration.md)

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
verified environment.

### ROCm version matrix on Frontier

The three Frontier-targeted backends in this repo do **not** all use the
same ROCm version. The combinations below are what is actually wired up
in the scripts and what you should expect at runtime:

| Backend | Module | Why this version |
|---|---|---|
| **HydraGNN Conda environment** | `rocm/7.2.0` | Current Frontier-supported PyTorch + ROCm path; created by HydraGNN's installer at `matsim-agents/.venv/` |
| **vLLM model server** | `rocm/7.2.0` | Isolated Python `venv` at `matsim-agents/.venv-vllm/`; built from source via [`deployments/frontier/setup/build-vllm-rocm72.sh`](deployments/frontier/setup/build-vllm-rocm72.sh) |
| **Quantum ESPRESSO GPU** | `rocm/6.2.4` *(forced)* | Frontier's `cray-mpich/8.1.31` GTL `libmpi_gtl_hsa.so` is hard-linked against `libamdhip64.so.6` (rocm 6.x SONAME). rocm/7.x ships `.so.7` and breaks the MPI Fortran link probe at CMake configure. Pin documented in [`docs/quantum-espresso-frontier.md`](docs/quantum-espresso-frontier.md). |

QE and the Python/ML stacks are deliberately never co-loaded in the same
shell; they couple through Slurm + the filesystem.

Environment overrides accepted by the installer:

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

HydraGNN and optional FairChem/UMA share `.venv`. Current upstream
`mace-torch==0.3.16` has a hard `e3nn==0.4.4` requirement, which conflicts with
HydraGNN's `e3nn==0.5.1`. Requesting `INSTALL_MACE=1` therefore creates
`.venv-mace` inside the matsim-agents checkout. It inherits the machine's
PyTorch stack from `.venv` without modifying `.venv`; MACE jobs run in a
separate Python process.

To bootstrap the local Ollama daemon and pull a model in one go:

```bash
BOOTSTRAP_OLLAMA=1 OLLAMA_MODELS="qwen2.5:14b llama3.1:8b" \
    ./scripts/setup_env.sh
```

---

## Quick start

Every command that runs an MLIP takes `--mlip-config FILE`, a YAML file whose
`mlip:` block selects the backend (`hydragnn`, `uma`, or `mace`) and its
settings. `examples/mlip/` has ready-made files for UMA, MACE, and HydraGNN; an
active-learning config works too. The examples below use a HydraGNN
foundation model:

```yaml
# mlip.yaml (relative paths resolve against this file)
mlip:
  backend: hydragnn
  logdir: ./multidataset_hpo-BEST6-fp64
  mlp_checkpoint: ./mlp_branch_weights.pt
```

### 1. Run the agent graph end-to-end

```bash
matsim-agents run \
  "Relax structures/mos2-B_Defect-Free_PBE.vasp and report the final energy." \
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

### 3. Supervisor orchestration (discovery -> UQ -> optional AL handoff)

```bash
matsim-agents supervisor-run Li2MnO3 \
  --mlip-config mlip.yaml \
  --al-config examples/active_learning/al_config.example.yaml \
  --al-dry-run
```

To execute AL instead of dry-run, replace `--al-dry-run` with `--al-run`.

### 3. Novelty-only exploration of exotic compositions

For compositions that have no AFLOW prototype match (e.g. 5-element
high-entropy alloys), disable the prototype branch entirely and let
pyXtal characterize the configuration space:

```bash
matsim-agents chat \
  --mlip-config mlip.yaml \
  --n-random 200 --random-seed 42
```

When the conversation introduces an unusual stoichiometry, the discovery
wrapper will report `No AFLOW prototype match` and rely on the pyXtal
pass; the resulting candidates are flagged `(novel)` in the stability
table.

---

## CLI reference

```text
matsim-agents doctor [--require BACKEND]   # Python/backend preflight without loading models
matsim-agents config list                  # configuration kinds, templates, and consuming commands
matsim-agents config init KIND OUTPUT.yaml # copy an annotated starter config; never overwrites
matsim-agents config schema KIND [-o FILE] # JSON Schema for editor validation and completion
matsim-agents tools list                   # training, dataset, and benchmark tools in the package
matsim-agents tools run NAME [ARGS...]     # run a tool; its own --help lists its options
matsim-agents run     OBJECTIVE [options]   # planner -> executor -> uq_gate -> analyst
matsim-agents plan    OBJECTIVE             # show the planner's task list
matsim-agents chat    [options]             # interactive discovery REPL
matsim-agents supervisor-run COMPOSITION [options]  # discovery -> UQ -> optional AL handoff
matsim-agents relax   CONFIG.yaml           # provenance-tracked MLIP/DFT relaxation
matsim-agents characterize CONTRACT.yaml   # answer a characterization contract (resumable)
matsim-agents explore check CONFIG.yaml    # static preflight for a bounded MACE phase search
matsim-agents explore run CONFIG.yaml      # generate, select, relax, and rank MLIP-only batches
matsim-agents explore summary RUN_DIRECTORY # inspect counts, termination reasons, and errors
matsim-agents explore status RUN_DIRECTORY # last saved checkpoint progress; no scheduler query
matsim-agents explore resume RUN_DIRECTORY # resume checkpointed exploration (one or more rounds)
matsim-agents workflow list                # advanced workflows and native restart capabilities
matsim-agents workflow init PRESET OUTPUT.yaml # annotated E3/exploration/debate presets; no execution
matsim-agents workflow check CONFIG.yaml   # validate wrapper/native config without scientific execution
matsim-agents workflow plan CONFIG.yaml    # validate and show the exact command without execution
matsim-agents workflow run CONFIG.yaml [--dry-run] # preview or invoke; E3 execution requires approval
matsim-agents workflow status RUN_DIRECTORY # last saved invocation status, not scheduler liveness
matsim-agents workflow summary RUN_DIRECTORY # human report and links to native evidence
matsim-agents workflow resume RUN_DIRECTORY [--native-run DIR] # native restart where supported
matsim-agents ledger  verify LEDGER.jsonl    # check the evidence ledger's hash chain and artifacts
matsim-agents ledger  show LEDGER.jsonl      # evidence counts and current hypothesis revisions
matsim-agents benchmark known-answers [--tier replay|live]  # grade the Si/SiC known-answer suite
matsim-agents capabilities check [--kind K] [--require NAME]  # which backends can run here
matsim-agents capabilities matrix [-o FILE] # backend capability matrix (docs/capabilities.md)
matsim-agents debate  CONFIG.yaml           # persisted multi-model scientific debate
matsim-agents llm-check CONFIG.yaml         # qualify an LLM deployment
matsim-agents e1-development MANIFEST.json --output DIR [--max-new-requests N] [--retry-errors]  # run one candidate's E1 development questions; held-out questions are not accepted
matsim-agents coordination list [--repo PATH] [--remote NAME]  # read shared HPC task board
matsim-agents inventory dft ROOT [--output FILE]  # inventory retained DFT run artifacts
matsim-agents al      run CONFIG.yaml [--mlip-config FILE]  # active-learning loop (MLIP <-> DFT)
matsim-agents al      validate-config CONFIG.yaml   # parse + dump resolved config as JSON
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

Common options (all commands that run an MLIP):

| Flag | Description |
|---|---|
| `--mlip-config FILE` | YAML with an `mlip:` block: `backend` plus that backend's settings. |
| `--llm-provider {ollama,vllm,openai,anthropic,huggingface}` | Chat backend. |
| `--llm-model NAME` | Provider-specific model identifier. |
| `--llm-base-url URL` | Override server URL (Ollama / vLLM). |

The `mlip:` block is flat; each backend accepts only its own fields:

| Backend | Fields |
|---|---|
| `hydragnn` | `logdir` (with `config.json` and checkpoint), `checkpoint`, `mlp_checkpoint` (BranchWeightMLP `.pt`), `inference_head`, `precision`, `mlp_precision`, `hydragnn_root`, `newhead_ft_config`, `ft_repo`, `radius`, `max_neighbours`, `charge`, `spin` |
| `uma` | `model_name`, `task_name` (`omat`, `omol`, `oc20`, `odac`, `omc`), `precision`, `charge`, `spin`, `dropout` |
| `mace` | `family`, `model`, `head`, `precision`, `dispersion`, `dropout` |
| all | `device` (`cuda`, `cpu`, `xpu`), `ensemble_models` |

HydraGNN uses fused 16-head inference by default, which requires
`mlp_checkpoint`. To pin one head for MLIP inference, set `inference_head` in the
`mlip:` block to a case-insensitive dataset name or its index, for example
`inference_head: OMat24`. Pinned mode invokes only that decoder, does not require
the branch MLP, and does not report branch-weight uncertainty. It preserves
the loader's precision/autocast context, including `bf16`, during energy and
force inference.

| Index | Dataset | Index | Dataset |
|---:|---|---:|---|
| 0 | Alexandria | 8 | OMol25 |
| 1 | ANI1x | 9 | OMol25-neutral |
| 2 | MPTrj | 10 | OMol25-non-neutral |
| 3 | OC2020 | 11 | OPoly2026 |
| 4 | OC2022 | 12 | Nabla2DFT |
| 5 | OC25 | 13 | QCML |
| 6 | ODAC23 | 14 | QM7X |
| 7 | OMat24 | 15 | transition1x |

`chat`-specific:

| Flag | Description |
|---|---|
| `--output-dir PATH` | Where discovery artifacts are written (default `./outputs`). |
| `--ase-structure-optimizer {FIRE,BFGS,BFGSLineSearch}` | ASE optimizer for relaxations. |
| `--maxiter INT` | Max relaxation steps per seed (default `200`). |
| `--fmax FLOAT` | Stop relaxation when max residual force is below this (eV/Å, default `0.02`). |
| `--n-random INT` | Number of supplementary pyXtal random structures per composition, in addition to every applicable AFLOW prototype decoration (default `50`). Set to `0` to disable the pyXtal pass; silently degrades to `0` if pyXtal is not installed. |
| `--random-seed INT` | RNG seed for the pyXtal sampler (reproducibility). |
| `--auto-confirm / --ask` | Skip the y/N prompt for every detected composition. |
| `--trigger-al-handoff / --no-trigger-al-handoff` | Enable or disable UQ-driven escalation to active learning. |
| `--al-config PATH` | Base AL YAML used when handoff is triggered. |
| `--al-dry-run / --al-run` | Plan/report AL handoff only, or execute AL loop. |
| `--uq-top-weight-threshold FLOAT` | Trigger handoff when mean top branch weight is below this value. |
| `--uq-min-unreliable-fraction FLOAT` | Trigger handoff when the low-confidence fraction exceeds this value. |
| `--uq-min-relaxations-for-handoff INT` | Minimum number of relaxations before evaluating handoff policy. |
| `--al-handoff-audit-path PATH` | Optional JSONL path for UQ and handoff audit artifacts. |
| `--resume ID_OR_PATH` | Continue a saved session (its id under `<output-dir>/chat/`, or its directory). |

`run`-specific:

| Flag | Description |
|---|---|
| `OBJECTIVE` | Natural-language task objective for planner/executor. |
| `--max-iterations INT` | Maximum executor iterations before forcing analysis. |
| `--trigger-al-handoff / --no-trigger-al-handoff` | Enable or disable UQ-driven AL escalation after run relaxations. |
| `--al-config PATH` | Base AL YAML used when run-path handoff is triggered. |
| `--al-dry-run / --al-run` | Plan/report run->AL handoff only, or execute AL loop. |
| `--uq-top-weight-threshold FLOAT` | Trigger handoff when mean top branch weight is below this value. |
| `--uq-min-unreliable-fraction FLOAT` | Trigger handoff when low-confidence fraction exceeds this value. |
| `--uq-min-relaxations-for-handoff INT` | Minimum relaxations before evaluating run-path handoff policy. |
| `--al-handoff-audit-path PATH` | Optional JSONL path for UQ and run->AL handoff audit artifacts. |

`supervisor-run`-specific:

| Flag | Description |
|---|---|
| `COMPOSITION` | Target composition for one supervisor pass (e.g. `Li2MnO3`). |
| `--trigger-al-handoff / --no-trigger-al-handoff` | Enable or disable UQ-driven AL handoff policy. |
| `--al-config PATH` | Base AL YAML used for optional handoff execution. |
| `--al-dry-run / --al-run` | Dry-run handoff planning or real AL execution. |
| `--uq-top-weight-threshold FLOAT` | UQ threshold on mean top branch weight. |
| `--uq-min-unreliable-fraction FLOAT` | UQ threshold on low-confidence fraction. |
| `--uq-min-relaxations-for-handoff INT` | Min relaxations required before handoff is considered. |
| `--al-handoff-audit-path PATH` | Optional JSONL path for decision artifacts. |

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
