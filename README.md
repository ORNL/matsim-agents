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
2. [Installation](#installation)
3. [LLM backends](#llm-backends)
4. [Downloading models for vLLM](#downloading-models-for-vllm)
5. [Quick start](#quick-start)
5. [Quick start](#quick-start)
6. [The agent graph](#the-agent-graph)
7. [Hypothesis-driven discovery chat](#hypothesis-driven-discovery-chat)
8. [Programmatic API](#programmatic-api)
9. [CLI reference](#cli-reference)
10. [Project layout](#project-layout)
11. [Configuration reference](#configuration-reference)
12. [Limitations and roadmap](#limitations-and-roadmap)
13. [Contributing](#contributing)
14. [License & citation](#license--citation)

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
- **Local & HPC ready**: setup script delegates to HydraGNN's own installers for laptops and DOE supercomputers (Frontier, Perlmutter, Aurora, Andes), and auto-relaxes HydraGNN's overly-tight `click==8.0.0` / `tqdm==4.67.1` pins so the env is conflict-free.
- **Pluggable LLMs**: Ollama, vLLM, OpenAI, Anthropic via a single factory.

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

# Frontier (OLCF, ROCm 7.1)
PLATFORM=frontier-rocm71 ./scripts/setup_env.sh

# Perlmutter (NERSC)
PLATFORM=perlmutter ./scripts/setup_env.sh
```

Available `PLATFORM` values:
`workstation` (default), `frontier-rocm71`, `frontier-rocm64`,
`perlmutter`, `aurora`, `andes`.

Environment overrides accepted by the installer:

| Variable | Purpose | Default |
|---|---|---|
| `PYTHON` | Python interpreter | `python3` |
| `HYDRAGNN_REPO` | HydraGNN git URL | `https://github.com/ORNL/HydraGNN.git` |
| `HYDRAGNN_REF` | Branch/tag/commit | `main` |
| `HYDRAGNN_DIR` | Reuse an existing HydraGNN checkout | `third_party/HydraGNN` |
| `HYDRAGNN_EXTRAS` | Args forwarded to `install_dependencies.sh` | `all dev` |
| `LLM_BACKENDS` | Subset of `ollama vllm openai anthropic` | `ollama vllm` |
| `BOOTSTRAP_OLLAMA` | Set to `1` to install the Ollama daemon, start it, and pull `OLLAMA_MODELS` (workstation only) | `0` |
| `OLLAMA_MODELS` | Space-separated list of models to pull when `BOOTSTRAP_OLLAMA=1` | `qwen2.5:14b` |
| `SKIP_HYDRAGNN` | Set to `1` to skip HydraGNN install | `0` |

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
export MATSIM_LLM_PROVIDER=ollama          # or vllm | openai | anthropic
export MATSIM_OLLAMA_BASE_URL=http://...    # optional
export MATSIM_VLLM_BASE_URL=http://node:8000/v1
export MATSIM_VLLM_API_KEY=EMPTY            # only if vLLM is auth-protected
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
| `--llm-provider {ollama,vllm,openai,anthropic}` | Chat backend. |
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

## Project layout

```
matsim-agents/
├── pyproject.toml
├── scripts/
│   └── setup_env.sh              # delegates to HydraGNN installers
├── src/matsim_agents/
│   ├── state.py                  # typed shared LangGraph state
│   ├── graph.py                  # planner -> executor -> analyst
│   ├── llm.py                    # Ollama | vLLM | OpenAI | Anthropic
│   ├── cli.py                    # `matsim-agents run|plan|chat`
│   ├── chat.py                   # interactive discovery REPL
│   ├── agents/
│   │   ├── planner.py
│   │   ├── executor.py
│   │   └── analyst.py
│   ├── tools/
│   │   └── relaxation.py         # HydraGNN + ASE relaxation tool
│   └── discovery/
│       ├── composition.py        # formula parsing
│       ├── phase_explorer.py     # crystal-phase seed enumeration
│       ├── stability.py          # ΔE/atom ranking & |F|max proxy
│       └── wrapper.py            # explore_composition()
├── examples/
│   ├── single_relaxation.py
│   └── discovery_chat.py
├── tests/
│   ├── test_state_and_graph.py
│   └── test_discovery.py
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

---

## Limitations and roadmap

- [ ] **Phonon-based dynamical stability** (phonopy / finite differences).
- [ ] **Formation-energy reference set** for absolute (not relative) chemical stability.
- [ ] **Richer phase enumeration** via pymatgen prototypes / CALYPSO / USPEX hooks.
- [ ] **Symmetry-aware ordering enumeration** via `enumlib` (currently random-shuffle + `StructureMatcher` dedup).
- [ ] **Anisotropic / per-axis lattice scans** (currently isotropic only).
- [ ] **AB / AA' stacking** for 2-D multilayers (currently AA-stacked only).
- [ ] **2-D heterostructures** (e.g. graphene/h-BN, MoS₂/WSe₂) with lattice-mismatch search.
- [ ] **MD agent**: NVT/NPT runs with the same HydraGNN calculator.
- [ ] **MCP tool server** so external clients (Claude Desktop, IDE agents) can call the discovery wrapper directly.
- [ ] **Distributed executor** for parallel composition exploration on HPC.
- [ ] **Pluggable MLFF backends** (MACE, NequIP, Orb) behind the same calculator interface.

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
