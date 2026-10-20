# Resource dispatch for LLMs, MLIPs, and DFT

Status: **phases 1–3 implemented; phase 4 implemented for the discovery
chat, `llm-check`, and `debate`; phase 5 implemented on Perlmutter** (planner,
presets, layouts, CLI; DFT and training read the placement; concurrent MLIP
relaxations; managed vLLM servers on one or several nodes, named servers,
in-process device limits).
Design questions are resolved in [section 11](#11-design-decisions);
implementation choices are recorded in
[section 12](#12-phase-1-implementation-notes),
[section 13](#13-phase-2-implementation-notes),
[section 14](#14-phase-3-implementation-notes),
[section 15](#15-phase-4-implementation-notes), and
[section 16](#16-phase-5-implementation-notes).

## 1. Problem

A matsim-agents job combines up to four kinds of work: language-model inference,
MLIP relaxations and MD, DFT calculations, and MLIP fine-tuning. Today each kind
chooses its hardware through a different mechanism, and nothing coordinates them.

| Component | How hardware is chosen today | Where it is decided |
| --- | --- | --- |
| DFT | `discover_allocation()` splits the Slurm/PBS node list into disjoint groups and runs jobs concurrently; `nodes_per_job`, `ranks_per_node`, `threads_per_rank` live in each backend block with Frontier-shaped defaults (8 ranks x 7 threads) | Python plus per-facility wrapper scripts |
| MLIP | One device string (`cuda`, `xpu`, `cpu`); always the first visible device; relaxations run one at a time | Job scripts set `CUDA_VISIBLE_DEVICES` / `ZE_AFFINITY_MASK` |
| LLM | vLLM servers are started by bash with GPU ids and tensor-parallel size in environment variables; Python only receives a URL. The Hugging Face provider spreads over every visible device | Job scripts (`MATSIM_VLLM_GPUS`, `MATSIM_VLLM_TP`, `MATSIM_CLIENT_GPUS`, ...) |
| Training | `nodes_for_train` / `ranks_per_node` in the trainer block plus a per-facility launcher | Config plus launcher scripts |

Consequences observed on Perlmutter:

- The vLLM discovery job reserves GPUs 1-3 for MLIP relaxations, but HydraGNN
  uses only GPU 1; GPUs 2-3 stay idle.
- The Hugging Face discovery job places a quarter of the LLM and all of HydraGNN
  on GPU 0 without saying so.
- About fifteen `MATSIM_*` environment variables and dozens of bash scripts
  carry placement logic; nothing rejects two components that collide on one
  device.
- Facility geometry (4 A100, 8 MI250X GCDs, 12 PVC tiles) is copied into
  configs and job scripts instead of being discovered.

## 2. Goals and non-goals

Goals:

1. One explicit, typed **resource plan** in which the user decides where every
   component runs.
2. **Exclusive by default.** Two components may share a device only when the
   plan declares it; undeclared overlap is a validation error.
3. **matsim-agents owns the LLM server lifecycle** on the devices assigned to
   it, while still accepting an existing external endpoint.
4. **MLIP pools run concurrently**: one independent relaxation per device in
   the MLIP pool.
5. The same plan works on Slurm (Perlmutter, Frontier) and PBS (Aurora), and on
   CUDA, ROCm, and Level Zero devices.
6. A **preview command** prints the resolved placement before anything runs.
7. Job scripts shrink to: request an allocation, load the environment, run one
   `matsim-agents` command with a plan.

Non-goals for the first implementation:

- Submitting or resizing scheduler allocations (the plan works inside an
  allocation that already exists).
- Dynamic rebalancing while a workflow runs.
- Multi-node tensor-parallel LLM serving managed from Python (phase 4; until then
  multi-node vLLM keeps using the existing serve jobs as an external endpoint).

## 3. Concepts

**Allocation.** What the scheduler granted, discovered at run time: scheduler
(`slurm`, `pbs`, `local`), ordered node list, and per-node geometry.

**Node geometry.** Accelerator vendor, logical devices per node, and usable CPU
cores per node. Discovered from the scheduler and the runtime, overridable per
facility preset:

| Facility | Vendor | Logical devices / node | Cores / node | Notes |
| --- | --- | --- | --- | --- |
| Perlmutter GPU | `cuda` | 4 (A100 40 or 80 GB) | 64 | `gpu&hbm80g` for 80 GB |
| Frontier | `rocm` | 8 (MI250X GCDs) | 56 usable | 7 cores per GCD |
| Aurora | `xpu` | 12 (PVC tiles, FLAT mode) | 104 | `ZE_FLAT_DEVICE_HIERARCHY=FLAT` |

**Pool.** A named, non-empty slice of the allocation: a set of nodes and, on
each of those nodes, a set of logical devices. Pools are declared per
component: `llm`, `mlip`, `dft`, `training`. A component without a pool does
not run. All nodes in a pool must have the same geometry (decision 11.1).

**Sharing.** Pools are exclusive unless they name each other in `share_with`.
Sharing is symmetric and must be declared on both pools, so it is always
deliberate and visible in the preview. A shared device can carry a memory
budget per pool (`memory_fraction`, decision 11.2).

**LLM server.** A named entry in the `llm` pool's `servers` list: one model
reachable at one endpoint, either started by matsim-agents on its own slice of
the pool or supplied as an external URL. A debate panel is one `llm` pool with
several servers (decision 11.3).

**Placement.** The resolved plan: for each pool, concrete hosts and device
indices, plus the launch details derived from them (device masks, `srun` or
`mpiexec` host lists, rank and thread counts, tensor-parallel size).

## 4. Configuration

The plan is a top-level `resources:` block. It can live in any workflow config
that already exists (AL config, campaign config) or in a standalone file passed
with `--resources`. Every field below has a validated type.

```yaml
resources:
  preset: perlmutter-gpu        # optional; supplies geometry and launch flavor
  pools:
    llm:
      nodes: 1                  # count, "all", "rest", or explicit host list
      devices: [0]              # explicit indices, a count, or "all"
      servers:
        - name: proposer        # referenced by workflows (roles, panel members)
          kind: managed-vllm    # managed-vllm | external | in-process
          model: ${MODEL_ROOT}/Qwen2.5-14B-Instruct
          served_name: Qwen2.5-14B-Instruct
          max_model_len: 16384
          # nodes/devices default to the whole pool; tensor_parallel_size
          # defaults to the server's device count
    mlip:
      nodes: 1                  # same node as llm (node 0 of the allocation)
      devices: [1, 2, 3]        # three concurrent relaxations
```

Field reference:

| Field | Type | Meaning |
| --- | --- | --- |
| `preset` | name | Facility geometry and launch flavor (`perlmutter-gpu`, `frontier`, `aurora`, `local`) |
| `pools.<name>.nodes` | int, `"all"`, `"rest"`, or list of hosts | Which allocation nodes the pool uses; `rest` takes nodes not used by any other pool |
| `pools.<name>.devices` | list of ints, int, `"all"` | Logical devices per node; an int takes the lowest free indices |
| `pools.<name>.share_with` | list of pool names | Pools allowed to overlap with this one |
| `pools.<name>.memory_fraction` | float in (0, 1] | Device-memory budget on shared devices; required for a `managed-vllm` server on a shared device |
| `pools.llm.servers` | list | One entry per model endpoint; a single-model setup is a one-entry list |
| `pools.llm.servers[].name` | string | Unique name used by workflows (proposer, critic, panel member) |
| `pools.llm.servers[].kind` | `managed-vllm`, `external`, `in-process` | How this model runs |
| `pools.llm.servers[].model` / `served_name` | path / string | Model directory and the identifier the server exposes |
| `pools.llm.servers[].nodes` / `devices` | as for pools | This server's slice of the pool; defaults to the whole pool when there is one server |
| `pools.llm.servers[].base_url` | URL | Required for `external`; such a server takes no nodes or devices |
| `pools.llm.servers[].tensor_parallel_size` | int | Defaults to the server's device count |
| `pools.llm.servers[].max_model_len` | int | Context length passed to vLLM |
| `pools.mlip.max_concurrent` | int | Defaults to the pool's device count |
| `pools.dft.nodes_per_job` | int | Nodes per DFT calculation |
| `pools.dft.ranks_per_device` | int | Defaults to 1; ranks per node = devices x ranks_per_device |
| `pools.dft.threads_per_rank` | int | Defaults to cores per node / ranks per node |
| `pools.training.devices` | int | Built-in HydraGNN fine-tuners require exactly 1 |

`nodes_per_job`, `ranks_per_node`, `threads_per_rank` move out of the
`dft.vasp` / `dft.qe` blocks; those blocks keep only scientific and executable
settings. `mlp_device` / `device` move out of the MLIP blocks.

### 4.1 Examples

Single Perlmutter node, vLLM on one GPU, three concurrent HydraGNN relaxations:

```yaml
resources:
  preset: perlmutter-gpu
  pools:
    llm:
      nodes: 1
      devices: [0]
      servers:
        - {name: proposer, kind: managed-vllm, model: /models/Qwen2.5-14B-Instruct}
    mlip: {nodes: 1, devices: [1, 2, 3]}
```

Single node with the LLM in-process across all GPUs and the MLIP sharing GPU 0
(the layout the Hugging Face discovery job uses today, now explicit, with a
memory budget on the shared GPU):

```yaml
resources:
  preset: perlmutter-gpu
  pools:
    llm:
      nodes: 1
      devices: all
      share_with: [mlip]
      memory_fraction: 0.75      # applies on GPU 0, the only shared device
      servers:
        - {name: proposer, kind: in-process, model: /models/Qwen2.5-72B-Instruct}
    mlip: {nodes: 1, devices: [0], share_with: [llm], memory_fraction: 0.2}
```

Active learning: one node for MLIP MD and training, the rest for DFT:

```yaml
resources:
  preset: frontier
  pools:
    mlip:     {nodes: 1, devices: [0, 1, 2, 3]}
    training: {nodes: 1, devices: [4]}
    dft:      {nodes: rest, devices: all, nodes_per_job: 1}
```

Campaign with a debate panel of locally served and remote models, plus MLIP and
DFT on the rest of the allocation:

```yaml
resources:
  preset: perlmutter-gpu
  pools:
    llm:
      nodes: 7
      servers:
        - {name: kimi,  kind: managed-vllm, model: /models/Kimi-K2.5, nodes: 4}
        - {name: glm,   kind: managed-vllm, model: /models/GLM-4.7, nodes: 2}
        - {name: gemma, kind: managed-vllm, model: /models/gemma-4-31B-it, nodes: 1, devices: [0, 1]}
        - {name: qwen,  kind: managed-vllm, model: /models/Qwen2.5-14B-Instruct, nodes: 1, devices: [2]}
        - {name: remote, kind: external, base_url: "https://llm.example.org/v1"}
    mlip: {nodes: 1, devices: all}
    dft:  {nodes: rest, devices: all, nodes_per_job: 1}
```

Server slices are taken from the pool in list order: `kimi` takes the pool's
first four nodes, `glm` the next two, and `gemma` and `qwen` share the last
node on disjoint GPUs. An `external` server owns no nodes or devices; a pool
whose servers are all external needs no `nodes` or `devices`.

## 5. Resolution and validation

`ResourcePlan.resolve(allocation) -> Placement` runs before any work starts.

1. Apply the preset, then auto-detected geometry, then explicit overrides.
2. Resolve `nodes` for each pool in declaration order; `rest` resolves last.
3. Reject a pool whose nodes differ in vendor, logical device count, device
   memory, or usable cores (decision 11.1).
4. Resolve `devices` per node; indices must exist on the node.
5. Reject any device claimed by two pools unless both list each other in
   `share_with`. On each shared device, the `memory_fraction` values of the
   pools using it must sum to at most 0.95; a `managed-vllm` server on a shared
   device must set one (decision 11.2).
6. Resolve LLM servers in list order inside the `llm` pool. Each server's
   `nodes` / `devices` must lie inside the pool, and servers must not overlap
   one another (decision 11.3). Server names must be unique.
7. Component checks:
   - `managed-vllm`: `tensor_parallel_size` equals or divides the server's
     device count and divides the model's attention heads.
   - `dft`: `nodes_per_job` divides the pool's node count; derived ranks and
     threads fit the node's cores.
   - `training`: built-in HydraGNN trainers get exactly one device.
8. Produce a `Placement` with concrete hosts, device indices, environment
   (`CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `ZE_AFFINITY_MASK`), and
   launcher arguments, and write it to the run directory as
   `placement.json` for provenance.

Errors name the pool, node, and device involved, for example:
`pool 'mlip' device 0 on nid001234 overlaps pool 'llm'; add share_with to both pools to allow sharing`.

## 6. Component mechanisms

### 6.1 LLM

- **managed-vllm.** A `VLLMServerManager` starts one server per
  `managed-vllm` entry on that server's slice, waits until `/v1/models` lists
  the exact served name, and stops the servers on exit. On a shared device the
  server's `memory_fraction` becomes `--gpu-memory-utilization`. The launch recipe is per vendor:
  `vllm serve` with a device mask on CUDA and ROCm; on Aurora, the qualified
  `mpiexec` + `aurora_vllm_entrypoint.py` recipe already used by
  `deployments/aurora/launchers/aurora-vllm.sh`. Logs go to the run directory.
- **external.** No lifecycle; the plan supplies `base_url`.
- **in-process.** The Hugging Face provider receives the server's devices and
  passes an explicit `max_memory` / device list to Accelerate, so the model can
  no longer silently spread onto MLIP devices. On a shared device, `max_memory`
  is the device's capacity times `memory_fraction`.
- `get_chat_model()` takes the endpoint for a named server from the placement
  instead of `MATSIM_VLLM_BASE_URL`. Workflows select servers by name: the chat
  proposer and critic, and each debate panel member, refer to entries of the
  `servers` list. The CLI `--llm-*` options remain for model choice, not
  placement.

### 6.2 MLIP

- An `MLIPWorkerPool` starts one worker process per device in the `mlip` pool,
  each with a single-device mask, and keeps its calculator loaded.
- Phase exploration, discovery chat, campaign candidate relaxation, and AL MD
  submit independent relaxations to the pool; results return in submission
  order so provenance is deterministic.
- A one-device pool behaves exactly like today, without worker processes.
- On a shared device, each worker sets PyTorch's per-process memory cap from
  the pool's `memory_fraction`.

### 6.3 DFT

- The `dft` pool replaces the whole allocation in `run_dft_batch`: groups are
  formed from the pool's nodes only, and ranks and threads come from the pool.
- Wrapper scripts keep their positional contract but receive values derived
  from the placement.

### 6.4 Training

- The training launcher receives the `training` pool's single device instead of
  assuming device 0, so training can run next to MLIP MD without colliding.

## 7. Interfaces

Python (proposed):

```text
from matsim_agents.execution.placement import ResourcePlan, Placement
plan = ResourcePlan.from_yaml(path)            # or the resources: block of a config
placement = plan.resolve(discover_allocation())
with placement.services() as services:         # starts and stops managed LLM servers
    services.llm["proposer"].base_url             # one entry per named server
    services.mlip.map(relax, structures)        # concurrent, one per device
```

CLI (proposed):

```text
matsim-agents resources plan <config-or-plan.yaml>      # print the resolved placement table
matsim-agents resources validate <config-or-plan.yaml>  # exit non-zero on conflicts
matsim-agents chat --resources plan.yaml ...
matsim-agents al run al.yaml                            # uses the config's resources: block
```

Job scripts reduce to allocation request, environment activation, and one
command; device masks and server launches move into the placement.

## 8. Facility presets

Presets are built into the package (`matsim_agents.execution.presets`) and
describe node hardware and launch flavor only: `perlmutter-gpu`,
`perlmutter-gpu-80gb`, `frontier`, and `aurora`. Named scenario layouts that
use them live in `deployments/<facility>/resources/`:

| Facility | Layouts |
| --- | --- |
| Perlmutter | `discovery-vllm`, `discovery-hf-80gb`, `active-learning`, `campaign-panel` |
| Frontier | `discovery-vllm`, `active-learning` |
| Aurora | `discovery-vllm`, `active-learning` |

Presets and layouts never contain accounts or site paths; model locations come
from `${MODEL_ROOT}`.

## 9. Migration

- Remove placement environment variables (`MATSIM_VLLM_GPUS`,
  `MATSIM_CLIENT_GPUS`, `MATSIM_VLLM_TP`, `MATSIM_DFT_RANKS_PER_NODE`,
  `MATSIM_DFT_THREADS_PER_RANK`, `MATSIM_DFT_NODES_PER_JOB`, `MATSIM_MLP_DEVICE`,
  `MATSIM_QE_MLP_DEVICE`, `MATSIM_VASP_MLP_DEVICE`, `MATSIM_GPUS_PER_NODE`) and the
  per-backend rank/thread and device fields.
- Rewrite job scripts to pass a preset layout or plan file.
- No compatibility layer: configs that still carry the removed fields fail
  validation with a message pointing to the `resources:` block.

## 10. Implementation phases

| Phase | Scope | Verification |
| --- | --- | --- |
| 1 | `ResourcePlan`, geometry discovery, presets, `resolve()`, validation, `placement.json`, `resources plan` / `validate` CLI | Unit tests for every resolution and conflict rule; plans for all three facilities |
| 2 | DFT and training read the placement; remove backend rank/thread/device fields | Existing DFT allocation tests; Perlmutter QE/VASP qualification |
| 3 | `MLIPWorkerPool` and concurrent relaxations in phase exploration, chat, campaign, AL | Determinism and failure-isolation tests; Perlmutter run with 3 devices |
| 4 | `VLLMServerManager` (CUDA, ROCm, XPU), named servers, memory budgets, in-process LLM device limits, `get_chat_model` from placement; job-script rewrite | Lifecycle tests with stub servers; Perlmutter chat, LLM check, campaign |
| 5 | Multi-node managed LLM servers (Ray) | Campaign panel on Perlmutter, Aurora serve job |

Each phase is a separate commit series in the new PR and keeps the full suite
green.

## 11. Design decisions

### 11.1 Pools are homogeneous

**Decision.** Every node in a pool must have the same geometry: accelerator
vendor, logical device count, device memory, and usable cores. A job that needs
two node types expresses each type as its own pool.

**Rationale.**

- On Perlmutter, Frontier, and Aurora an ordinary job receives one node type.
  Mixing types (for example 40 GB and 80 GB A100 nodes) requires a Slurm
  heterogeneous job, which no current workflow uses.
- DFT rank and thread arithmetic, vLLM tensor-parallel sizing, and one MLIP
  worker per device all assume identical nodes. Mixed pools would need
  per-node rules everywhere for a case that does not occur.
- The planner already discovers each node's geometry, so it can reject a mixed
  pool with a precise message. If heterogeneous jobs are needed later, each
  component of the job maps naturally to its own pool; the design does not
  block that extension.

**Consequence.** Resolution rule 3 (section 5) rejects pools whose nodes differ
in vendor, device count, device memory, or usable cores.

### 11.2 Shared devices carry optional memory budgets

**Decision.** A pool may set `memory_fraction`, the share of each shared
device's memory it may use. It is required for a `managed-vllm` server on a
shared device and optional otherwise. On every shared device, the fractions of
the pools using it must sum to at most 0.95.

**Rationale.**

- vLLM reserves 90% of a device's memory by default. Sharing a device with
  vLLM without a budget always starves the other component, which is why the
  budget is mandatory in that case.
- Each backend has a native mechanism to honor the budget:
  - vLLM: `--gpu-memory-utilization`;
  - in-process Hugging Face: Accelerate's per-device `max_memory`;
  - MLIP workers: PyTorch's per-process memory fraction.
- The 0.95 ceiling leaves headroom for the driver, CUDA/HIP/Level Zero
  contexts, and allocator fragmentation.

**Limitation.** These are budgets that each library respects, not hardware
isolation. A process that ignores them can still exhaust the device. True
partitioning (for example A100 MIG) is generally not available to users on
these systems and is out of scope.

**Consequence.** Resolution rule 5 (section 5) validates the budgets; the LLM
and MLIP mechanisms (section 6) apply them.

### 11.3 A debate panel is one LLM pool with named servers

**Decision.** The `llm` pool holds a `servers` list. Each entry is one model
endpoint with a unique name, a `kind` (`managed-vllm`, `external`,
`in-process`), and, for managed and in-process servers, its own slice of the
pool's nodes or devices. A single-model setup is a one-entry list. This
replaces a single `service` block with a server count.

**Rationale.**

- The panel is one unit. Debate, chat, and campaign code refer to models by
  role or name (proposer, critic, panel member), not by pool; one list keeps
  those names together instead of creating a pool per model (seven pools for
  the current campaign panel).
- The same exclusivity checks still apply: servers must lie inside the pool
  and must not overlap one another.
- Local and remote models can be mixed in one panel, which separate pools
  would make awkward.
- The common single-model case stays a one-entry list.

**Consequence.** Resolution rule 6 (section 5) assigns and validates server
slices in list order; `get_chat_model()` and the workflows select servers by
name (section 6.1).

## 12. Phase 1 implementation notes

Phase 1 adds `execution/placement.py` (`ResourcePlan`, `Placement`),
`execution/presets.py`, `execution/variables.py` (the `${VAR}` substitution now
shared with `ALConfig.from_yaml`), the layouts listed in section 8, and the
`matsim-agents resources plan|validate` commands. Nothing in the workflows reads
the placement yet; phases 2–4 connect it.

Choices made while implementing, relative to sections 4–8:

- **Presets are code, not YAML.** Geometry is part of the validated schema and
  is needed by tests without a checkout of `deployments/`. Layouts stay in
  `deployments/`. There is no `local` preset: a workstation plan sets
  `geometry:` explicitly (`vendor`, `devices_per_node`, `cores_per_node`) and,
  optionally, `launcher: local`. CPU-only nodes use `vendor: cpu` with
  `devices_per_node: 0`; pools then claim whole nodes.
- **One geometry per plan.** All nodes of an allocation share one geometry, so
  decision 11.1 holds by construction. Precedence is preset, then the GPU count
  reported by the scheduler, then explicit `geometry:` fields; an explicit
  device count that contradicts the scheduler is an error.
- **Node selection.** An integer `nodes` takes the first allocation nodes on
  which the requested devices are free for that pool (respecting
  `share_with`); `rest` takes nodes no other pool touches and resolves last;
  explicit host lists are checked for conflicts. An integer `devices` takes the
  lowest free indices on each node.
- **Loading.** `ResourcePlan.from_yaml` reads only the top-level `resources:`
  (and `vars:`) keys, so the same block can sit in any workflow config without
  that config's schema knowing about it yet. A value that is exactly one
  variable (`nodes: ${LLM_NODES}`) takes the type of its expansion. Layout
  files carry a `preview: {nodes: N}` hint used by the tests and readers.
- **LLM servers.** When the `llm` pool has one non-external server, the server
  defaults to the whole pool; with several, each must set `nodes`. A
  `managed-vllm` server's tensor-parallel size defaults to its devices per node
  and the pipeline-parallel size is derived as devices / tensor-parallel size.
  The attention-head check runs when the model's `config.json` (or its
  `text_config`) is readable.
- **Training** requires exactly one node and one device; **MLIP**
  `max_concurrent` may not exceed the pool's device count.
- **Diagnostics.** Unused devices and unused LLM-pool devices are reported as
  warnings, not errors. `resources plan --nodes N` or `--hosts a,b` previews a
  multi-node plan from a login node; `--output` writes `placement.json`.

## 13. Phase 2 implementation notes

DFT and training now take their nodes, ranks, threads, and devices from the
placement:

- **Removed fields.** `nodes_per_job`, `ranks_per_node`, `threads_per_rank`
  (`dft.qe`, `dft.vasp`), `dft.max_concurrent_jobs`, and
  `trainer.nodes_for_train` / `trainer.ranks_per_node` are rejected with a
  message naming the `resources:` field that replaces each one.
  `MATSIM_GPUS_PER_NODE`, `MATSIM_DFT_RANKS_PER_NODE`,
  `MATSIM_DFT_THREADS_PER_RANK`, `MATSIM_DFT_NODES_PER_JOB`, and
  `MATSIM_DFT_ASSIGNED_NODES` are gone.
- **`ProcessLaunch`.** Each DFT job group and the training pool resolve to a
  `ProcessLaunch` (hosts, devices, ranks per node, threads per rank, device
  mask). Step wrappers keep their positional
  `<nodes> <ranks_per_node> <threads_per_rank>` contract and read
  `MATSIM_ASSIGNED_NODES` and `MATSIM_DEVICES_PER_NODE`; Slurm wrappers request
  that many GPUs per node, so several ranks can share a device
  (`ranks_per_device`).
- **Required plans.** `ALConfig.resources` is required, with a `dft` pool and,
  when `trainer.enabled`, a `training` pool. `matsim-agents al run` resolves the
  plan once and writes `placement.json` into `loop.out_dir`. The DFT execution
  signature records the launch layout but not host names, so cached elemental
  references stay valid across allocations with the same layout.
- **Other entry points.** Elemental references, campaign DFT refinement, and
  `relax` use the first DFT job group. `relax` launchers may use `{nodes}`,
  `{ranks_per_node}`, and `{threads_per_rank}`; a plain `pw.x` launcher needs no
  plan. `elemental_dft` takes `--config` with `dft:` and `resources:` blocks.
- **Sequential layouts.** AL stages run one after another, so shipped AL
  configs and `deployments/<facility>/resources/active-learning-sequential.yaml`
  give DFT every device and let training share one device with it
  (`share_with`). `deployments/common/run-active-learning.sh` reads the layout
  from `MATSIM_RESOURCES`.
- **Training launchers** pin the trainer to the placed host
  (`--nodelist` / `--hosts`); the Aurora trainer requires the tile mask from the
  placement instead of `MATSIM_TRAIN_TILE`.

## 14. Phase 3 implementation notes

`matsim_agents.backends.mlip.pool.MLIPWorkerPool` runs independent MLIP
relaxations concurrently:

- **Workers.** One `spawn`-started process per device of the `mlip` pool (up to
  `max_concurrent`), each started with that device's mask and, on a shared
  device, PyTorch's per-process memory cap from `memory_fraction`. `spawn` is
  required because the parent may already hold CUDA state (for example an
  in-process LLM). Workers inherit the parent's `sys.path`.
- **Model reuse.** `relax_with_mlip(..., calculators=CalculatorCache())` reuses
  a loaded calculator for identical model settings (backend, checkpoint paths,
  head, precision, device, charge, spin); the cache keeps the two most recent
  models. Every worker owns one cache, so each device loads a model once
  instead of once per relaxation.
- **Ordering and failures.** `map` / `relax` return outcomes in submission
  order. A task's exception is its outcome; a worker that exits fails only the
  task it was running and is replaced before the next task.
- **Local mode.** Without an `mlip` pool, or on CPU-only geometry, relaxations
  run in the calling process, now with model reuse. This departs from the
  original "a one-device pool behaves exactly like today": a one-device `mlip`
  pool still gets a worker so its device mask is honoured.
- **Driver node.** Worker processes run on the node of the calling process, so
  the `mlip` pool must have one node and that node must be the driver's;
  otherwise resolution fails with a message naming both hosts.
- **Callers.** `explore_composition(..., mlip_pool=...)` builds every input
  first, calling the start and attempt callbacks in candidate order (so
  per-attempt budget accounting is unchanged), then records results and
  completion callbacks in candidate order. The discovery chat uses
  `--resources` (only the `mlip` pool is used until phase 4) and reuses the pool
  for the whole session, including `/relax`. Campaign formula exploration uses
  the AL template's `mlip` pool when declared; perturbation-robustness trials
  reuse one in-process model.
- **AL MD.** Each seed draws from its own random stream, spawned from
  `md.random_seed` with NumPy's `SeedSequence`, so a seed's trajectory depends
  only on `random_seed` and the seed's position in the list. Seeds then run
  concurrently on the `mlip` pool, one per device, and candidates are returned
  in seed order; the result is identical to running them one after another.
  Trajectories differ from runs made before this change, which drew every seed
  from one shared stream. The MD workers exit after sampling so their device
  memory is free for DFT. Shipped AL configs and the `active-learning-sequential`
  layouts declare an `mlip` pool on the first node, sharing devices with `dft`
  and `training`.
- **Driver device.** MLIP work done in the driver itself (AL model loading,
  single-seed MD, acquisition scoring, candidate comparison) runs with the
  `mlip` pool's first device as the current device and its `memory_fraction`
  cap (`driver_on_mlip_device`), so it no longer lands on device 0. The AL loop
  frees its surrogate after candidate selection, and campaign and chat
  exploration workers are stopped before active learning starts and restarted
  for reevaluation, so no exploration model stays resident during DFT or
  training on shared devices.
- **Slurm device choice.** A Slurm step that requests *N* GPUs receives *N*
  GPUs of Slurm's choosing. Launches therefore also export `MATSIM_DEVICE_MASK`
  (the placed indices) and `MATSIM_NODE_DEVICES`. When a step uses only some of
  a node's GPUs, the Perlmutter and Frontier DFT and training wrappers request
  all of the node's GPUs with `--gpu-bind=none` and set `CUDA_VISIBLE_DEVICES`
  (or `ROCR_VISIBLE_DEVICES`) to the placed indices inside the step, and use
  `--overlap` instead of `--exclusive` because other pools may hold the node's
  other GPUs; whole-node steps keep `--exclusive --gpu-bind=closest`. Aurora
  DFT wrappers bind each rank to `threads_per_rank` cores
  (`--depth --cpu-bind=depth`) and, for a subset of tiles, map ranks to the
  placed tiles with `--gpu-bind=list:...`. The Frontier vLLM launcher likewise runs
  the server in an `srun --overlap` step that takes the node's GCDs and exposes
  only the placed ones through `HIP_VISIBLE_DEVICES`, as the qualified Frontier
  vLLM recipes do.
- **Driver masks.** When the driver already runs under a device mask (for
  example a Slurm step), worker masks are translated through it; a pool device
  the driver cannot see is an error. Campaign configs declare no `mlip` pool
  because the campaign step sees one GPU.

## 15. Phase 4 implementation notes

`matsim_agents.backends.llm.servers.LLMServices` turns the `llm` pool into
named endpoints:

- **managed-vllm.** One process per server, started on the server's node with
  its device mask: directly when that is the driver's node (bound to
  `127.0.0.1`), otherwise through `srun --overlap ... env <mask> ...` or
  `mpiexec --hosts ... env <mask> ...` (bound to `0.0.0.0`, reached by host
  name). A server is ready once `GET /v1/models` lists its served name; a
  server that exits or misses `startup_timeout_s` (default 2400 s) fails the
  start with the tail of its log (`<output>/llm-servers/<name>.log`), and every
  server already started is stopped. Servers stop with `SIGTERM`, then `SIGKILL`
  after 30 s, when the session ends.
- **Server fields.** `port` (default 8000 + the server's position among managed
  servers; two servers on one node must differ), `launcher` (command that runs
  `vllm serve`, default `vllm serve` on `PATH`), `extra_args`, and
  `startup_timeout_s`. The launcher receives the model and `--served-model-name`,
  `--host`, `--port`, `--tensor-parallel-size`, `--pipeline-parallel-size` (when
  above 1), `--max-model-len`, and `--gpu-memory-utilization` (the pool's
  `memory_fraction`, only when the server uses a shared device), followed by
  `extra_args`. Facility environments belong in the launcher:
  `deployments/perlmutter/launchers/vllm-server-perlmutter.sh` runs vLLM from
  `venv_vllm` with node-local JIT caches.
- **Several nodes.** See section 16: a managed server on several nodes needs a
  facility launcher that starts the Ray cluster.
- **in-process.** The server must sit on the driver's node. Its devices are
  registered with the Hugging Face provider, which loads the model with
  `device_map="auto"` and an explicit `max_memory` for exactly those devices
  (free memory, or `memory_fraction` of the total on shared devices), so the
  model can no longer spread onto MLIP devices or the CPU.
- **In-process caching.** A loaded in-process model is keyed by its device
  assignment as well as its path; registering the same model for different
  devices in one process is an error rather than silently reusing the old
  devices. `llm-check` verifies an in-process model's device map lies within
  its assigned devices (not every GPU the driver sees), and a vLLM server's
  workers as tensor x pipeline parallel size.
- **API keys.** A server may set `api_key_env`, the name of an environment
  variable holding its key. Clients (chat, critics, debate participants,
  `llm-check`, readiness probes) send it as a bearer token, and a managed vLLM
  server receives it as `VLLM_API_KEY` (never on its command line). Only the
  variable name appears in plans, `placement.json`, `llm-endpoints.json`, and
  run records. Without `api_key_env`, the provider default applies
  (`MATSIM_VLLM_API_KEY`, else `EMPTY`).
- **Failure records.** When a plan server cannot start or a server name cannot
  be resolved, `llm-check` still writes a failed `result.json` with the reason.
- **external.** Endpoint only, reached as an OpenAI-compatible server.
- **Shared devices.** `NodeSlice.shared_devices` records which devices of a
  slice another pool also uses; it drives the vLLM memory flag and the
  in-process budgets.
- **Device indices** are positions among the devices the driver can see
  (`execution/devices.py`): under a Slurm step, which renumbers its GPUs from
  zero, index `i` is the step's `i`-th GPU.
- **Discovery chat.** `--resources` resolves the plan once per session, writes
  `placement.json`, starts the `llm` pool's servers, and uses `--llm-server`
  (default: the only server) as proposer and `--critic-llm-servers` as the
  critic panel; MLIP relaxations use the `mlip` pool. The Perlmutter vLLM and
  Hugging Face discovery jobs now pass a layout instead of starting servers and
  masking GPUs themselves.
- **`resources serve`.** `matsim-agents resources serve PLAN -- CMD` starts the
  plan's servers (managed and external; in-process servers cannot be shared
  with another process), writes `llm-endpoints.json`, runs `CMD` with
  `MATSIM_LLM_ENDPOINTS` pointing at it (and `MATSIM_LLM_PROVIDER`,
  `MATSIM_LLM_MODEL`, `MATSIM_VLLM_BASE_URL` for the selected server), stops
  the servers, and exits with the command's code. `SIGTERM` from the scheduler
  also stops the servers.
- **llm-check and debate.** `LLMCheckConfig` takes `server` and/or
  `resources`; the check then uses that server's endpoint and defaults its
  tensor-parallel size, device count, and context length from the placement.
  Debate participants may name a `server` instead of `provider`, `model`, and
  `base_url`, with servers from the config's `resources:` block. Without a
  `resources:` block, names resolve through `MATSIM_LLM_ENDPOINTS`, so commands
  run under `resources serve` share its servers.
- **Migrated jobs.** The Perlmutter, Frontier, and Aurora `llm-check` jobs run
  under `resources serve` with per-facility `llm-check.yaml` layouts and
  launchers (`vllm-server-{perlmutter,frontier,aurora}.sh`); the shared steps
  live in `deployments/common/run-llm-check.sh`. The Perlmutter hypothesis
  debate job names its panel in `hypothesis-debate.yaml` (or
  `hypothesis-debate-smoke.yaml`). The Frontier and Aurora launchers start
  their own `srun`/`mpiexec` step for a server on the driver's node; for a
  server matsim-agents already launched on another node (`MATSIM_SERVER_IN_STEP=1`)
  they run vLLM directly instead of nesting a second step.
- **Not yet migrated.** Campaign formula discovery and the all-model debate
  benchmarks serve multi-node models (phase 5), and the Frontier and Aurora
  discovery-chat jobs still start their servers in their job scripts.

## 16. Phase 5 implementation notes

A `managed-vllm` server may span several nodes:

- **Launch.** matsim-agents starts the server's launcher on its first (head)
  node, directly when that is the driver's node and otherwise through the
  facility launcher, and passes the layout as `MATSIM_SERVER_SLICES`
  (`host=devices;host=devices`), `MATSIM_SERVER_RAY_PORT` (server port + 1000),
  `MATSIM_NODE_DEVICES`, and `MATSIM_NODE_CORES`. Readiness, logs, and stopping
  work as for a single-node server; the API listens on the head node.
- **Perlmutter launcher.** With more than one node,
  `vllm-server-perlmutter.sh` follows the qualified
  `job-serve-multinode-perlmutter.sh` recipe: it starts the Ray head with the
  head node's GPUs, starts one Ray worker per other node through
  `srun --overlap --gpu-bind=none` with that node's placed GPUs in
  `CUDA_VISIBLE_DEVICES`, waits until the cluster reports every GPU, then runs
  `vllm serve --distributed-executor-backend ray`. Stopping the launcher stops
  vLLM and the Ray cluster.
- **Parallelism.** `tensor_parallel_size` defaults to the server's devices per
  node, with pipeline stages across nodes (`--pipeline-parallel-size`). The
  campaign panel sets tensor parallelism over all of a server's GPUs, as the
  qualified campaign runs used.
- **Layouts.** `campaign-panel.yaml` describes the seven-model 15-node panel
  with the Perlmutter launcher; `llm-check-multinode.yaml` serves
  Qwen3-235B-A22B-Instruct-2507 across two nodes for qualification.
- **Other facilities.** A multi-node server without a launcher is rejected. The
  Frontier and Aurora launchers serve on one node; their multi-node recipes
  (`job-serve-multinode-frontier.sh`, `job-serve-multinode-aurora.sh`) are not
  yet folded into the launchers.

