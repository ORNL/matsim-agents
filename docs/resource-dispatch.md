# Resource dispatch for LLMs, MLIPs, and DFT

Status: **design proposal** — not implemented. The open design questions are
resolved in [section 11](#11-design-decisions). The implementation lands in a
separate PR after #14.

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

Presets live in `deployments/<facility>/resources/` as small YAML files with
geometry and launch flavor (`srun` vs `mpiexec`, device-mask variable, cores per
node), plus a few named layouts per scenario (single-node discovery, AL with DFT
nodes, campaign). Presets never contain accounts or site paths.

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
