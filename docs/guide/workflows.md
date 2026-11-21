# Choosing a scientific workflow

## Workflow selection matrix

### Composable scientific workflow hierarchy

The public workflows form a strict dependency ladder. Higher-level workflows
reuse lower-level services and their typed results instead of duplicating
relaxation, labeling, ranking, or provenance logic:

```text
scientific run directory + provenance + validation
    -> structure relaxation (MLIP | DFT | MLIP-to-DFT)
    -> uncertainty acquisition + DFT labeling (+ optional retraining)
    -> composition/phase exploration (+ optional embedded AL)
    -> property-driven agentic investigation and hypothesis revision
```

Every workflow writes a collision-resistant directory such as
`runs/2026-09-01T14-32-18Z_a8f31c/` containing the resolved request,
provenance, event stream, structures, calculations, datasets, models, and
results. Failed and unconverged calculations are retained with an explicit
reason; they are not silently dropped or included in phase rankings.

### Scientific workflow contracts

The authoritative guide to relaxation, active learning, phase exploration,
agentic investigation, approval gates, evidence levels, and current execution
status is [docs/scientific-workflows.md](../../docs/scientific-workflows.md).
Run-directory layout and restart rules are documented in
[docs/run-artifacts-and-restarts.md](../../docs/run-artifacts-and-restarts.md), while
scheduler-neutral ensemble DFT execution is documented in
[docs/distributed-dft-dispatch.md](../../docs/distributed-dft-dispatch.md).

#### Unified structure relaxation

Use `matsim-agents relax CONFIG.yaml` with `mode: mlip`, `mode: dft`, or
`mode: mlip-dft`. The last mode performs an MLIP warm start and passes its
optimized geometry to VASP or Quantum ESPRESSO. See
`examples/relaxation/scientific_relaxation.example.yaml`. DFT execution
requires explicit approval by default.

#### Characterization contracts

`matsim-agents characterize CONTRACT.yaml` answers one question about given
structures, without an optimization objective: relaxed structure, energy
ranking of polymorphs, or thermodynamic stability against declared DFT
references. The contract fixes observables, units, minimum fidelity, methods,
tolerances, budget, and approvals; the report concludes only when the evidence
suffices (MLIP energies never establish stability) and otherwise lists what is
missing. See [docs/characterization.md](../../docs/characterization.md) and
`examples/characterization/`.

Campaigns and characterizations record every result, failure, hypothesis
revision, and review decision in a hash-chained, append-only evidence ledger
(`evidence-ledger.jsonl`), checked with `matsim-agents ledger verify`. See
[docs/evidence-ledger.md](../../docs/evidence-ledger.md).

#### Active-learning safety defaults

Active learning defaults to DFT label collection without retraining:

```yaml
trainer:
  enabled: false
  promote_model: false
  promotion_approved: false
```

`enabled: true` creates a candidate model. `promote_model: true` is a separate,
explicit decision that permits the next iteration to use it and additionally
requires `promotion_approved: true` after validation metrics have been reviewed. Newly labeled
frames are checked for finite energies/forces, correct force shapes, and exact
duplicate geometries; a SHA-256 dataset manifest records the DFT energy
reference and validation summary.

**Training needs elemental DFT references.** Every fine-tune trains against
pure-element DFT energies computed with the same DFT method as the labels:
HydraGNN trains on formation energies relative to them, and UMA and MACE use
them to put the labels on their foundation model's energy scale (DFT totals
from Quantum ESPRESSO, or VASP with other settings, differ from the foundation's
pretraining energies by large per-element constants). Set
`trainer.training_references` (a verified manifest or an approved elemental
phase plan) or `trainer.validation_reference_set`; `enabled: true` without one
is rejected when the configuration loads, before any MD or DFT runs. UMA and
MACE fine-tune with low-rank adapters by default (`trainer.finetune_mode: lora`;
`full` trains all weights), and `trainer.start_from: foundation` retrains the
configured model each time instead of the last promoted one. See
[Fine-tuning UMA and MACE](../../docs/scientific-workflows.md#fine-tuning-uma-and-mace).

**Evaluating fine-tuned MLIPs.** `python -m matsim_agents.active_learning.finetune_eval
--mlip-config FILE` fine-tunes the selected model (any backend) with the recipe in
the evaluation YAML's `fine_tune:` block. It splits a labelled dataset by a declared strategy (random frames, time blocks,
trajectories, structures, compositions, conditions, or AL iterations) into
train, validation, and test partitions with a content-addressed manifest and
leakage checks, selects between the foundation and fine-tuned model on
validation only, and evaluates the frozen choice on test and named test
domains with per-group errors. See [MLIP evaluation](../../docs/mlip-evaluation.md).

Held-out model energy comparisons and promotion use **formation-energy
errors**, subtracting each method's own elemental energies on the same fixed
reference geometries. These comparisons require a DFT-labelled elemental JSON
manifest in `trainer.validation_reference_set`. Force gates remain in place;
MD acquisition, polymorph ranking, and hull construction are not redefined.
See [Energy comparison and promotion](active-learning.md#energy-comparison-and-promotion).

#### Consistent multi-node DFT labeling

Where DFT runs is declared in the config's `resources:` block
([resource dispatch](../../docs/resource-dispatch.md)): the `dft` pool's nodes are
split into stable, disjoint groups of `nodes_per_job`, and each group processes
one serial queue, so a node is never reused while its preceding VASP/QE step is
still running. Ranks per node are the pool's devices per node times
`ranks_per_device`; threads per rank default to the cores per node divided by
the ranks. The plan is checked against the Slurm or PBS allocation before any
calculation starts, and the resolved layout is written to `placement.json`.
Set `dft.max_concurrent_jobs` to impose a lower concurrency cap; when omitted,
all complete `nodes_per_job` partitions are used.

#### Thermodynamic claims

`relative_phase_ranking` compares only converged candidates generated in one
exploration. `convex_hull_ranking` additionally requires a compatible,
method-identified elemental and competing-phase reference set and reports
formation energy, energy above hull, and decomposition. Residual forces are a
convergence filter, not a thermodynamic ranking term.

MLIP surrogate hulls perform a separate unary-reference search for every
model. Each target element starts from curated structures plus all compatible
AFLOW prototypes and, by campaign default, 50 pyXtal structures. Initial and
relaxed duplicates are identified with `StructureMatcher`; every relaxation
outcome remains in `unary_reference_search.json`, while only unique converged
polymorphs enter the phase diagram. The lowest corrected energy per atom for
each model defines that model's elemental endpoint. A missing converged unary
endpoint makes the hull provisional and suppresses formation/hull energies.
Isolated atoms are not hull endpoints; they are appropriate only for optional
cohesive or atomization-energy diagnostics. Competing compound references are
currently model-scored at their manifest geometries, so these results remain
explicitly labeled `mlip_proxy` rather than DFT thermodynamics.

Use this table to choose the right entry point quickly.

| Goal | Recommended entry point | Required inputs | Typical outputs |
|---|---|---|---|
| One-shot objective execution with planning, UQ gate, and summary | `matsim-agents run` | natural-language objective; `--mlip-config` for relaxations; optional MCP and AL handoff configuration | planner/executor/UQ/analyst result; optional MCP records and audit, or AL handoff + JSONL audit |
| Call approved external tools or read resources | `matsim-agents run` with `--mcp-config` | MCP extra, server configuration and an objective; no MLIP for MCP-only tasks | external MCP call records and JSONL audit; see [MCP guide](../mcp.md) |
| Interactive hypothesis generation with optional atomistic exploration | `matsim-agents chat` | `--mlip-config`, LLM provider/model | chat transcript under `output_dir/chat/<session>/` + discovery artifacts under `output_dir/discovery/` |
| Optional single structure relax during chat | `matsim-agents chat` + `/relax <path>` | same as chat + structure path | one relaxation summary + optimized structure under `output_dir/single_relax/` |
| Trigger active-learning handoff during chat | `matsim-agents chat` + `/al [composition]` | same as chat + `--al-config`; composition optional (defaults to last discussed) | AL handoff (dry-run or run) + JSONL audit record |
| Reset conversation/discovery state during chat | `matsim-agents chat` + `/clear` | none | cleared session history and seen-composition state |
| Automated discovery -> UQ policy -> optional AL handoff | `matsim-agents supervisor-run` | composition, `--mlip-config`, optional `--al-config` | supervisor summary + optional AL handoff + JSONL audit records |
| Standalone active-learning loop (MD -> UQ -> DFT -> retrain/frozen) | `matsim-agents al run <config.yaml>` | AL YAML config (`mlip`, `md`, `acquisition`, `dft`, `trainer`, `loop`) + optional `--mlip-config` replacing its `mlip` block | iteration state dirs, dataset, optional retrained model logdirs |
| Config-only validation (no execution) | `matsim-agents al validate-config <config.yaml>` | AL YAML config + env vars used in `${VAR}` placeholders | resolved/validated JSON config dump |

Notes:

- `run`, `chat`, and `supervisor-run` can escalate to AL using `--al-config` plus UQ policy flags.
- Handoff audit artifacts default to `<output_dir>/discovery/al_handoff_events.jsonl` unless overridden.

### Minimum viable commands

Use these as starter templates; replace paths with your environment.
`hydragnn.yaml` stands for an `mlip:` block with your HydraGNN model:

```yaml
mlip:
  backend: hydragnn
  logdir: /path/to/hydragnn_logdir
  mlp_checkpoint: /path/to/mlp_branch_weights.pt
```

```bash
# 1) Core graph: planner -> executor -> uq_gate -> analyst
matsim-agents run \
  "Relax structures/mos2-B_Defect-Free_PBE.vasp and summarize results." \
  --mlip-config hydragnn.yaml

# 1a) Core graph with UMA relaxation backend (no HydraGNN checkpoints needed)
matsim-agents run \
  "Relax structures/mos2-B_Defect-Free_PBE.vasp and summarize results." \
  --mlip-config examples/mlip/uma-s-1p1-omat.yaml

# 1b) Core graph with UQ-triggered AL handoff planning
matsim-agents run \
  "Relax structures/mos2-B_Defect-Free_PBE.vasp and summarize results." \
  --mlip-config hydragnn.yaml \
  --trigger-al-handoff \
  --al-config examples/active_learning/al_config.example.yaml \
  --al-dry-run

# 2) Interactive discovery chat
matsim-agents chat \
  --mlip-config hydragnn.yaml

# 2b) Interactive chat with proposer/critic multi-LLM hypothesis debate
matsim-agents chat \
  --mlip-config hydragnn.yaml \
  --llm-peer-review \
  --critic-llm-provider ollama \
  --critic-llm-model qwen2.5:14b \
  --peer-review-rounds 2

# 2c) True multi-critic panel mode (multiple critics + cross-critique)
matsim-agents chat \
  --mlip-config hydragnn.yaml \
  --llm-peer-review \
  --critic-panel-models "qwen2.5:14b,llama3.1:8b,mistral:7b" \
  --critic-panel-providers "ollama,ollama,ollama" \
  --peer-review-rounds 2 \
  --critic-cross-critique

# 2a) Interactive discovery chat with UMA relaxations
matsim-agents chat \
  --mlip-config examples/mlip/uma-s-1p1-omat.yaml

# 3) Supervisor orchestration with AL handoff planning
matsim-agents supervisor-run Li2MnO3 \
  --mlip-config hydragnn.yaml \
  --al-config examples/active_learning/al_config.example.yaml \
  --al-dry-run

# 3a) Supervisor orchestration with UMA relaxations
matsim-agents supervisor-run Li2MnO3 \
  --mlip-config examples/mlip/uma-s-1p1-omat.yaml \
  --al-config examples/active_learning/al_config.example.yaml \
  --al-dry-run

# 4) Standalone active learning run
matsim-agents al validate-config examples/active_learning/al_config.example.yaml
matsim-agents al run examples/active_learning/al_config.example.yaml

# 4b) Same AL config, UMA surrogate instead of HydraGNN
matsim-agents al run examples/active_learning/al_config.example.yaml \
  --mlip-config examples/mlip/uma-s-1p1-omat.yaml
```

---
