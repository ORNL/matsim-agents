# Scientific workflow contracts

`matsim-agents` is organized as composable scientific services. Higher-level
workflows call lower-level workflows and consume their typed results; they do
not reproduce relaxation, labeling, ranking, validation, or provenance logic.

```text
relaxation
  └─ active learning
       └─ composition and phase exploration
            └─ property-driven agentic investigation
```

## Capability and execution status

| Workflow | Public implementation | Current execution status |
| --- | --- | --- |
| Structure relaxation | `matsim-agents relax CONFIG.yaml` | End-to-end for configured MLIP, QE, and VASP environments |
| Active learning | `matsim-agents al run CONFIG.yaml` | End-to-end candidate generation, acquisition, DFT labeling, dataset growth, and optional retraining |
| Phase exploration | `matsim_agents.workflows.run_phase_exploration` | Programmatic workflow; relaxation and optional AL are composed through callbacks |
| Agentic investigation | `matsim_agents.workflows.run_investigation` | Programmatic orchestration and persisted hypothesis revision; numerical work is delegated to phase exploration |
| Cross-facility benchmark | `benchmarks/portability/run.py` | Separates deterministic workflow contracts from `--qualification compute`, which executes real MLIP and QE relaxation configs and emits a mandatory scientific summary |
| Scientific debate | `matsim-agents debate debate.yaml` | Runs independently configured LLMs that challenge a shared hypothesis for a user-selected number of rounds, then persists the transcript and synthesis |

## Multi-model scientific hypothesis debate

The debate workflow is separate from phase exploration and numerical evidence.
It assigns the same hypothesis to at least two LLM participants. Within each
round, every participant sees the prior transcript and must identify which peer
claims it supports or disputes, expose assumptions, and propose falsification
tests. Speaking order rotates between rounds, so one model does not permanently
receive either the first-turn or last-turn advantage. In equal mode, every
participant produces an independent final verdict; no designated model is
allowed to reinterpret the panel or manufacture consensus.

```yaml
hypothesis: "Pressure stabilizes a metastable silicon phase at room temperature."
rounds: 3
output_root: ./runs
debate_mode: equal
synthesis_method: independent_verdicts
participants:
  - name: theorist
    role: first-principles condensed-matter theorist
    provider: vllm
    model: Qwen/Qwen2.5-72B-Instruct
    base_url: http://localhost:8000/v1
  - name: experimentalist
    role: skeptical high-pressure experimentalist
    provider: ollama
    model: qwen2.5:14b
  - name: reviewer
    role: independent materials-science reviewer
    provider: openai
    model: gpt-4o-mini
```

Run `matsim-agents debate debate.yaml`. The run directory records resolved
participant identities, complete ordered transcript, synthesis, and provenance.
LLM statements remain hypothesis-level evidence until calculations or
experiments verify them. `max_transcript_chars` bounds the context sent to each
model (default 60,000) without truncating the transcript saved on disk.

`debate_mode: equal` is the default. Participant roles are ignored, all models
receive the same neutral system instructions, and
`synthesis_method: independent_verdicts` saves one conclusion per model. To run
an intentionally asymmetric panel instead, select `debate_mode: role_based`,
assign participant roles, and optionally use
`synthesis_method: designated_model` with `synthesis_participant`.

For all four supported combinations, complete configuration examples, output
semantics, and artifact definitions, see
[Scientific hypothesis debate](scientific-debate.md).

For deployment qualification across the entire first-class model catalog, use
`benchmarks/portability/all_model_scientific_debate.py`. Unlike a user-selected debate, this
benchmark fails closed unless every catalog model completes at least two rounds
and the saved dialogue assigns a unique contribution ID to every model turn.

“Supported” means that the workflow contract exists. It does not imply that a
licensed VASP binary, POTCAR library, QE pseudopotentials, or a particular MLIP
checkpoint is installed at a site.

## Structure relaxation

`ScientificRelaxationConfig` supports three modes:

- `mlip`: relax entirely with the selected machine-learned potential;
- `dft`: relax directly with Quantum ESPRESSO or VASP;
- `mlip-dft`: perform an MLIP warm start, then refine the resulting geometry
  with DFT.

The configuration declares the structure, output root, geometry controls,
force tolerance, maximum steps, backend settings, and approval policy. Atomic
and cell relaxation are independent choices. Fixed atoms, charge, spin,
pressure, symmetry preservation, and parent-run lineage are explicit inputs.

See `examples/relaxation/scientific_relaxation.example.yaml` for the complete
shape. DFT modes require a `dft` block and, by default, explicit approval:

```yaml
mode: mlip-dft
structure_path: structures/Si.vasp
output_root: runs
geometry:
  relax_atoms: true
  relax_cell: false
approvals:
  before_dft: true
dft_approved: false
dft:
  backend: qe
  pseudo_dir: /path/to/pseudopotentials
```

## Active learning: three separate decisions

DFT labeling, retraining, and model promotion are intentionally independent:

1. Acquisition selects uncertain structures and DFT labels them.
2. `trainer.enabled: true` trains a candidate model from the augmented dataset.
3. `trainer.promote_model: true` allows that candidate to drive the next
   iteration, but only when `promotion_approved: true` is also recorded.

The safe default only accumulates validated DFT labels:

```yaml
trainer:
  enabled: false
  promote_model: false
  promotion_approved: false
```

This is not a frozen or incomplete form of active learning. It is a valid data
acquisition workflow whose output can be reviewed and trained offline. Enabling
training does not silently replace the deployed model.

New DFT frames are checked for finite energy and forces, correct force shape,
and duplicate geometry. Dataset manifests preserve hashes, backend identity,
energy-reference metadata, and validation outcomes. VASP and QE energies must
not be mixed without an explicit, recorded reference transformation.

## Phase exploration

`PhaseExplorationPolicy` controls four independent behaviors:

```yaml
relax_structures: true
active_learning: false
retrain_mlip: false
reevaluate_after_retraining: false
ranking_mode: relative_phase_ranking
```

Retraining requires active learning. Re-evaluation requires both retraining and
successful model promotion. Compute budgets may cap candidates, MLIP
relaxations, DFT calculations, AL iterations, and node-hours.

`relative_phase_ranking` compares converged candidates within one exploration.
It is not a convex-hull claim. `convex_hull_ranking` additionally requires
method-compatible elemental and competing-phase references. Residual forces
filter unconverged structures; they are not added to formation energies.

## Agentic investigation

The investigation layer stores the original user request, the LLM-generated
scientific hypothesis, explicit property tasks, each phase-exploration result,
and subsequent hypothesis revisions. A new interaction can consume a previous
result without overwriting it. Run identifiers combine a UTC timestamp and a
random suffix, preventing concurrent studies from sharing a directory.

LLM proposals have `hypothesis` evidence. They do not become DFT or
experimental claims merely because a lower-level workflow was dispatched.

## Approval, evidence, and failure semantics

`ApprovalPolicy` exposes gates before DFT, retraining, and model promotion.
`EvidenceLevel` distinguishes hypotheses, MLIP predictions/relaxations,
low-fidelity DFT, converged DFT, higher-accuracy DFT, and experiment.

Every failed or rejected result must include a reason. Unconverged jobs remain
in the run record and are excluded from rankings rather than silently dropped.

Related documentation:

- [Distributed DFT dispatch](distributed-dft-dispatch.md)
- [Run artifacts and restarts](run-artifacts-and-restarts.md)
- [Cross-facility portability benchmark](../benchmarks/portability/README.md)
