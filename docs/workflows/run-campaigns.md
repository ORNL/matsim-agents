# Run a workflow or E3 campaign

Choose a task first:

- A small MLIP-only search: [bounded exploration](simple-exploration.md).
- A frozen E3-MLIP production stream: the preset below.
- The frozen E3 Round-1 BCC VASP batch on Aurora: the DFT preset below.
- An advanced workflow: the common interface and capability table below.
- After execution: [interpret results](interpret-results.md).
- To implement scientific changes: [extend the library](extend-library.md).

## The common interface

```bash
matsim-agents workflow list
matsim-agents workflow init explore ./search.yaml
# Edit the native config identified by config: in search.yaml.
matsim-agents workflow check ./search.yaml
matsim-agents workflow run ./search.yaml
# Use the invocation directory printed by run, not the native scientific directory.
matsim-agents workflow status WORKFLOW_RUN_DIRECTORY
matsim-agents workflow summary WORKFLOW_RUN_DIRECTORY
matsim-agents workflow resume WORKFLOW_RUN_DIRECTORY --native-run SCIENTIFIC_RUN_DIRECTORY
```

`init`, `status`, and `summary` do not calculate. `check` validates configuration,
native CLI arguments, and the relevant existing native configuration model.
It does **not** qualify authentication, model/device ABI, live services, or
scheduler readiness. E3-MLIP uses the native protocol-only validator; E3-DFT
additionally executes the existing **dry-run**
input/PAW/executable/collision validator, never `qsub`.

`run` delegates to the existing native workflow. It streams native output to
the console and retains it in an invocation log. Interactive chat retains stdin.
An invocation has its own configuration, status, JSON summary, and Markdown
report; the native workflow continues to own scientific results and checkpoints.
Two processes cannot execute the same invocation concurrently.

Relative `config`, `output_root`, and E3 settings paths are resolved against the
wrapper YAML. Paths inside a native config retain **that workflow's** existing
semantics. The `arguments` list is passed literally without a shell; argument
paths retain the current working directory's semantics. Use absolute paths when
moving a wrapper between directories. Native execution records its working
directory and reuses it on resume, preserving those relative paths. Do not put credentials in configs or
arguments: invocation evidence records them.

## E3-MLIP: current three-round benchmark

```bash
matsim-agents workflow init e3-mlip ./mlip-campaign.yaml
# Select a material class and set the prepared worker_python and paths.
matsim-agents workflow check ./mlip-campaign.yaml
```

The annotated [MLIP preset](../../examples/workflows/e3-mlip.yaml) uses the
[current production manifest](../../benchmarks/e3/manifests/e3-mlip-production.json).
The E3 presets require a source checkout containing the native campaign files.
The wrapper gives their subprocesses that checkout's package paths, so copied
presets can be checked from outside the repository.
The benchmark runs one LLM-guided stream per class: three adaptive rounds,
seed `20261113`, and at most 64 candidate relaxations per round. It does not
include a rule-based arm or multiple seeds, so report descriptive results only.

- Materials: `bcc_refractory`, `fcc_cantor`, `lifepo4`, `phosphorene`,
  `nb_ta_o`, `zn_formate_mof`.
- Strategy: `llm_guided_adaptation`.
- Maximum total: 1,152 candidate relaxations across all six classes.
- Use the prepared inference environment and `MACE_ARTIFACT_DIR` from the
  [Perlmutter execution runbook](../../research/paper/perlmutter-e3-e4-execution-runbook.md).
  The native runner verifies the frozen model artifact; wrapper static preflight
  does not load MACE or qualify CUDA.

Each material's allocation is limited to 16 premium-QOS GPU nodes for two
hours; it uses 64 workers, a five-minute parent-enforced geometry-task cutoff,
and a hard 30-minute round deadline. The six allocations reserve at most
768 physical GPU-hours (192 GPU-node-hours). Premium billing multipliers are
separate; verify current project charges. A five-element array concurrency
limit does not guarantee simultaneous start or continuous GPU utilization.
The round deadline includes generation and LLM work as well as GPU evaluation.
GPU workers may use up to 15 minutes of that round budget to initialize MACE;
the remaining time is still bounded by the 30-minute round deadline.

```bash
# Keep the existing manifest on resume. This performs no inference or submission.
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -m benchmarks.e3.production validate \
  --manifest benchmarks/e3/manifests/e3-mlip-production.json

# Example: actual CPU generation, then authorized GPU submission for LiFePO4 (index 2).
# This stops without submission if allocation, capacity, or real generation fails.
MATSIM_E3_MLIP_RESOURCE_AUTHORIZED=m5216:premium:768 \
  PROJECT_ROOT="$PWD" bash \
  deployments/perlmutter/jobs/submit-e3-mlip-production-perlmutter.sh 2
```

Use the [CPU-generation submission helper](../../deployments/perlmutter/jobs/submit-e3-mlip-production-perlmutter.sh),
not direct `sbatch`, for each stream. Index mapping is 0 BCC, 1 FCC, 2 LiFePO4,
3 phosphorene, 4 Nb-Ta-O, 5 Zn-formate. Do not relaunch completed streams.
For a partially completed stream, set `MATSIM_E3_MLIP_START_ROUND` to its first
incomplete round; all earlier round summaries must be complete.

The CPU gate executes the selected actions with production seeds, validates
composition and geometry, and applies the unchanged symmetry-aware deduplication.
Only a full 64-candidate workload receives `generation-preflight.json`; the GPU
launcher checks this receipt and all structure checksums before reservation.
Generation time is deducted from the round's 1,800-second budget, excluding queue
waiting time. Later adaptive rounds are generated only after the previous round's
optimization results exist, and before dispatching that round's GPU tasks.

Both acquisition policies use the same executable catalog and joint capacity
validator, although the production experiment remains LLM-only. The catalog
records installed AFLOW prototypes, template availability, perturbation bounds,
capacity-group aliases, and remaining operational capacity. Small four-atom
phosphorene reconstructions and 0.5% homogeneous MOF strains have a conservative
capacity of one per group/tier; this is not a physical limit on possible structures.
The existing affine-strain bounds and equivalence tolerances are unchanged.
Generator revision 2 adds explicitly recorded, bounded non-affine displacements
of at most 0.04 Angstrom to the strained-olivine family (small relative to
interatomic bond lengths), preserving composition and the olivine template.
Pure single-antisite actions have their template-symmetry orbit capacities
of 2, 4, 8, and 16 for tiers 4, 8, 16, and 32; aliases share those limits.

Round-local `acquisition/` records contain versioned catalogs, prompts, unmodified
LLM responses saved before parsing, parsed and accepted allocations, and precise
rejection/shortfall diagnostics. Partial generation artifacts are retained and
never counted as MLIP evaluations. The deterministic policy may make another
explicit decision using generation-shortfall evidence. The frozen production
protocol allows **one LLM call per round**: a rejected or exhausted proposal does
not permit an automatic extra call. Replacing an already-consumed legacy
decision requires scientific authorization; missing historical raw responses
are recorded as unavailable, never reconstructed or invented.

The CPU submission helper reserves the full allocation in the shared resource
ledger before `sbatch`. The GPU launcher verifies the allocation's actual
account, QOS, nodes, time limit, and reservation; it does **not** reconcile
usage on exit. After Slurm reports the allocation as terminal, manually run
`budget-reconcile` with the helper's printed reservation ID and the same ledger
path, following the [accounting command](../../benchmarks/e3/README.md).
Until verified terminal `sacct` usage is reconciled, the full reservation
remains charged against the 768-GPU-hour cap and can block further submissions.
This includes startup failures and unavailable or ambiguous scheduler records.
The launcher uses a private, unique node-local `COPILOT_HOME`
for Copilot sessions, initializing it from the user's CLI config when present
and removing it at exit. This keeps runtime metadata locks off CFS. Array
indices 0-5 map to the six classes listed above. The scheduler rejects all six
submitted tasks at once with its current five-task submit limit; submit the
last task only after a slot opens. Do not bypass these checks or run the
workflow wrapper as a second writer in the same allocation.

### Hard task and round cutoffs

Each geometry task runs in a spawned process. At 300 seconds its controller
kills the process even if it is stuck inside a force evaluation/optimizer step,
then records it as terminal and nonconverged. Partial artifacts are retained;
no final energy, force, or convergence claim is invented.

The parent supervises each complete round for 30 minutes, including structure
generation, the remote LLM decision, GPU evaluation, and cleanup. Timeout
terminates the round process group and writes an incomplete campaign report.
The 2-hour Slurm allocation is the outer limit; the three 30-minute round
budgets leave up to 30 minutes for startup and finalization.

The run writes `report.md`, `campaign-summary.json`, round manifests/summaries,
and per-task structure/trajectory/result evidence under the configured output
root. Failed or nonconverged attempts are retained and not silently replayed.
The lowest MLIP energy is not a ground-state or DFT-validation claim.

## E3-DFT/VASP: validate the existing frozen Aurora batch

```bash
matsim-agents workflow init e3-dft ./vasp-campaign.yaml
matsim-agents workflow check ./vasp-campaign.yaml
# Still validation-only; writes an invocation record/report:
matsim-agents workflow run ./vasp-campaign.yaml
```

The [DFT preset](../../examples/workflows/e3-dft.yaml) is specifically
`E3-round1-bcc-01`: three BCC calculations, Aurora/PBS, HydraGNN allocation,
and the frozen 72 node-hour cap. It is **not** a generic VASP campaign or a
Perlmutter/Slurm submission preset. Follow the
[batch staging and authorization guide](../../deployments/aurora/jobs/e3-round1-bcc/README.md)
for private inputs, PAWs, executable qualification, and authorized execution.
Neither preset initialization nor checking stages proprietary inputs.

Submission requires `--approve-execution` **and** native manifest authorization:

```bash
# Only after independent authorization and successful input checks:
matsim-agents workflow run ./vasp-campaign.yaml --approve-execution
```

The existing submission helper retains executable/PAW/input hashes, output
collision checks, and its accepted-job ledger. Successful invocation means the
validator/submission helper exited successfully, **not** that VASP converged or
that a submitted job completed. Accepted calculation IDs are never resubmitted
by this interface. `workflow resume` is unsupported for this batch; inspect the
native scheduler/jobs and use its documented same-ID restart procedure.

## Advanced workflow catalog

All entries expose `check`, `run`, `status`, and `summary` through the same
wrapper. `resume` is capability-based: a command is available, but rejects
workflows without a safe native restart contract instead of silently replaying.

| Workflow | Native command | Wrapper input | Native resume |
| --- | --- | --- | --- |
| `explore` | `explore run` | `config` | Version-2 scientific directory via `--native-run` |
| `relax` | `relax` | `config` | Unsupported |
| `characterize` | `characterize` | `config` | Same contract and native step journal |
| `debate` | `debate` | `config` | Unsupported |
| `llm-check` | `llm-check` | `config` | Unsupported |
| `al` | `al run` | `config` | Native loop with `loop.resume: true` |
| `objective` | `run` | Objective/options in `arguments` | Unsupported |
| `plan` | `plan` | Objective/options in `arguments` | Unsupported |
| `supervisor` | `supervisor-run` | Composition/options in `arguments` | Unsupported |
| `chat` | `chat` | Options in `arguments` | Native session via `--native-run` |
| `e1` | `e1-development` | `config`; `--output DIR` in `arguments` | Same manifest/output, native committed-request reuse |
| `known-answers` | `benchmark known-answers` | Options in `arguments` | Unsupported |
| `agent-eval` | `benchmark agent-eval` | `config` | Unsupported |
| `e3-mlip` | Frozen production runner | Typed `settings` | Native production stream |
| `e3-dft` | Frozen Aurora batch runner | Typed `settings` | Unsupported; never resubmit accepted IDs |

Native approvals, backend selection, evidence policies, and compute budgets
remain in their existing native config/options. `--approve-execution` is an
additional E3 gate; it does not authorize DFT/LLM work in unrelated workflows.
For E1, error retries remain governed by its explicit native `--retry-errors`;
they are not enabled automatically by wrapper resume.

For example, an objective wrapper is:

```yaml
workflow: objective
arguments:
  - "Find a candidate Nb oxide structure"
  - --mlip-backend
  - mace
output_root: ./workflow-records
```

Use native `--help` to choose options. Backend/service/scheduler utilities
(`capabilities`, `resources`, `inventory`, `ledger`, `coordination`) are operational
tools, not computational workflows; they retain their own interfaces. Native
CLI and public Python workflow APIs are still directly usable.
