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

## E3-MLIP: choose one frozen production stream

```bash
matsim-agents workflow init e3-mlip ./mlip-campaign.yaml
# Inspect and edit material, strategy, seed, output directories, worker_python,
# and resource_state. Do not change the frozen scientific manifest.
matsim-agents workflow check ./mlip-campaign.yaml
```

The annotated [MLIP preset](../../examples/workflows/e3-mlip.yaml) uses the
[frozen production manifest](../../benchmarks/e3/manifests/e3-mlip-production-v1.json).
The E3 presets require a source checkout containing the native campaign files.
The wrapper gives their subprocesses that checkout's package paths, so copied
presets can be checked from outside the repository.
One invocation runs one material/strategy/seed stream, including its ten adaptive
rounds. This is **not** a smoke-test budget. The entire frozen campaign comprises
six materials, two strategies, and three seeds (360 material-stream rounds).

- Materials: `bcc_refractory`, `fcc_cantor`, `lifepo4`, `phosphorene`,
  `nb_ta_o`, `zn_formate_mof`.
- Strategies: `llm_guided_adaptation`, `rule_based_adaptation`.
- Seeds: `20261111`, `20261112`, `20261113`.
- Use the prepared inference environment and `MACE_ARTIFACT_DIR` from the
  [Perlmutter execution runbook](../../research/paper/perlmutter-e3-e4-execution-runbook.md).
  The native runner verifies the frozen model artifact; wrapper static preflight
  does not load MACE or qualify CUDA.

Actual execution requires **both** explicit approval and an existing Slurm
allocation with an active reservation for `SLURM_JOB_ID` in `resource_state`:

```bash
# Only inside an already approved and accounted allocation:
matsim-agents workflow run ./mlip-campaign.yaml --approve-execution
```

The wrapper does not submit Slurm jobs, create reservations, or reconcile
allocation charges. Preserve the reserve/release lifecycle and walltime/resource
limits in the [production launcher](../../deployments/perlmutter/jobs/job-e3-mlip-production-perlmutter.sh).
If using its command slot for the wrapper, keep its accounting and exit trap;
never run a second stream concurrently against the same reservation. The normal
launcher already runs the native stream directly and remains a supported choice.
Missing approval, allocation, or reservation is an error, not a CPU fallback.

### Five-minute hard geometry cutoff

The MLIP preset now explicitly sets `hard_optimization_seconds: 300`.
Each GPU's controller runs geometry tasks in a spawned process. At 300 seconds
it kills that process even if it is blocked inside a force evaluation/optimizer
step, then atomically records `status: nonconverged`, `converged: false`, and
`termination_reason: optimization_time_limit`. Partial artifacts are retained,
but no final energy/force or convergence claim is invented. Completed timeout
records are not automatically retried on resume.

The task budget includes model loading on first use, initial evaluation,
relaxation, and final scientific measurements. Worker-process initialization
has a separate bounded startup allowance; this is not a whole-round or
allocation walltime limit. Calculators are reused across successful tasks.
After a forced timeout the next task gets a fresh process/calculator.
Process scheduling and cleanup can add small overhead beyond 300 seconds,
but a stuck native calculation cannot continue indefinitely.

This is an explicit **new execution policy**, not a silent modification of the
frozen 60-second between-step protocol. It is stored in `execution-policy.json`,
each task/result's optimizer metadata, and the round's `execution_optimizer`.
Use fresh output streams; switching the deadline when resuming is rejected.
The underlying frozen manifest remains unchanged. Native invocation accepts
`--hard-optimization-seconds 300`; omitting it retains the frozen protocol.
This option does not itself change ten rounds to three or enforce the proposed
16-node/two-hour class-level schedule.

For an interrupted invocation, native round manifests remain authoritative:

```bash
matsim-agents workflow resume WORKFLOW_RUN_DIRECTORY --approve-execution
```

Resume needs an active allocation/reservation again and delegates native
production replay/reuse; it does not reinterpret scientific results.

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
