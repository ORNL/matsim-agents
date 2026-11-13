# Perlmutter Runbook: Unfinished E4 DFT and E3-MLIP

Status snapshot: 2026-11-11. This runbook is for the Perlmutter session. Slurm job files and local environment-variable setup are intentionally left to the operator. **No Perlmutter jobs are submitted or authorized by this document.** The previous 1,488 node-hour approval was for the Aurora `capacity` queue and `HydraGNN` allocation; it does not transfer to NERSC.

## 1. Synchronize the repository

On Perlmutter, fetch the branch containing the frozen manifests, code and this runbook:

```bash
git fetch origin
git switch research/e3-e4-completion
git pull --ff-only origin research/e3-e4-completion
git log -1 --oneline
```

Keep raw runs and temporary model caches out of Git. Use separate Perlmutter run roots and unique platform-specific run IDs so Perlmutter output never overwrites Aurora output.

## 2. E4 DFT on Perlmutter

### 2.1 What remains

The E4 first-production batch manifest contains 20 cases: 8 BCC, 4 FCC and 8 LiFePO4. The last inspected Aurora records (2026-11-11) showed:

| Case group | Aurora result | Perlmutter handling |
|---|---|---|
| E4-BCC-0001, E4-BCC-0561 | `converged:N0` | Already qualified as DFT data. Re-run only as a separately authorized cross-platform timing/reproducibility measurement; do not count as new unique structures. |
| E4-BCC-1121 | Low-stage `stage-step-limit` failure | Preserve failed attempt. Any new attempt needs a fresh platform attempt/output ID and explicit recovery/authorization decision. |
| E4-BCC-1201, E4-BCC-1501 | Normal-stage exit 127 | Failed launch, no qualified DFT result. Preserve the records; diagnose before any Perlmutter rerun. |
| E4-BCC-1701, E4-BCC-1801, E4-BCC-1805 | No first-batch run record observed | Unexecuted candidates, subject to the batch's existing authorization and duplicate policy. |
| E4-FCC-0001 | PBS exit 127, no VASP run record | Failed launch, not a DFT result. |
| E4-FCC-0006, E4-FCC-0011, E4-FCC-0016 | No run record observed | Unexecuted candidates, but FCC magnetic method/initialization and qualification remain a separate gate. |
| E4-LFP-0002 | PBS exit 127, no VASP run record | Failed launch, not a DFT result. |
| E4-LFP-0003, E4-LFP-0010, E4-LFP-0025, E4-LFP-0050, E4-LFP-0100, E4-LFP-0505, E4-LFP-0506 | No run record observed | Unexecuted candidates; retain the frozen 112-atom input and spin setup. |

Reconcile this table against current Perlmutter/Aurora scheduler history and each case's `run-record.log` before submission. A PBS/Slurm job exit code is not itself a qualification result; require converged SCF at every ionic step, the intended ionic termination and the frozen force threshold.

The E4 scaling manifests are a separate BCC workload:

- Strong: fixed eight BCC structures; 1 node in two four-task segments (96 h each), 2 nodes/96 h, 4 nodes/48 h, 8 nodes/24 h.
- Weak: matched compositions with two frozen disorder realizations per node; 1/2/4/8 nodes, each tier capped at 48 h; two tasks per node run sequentially.
- Manifest caps: 768 node-hours strong plus 720 node-hours weak = 1,488 node-hours total. These caps were approved for Aurora only; request and record a separate Perlmutter allocation before running them.

The Aurora scaling jobs 9069490 (weak, one node) and 9069491 (strong, eight nodes) were queued in the last checked snapshot and had no dispatcher output yet. Check their current status before deciding whether a Perlmutter run is a platform comparison or a new execution.

### 2.2 Do not submit the Aurora entrypoint on Perlmutter

The current files are not portable as-is:

- `deployments/aurora/jobs/job-e4-scaling.pbs` is PBS and the dispatcher requires `PBS_JOBID`/`PBS_NODEFILE`.
- `benchmarks/e4/aurora_dft/scaling.py` invokes the Aurora PBS wrapper.
- `deployments/aurora/jobs/e4-first-batch/job-bcc.pbs` assumes PBS, an Aurora case layout, and the Aurora launcher.
- The current `deployments/perlmutter/jobs/job-al-dft-scaling-perlmutter.sh` is a different 32-case FCC active-learning experiment, not the fixed E4 BCC scaling campaign.

The Perlmutter VASP implementation to reuse is `deployments/perlmutter/launchers/_vasp-step-perlmutter.sh`; the existing VASP build path is VASP 6.6.1 (`deployments/perlmutter/setup/build-vasp-gpu-perlmutter.sh`, with the NERSC facility module preferred if it passes its smoke test). The step launcher uses Slurm `srun`, four A100 GPUs/node and a default four-rank/16-thread layout. Do not copy Aurora's eight-rank/eight-thread, `KPAR=8` placement blindly. Keep physical DFT parameters unchanged; adapt only hardware-parallel tags and verify `KPAR` is compatible with the eight-point BCC mesh. The 112-atom LiFePO4 mesh and spin treatment also need to remain the frozen protocol.

### 2.3 Perlmutter port sequence

The operator should create a Perlmutter-specific E4 Slurm entrypoint/adapter, separate from the PBS file, with these responsibilities:

1. Read the same authorized task manifest and reject duplicate task IDs, changed hashes, unsupported tiers and missing Perlmutter authorization.
2. Stage frozen case inputs and licensed PAWs onto Perlmutter storage using an approved data-transfer route. The manifests' case paths point to Aurora-adjacent directories and won't exist automatically in a Perlmutter clone.
3. Reconstruct each case directory from the committed source structures/templates and the site's licensed PAWs, then record hashes for START/INCAR, POSCAR, KPOINTS, POTCAR checksum/components and executable. Do not commit POTCAR bytes.
4. Discover the Slurm allocation's hosts, verify its node count, assign disjoint nodes to parallel tasks, and use the Perlmutter VASP step launcher. For weak scaling, run two tasks per node sequentially.
5. Use a fresh output root per `platform/run-id/experiment/tier/segment/calculation-id`. Never overwrite or resume an Aurora case under the same attempt identity.
6. Record Slurm job and step IDs, node/CPU/GPU layout, timestamps, requested and elapsed node-hours, every SCF and ionic status, final force, restart/checkpoint identity, exit code and final `converged:N0` status.
7. Stop a later task wave when a required task fails or is incomplete. Keep failed attempts and consumed node-hours in the report.

For classical strong-scaling interpretation, all node counts must use the exact same fixed eight tasks and compare application makespan; these repeated structures are performance replicates, not new scientific discoveries. For weak scaling, preserve the frozen two-unique-structures-per-node assignments and report per-node throughput. Do not count queue wait as application runtime.

NERSC QOS/account limits are time-dependent. The repository documents premium QOS as having a 48-hour maximum; verify current Perlmutter account/QOS rules before scheduling a 96-hour point. If a requested point exceeds the current wall limit, do not silently truncate it: design an explicit restartable segmentation method and retain one scientific calculation identity across attempts.

### 2.4 E4 data handoff and analysis

After a Perlmutter run, preserve its outputs and accounting under a Perlmutter-specific root. Compare each completion against the original manifest and inputs; qualify results by raw VASP outputs, not summary files alone. Keep Aurora and Perlmutter ledgers separate, then merge only the measured performance records into the scaling analysis. First-batch structures already qualified on Aurora remain one unique structure even if rerun to compare hardware timing.

## 3. E3-MLIP on Perlmutter

E3 explicitly sets `relative_increase_threshold=None`,
`max_cell_step_strain=None`, `max_cell_total_strain=None`, and
`stress_tolerance_eV_per_A3=None` (JSON `null`). This disables the force-growth
and cell-deformation abort thresholds and the separate numerical stress
convergence criterion. A transient force increase must not terminate FIRE
relaxation. Filter forces include cell pseudo-forces: they are not
the same as atomic forces. Principal strain is not volume change.
The disabled growth/deformation thresholds were screening limits, not
convergence criteria or proof of stability.
The atomic and generalized-force convergence target remains 0.02 eV/A.
Stress still drives cell relaxation through ExpCellFilter; `None` does not
freeze the cell or remove its generalized forces from convergence.

The shared relaxation input, discovery wrapper, orchestration, and CLI retain
their general 20% force-growth default; E3 explicitly overrides it with `None`.
Cell-strain and stress tolerances are explicitly unset in the
E3 protocol; general non-E3 relaxation inputs retain optional, unset cell
limits to preserve fixed-cell workflows. Each abort threshold accepts Python
`None` (JSON `null`) in the relaxation input; in E3, changing the corresponding
optimizer setting to `None` also requires regenerating the manifest. Disabling
a safeguard does not disable finite-value/valid-cell checks, MOF connectivity
protection, or step/time limits. Existing attempts use a different optimizer
contract and must not be silently reused under the revised manifest.
CPU-only first-round preflight is compatible with a fresh hard-deadline
stream: execution verifies its receipt before registering the execution policy.
Existing optimization attempts or completed rounds without that policy are
still rejected.

The submission helper accepts `MATSIM_E3_MLIP_ACCOUNT` (default `m5216`) from
the authorized project list `m5216`, `m5048`, `m4828`, `amsc001`, `amsc004`,
and `dasrepo`. Set `MATSIM_E3_MLIP_RESOURCE_AUTHORIZED=ACCOUNT:premium:768`
to match, and use the same `MATSIM_E3_MLIP_RESOURCE_STATE` across accounts.
Slurm must permit premium QoS for the selected account; account selection
does not enlarge or reset the shared GPU-time cap. The frozen manifest's
account is the default, while allocation metadata records the actual account.

### Optional MACE inference import isolation

For a **dedicated MACE inference process**, the opt-in
[auxiliary launcher](../../scripts/mace-inference.sh) avoids inadvertent
Transformers initialization without uninstalling anything:

```bash
PYTHONPATH="$PWD/src:$PWD" bash scripts/mace-inference.sh .venv-mace/bin/python -m benchmarks.e3.production --help
```

For an authorized Slurm run, put the launcher **inside** `srun`, before the
Python executable, retaining all existing task/GPU options. Alternatively,
export `MATSIM_MACE_SKIP_TRANSFORMERS=1` before launching matsim MACE workers.
The calculator factory applies the shim before importing MACE, including in
spawned workers, and logs a warning once per process. The launcher does not
submit a job, change a scientific protocol, or authorize additional compute.

**Why:** MACE 0.3.16 imports training utilities during calculator initialization;
these import TorchMetrics, whose optional text/image metrics can import
Transformers simply because it is installed. The installed Transformers version
recursively scans its model source directories. In the Perlmutter diagnostic
benchmark, all four workers waited on shared-filesystem metadata for unrelated
Transformer source files, while a separate 79.5 MB checkpoint read took 0.205 s.
This identifies dependency discovery as an observed bottleneck, not MACE
checkpoint size or GPU arithmetic. It does not measure total model load time.

**Cautions:** this is a process-local compatibility shim, not an upstream MACE
setting. It overrides Lightning's private requirement check **only for
Transformers**, allowing TorchMetrics to skip its optional features. Actual
Transformers imports then fail explicitly; no dummy model or metric is supplied.
Enable it before importing MACE, TorchMetrics, or Transformers; late activation
fails instead of claiming isolation succeeded. Other packages and ordinary
launches remain unchanged. The setting persists for the process and its children:
do not use it for Hugging Face LLMs, Transformer-based metrics, or combined
inference/training workflows that need Transformers. It does not prevent other
dependencies from accessing shared storage.

Only matsim's MACE calculator factory automatically installs the shim; scripts
that import `mace` directly must call
`configure_mace_inference_imports()` from
`matsim_agents.backends.mlip.mace_inference` first. Do not enable it globally in
shell startup files. Revalidate it after Lightning/TorchMetrics/MACE dependency
upgrades: check clean startup with no Transformers modules loaded, identical
energy/force/stress predictions with the same checkpoint and precision, and
worker inheritance. GPU startup and throughput still require an authorized
compute-node benchmark; CPU checks alone do not establish GPU speedup.

Local validation (2026-11-12) used separate CPU processes, the same frozen
MACE-OMAT-0 medium checkpoint, fp64, and one CPU thread. Calculator construction
took 116.55 s normally and 8.38 s with isolation. Energies, every force component,
and every stress component matched exactly for the 4-atom phosphorene and
14-atom oxide diagnostic structures; no Transformers modules were loaded in the
isolated process. These sequential startup timings are subject to filesystem
caching and are not a controlled GPU speedup measurement. Regression tests cover
default behavior, selective suppression, explicit import failure, idempotency,
late activation, invalid settings, and spawned-worker inheritance:

```bash
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -m pytest -q tests/test_mace_inference_imports.py tests/test_mace_backend.py tests/test_documentation.py
```

### 3.1 Model and benchmark scope

The benchmark uses **MACE-OMAT-0 medium**, trained on OMat24, loaded with
MACE-Torch alias `medium-omat-0`. Its artifact SHA-256 is
`d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`.
MOF transfer applicability is not independently established and must be
reported as a limitation.

The current benchmark supersedes the former large paired design. It runs one
LLM-guided stream with seed `20261113` for each of the six material classes,
with three adaptive rounds and at most 64 candidate relaxations per round.
That is at most 192 tasks per class and 1,152 tasks total. There is no
rule-based arm or multiple-seed replicate; report descriptive outcomes, not
comparative effects or confidence intervals. Use generated geometry and MLIP
outcomes only. Do not read DFT references or submit VASP; E3-DFT remains
separate.

### 3.2 Resource limits and launch

Submit through the
[`CPU-generation helper`](../../deployments/perlmutter/jobs/submit-e3-mlip-production-perlmutter.sh),
which reserves the entire allocation against the shared ledger before `sbatch`.
The GPU entrypoint is
[`job-e3-mlip-production-perlmutter.sh`](../../deployments/perlmutter/jobs/job-e3-mlip-production-perlmutter.sh).
It maps one array element to each class and requests 16 GPU nodes, four A100
GPUs per node, 64 workers, premium QOS, and a two-hour allocation. Each of the
three rounds has a hard 30-minute wall-clock deadline, including generation,
LLM selection, evaluation, and cleanup. Each geometry task is terminated by
its parent process after five minutes and recorded as nonconverged. GPU workers
may use up to 15 minutes of the round budget to initialize MACE.

The maximum allocation is 192 GPU-node-hours, or 768 physical GPU-hours, across
the six classes. The GPU launcher retains each full reservation through startup,
execution, and teardown. After a job is terminal, run `budget-reconcile` using
the helper's printed reservation ID; only verified final `sacct` usage releases
unused time. Failed/ambiguous submissions and unavailable scheduler records
remain conservatively charged. See the [accounting instructions](../../benchmarks/e3/README.md)
for the command. Premium billing multipliers are separate from physical GPU
hours; verify current project charges and authorization before launch.
Five array elements are allowed concurrently. The final element may start
later, so the array does not promise all six jobs start together.

```bash
# Keep the existing production manifest and attempt records on resume.
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -m benchmarks.e3.production validate \
  --manifest benchmarks/e3/manifests/e3-mlip-production.json

# Actual CPU generation must pass before this helper requests GPU resources.
# Example: material index 2 is LiFePO4. Use only the affected stream indices.
MATSIM_E3_MLIP_RESOURCE_AUTHORIZED=m5216:premium:768 \
  PROJECT_ROOT="$PWD" bash \
  deployments/perlmutter/jobs/submit-e3-mlip-production-perlmutter.sh 2
```

The [generation gate and recovery contract](../../docs/workflows/run-campaigns.md)
describe shared capacities, installed-prototype filtering, raw-proposal retention,
and generation-exhaustion evidence. No first-round GPU submission is allowed
without a full real-generation receipt. Generation time remains part of the
30-minute round budget. Later rounds are checked only after preceding optimization
outcomes exist. An already-consumed LLM decision cannot be replaced under the
frozen one-call-per-round limit without additional scientific authorization.
Generation, source replays, and execution retries have durable cumulative time
ledgers; retries do not receive a new 30-minute allowance. Hard-interrupted
attempts retain their remaining time reservation until verified reconciliation.
Legacy interrupted evidence without timing records is not assigned zero duration.

The launcher verifies the allocation's actual account, QOS, nodes, tasks,
manifest, model environments, and resource reservation before calculation.
It uses `$PROJECT_ROOT/.venv` and `$PROJECT_ROOT/.venv-mace` by default; set
`MATSIM_E3_MLIP_BASE_PYTHON` and `MATSIM_E3_MLIP_WORKER_PYTHON` only when
using prepared environments elsewhere.
The current QOS rejects more than five submitted tasks; wait for capacity before
submitting another material. The launcher places Copilot session state in a unique node-local
`COPILOT_HOME`, seeds it from the user's Copilot CLI config when present, and
removes it on exit. This keeps runtime metadata locks off CFS.

The registered model is selected in
`benchmarks/e3/scientific-objectives.yaml`; one configured Copilot model is
queried once per round. Each attempt preserves its initial geometry,
optimizer trajectory, per-step log, terminal reason, hardware, and timing.
The Markdown campaign report distinguishes incomplete rounds and counts
nonconverged and failed outcomes explicitly. No throughput guarantee or
continuous 64-GPU utilization is implied. model are available
from compute nodes. The 60-second optimization target is checked between
completed steps, excludes loading and other allocation overhead, and can be
exceeded by a slow or stalled step. After all streams finish, generate the
campaign-wide relaxed-minimum report offline (not during acquisition):

```bash
python -m benchmarks.e3.production cluster-minima \
  --output-root /path/to/e3-mlip-results \
  --output /path/to/e3-mlip-results/relaxed-minima.json
```

### 3.3 E3-MLIP pre-production gate

Follow the six production-start gates in the E3-MLIP plan: valid class-specific
generators including COD-derived connectivity-checked Zn-formate variants;
checkpoint hash and species checks; identical frozen optimization settings;
restartable task manifests and round barriers; matched adaptive budgets; and
explicit Perlmutter resource authorization. No standalone pilot is required
or introduced. The production job checks the exact authorization token and
enforces the cumulative GPU-hour cap from its persistent ledger.

## 4. Perlmutter decision checklist

- [ ] Fetch the current `research/e3-e4-completion` branch.
- [ ] Reconcile the current E4 Aurora queue and run records; do not duplicate qualified tasks unless measuring platform performance.
- [ ] Stage E4 inputs/PAWs and verify hashes; obtain separate Perlmutter allocation approval.
- [ ] Write the E4 Slurm dispatcher/entrypoint around the existing Perlmutter VASP step launcher; do not call PBS wrappers.
- [ ] Run and qualify one E4 Perlmutter pilot before the scaling tiers.
- [ ] Confirm the E3-MLIP premium GPU-hour authorization and current project billing multiplier before submission.
- [ ] Verify the prepared MACE/Copilot environments and checkpoint on Perlmutter.
- [ ] Validate the generated E3-MLIP manifest and the actual six-element array/QOS behavior.
- [ ] Keep E3-MLIP MLIP-only; any DFT/VASP work requires the separate E3-DFT protocol and authorization.
