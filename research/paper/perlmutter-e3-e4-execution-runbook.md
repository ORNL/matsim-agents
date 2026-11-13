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

### 3.1 Selected frozen model and campaign contract

The selected primary checkpoint is **MACE-OMAT-0 medium**, trained on OMat24,
loaded with MACE-Torch alias `medium-omat-0` (E3 CLI alias
`mace-omat-0-medium`). See the [E3-MLIP exploration
plan](./e3/e3-mlip-exploration-plan-v1.md) for the frozen generation,
optimization, and adaptive-comparison design. Checkpoint species coverage and
class-specific transferability must pass the plan's feasibility gates; the MOF
domain is not established.

E3-MLIP uses generated geometry and MLIP optimization outcomes only: do not
load DFT references, labels, energies, forces, or results, and do not submit
VASP. E3-DFT remains separate and unchanged. The design target is 5,000 unique
valid starts per class **per strategy** (60,000 logical optimization jobs,
before matched-size controls and any physical cache reuse); this is not
submission authorization.

### 3.2 Current entrypoint and planned workflow

The [CPU preparation job](../../deployments/perlmutter/jobs/job-e3-mlip-preparation-perlmutter.sh)
remains a diagnostic and is not the production path. The production entrypoint
is
[`job-e3-mlip-production-perlmutter.sh`](../../deployments/perlmutter/jobs/job-e3-mlip-production-perlmutter.sh):
it runs one paired-seed/material/strategy stream per Slurm array element,
launches one MACE worker per GPU on one node, enforces round barriers, and
resumes only tasks without a terminal result. A persistent GPU-time ledger
reserves each allocation before work starts and caps cumulative use at 2,000
GPU-hours, including restarts. The complete adaptive schedule is generated
with `python -m benchmarks.e3.production plan`; future-round structures are
generated only after that round's strategy decision.

```bash
# Diagnostic only; this is not the revised exploration campaign.
PROJECT_ROOT="$PWD" sbatch -A <allocation> \
  deployments/perlmutter/jobs/job-e3-mlip-preparation-perlmutter.sh

# Generate all 360 adaptive round manifests (no structures or jobs are run).
python -m benchmarks.e3.production plan \
  --output benchmarks/e3/manifests/e3-mlip-production-v1.json

# After the 2,000 GPU-hour resource authorization:
MATSIM_E3_MLIP_RESOURCE_AUTHORIZED=m5216:regular:2000 \
  PROJECT_ROOT="$PWD" sbatch --account=m5216 \
  deployments/perlmutter/jobs/job-e3-mlip-production-perlmutter.sh
```

Submit with project account `m5216`; NERSC records GPU usage against billing
account `m5216_g` and maps `regular` to GPU QoS `gpu_regular`.

The production launcher uses `$PROJECT_ROOT/.venv` and
`$PROJECT_ROOT/.venv-mace` by default. Set
`MATSIM_E3_MLIP_BASE_PYTHON` and `MATSIM_E3_MLIP_WORKER_PYTHON` when a source
worktree shares the prepared base and MACE environments from another checkout.

The cell-relaxation protocol now uses native MACE stress for all families,
in-plane-only strain for phosphorene, explicit stress convergence, and
strain/connectivity guards; see the E3-MLIP exploration plan. The former
fixed-cell array was cancelled. Before submitting a replacement, verify the
updated manifest, qualify the cell-relaxation workflow and GPU throughput, and
obtain authorization for any revised resource request. Do not mix prior
fixed-cell results with this protocol.

The cap is 500 GPU-node-hours at four GPUs per node. A single full array is
limited to 1,999.2 GPU-hours (36 elements, one node each, 13 hours 53 minutes);
unused reservation is reconciled to actual allocation time for safe restarts.
The cap does not guarantee completion of all 61,120 logical optimizations:
that would require an average below 118 GPU-seconds per optimization, which
has not been measured. The controller preserves partial progress rather than
exceeding the authorization.

The production job requires the verified offline MACE cache and the base and
MACE virtual environments. Each structure attempt preserves its initial
geometry, optimizer trajectory, per-step log, terminal reason, hardware and
timing. It uses no DFT labels or references. The legacy reference-backed
E3-DFT commands and protocol below are unchanged and must not be used by
E3-MLIP.

The frozen LLM provider, model identifier, temperature, and token limit are
versioned in `benchmarks/e3/scientific-objectives-v1.yaml`; runtime environment
variables cannot override them. After all streams finish, generate the
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
- [ ] Prefetch MACE-OMAT-0 medium using the existing Perlmutter downloader (`mace_mp medium-omat-0`).
- [ ] Complete six-class E3 candidate generation, scientific/reference gates and updated frozen model protocol.
- [ ] Implement a live paired E3-MLIP Slurm/DFT controller; current E3 commands only prepare, replay or freeze decisions.
- [ ] Obtain a separate bounded Perlmutter E3-MLIP DFT allocation approval before any campaign DFT submission.
