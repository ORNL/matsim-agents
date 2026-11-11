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

The current [E3-MLIP Slurm job](../../deployments/perlmutter/jobs/job-e3-mlip-preparation-perlmutter.sh)
is still CPU-only geometry materialization plus single-point MACE inference.
It is a diagnostic only and does not satisfy the revised experimental
protocol. The adaptive generator, geometry-optimization controller, and
GPU-runner described in the plan must be implemented and pass its pilot gates
before production.

The repository already has an in-node `MLIPWorkerPool`: on a Perlmutter GPU
node it can run one persistent relaxation worker per A100. The planned pilot
uses one node and all four GPUs; adaptive rounds synchronize between batches.
This does not submit or coordinate a production allocation.

```bash
# Diagnostic only; this is not the revised exploration campaign.
PROJECT_ROOT="$PWD" sbatch -A <allocation> \
  deployments/perlmutter/jobs/job-e3-mlip-preparation-perlmutter.sh

# Candidate-only diagnostic preparation; does not generate or optimize structures.
python -m benchmarks.e3.run prepare-candidates \
  --output runs/e3-perlmutter/prepared \
  --materials lifepo4 bcc_refractory fcc_cantor phosphorene zn_formate_mof nb_ta_o

# Predict the selected frozen MACE model on all candidate geometries.
python -m benchmarks.e3.mlip_predict \
  --model mace-omat-0-medium \
  --input runs/e3-perlmutter/prepared/bcc_refractory/candidates.extxyz \
  --output runs/e3-perlmutter/predictions/bcc_refractory.mace-omat-0-medium.json \
  --device cpu
```

For this diagnostic only, populate the offline MACE cache on a Perlmutter login node:
`PROJECT_ROOT="$PWD" bash deployments/perlmutter/download/download-mace-perlmutter.sh mace_mp medium-omat-0`.
The diagnostic writes a unique
`runs/e3-perlmutter/preparation-<job-id>/` directory (or configured
`MATSIM_RUNS_ROOT`) with geometry files, single-point predictions, and source
revision/status. It does not run adaptive selection or geometry optimization.
The legacy reference-backed E3-DFT commands and protocol below are unchanged
and must not be used by E3-MLIP.

### 3.3 E3-MLIP pre-production gate

Follow the plan's six pre-production gates, including the incomplete MOF
seed stoichiometry, checkpoint species checks, and 280-relaxation GPU pilot.
No production run is authorized until pilot-derived cost estimates and an
explicit Perlmutter allocation are approved.

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
