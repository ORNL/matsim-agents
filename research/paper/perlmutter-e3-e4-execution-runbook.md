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

The user selected **MACE-OMAT-0 medium**, MACE-Torch alias `medium-omat-0`, default head, as the common frozen primary checkpoint for both E3-MLIP adaptation arms. The MACE model cache helper is `deployments/perlmutter/download/download-mace-perlmutter.sh mace_mp medium-omat-0`. No fine-tuning is part of the primary comparison.

E3-MLIP is distinct from E3-DFT: it uses MLIP predictions as acquisition evidence, while any selected DFT labels are independent higher-fidelity outcomes. Keep the LLM-guided and Rule-based evidence ledgers separate, match candidate pools/actions/budgets, and do not treat MLIP agreement as independent validation. The planned scale is 5,000 unique valid configurations per class (30,000 total), a planning target rather than submission authorization.

### 3.2 Commands that are available (preparation only)

From the Perlmutter repository checkout:

```bash
# Inventory and blockers; no inference or DFT submission.
python -m benchmarks.e3.run inventory --output research/paper/e3/inventory-perlmutter.json

# Freeze the currently implemented pools/splits (only materialized pool types work).
python -m benchmarks.e3.run prepare \
  --output runs/e3-perlmutter/prepared \
  --materials lifepo4 bcc_refractory fcc_cantor nb_ta_o

# Predict the selected frozen MACE model on a prepared acquisition set.
python -m benchmarks.e3.mlip_predict \
  --model mace-omat-0-medium \
  --input runs/e3-perlmutter/prepared/bcc_refractory/acquisition.extxyz \
  --output runs/e3-perlmutter/predictions/bcc_refractory.acquisition.mace-omat-0-medium.json \
  --device cpu
```

Repeat the prediction command for each existing acquisition/reference file and class where it exists. These are zero-shot predictions on frozen geometries; they do not relax structures, create a 5,000-structure pool, select both strategy arms, or request DFT.

Current inventory limitations:

- LiFePO4: 120 candidate POSCARs, but no adequate E3-qualified validation/test groups; public Codabench has one diagnostic case only.
- BCC refractory: 120 candidate occupations; the 54 available pilot frames are one correlated group, not the required >=2 validation plus >=1 test groups.
- FCC Cantor: 120 candidates, but the spin-polarized DFT signature/MAGMOM protocol is not frozen and qualified references are absent.
- Phosphorene: two seeds, zero materialized candidates; a symmetry-/strain-/defect-aware pool generator and qualified 2D reference method are missing.
- Zn-formate MOF: two seeds, zero materialized candidates; the periodic MOF candidate workflow and dispersion/reference protocol are missing. MACE-OMAT-0 medium has no direct Codabench MOF validation and is not established as MOF-capable.
- Nb-Ta-O: 14 structure files, but no common E3 VASP method signature or qualified reference hull; historical QE records are not E3-qualified VASP labels.

### 3.3 E3-MLIP launch gate: a live paired runner is not present

The current `benchmarks/e3/run.py` exposes `inventory`, `dry-run`, `prepare`, `replay`, `round1` and `ingest`. `replay` and `round1` freeze/record decisions and explicitly set `dft_dispatched: 0`; they are not live campaign launchers. The existing protocol is `replay_only`, and its Rule-based committee still names MACE-MP-0-small plus UMA, not the selected MACE-OMAT-0 medium. Before live E3-MLIP on Perlmutter, the new campaign must have a versioned protocol that:

1. Pins MACE-OMAT-0 medium (checkpoint hash/default head) consistently in both strategy arms and resolves whether any comparator model, including UMA, is also part of the frozen shared evidence.
2. Materializes admissible, deduplicated class-specific candidate pools and the intended 5,000/class exploration plan without arbitrary perturbations.
3. Freezes scientific endpoints, class-specific thresholds, qualified reference roles (>=2 validation groups and >=1 test group), matched strategy budgets and run seeds. Insufficient coverage must be reported as `discovery_performance_not_established`.
4. Adds a Perlmutter live-round controller that submits only accepted paired DFT selections through the Perlmutter Slurm/VASP layer, preserves distinct ledgers, and records scheduler IDs, failures, retries, node-hours and provenance.
5. Uses new E3-MLIP run IDs and artifacts; it must not alter historical E3-DFT IDs, ledgers, results, execution protocol or budgets.

The Perlmutter active-learning paper-case job is not this paired campaign. It runs one AL loop, not matched LLM and Rule-based strategies. Do not submit it as an E3-MLIP surrogate.

No E3-MLIP DFT campaign is authorized by the MACE checkpoint selection alone. Get a separate bounded Perlmutter DFT/resource approval after the six-class protocol and live driver are ready.

## 4. Perlmutter decision checklist

- [ ] Fetch the current `research/e3-e4-completion` branch.
- [ ] Reconcile the current E4 Aurora queue and run records; do not duplicate qualified tasks unless measuring platform performance.
- [ ] Stage E4 inputs/PAWs and verify hashes; obtain separate Perlmutter allocation approval.
- [ ] Write the E4 Slurm dispatcher/entrypoint around the existing Perlmutter VASP step launcher; do not call PBS wrappers.
- [ ] Run and qualify one E4 Perlmutter pilot before the scaling tiers.
- [ ] Prefetch the MACE-OMAT-0 medium model using the existing Perlmutter downloader.
- [ ] Complete six-class E3 candidate generation, scientific/reference gates and updated frozen model protocol.
- [ ] Implement a live paired E3-MLIP Slurm/DFT controller; current E3 commands only prepare, replay or freeze decisions.
- [ ] Obtain a separate bounded Perlmutter E3-MLIP DFT allocation approval before any campaign DFT submission.
