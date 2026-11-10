# E4-B: single-node MLIP scaling

Status: **implementation and CPU-only preflight; NO-GO for execution**.
No PBS submission, GPU benchmark, training, DFT, manuscript edit, commit, or
push is authorized by this package. E4-A submission status is not inferred
from the request's description.

## Existing-capability audit

- `MLIPWorkerPool`: spawn processes, one disjoint visibility mask per device,
  dynamic dispatch, persistent `CalculatorCache`, explicit failed-task outcomes,
  replacement of a crashed process. Placement is restricted to the calling node.
- `CalculatorCache`: cache key excludes structure/output path, so the same
  checkpoint is reused across all structures; loading occurs inside workers.
- `relax_with_mlip`: existing ASE FIRE/BFGS/BFGSLineSearch implementation and
  trajectory/CSV/result handling. This benchmark calls it unchanged, using
  FIRE, fixed cell, fmax=0.02 eV/A, maxiter=200, maxstep=0.01 A.
- HydraGNN fused ASE calculator predicts energy and analytical forces; its
  force path uses negative autograd energy gradients. It provides no native
  stress requirement for this fixed-cell benchmark.
- `ResourcePlan` Aurora preset describes 12 logical XPU tiles. Device counts
  1, 2, 4, 8, 12 all resolve through existing placement; these are NOT nodes.
- One narrow pool addition, `on_each_worker`, initializes every slot exactly
  once. Dynamic `map` cannot guarantee one warmup per slot, so it is not a
  valid initialization barrier for steady-state measurement.

## Model audit

The user-approved frozen code is **allaffa/HydraGNN commit
81d5b88304f20bb45c840571e2e26c490d1b19da**. Its branch context is
`fix/multibranch-bf16-autocast-dtype`, but the commit, not the moving branch,
is the reproducibility pin. Selection is recorded in `frozen-source.json`.
Existing setup docs exposed `HYDRAGNN_REF` without specifying a frozen E4-B
revision; do not equate the dependency installer's main branch with this model
source revision.

A separate checkout was created at
`/lus/flare/projects/HydraGNN/mlupopa/AgenticAI4Materials/HydraGNN-e4b-frozen-81d5b883`.
The original checkout and installed environments were preserved. Upload the
model files under that checkout's `examples/multidataset_hpo_sc26`:
`mlp_branch_weights.pt` and
`multidataset_hpo-BEST6-fp64/{config.json,multidataset_hpo-BEST6-fp64_epoch_97.pk}`.
The user uploaded all three model files, and the CPU-only audit on
2026-11-08 verified the checkpoint and branch-weight hashes against the
previously inspected files. The configuration parsed successfully and the
tracked frozen source remained unchanged. This clears the missing-upload
blocker, not species support or XPU compatibility. Re-run `prepare` against
the checkout after any further upload; historical audit hashes do not
automatically approve new files. Execution verifies code revision and
that the imported HydraGNN package originates from the pinned checkout.

The default Aurora job names `multidataset_hpo-BEST6-fp64` plus
`mlp_branch_weights.pt` in HydraGNN's multidataset example. The mounted model is
PAINN, hidden_dim=337, two convolution layers, radius=5 A, max_neighbours=20,
graph-attribute conditioning, fp64, epoch 97. `prepare` records the full
architecture, exact file hashes, HydraGNN revision and software versions without
deserializing weights or initializing an accelerator.

Species support is **not** proven by universal atomic-number inputs or the
training dataset names. The mounted configuration lacks an explicit supported
species declaration. A reviewed species list plus evidence is required; runtime
execution also verifies model XPU residency and finite energy/force outputs.
Supported benchmark materials therefore remain unverified, not all three.

UMA 2.23.0 and MACE 0.3.16 are present in isolated environments with the
framework Torch 2.13 XPU build. No compatible local model checkpoint, species
contract, or matching XPU operations was verified. They are not admitted into
this sweep. No environments are modified and no checkpoints are downloaded.

## Frozen workload

The primary workload requires **exactly 120 unique reviewed geometries** with
source, ID, family, atom count, species, composition, initial/relaxed/intermediate
status, provenance, file SHA-256 and geometry SHA-256.
128 atoms are required for BCC/FCC and 112 for LiFePO4. One source E4 BCC
structure is mounted; the pilot attempts are copies or relaxed descendants,
not a chemically diverse 120-case E4 dataset. Do not copy the same geometry to
meet the count or substitute E2 geometries silently. Distinct reproducibly
generated occupations are permitted when their generation and applicability
are documented. Family proportions depend on checkpoint evidence, not a
mandatory balanced split: use 40/40/40 only if all three families are supported;
120 BCC configurations are an option only with adequate BCC coverage, or with
explicit out-of-domain systems-benchmark labeling and approval.

Provide a JSON source list with entries:

```json
[
  {
    "structure_id": "E4-BCC-0001",
    "family": "bcc_refractory",
    "path": "/absolute/path/to/POSCAR",
    "source": "reviewed E4 dataset and source identifier"
  }
]
```

`prepare --sources` verifies all 120 before publishing the manifest. The optional
`--bcc-parent /path/to/staged/POSCAR` instead freezes 120 different seeded
species permutations on an existing 128-site BCC parent. It preserves cell,
coordinates and composition, records parent hashes, RNG/version/seed and
initial status, and labels the entire result **provisional, not scientifically
qualified**. It does not create DFT labels, imply a relaxed geometry, or
establish symmetry-canonical uniqueness. Every device-count and repetition uses the same
manifest hash and IDs. Optional weak scaling is deferred: disjoint 10-per-tile
subsets across all five points require 270 structures, not 120.

## CPU-only preparation and validation

```bash
.venv/bin/python -m benchmarks.e4_mlip.prepare \
  --example /path/to/HydraGNN/examples/multidataset_hpo_sc26 \
  --output /path/to/new/preparation-directory
```

This writes model metadata, readiness blockers, and five one-node PBS scripts.
Add `--sources /path/to/reviewed-sources.json` to freeze a complete inventory.
Output directories are no-clobber. Neither preparation nor default run mode
imports Torch, loads a model, invokes MPI, or submits jobs.

To audit existing local structure archives without modifying them:

```bash
.venv/bin/python -m benchmarks.e4_mlip.inventory \
  --root /path/to/staged/e4 --root /path/to/local/structure/archive \
  --output /path/to/new/inventory.json
```

The inventory counts files and unique ordered geometry hashes separately,
records unreadable files explicitly, and applies chemistry/size/periodicity
filters. These filters are not crystallographic or checkpoint qualification.
CONTCAR alone is not evidence of convergence. Archived initial files and
intermediate snapshots must not be counted as independent material sources.

```bash
.venv/bin/python -m benchmarks.e4_mlip.run \
  --manifest /path/to/structures.json --metadata /path/to/model-metadata.json \
  --mode inference --devices 4 --repetition 1 --output /path/to/new/preflight
```

The default is dry-run. Execution requires explicit `--execute`, an existing
single-node PBS allocation, verified species metadata and approved numerical
tolerances and verified energy/force units and energy-reference evidence.
Optional `--checkpoint` must match the pinned checkpoint. Precision
is pinned to fp64; other precision settings require separate validation.

## Bounded hardware pilot (preparation only)

Use `python -m benchmarks.e4_mlip.pilot --manifest /path/to/pilot-representatives.json
--metadata /path/to/model-metadata.json --output /path/to/new/pilot-preflight`
for CPU-only validation. The representative manifest contains one structure
per supported family, rather than requiring all 120 structures. Generated
BCC candidates include a one-structure representative manifest, but it remains
blocked until coverage and units are established; it is not silently admitted
as a supported chemistry.

The exact job template is
[`job-e4b-mlip-pilot.pbs`](../../deployments/aurora/jobs/job-e4b-mlip-pilot.pbs).
It requests **one node, allocation HydraGNN, capacity, 15 minutes, at most
0.25 node-hours** for the whole experiment, not per individual test.
One-tile and two-tile stages run sequentially inside the same allocation.
Both one-/two-tile inference stages finish before optional relaxation stages.
Every inference worker evaluates each representative initially and twice warm.
Optional fresh relaxation workers perform fixed-cell FIRE with fmax=0.02 eV/A,
maxiter=200, maxstep=0.01 A. Inference pools close before relaxation pools,
so relaxation initialization is explicitly cold, not mislabeled as cache reuse.
Only the first model load per worker is truly cold; per-record cache/load
fields distinguish subsequent representatives.

The job defaults to CPU-only dry-run. After separate execution approval,
`E4B_EXECUTE=1` uses one MPI rank to start the parent on the allocated compute
node, then existing spawn workers isolate the tiles. The entire launched
pilot has an 840-second timeout plus a 30-second termination grace period;
PBS provides the final 900-second cap. A timeout is a failed/incomplete
qualification, not a performance point. No submission helper is supplied.
The production PBS templates use the same one-rank compute-node launch only
in explicitly enabled execution mode.

An explicitly approved controlled systems-only pilot may declare
`pilot_scope=approved_controlled_systems_only` in a separate metadata file.
This permits technical execution without claiming species/family validation.
Artifact hashes are still checked. Unknown physical units remain unknown:
raw differences are reported but eV/eV/A tolerance verdicts are deferred until
the convention is established. Production gates are unchanged.
The script also requires `EXPECTED_MANIFEST_SHA256` and
`EXPECTED_METADATA_SHA256`; both are checked before launch. It uses the
existing trusted-checkpoint loader setting `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`,
matching the established Aurora qualification workflow.

Submission (after approval) uses a new `OUTPUT_ROOT` per attempt; outputs go to
`${OUTPUT_ROOT}/pilot-${PBS_JOBID}` and never overwrite earlier pilots:

```bash
qsub -o "${OUTPUT_ROOT}/pilot-pbs.log" -v \
REPO=/path/to/matsim-agents,\
MANIFEST=/path/to/pilot-representatives.json,\
MODEL_METADATA=/path/to/approved-pilot-metadata.json,\
OUTPUT_ROOT=/path/to/new/output-root,\
HYDRAGNN_ROOT=/path/to/HydraGNN-frozen-81d5b883,\
EXPECTED_MANIFEST_SHA256=<sha256>,EXPECTED_METADATA_SHA256=<sha256>,\
E4B_EXECUTE=1 deployments/aurora/jobs/job-e4b-mlip-pilot.pbs
```

The script derives `HYDRAGNN_EXAMPLE` from `HYDRAGNN_ROOT` and puts the frozen
HydraGNN example, HydraGNN, and this repository on `PYTHONPATH`. Aurora's
per-user job limit applies; a `would exceed queue generic's per-user limit`
rejection creates no job and consumes nothing.

### DDP checkpoint loading (fixed after job 9051385)

The HydraGNN checkpoint was saved from a DistributedDataParallel-wrapped
model, so its keys are `module.model...`. The frozen upstream loader calls
`load_state_dict(..., strict=False)` without removing that prefix, which
silently left all 200 GNN tensors uninitialized in pilot job 9051385 while
still printing "Loaded checkpoint". That run's numbers are invalid.

`_load_fused_stack` now reapplies the weights through the existing
`restore_hydragnn_checkpoint` helper: one leading `module.` is removed, `model.`
and lazily created conditioning modules are preserved, and loading is strict
(missing, unexpected, or mis-shaped tensors raise before inference). The pilot
re-verifies the restored key set before every inference and relaxation and
records `model_tensors_matched`. Checkpoint files are only read, never
rewritten.

Progress is written after each inference operation and each FIRE boundary.
Optional relaxations stop cooperatively at a 720-second whole-pilot deadline,
leaving headroom under the launcher/PBS caps. A resource-limited relaxation has
`converged=null` and `INCOMPLETE_RESOURCE_CAP`, not a nonconvergence verdict.
The pilot's technical result is separate from numerical and scientific
qualification; successful inference does not require every optional relaxation
to finish within the cap.

Qualification requires finite outputs, one visible tile per worker, distinct
two-worker PIDs and masks, synchronized evaluations, no unintended checkpoint
reloads, numerical agreement at the provisional thresholds, converged bounded
relaxations, and at least 20% headroom relative to reported peak allocated
memory. Allocation headroom is not measured free/total device utilization.
A technically successful pilot does not establish DFT accuracy or authorize
production submission. Inspect its measured cold/steady/relaxation costs before
choosing production walltime; no runtime estimate is fabricated beforehand.

## Timing and correctness

The fused path reuses the strict DDP checkpoint helper: only one leading
`module.` wrapper prefix is removed; `model.` and lazy conditioning modules are
preserved. Restoration must succeed before inference. The pilot checks the
normalized state keys before each numerical operation and records
`model_tensors_matched` per result and worker. Checkpoint shape compatibility is
enforced by strict loading, not by a permissive post-inference check.
The worker calculator cache remains active: strict restoration occurs only on
calculator creation, not for each structure or warm inference. The upstream
stack constructor still opens the checkpoint before the helper restores it;
this additional initial read is part of measured cold model-load time, not a
warm-inference cost. Original GNN and branch-MLP files are never rewritten.

Each process verifies it sees exactly one XPU tile, sets local device 0, and
verifies the model (and branch MLP) resides on XPU. No CPU fallback is allowed.
The parent never loads calculators. Every slot performs an untimed workload
warmup, recording process-entry latency, checkpoint initialization and a cold
energy/force operation; all workers finish warmup before the steady timer.

Accelerator synchronization brackets model loading and every actual calculator
evaluation. Separate fields record input parsing/hash validation, numerical
operations, whole relaxation/inference operation, worker elapsed time, result
writing, cold-phase and steady-phase walltime, task distribution, idle/dispatch
time and peak allocated device memory. Non-numerical operation time includes
optimizer, trajectory writing and internal structure conversion; those internal
components are not individually instrumented by the existing API. Graph building
inside the calculator is included in numerical-call walltime, not mislabeled as
pure GPU kernel time. Peak allocation is not total device memory utilization.

Full elapsed time retains scheduling, transfers and I/O. A replacement worker
produces cold-model measurements and invalidates the steady-state scaling point;
failed tasks remain in the results. Unconverged finite relaxations remain explicit
and do not count toward converged-relaxation throughput. Hardware utilization
is unmeasured (`null`), not estimated from task occupancy; an approved facility
telemetry collector is needed for that metric.

The proposed fp64 absolute tolerances are 1e-6 eV energy, 1e-5 eV/A force
components and 1e-4 A periodic site displacements. They require approval and a
real consistency pilot; they are not claimed to be validated. Analysis checks
input/model identity, convergence, energies, forces and fixed-cell final geometry
against each repetition's one-device baseline. Step-count differences are
reported, not hidden. Different floating-point results are not required to be
bitwise identical.

## Analysis

```bash
.venv/bin/python -m benchmarks.e4_mlip.analyze \
  /path/to/runs/*/results.json --metadata /path/to/model-metadata.json \
  --output /path/to/new/analysis
```

Only valid, synchronized, numerically consistent measured records are plotted.
Inference throughput counts finite evaluations; relaxation throughput counts
converged relaxations. Fixed-workload speedup is T(1)/T(d), efficiency S(d)/d.
The supporting CSV retains every repetition; figures show mean and sample
standard deviation. Missing repetitions remain visible in summary metadata.
No synthetic performance conclusions or plots are generated by preparation.

## Budget and execution gate

Propose HydraGNN for the entire initial MLIP benchmark, subject to explicit
allocation approval; material identity does not automatically switch projects.
Five device counts x two modes x three repetitions = **30 one-node jobs**.
The prepared scripts request one hour in capacity: **30 reserved node-hours**,
plus any separately authorized pilot. This is a reservation envelope, NOT a
runtime forecast. An unfinished one-hour benchmark produces no full-workload
scaling point; walltime needs adjustment after measured runtime. Peak memory
and completion time cannot be predicted reliably from the 291 MB checkpoint:
autograd, graph size, activations and replicas also consume memory.

Scripts default to dry-run (`E4B_EXECUTE=0`). Required variables are REPO,
MANIFEST, MODEL_METADATA, OUTPUT_ROOT, HYDRAGNN_EXAMPLE, HYDRAGNN_ROOT. The
pilot script derives HYDRAGNN_EXAMPLE itself and additionally requires
EXPECTED_MANIFEST_SHA256 and EXPECTED_METADATA_SHA256.
MODE and REP select mode/repetition. No submission helper is included.

Minimum proposed pilot after approval: one compatible representative structure
on one tile, repeat a warm evaluation, a bounded fixed-cell relaxation, then
verify two disjoint workers agree numerically and fit memory. The full sweep
remains NO-GO until the real inventory, species evidence, approved tolerances,
memory and elapsed-time checks exist.
