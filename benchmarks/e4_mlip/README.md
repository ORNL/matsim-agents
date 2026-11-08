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
source, ID, family, atom count, species, file SHA-256 and geometry SHA-256.
128 atoms are required for BCC/FCC and 112 for LiFePO4. One source E4 BCC
structure is mounted; the pilot attempts are copies or relaxed descendants,
not a chemically diverse 120-case E4 dataset. Do not replicate it to meet the
count or substitute E2 geometries silently.

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

`prepare --sources` verifies all 120 before publishing the manifest. No
structures are generated. Every device-count and repetition uses the same
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

```bash
.venv/bin/python -m benchmarks.e4_mlip.run \
  --manifest /path/to/structures.json --metadata /path/to/model-metadata.json \
  --mode inference --devices 4 --repetition 1 --output /path/to/new/preflight
```

The default is dry-run. Execution requires explicit `--execute`, an existing
single-node PBS allocation, verified species metadata and approved numerical
tolerances. Optional `--checkpoint` must match the pinned checkpoint. Precision
is pinned to fp64; other precision settings require separate validation.

## Timing and correctness

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
MANIFEST, MODEL_METADATA, OUTPUT_ROOT, HYDRAGNN_EXAMPLE, HYDRAGNN_ROOT.
MODE and REP select mode/repetition. No submission helper is included.

Minimum proposed pilot after approval: one compatible representative structure
on one tile, repeat a warm evaluation, a bounded fixed-cell relaxation, then
verify two disjoint workers agree numerically and fit memory. The full sweep
remains NO-GO until the real inventory, species evidence, approved tolerances,
memory and elapsed-time checks exist.
