# E4 Scaling Priority

For Perlmutter staging, VASP portability limits, and unfinished-case instructions,
see the [Perlmutter E3-MLIP and E4 execution runbook](../../../perlmutter-e3-e4-execution-runbook.md).

This package records the immediate BCC DFT scaling workload, its blockers, and
resource caps. The original BCC manifests and authorizations remain immutable.
The versioned six-class extension is
[`../six-class-workflow-v2.json`](../six-class-workflow-v2.json) (v1 is the
superseded infrastructure baseline); it does not authorize additional work.

## Six-Class Extension

The v2 manifest consumes the qualified inputs in
[`../six-class-inputs-v1/manifest.json`](../six-class-inputs-v1/manifest.json),
staged by `python -m benchmarks.e4.aurora_dft.six_class_inputs` into the
private run directory `../e4-six-class-inputs-v1/` (POTCARs are never
committed). Each VASP task remains one node, and nodes execute independent
calculations. It does not measure solver-level scaling of one calculation
across nodes.

| Class | Unique validated inputs | Protocol | Strong ready (8 fixed tasks) | Weak ready (2 tasks/node) | Proposed node-hour cap | Remaining blocker |
|---|---|---|---|---|---|---|
| BCC refractory | 16 | `bcc-pbe-low-normal-v1-approved-ediffg-0.01` | 1/2/4/8 | 1/2/4/8 | existing 1,488 authorization | none; running jobs preserved |
| FCC Cantor | 16 (4 frozen + 12 new realizations) | `fcc-pbe-verified-v1` | 1/2/4/8 | 1/2/4/8 | 1,488 | E4 runtime unmeasured |
| LiFePO4 | 16 (8 frozen + 8 inventory) | `lifepo4-pbe-verified-v1` | 1/2/4/8 | 1/2/4/8 | 1,488 | E4 runtime unmeasured |
| Phosphorene | 16 rebuilt puckered monolayer cells | `phosphorene-pbe-2d-inplane-relax-v1` | 1/2/4/8 | 1/2/4/8 | 744 | no qualified E2 geometry exists; runtime extrapolated |
| Zn(HCOO)2 MOF | 16 EOS points of 2 E2-qualified frameworks | `znformate-pbe-d3bj-fixed-volume-eos-v1` | 1/2/4/8 | 1/2/4/8 | 744 | only two distinct frameworks; cross-build check |
| Nb-Ta-O | 14 symmetry-distinct NbTaO5 orderings | `nbtao5-fixed-kmesh-v1` + `KPAR=12` | 1/2/4/8 | 1/2/4 | 230 | weak-8 needs 16 orderings; cross-build check |

Input qualification:

- **Nb-Ta-O**: POSCAR, INCAR, and KPOINTS hashes match the frozen E2 manifest.
  INCAR equals the converged E2 ordering-01 run (50 ionic steps) apart from
  `KPAR=12`, the only E4 adapter tag. PAWs match `POTCAR.order`. Strong uses the
  five production candidates plus orderings 02-04.
- **Zn-formate**: alpha-Zn3(HCOO)6 (COD 4515120, 108 atoms) and lonsdaleite
  Pbcn Zn(HCOO)2 (COD 7205949, 216 atoms) come from the accepted E2 PBE-D3(BJ)
  relaxations (forces < 0.02 eV/A). Each staged cell has octahedral Zn, intact
  HCOO- linkers including H, and one connected periodic bond graph. Isotropic
  strains of +/-0.5 to +/-2 percent at fixed volume (ISIF=4) form an
  equation-of-state set. The assembled POTCAR equals the E2 hash `f8ebdae2`.
  The dihydrate (COD 7102783) is qualified but excluded as a different formula.
- **Phosphorene**: archived E2 P4/P36/P35 inputs have no P-P bond below 2.5 A
  because the generator placed the puckering on the zigzag axis, and the
  relaxed results fragmented. Inputs are rebuilt from bulk black P (Cmce,
  Brown & Rundqvist 1965) in the E2 cell (4.376 x 3.314 A, 20 A vacuum).
  All atoms are three-fold coordinated (2.224/2.244 A bonds). The set covers
  the 1x1 cell and 2x3-4x5 supercells, pristine plus mono- and divacancies,
  using the E2 numerics with in-plane-only cell relaxation.
- **FCC/LiFePO4**: FCC realizations 2-4 of the four frozen compositions use
  the pre-assigned inventory seeds; the generator reproduces the frozen
  realization-1 POSCARs byte for byte. LiFePO4 adds eight canonical
  inventory structures (1/2/3 antisite pairs). Symmetry-canonical duplicates
  are rejected for both.

The proposed additional ceiling is 4,694 node-hours. These ceilings are task
walltime caps, not runtime predictions, and are proposals, not approvals. The
existing BCC 1,488 node-hour authorization is not counted again.

The six-class manifest also records 16/32/64/128-node weak-scaling points. The
capacity queue admits at most 16 nodes/job; 32/64/128 are explicitly marked
ineligible pending a different allocation/queue. At 2 calculations/node and a
24-hour/task cap, those hypothetical weak points require 64/128/256 distinct
inputs and 1,536/3,072/6,144 node-hours, respectively. Strong-scaling points
above 8 nodes are blocked because the fixed strong workload contains only eight
calculations; do not call idle-node runs strong scaling.

Generate/validate the plan and inspect a tier without submitting:

```bash
python -m benchmarks.e4.aurora_dft.scaling plan-six-class
python -m benchmarks.e4.aurora_dft.scaling validate-six-class
python -m benchmarks.e4.aurora_dft.scaling dispatch-six-class \
  --manifest research/paper/e4/aurora-dft-scaling/six-class-workflow-v2.json \
  --material-class fcc_hea --experiment strong --nodes 4 --dry-run
```

The six-class manifest has `execution_authorized=false`. No jobs were submitted
for this extension. In the last checked state, weak BCC scaling had completed
two one-node calculations and strong BCC job 9069491 was still running on eight
nodes. Do not edit or reuse their output roots.

After separately authorized work completes, `benchmarks/e4/analyze.py` reads
the dispatcher-written `measurements.jsonl` records, groups results by class,
and writes per-class CSV aggregates, a planned-versus-measured JSON summary, and
plots. For example:

```bash
find runs/e4-scaling -name measurements.jsonl -type f -print0 \
  | xargs -0 -r cat > runs/e4-scaling/measurements-all.jsonl
python -m benchmarks.e4.analyze \
  runs/e4-scaling/measurements-all.jsonl --kind scaling \
  --workload-manifest research/paper/e4/aurora-dft-scaling/six-class-workflow-v2.json \
  --output runs/e4-scaling/analysis/scaling.csv
```

The analyzer merges one-node continuation segments for the same fixed strong
workload before computing speedup. It leaves incomplete points without speedup
or efficiency and never interpolates unmeasured tiers.

## Proposed Submission Order

1. Preserve and finish the active BCC jobs; validate their dispatch and measurement records, then use the existing BCC authorization only for its frozen manifests.
2. One-node pilots (strong n1 segment 1, or weak n1) for Nb-Ta-O, Zn-formate, and phosphorene. These measure E4 runtime and check the ALCF build against E2 (NbTaO5 ordering-01 and the zero-strain-adjacent MOF points).
3. FCC and LiFePO4 strong 1/2/4/8 and weak 1/2/4/8.
4. Remaining Nb-Ta-O, Zn-formate, and phosphorene tiers, with task caps resized from pilot measurements.

Each order step requires its own scientific and resource approval. The workload
manifest and dry-run commands alone authorize no production calculation.

## Workload Audit

The approved first-batch manifest pins eight BCC records to
`bcc-pbe-low-normal-v1-approved-ediffg-0.01`, with an 8-rank, 8-thread task
layout. Eight additional disorder realizations were frozen from the same
audited BCC parent and matched compositions. All 16 POSCAR hashes and
symmetry-canonical occupation IDs are distinct; per-case PAW component hashes,
protocol inputs, and assembled POTCAR hashes are recorded in
`weak-input-manifest.json`. This is a local structure-identity check only:
external archive duplicate screening remains incomplete.

The `capacity` queue currently reports a 16-node maximum, 168-hour job
walltime, and two running jobs per user. `benchmarks/e4/aurora_dft/scaling.py`
dispatches one existing E4 case runner per assigned node in task waves, records
per-task exit status and execution timing, and gives every task a fresh output
root. `deployments/aurora/jobs/job-e4-scaling.pbs` is the PBS entrypoint; it
does not submit jobs. The AL DFT scaling harness is unrelated because it
generates a different 32-case FCC workload through an active-learning loop.

The manifests reference approved input cases, not their existing output or
checkpoint directories. Any authorized scaling run must use fresh scaling run
and task IDs under the listed output roots; it must not resume or overwrite
first-batch or terminal-failure records. The initial 480 node-hour approval
does not authorize these new calculations.

## Blocker Corrections

The earlier GPU launch failure came from `--gpu-bind=closest`, which PALS
MPICH 1.8 does not support. The Aurora launcher now uses an explicit tile list.
The isolated capacity-queue probe (PBS 9067531) ran VASP through the pinned
`.51a9474` launcher; VASP reported 8 MPI ranks, 8 threads per rank, 8 GPUs
initialized, and accepted the frozen 128-ion BCC inputs. Its separate mask
subprobe ran under facility-default `.87e2045` and saw masks 0-7 once; that is
not treated as proof for the pinned module. A pinned-module follow-up on the
debug queue (PBS 9067675) failed before VASP because that node exposed only six
GPU devices (`Requested GPU 6 but only 6 exist`). Do not use that debug node for
the eight-rank placement check. The capacity run was deliberately interrupted
before convergence and is not a scaling measurement; PBS exit 1 followed the
intentional SIGINT, which the smoke script now recognizes explicitly.

The PAW hash failure was a missing-path issue: the new worktree lacked the
private/ignored `external/vasp6/potcar` files. The exact archived PAWs are
restored locally as read-only hardlinks. `submit_batch.py --dry-run --only
E4-BCC-1701` validates the frozen hashes and reports no qstat/qsub execution;
no hash checks were weakened.

## Resource Caps

All values are worst-case caps using a 24-hour per-task walltime. The 1,488
node-hour envelope below is approved only for the frozen BCC manifests; it does
not authorize any additional class or tier.

| Experiment | Nodes | Tasks | Max walltime | Node-hour cap | Status |
|---|---:|---:|---:|---:|---|
| Strong | 1 | 8 | 2 x 96 h | 192 | Authorized; two 4-task PBS segments |
| Strong | 2 | 8 | 96 h | 192 | Authorized under BCC manifest |
| Strong | 4 | 8 | 48 h | 192 | Authorized under BCC manifest |
| Strong | 8 | 8 | 24 h | 192 | Authorized; job 9069491 running |
| Weak | 1 | 2 | 48 h | 48 | Authorized; both calculations converged |
| Weak | 2 | 4 | 48 h | 96 | Authorized under BCC manifest |
| Weak | 4 | 8 | 48 h | 192 | Authorized under BCC manifest |
| Weak | 8 | 16 | 48 h | 384 | Authorized under BCC manifest; external duplicate audit remains incomplete |

The bounded request for one measurement at every listed point is **1,488
node-hours maximum**: 768 strong plus 720 weak. This is the existing BCC-only
authorization, not a new request. No repeats or larger tiers are included. The
root `strong-manifest.json` and `weak-manifest.json` remain
planning-only; corresponding copies under `authorized/` carry the existing
BCC-only authorization. The six-class manifest remains
`execution_authorized: false`. The dispatcher requires a separately authorized
manifest and matching `E4_SCALING_AUTHORIZATION` ID. For the 1-node strong point, request two
separate 96-hour jobs (`segment-1`, `segment-2`), each with four tasks. The
other per-job requests are 2 nodes/96 h, 4 nodes/48 h, 8 nodes/24 h for strong,
and 1/2/4/8 nodes/48 h for weak. All requests use `capacity` and the
`HydraGNN` allocation. Queue wait is excluded from task execution timing.

For weak scaling, each node runs its two tasks sequentially. Concurrently
running two unchanged 8-rank x 8-thread tasks on one node would oversubscribe
the node's CPU allocation and would change the resource layout.

## Path To 128 Nodes

Keep the DFT workflow study separate from MLIP device scaling. First qualify
the authorized 1/2/4/8-node BCC points. The queue's 16-node maximum does not make
a 16-node BCC strong point meaningful: the fixed workload contains only eight
independent calculations. A 16-node weak point would require 32 unique frozen
tasks and a new authorization. The 32/64/128-node weak points require 64/128/256
unique tasks and a queue/allocation capable of those requests. At two sequential
24-hour tasks per node, a hypothetical 128-node point would need 48 hours and
6,144 node-hours. No curve is extrapolated from unmeasured tiers.

## Existing Components

- The E4 per-case PBS wrapper, frozen-input validator, PAW hash checks, and
  corrected explicit Aurora tile-list launcher remain reusable. The scaling
  dispatcher runs the per-case wrapper concurrently on disjoint nodes and
  serially for each node's assigned weak-scaling tasks.
- The AL DFT scaling harness is an executable but different FCC/active-learning
  experiment; do not use it as E4 BCC evidence.
- `benchmarks/e4_mlip/run.py` is a separate single-node device-scaling runner.
  It still requires the reviewed 120-geometry manifest and frozen model
  metadata. Its timing outputs do not qualify HydraGNN energy/force accuracy.
- `benchmarks/e4/analyze.py` consumes recorded measurements only. Its strong
  baseline now uses the smallest measured node count; the focused planning
  tests pass (7 tests).

The isolated GPU-binding probe is
`deployments/aurora/jobs/e4-first-batch/smoke-vasp-gpu-binding.pbs`. It uses a
frozen BCC input copy in a unique temporary directory and is not a scaling or
recovery job. The initial 480 node-hour production-batch cap does not cover
the separately approved BCC scaling manifests or the proposed six-class tiers.