# E3-MLIP adaptive crystallographic exploration plan

Status: **feasibility-pilot design; production execution is not authorized**.
This plan changes E3-MLIP only. It does not change E3-DFT, use DFT labels or
references, or submit VASP calculations.

## 1. Experimental contract

- Freeze **MACE-OMAT-0 medium**, trained on OMat24, for both Rule-based and
  LLM-guided adaptation. Its MACE-Torch loader alias is `medium-omat-0` and its
  E3 CLI alias is `mace-omat-0-medium`. Record the checkpoint hash and do not
  silently substitute another model. Before the pilot, verify checkpoint
  species coverage and calculator loading for every proposed composition.
  Exact species coverage is not yet recorded and MOF transferability is
  unestablished.
- Each strategy receives a budget of **5,000 unique valid starting
  configurations per material class**, not 5,000 shared across the strategies.
  That is 60,000 logical optimization jobs across six classes and two
  strategies. Identical structure hashes may share one physical relaxation;
  both strategies are still charged one logical evaluation and receive the
  same result.
- Run three paired random-seed replicates per strategy, dividing each
  class/strategy budget as 1,667, 1,667, and 1,666 configurations. Use ten
  acquisition rounds per replicate. Each strategy sees only its own prior
  generation and optimization results.
- Every generated structure that passes the pre-optimization validity and
  uniqueness checks is optimized. An optimization failure still consumes a
  logical evaluation and is retained with its trajectory and failure reason.
  Invalid proposals are recorded but are not called valid configurations and
  are not optimized.
- Geometry is the only input evidence. Strip calculator results and labels from
  source structures. Report candidates as **MLIP-predicted**, not independently
  validated discoveries.

## 2. Six-class generation and size matrix

The tier counts below are minimum exploration quotas **per strategy**. The
remaining within-tier generator choices are made adaptively. Atom counts for
composition-variable materials are derived from the exact integer composition,
not rounded after generation.

| Class | Composition and generation families | Starting size tiers | Minimum starts per tier |
|---|---|---|---:|
| BCC refractory | Nb-Ta-V-Hf-Zr-Ti equiatomic baseline; controlled 4- and 5-component subsets and integer-ratio composition perturbations. AFLOW BCC prototypes, symmetry-distinct lattice occupations/SQS-like orderings, and filtered PyXtal bulk proposals. | 4, 8, 16, 32, 64 formula units. For the six-component equiatomic formula this is 24, 48, 96, 192, 384 atoms; four-component variants give 16, 32, 64, 128, 256 atoms. | 1,000 |
| FCC Cantor | Equiatomic Cr-Mn-Fe-Co-Ni baseline plus commensurate, controlled substitution/composition variants. AFLOW FCC prototypes and symmetry-distinct FCC occupations; PyXtal proposals are retained only when they satisfy the declared FCC/family filters. | 4, 8, 16, 32, 64 formula units: 20, 40, 80, 160, 320 atoms. | 1,000 |
| LiFePO4 | Olivine prototype and symmetry-distinct Li/Fe antisite pairs and pair separations; separately tagged Li-vacancy/compensation and strain families. AFLOW prototypes plus PyXtal proposals filtered to valid bulk chemistry and connectivity. | 4, 8, 16, 32 formula units: 28, 56, 112, 224 atoms. | 1,250 |
| Phosphorene | Black and blue phosphorus allotropes, symmetry-distinct vacancy/reconstruction families, and bounded in-plane strains. Use a class-specific 2D generator; the current PyXtal helper is 3D-only and must not generate slabs for this class. | 4, 16, 36, 64, 144 P atoms, with near-isotropic 2D supercells of each allotrope. Keep the out-of-plane vacuum and periodicity fixed. | 1,000 |
| Nb-Ta-O oxides | Start with Nb2O5, Ta2O5, NbTaO5, NbO2, TaO2, and NbTaO4; add charge-balanced integer substitutions and separately tagged oxygen-defect families. AFLOW prototypes and 3D PyXtal proposals are filtered by formula, density, coordination, and valid periodic geometry. | 2, 4, 8, 16, 32 formula units per formula. This gives 14-224 atoms for the 7-atom pentoxide formulas and 6-96 atoms for the 3-atom dioxide formulas. | 1,000 |
| Zn(HCOO)2 MOF | Alpha/beta framework topologies; connectivity-preserving linker/coordination variants, proton configurations, bounded framework strain, and larger-cell defects. Use a MOF-specific generator; generic AFLOW decoration or unconstrained 3D PyXtal is not accepted as a MOF generator. | Corrected primitive cell and 2x, 4x, 8x supercells. Record the actual atom count from the corrected CIF and its formula-unit count; do not infer it from the current seed filenames. | 1,250 |

For the four-tier classes, the 5,000-configuration quota is divided evenly
(1,250 per tier); for the five-tier classes it is 1,000 per tier. Formula and
family subquotas are divided as evenly as integer counts permit. At least 10%
of each replicate's budget must cover every size tier; the remaining budget is
available for adaptive prioritization. This prevents either strategy from
ignoring large or small systems while still allowing it to choose which tiers
to emphasize.

For each class, additionally optimize **20 matched structural families at all
size tiers** as finite-size controls. These replicated controls are explicitly
reported as size controls, not counted as new structure families or toward the
5,000 unique-configuration target. They add 560 logical optimizations per
strategy across all six classes. Thus the planned total is 61,120 logical
relaxations before exact-hash cache reuse.

### Crystallographic and MLIP feasibility gates

1. AFLOW matching supplies prototype hypotheses, not labels. Retain its
   prototype ID, space group, decoration, formula-unit count, atom count,
   lattice parameters, cell matrix, and source geometry hash. The generic
   generator now exposes `SeedGenerationAction`: a strategy can request an
   explicit formula-unit tier, AFLOW candidate cap, PyXtal count, prototype or
   space-group filters, and random seed. The action manifest is persisted, and
   `explore_composition(..., generation_action=...)` sends every generated
   candidate through the same relaxation path unless an explicit relaxation
   cap is supplied.
2. The action interface currently covers AFLOW prototypes and 3D PyXtal
   generation only. `benchmarks.e3.mlip_exploration.run_generation_round`
   executes strategy-supplied actions with the same MACE-OMAT-0 medium
   optimizer and no relaxation cap, or selects from a shared allow-listed
   action space. The deterministic selector scores under-covered composition,
   size, generator, prototype and space-group dimensions, with a secondary
   smoothed convergence-rate term from prior same-composition/size MLIP
   results. The LLM selector receives only its strategy-local round summaries
   and the same allowed actions; it must cite valid evidence and cannot submit
   unsupported or previously used action IDs. Class-specific actions for alloy
   occupations, defects, and framework perturbations remain to be implemented.
   The older `generate_seeds` defaults remain for backward compatibility and
   must not be used by the adaptive E3 runner. The current PyXtal path accepts
   formula-unit and space-group controls, but remains 3D; implement and validate
   a separate 2D path before considering PyXtal for phosphorene.
3. Use symmetry-aware occupation enumeration for BCC/FCC alloys and
   connectivity-preserving perturbations for LiFePO4 and the MOF. Reject
   unconstrained coordinate noise, broken MOF connectivity, non-periodic
   overlaps, and supercell copies masquerading as new structural families.
4. **MOF seed blocker:** the two checked-in alpha/beta EXTXYZ seeds contain
   Zn4C8O8 (20 atoms) and no hydrogen, rather than the stated
   Zn4C8H8O16 stoichiometry for four Zn(HCOO)2 formula units. Do not use them
   until their source is verified and chemically complete framework geometries
   are supplied or reconstructed.
5. Before freezing pools, inspect the pinned MACE checkpoint for every element
   and test calculator creation and one bounded relaxation per class and
   atom-count tier. Element support is only a software-compatibility check, not
   evidence of transferability. A failure blocks that class; do not swap models
   without a recorded protocol amendment.

All generation outputs record: formula/composition, formula-unit count, atom
count, cell matrix and lengths/angles, PBC, space group (or `unknown` with
reason), prototype/family, defect/disorder/strain operations, generator and
version, random seed, parent structure, and a canonical geometry hash.

## 3. Adaptive actions and matched strategies

Both strategies receive the same allow-listed action interface. One action
specifies a material, composition, atom-count tier, generator family
(AFLOW, PyXtal, or class-specific), prototype/space-group constraints,
occupation/defect/disorder operation, strain/displacement range, requested
count, and RNG seed. Both have the same limits, model, validity filters,
optimizer, and per-round optimization budget. They do **not** receive a common
static 5,000-structure inventory.

Each replicate has ten round barriers. A round generates, validates,
deduplicates, and optimizes its selected batch in parallel; the next action is
not chosen until all outcomes for that round are recorded. Generator attempts,
invalid proposals, duplicate rejections, valid starts, completed optimizations,
and failures have separate counters. The controller generates replacement
proposals when a tier's valid-unique quota is short, subject to a predeclared
generation-attempt cap.

**Rule-based policy (deterministic, versioned):**

- The executable generic selector is currently a bounded prototype: greedily
  score under-covered composition, size, generator, prototype and space-group
  dimensions, add `0.25 * (converged + 1) / (attempted + 2)` from prior
  same-composition/size relaxations (or a neutral 0.5 prior), and break ties
  by action ID. It chooses only among the explicitly supplied allow-listed
  actions. It does not yet implement the structural descriptor, promising
  relaxed-parent ranking, or perturbation actions specified below; therefore
  it is not the final frozen production rule-based policy.
- Reserve the first 10% of each replicate's class budget for balanced coverage
  of all allowed size tiers and seed structural families.
- For each later round, rank available generation actions using
  `0.40 * undercovered-stratum + 0.30 * structural-novelty + 0.20 * promising-parent + 0.10 * relaxation-response`.
  Each component is normalized within the same class and composition. Novelty
  uses a frozen structural descriptor and nearest-neighbor distance. A
  promising parent is selected by low relaxed energy per atom **only within
  the same composition and family**. Relaxation response is the normalized
  energy decrease and geometry change of prior perturbations. Break ties by
  action ID.
- Alternate one highest-ranked undercovered/exploratory action with one
  highest-ranked promising-parent refinement action, subject to the tier
  minimums and action bounds. Do not claim force disagreement is calibrated
  uncertainty; the primary comparison uses one frozen MLIP.

**LLM-guided policy:**

- Make one high-level decision per material/replicate/round, not one call per
  structure. Provide the same allowed actions and an evidence summary of that
  strategy's own prior starts, relaxed results, failed relaxations, novelty,
  cost, and coverage gaps.
- Require a schema-valid action list, requested counts, and evidence IDs for
  every rationale. Reject unsupported actions and requests over budget; never
  silently repair them. Freeze model/provider/prompt/temperature and record
  tokens and time.
- Permit the strategy to change compositions, prototype/space-group focus,
  size tiers, generator mix, and perturbation family within the same class
  validity rules.

The shared physical-result cache is keyed by the canonical initial geometry
hash, model/checkpoint hash, optimizer version, and optimization settings.
Cache reuse saves physical work but does not reduce either strategy's logical
budget or erase its independent action provenance.

## 4. Mandatory MLIP geometry optimization

Use the existing MACE relaxation path with a single frozen checkpoint per
worker. Proposed common defaults for the feasibility pilot are:

- Initial energy and forces recorded before optimization.
- ASE FIRE optimizer; maximum 200 optimization steps; maximum atomic step
  0.01 A; convergence at maximum atomic force below 0.02 eV/A.
- Store initial structure, every available trajectory frame, per-step energy
  and maximum force, final structure, final energy/forces, and termination
  outcome. A single-point evaluation is never marked as an optimized result.
- Fixed-cell atomic relaxation is the default. Enable variable-cell relaxation
  only for a class/tier after native MLIP stress has passed a finite, symmetric
  strain check. Do not use finite-difference stress as a substitute for an
  unverified stress model. Phosphorene keeps vacuum and out-of-plane cell
  fixed; selective in-plane cell optimization requires a validated 2D filter.
  MOF cell relaxation also remains off until framework stress and connectivity
  are shown stable in the pilot.
- Same criteria and optimizer for both strategies within each class. Any
  class-specific change is frozen before either strategy starts.

Explicit outcomes are `converged`, `max_steps`, `force_increase_abort`,
`nonfinite_energy_or_force`, `unphysical_geometry`, `unsupported_species`,
`worker_crash`, and `walltime_interrupted`. Do not delete or omit failed
structures. Preserve partial trajectories and costs; retry only infrastructure
failures, at most once, using the same structure ID and cumulative step cap.

Each result records the input/action IDs, structure and checkpoint hashes,
model/head, hardware/device, optimizer settings, initial/final geometry,
number of steps, runtime, device time, termination reason, and artifact paths.
Do not compare raw total energies across different compositions or atom counts.
Use per-atom energies only within a fixed composition and compatible model
convention; report composition and size strata separately.

## 5. Unique starting structures and relaxed minima

Use two separate deduplication stages:

1. **Starting configurations:** exact canonical geometry hash followed by a
   composition-aware symmetry match. A supercell of an existing primitive
   motif is a size-control relation, not a new structural family. Different
   chemical occupations, defects, or strain states remain distinct only when
   they are not symmetry-equivalent under the frozen tolerances.
2. **Relaxed minima:** match only structures with the same composition,
   defect state, and dimensionality. Initial proposed matcher tolerances are
   `ltol=0.2`, `stol=0.3 A`, and `angle_tol=5 degrees`, with primitive-cell and
   supercell matching enabled. Freeze these after a blind pilot review and
   report a tolerance-sensitivity check. Record the relaxed-minimum cluster ID
   separately from the starting-configuration ID and size-control family.

Report counts of proposals, valid unique starts, optimized starts,
convergences, failures, relaxed-minimum clusters, and cross-size family
relations separately. A converged relaxation is not automatically a distinct
material discovery.

## 6. Feasibility pilot and cost model

### Pilot

Run five valid, unique starts per class/tier/strategy on one Perlmutter GPU
node: `5 * (5 + 5 + 4 + 5 + 5 + 4) * 2 = 280` logical relaxations. Include the
matched-size controls only after the first pilot passes. Request one node with
four A100 GPUs, one MACE worker per GPU, and a 24-hour walltime ceiling under a
currently approved QoS. Confirm the account/QoS limit before submission. No
DFT or VASP allocation is part of this pilot.

Measure per class/tier: structures per GPU-hour, median and p90 relaxation
time, optimizer-step distribution, convergence/failure rates, GPU memory,
model load time, and I/O overhead. Do not extrapolate production hours until
this pilot is complete.

### Production cost estimate

Only one model is currently selected, so there is no second-model cost
estimate. For each class/tier, let `t90[class,tier]` be the measured p90
single-structure GPU seconds from the pilot and `n[class,tier]` the logical
structure count. A conservative four-GPU-node estimate is:

```
node_hours = 1.20 * sum(n[class,tier] * t90[class,tier]) / (4 * 3600)
```

The 20% factor covers controller, staging, and result I/O. Use actual pilot
throughput if GPU occupancy or the large structures make the four-worker
assumption inaccurate. At the proposed maximum of 200 steps, the 60,000
exploration relaxations have a 12,000,000-step upper bound; matched-size
controls add their own measured work. This is a step-count bound, not a
walltime claim.

### GPU dispatch and scale-out

Structure relaxations are embarrassingly parallel across configurations;
optimizer steps within a single structure remain sequential. The repository's
`MLIPWorkerPool` already launches one persistent worker process per assigned
device, caches the loaded calculator in that worker, dispatches independent
relaxations concurrently, and preserves result order. Perlmutter GPU nodes have
four A100 devices, so a one-node pilot can run four independent MACE
relaxations at once.

Use a one-node `perlmutter-gpu` resource placement with all four devices in the
`mlip` pool and `max_concurrent: 4`. Each worker owns one GPU; keep CPU thread
counts bounded to avoid oversubscription. At round boundaries, wait for the
round's results before issuing the next adaptive decisions. Scale beyond one
node with Slurm array/chunk jobs and a manifest-backed round barrier; the
in-node worker pool does not itself coordinate multiple nodes.

The current
[`job-e3-mlip-preparation-perlmutter.sh`](../../../deployments/perlmutter/jobs/job-e3-mlip-preparation-perlmutter.sh)
is CPU-only single-point prediction and does not implement this plan. A new
GPU relaxation entrypoint and adaptive controller are required before the
pilot. The plan requires, at minimum, an action schema, size-aware/class-aware
generators, resumable optimization manifests, a round controller, and the
coverage/cost analysis outputs described above.

## 7. Pre-production gates

Do not start the 5,000-per-class campaign until all of these pass:

1. The MOF seed-composition issue is resolved and framework validity is
   checked.
2. The pinned checkpoint's species support and one relaxation in every
   class/tier are verified offline.
3. The 280-relaxation pilot meets the force, failure-recording, restart, and
   GPU-utilization requirements; measured costs are used to size the campaign.
4. AFLOW/PyXtal/class-specific generation actions are reproducible, size-aware,
   and produce no invalid or trivially duplicated structures in the pilot
   acceptance set.
5. Rule and LLM policies share the same action schema, candidate validity
   rules, model, optimizer, budgets, and result-cache semantics.
6. The Perlmutter allocation and campaign node-hours are explicitly authorized.

Until then, the current inference-only Slurm job remains a diagnostic and must
not be represented as the E3-MLIP exploration campaign.
