# E3-MLIP adaptive crystallographic exploration plan

Status: **production implementation incomplete; production execution is not authorized**.
This plan changes E3-MLIP only. It does not change E3-DFT, use DFT labels or
references, or submit VASP calculations.

## 1. Experimental contract

- Freeze **MACE-OMAT-0 medium**, trained on OMat24, for both Rule-based and
  LLM-guided adaptation. Its MACE-Torch loader alias is `medium-omat-0` and its
  E3 CLI alias is `mace-omat-0-medium`. The frozen artifact is
  `${MACE_ARTIFACT_DIR}/mace/maceomat0mediummodel`, SHA-256
  `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`.
  The loaded checkpoint supports all campaign species (C, Co, Cr, Fe, H, Hf,
  Li, Mn, Nb, Ni, O, P, Ta, Ti, V, Zn, Zr). Verify the hash at every round
  start and do not silently substitute another model. MOF transferability is
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
  source structures. Use MLIP-only evidence for acquisition and assessment.
  Report candidates as **MLIP-predicted**, not independently validated
  discoveries; independent DFT validation is outside E3-MLIP.

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
| Zn(HCOO)2 MOF | COD 7205949 experimental Pbcn zinc-formate framework as the verified parent; additional topologies, proton configurations, bounded strain, and defects require connectivity-preserving generation. Generic AFLOW decoration or unconstrained 3D PyXtal is not accepted as a MOF generator. | 216-atom, 24-formula-unit CIF cell and 2x, 4x, 8x multiplicities (432, 864, 1,728 atoms). | 1,250 |

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
   optimizer and no relaxation cap, or selects from a shared compact,
   hierarchical allow-list. The deterministic and LLM selectors validate
   against that same full action space; the LLM receives strategy-local
   summaries and compact dimension codes, and must cite valid evidence.
   Class-specific generators now cover all six materials. The pristine
   LiFePO4 parent is a finite-capacity action: at most one parent structure
   per tier is allocated across rounds, with exhausted actions removed from
   both selectors. The older `generate_seeds` defaults remain for backward
   compatibility and must not be used by the adaptive E3 runner.
3. Use exact geometry hashes before symmetry matching. Acquisition maintains
   composition-, size-tier-, and atom-count-scoped symmetry indexes of accepted
   representatives, so equivalent structures within a batch, round, or prior
   round do not consume additional unique-configuration budget. Finite-size
   controls retain their valid material generator family and a separate
   control-family identity; they are not counted as new structure families.
   Offline relaxed-minimum clustering remains the comprehensive final
   structural-diversity analysis. Reject unconstrained coordinate noise,
   broken MOF connectivity, non-periodic overlaps, and supercell copies
   masquerading as new structural families.
4. **MOF seed source:** E3-MLIP preparation uses the structure-only experimental
   CIF COD 7205949 (DOI `10.1039/c1ce05031a`), which expands to
   Zn24C48H48O96 (216 atoms, 24 Zn(HCOO)2 formula units, Pbcn); its source
   SHA-256 is `df49d464c2cffdf50abe5c5442f0e2e2719fff77e950a43463ab8476a1887d10`.
   The two older alpha/beta EXTXYZ files remain excluded: they contain Zn4C8O8
   and omit hydrogen. The full campaign still requires connectivity-preserving
   class-specific generation beyond this verified parent structure.
5. Before freezing production manifests, inspect the pinned MACE checkpoint
   for every element and automatically verify calculator loading and species
   compatibility for every class. Element support is only a
   software-compatibility check, not evidence of transferability. A failure
   blocks that class; do not swap models without a recorded protocol amendment.

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
  silently repair them. Freeze model/provider/prompt and record tokens and
  time; the Copilot SDK uses provider-default decoding rather than a fixed
  temperature.
- Permit the strategy to change compositions, prototype/space-group focus,
  size tiers, generator mix, and perturbation family within the same class
  validity rules.

The shared physical-result cache is keyed by the canonical initial geometry
hash, model/checkpoint hash, optimizer version, and optimization settings.
Cache reuse saves physical work but does not reduce either strategy's logical
budget or erase its independent action provenance.

## 4. Mandatory MLIP geometry optimization

Use the existing MACE relaxation path with a single frozen checkpoint per
worker. The frozen common protocol for both strategies is:

- Initial energy and forces recorded before optimization.
- ASE FIRE optimizer; maximum 500 optimization steps; maximum atomic step
  0.05 A; convergence requires maximum atomic force and filter force below
  0.02 eV/A and every active native stress component at or below 0.01 eV/A^3.
  The stress threshold is an exploratory screening choice, not a validated
  zero-pressure accuracy guarantee. The larger
  displacement cap avoids artificially slow movement from strongly distorted
  starts, while remaining conservative for local MLIP relaxation; the larger
  step cap prevents early termination from being mistaken for convergence.
- Store initial structure, every available trajectory frame, per-step energy
  and maximum force, final structure, final energy/forces, and termination
  outcome. A single-point evaluation is never marked as an optimized result.
- Relax atomic positions and the cell for all classes using native MACE stress;
  numerical stress fallback is forbidden. Phosphorene uses only xx, yy and xy
  strain, preserving its out-of-plane cell and vacuum. The filter requires the
  vacuum vector aligned with z. Stop and roll back any step exceeding 10%
  principal strain in one step or 50% cumulative principal strain, or changing
  the initial MOF atom-indexed Zn-O, C-O, or C-H bond graph. The periodic
  neighbor-list guard uses respective distance cutoffs of 2.75, 1.68, and
  1.25 A; bonds between the same atom indices in different periodic images
  are treated as one bond. These limits are screening safeguards, not proof
  of physical stability. Record active stress, atomic force and volume at
  each accepted step.
- Abort when the maximum optimizer force (including cell pseudo-forces)
  increases by more than 20% between completed steps. This does not loosen
  the final force/stress convergence targets. The force-growth and both
  strain safeguards accept `None` to disable them explicitly; such E3
  protocol changes require a regenerated manifest and separate validation.
- Each geometry optimization has a 60-second wall-clock target, starting at
  the optimization loop after model loading and initial energy/force evaluation.
  The clock is checked after each completed, validated step: a long or stalled
  step can exceed the target without bound. If the limit is reached before
  convergence, save the last valid structure and trajectory as nonconverged
  with `optimization_time_limit`. If a step converges or triggers a geometry
  safeguard, that result takes precedence over the time limit. The loaded
  model remains available for subsequent structures in the same GPU worker
  and is reloaded when the worker restarts at a round boundary. Initialization,
  result processing, generation, LLM calls, and scheduling are outside this
  target, so it is not a cap on total GPU allocation time.
- LLM-guided allocation uses Copilot SDK with `gemini-4-carbon` and default
  provider decoding. Temperature is not controllable through the SDK; its
  requested output-token limit is not guaranteed by Copilot-hosted models.
- Preliminary energy finite-difference checks covered xx and xy at two strain
  magnitudes for one representative per family; they do not qualify every
  stress component, structure, cell filter, or GPU backend. The exploratory
  relaxation checks did not converge for most families, and the oxide hit its
  cumulative-strain limit. Validate remaining cases and measure throughput
  before a new production submission; do not reuse fixed-cell campaign results.
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
   supercell matching enabled. Freeze these before round 1 and report a
   tolerance-sensitivity check. Record the relaxed-minimum cluster ID
   separately from the starting-configuration ID and size-control family.

Report counts of proposals, valid unique starts, optimized starts,
convergences, failures, relaxed-minimum clusters, and cross-size family
relations separately. A converged relaxation is not automatically a distinct
material discovery.

## 6. Production throughput and resource controls

There is no separate disposable pilot. After automated model and species
checks, the first acquisition round is production data and remains in the
scientific dataset. Record per class and size tier the structures per GPU-hour,
relaxation-time distribution, optimizer steps, convergence/failure rates,
GPU memory, model load time, and I/O overhead. Use those measurements to size
later round jobs without changing the frozen optimizer protocol or either
strategy's logical budget.

The current plan contains 60,000 logical optimizations for the matched
strategy budgets plus 1,120 finite-size-control optimizations, before exact
hash cache reuse. At the 500-step cap this is at most 30,560,000 attempted
optimizer steps. This is a step-count bound, not a walltime or GPU-hour
estimate. A defensible aggregate node-hour request cannot be calculated until
the checkpoint and production runtime per class/size are known. Submit no job
until the required Perlmutter account, QoS, and aggregate resource authorization
are confirmed.

### GPU dispatch and scale-out

Structure relaxations are embarrassingly parallel across configurations;
optimizer steps within a single structure remain sequential. Use one
independently managed worker per configured A100 GPU, subject to runtime
hardware discovery. Perlmutter has four A100 devices per GPU node; verify this
at launch rather than assuming it. Multi-node scheduling, persistent
per-structure manifests, automatic restart, and round barriers are required;
the existing single-node `MLIPWorkerPool` alone is not the production
orchestrator.

The current
[`job-e3-mlip-preparation-perlmutter.sh`](../../../deployments/perlmutter/jobs/job-e3-mlip-preparation-perlmutter.sh)
is CPU-only single-point prediction and does not implement this plan. A
multi-node GPU geometry-optimization entrypoint, manifest-backed scheduling,
automatic restart, and adaptive round barriers are still missing. Do not
represent the current CPU single-point job as an E3-MLIP campaign.

## 7. Production start conditions

Production may start only after all of these concrete conditions pass:

1. All six class-specific generators produce valid, provenance-complete
   structures, including a chemically complete, connectivity-checked MOF
   source; class omissions are not permitted.
2. The exact MACE checkpoint is frozen by identifier and hash, and automated
   loading/species checks pass for every class.
3. The 500-step, 0.05 A, 0.02 eV/A FIRE protocol and all termination outcomes
   are recorded in the manifests and applied identically to both strategies.
4. Persistent per-structure/per-round manifests, restart after worker failure,
   distributed scheduling, and strategy-specific round barriers are tested.
5. Rule and LLM policies share the same generator capabilities, validity
   rules, model, optimizer, matched logical budgets, and cache accounting.
6. The Perlmutter allocation and aggregate campaign resource budget are
   explicitly authorized.

The first acquisition round is part of production, not a separate pilot; its
valid results and resource measurements are retained.
