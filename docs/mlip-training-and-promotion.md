# MLIP training references and model promotion

Active-learning fine-tuning, held-out energy comparisons, and model promotion
compare energies through elemental DFT references. This page describes those
references, how each MLIP backend is fine-tuned, and when a candidate model may
replace the incumbent. The surrounding workflow contracts are described in
[scientific workflow contracts](scientific-workflows.md), and the active-learning
configuration in [active learning](guide/active-learning.md). The label and
reference conventions of each backend are summarized in
[MLIP and DFT energy conventions](mlip-energy-conventions.md).

## Training references

Every AL fine-tune trains against elemental DFT references, so
`trainer.enabled: true` requires `trainer.training_references` or
`trainer.validation_reference_set`. The configuration is rejected when it
loads, and `run_active_learning` checks again before any MD or DFT runs (for
callers that edit a loaded configuration), instead of failing at the first
retrain after that iteration's DFT. HydraGNN trains on formation energies
relative to the references (below); UMA and MACE use them to align the labels
to their foundation model's energy scale
([Fine-tuning UMA and MACE](#fine-tuning-uma-and-mace)). The frozen reference
snapshot is shared by every backend.

### HydraGNN formation energies

Both routed and new-head fine-tuners now require verified elemental DFT
references and train on physical formation energies plus unchanged forces.
Mixture-fitted composition offsets have been removed: a single-formula dataset
cannot uniquely identify physical elemental coefficients.

The shared formation-label preparation layer is implemented. Given a verified
native-total-energy dataset sidecar and compatible elemental DFT manifest, run:

```bash
python -m matsim_agents.data.formation_training \
  --dataset /path/to/raw-dft.extxyz \
  --elemental-reference-manifest /path/to/elemental-references.json \
  --output-dir /path/to/new-formation-snapshot
```

The output directory must not already exist. It contains `formation.extxyz`,
a method/hash/split-role sidecar, `energy-convention.json`, and a self-contained
copy of the elemental-reference geometries and manifest. Energy labels are
total-cell formation energies; forces, available stress, and geometries are
preserved. Each frame also records its original DFT total and elemental baseline
so total energy can be reconstructed. Raw input data and partition membership
are unchanged. Missing references/forces, incompatible methods, invalid labels,
and already-transformed input are rejected.

This command does not launch DFT, select elemental phases, or train a model.
The fine-tuners accept the **raw total-energy** dataset and the required
`--elemental-reference-manifest` argument; they create their own immutable
`training-reference/` snapshot before building graphs. Do not pass an already
converted dataset: double conversion is rejected. The standalone comparison
driver forwards its existing elemental manifest to both fine-tuners.
Force labels are read without applying ASE constraints, so constrained atoms
retain their raw DFT forces in both the snapshot and fine-tuning targets.

The shipped AL YAMLs use `finetune_hydragnn_newhead.py` and one process for
formation-aware retraining. Configurations that train with a branch-weighting
MLP should select `finetune_hydragnn.py` instead and provide that MLP checkpoint.
HydraGNN's `examples/mptrj/train.py` is a pretraining entry point, not an AL
fine-tuner, and is not used by these configurations.
AL forwards the configured charge/spin conditioning to both built-in trainers
and `mlip.ft_repo` to the new-head trainer, including through HPC launchers.

AL training requires references even with comparison and promotion disabled.
Configure a verified manifest explicitly:

```yaml
trainer:
  training_references:
    manifest: /path/to/elemental-references.json
```

Alternatively use an approved elemental-reference phase plan automatically:

```yaml
trainer:
  training_references:
    phase_plan: /path/to/approved-phases.yaml
    cache_dir: /path/to/elemental-dft-cache
    phases_approved: true
    dft_approved: true
    max_dft_calculations: 4
```

The reference calculation cap is **separate** from the compound-label cap.
Uncached calculations require explicit DFT approval; fully cached references
can be reused with `dft_approved: false` and a zero cap. The existing
`validation_reference_set` can supply training references when no explicit
training reference source is configured. AL freezes the selected reference
snapshot under its output directory and rejects changed reference inputs or
DFT protocols on subsequent iterations/restarts; use a new dataset for changes.
Dataset method metadata records the label-stage element scope, including
held-out species, so reference preparation uses the same potential set as
labeling rather than only the training split's elements. Relative reference
cache paths are resolved before launching relaxation or single-point jobs.

Saved checkpoints bind `energy-convention.json` to the checkpoint hash,
inference configuration, the exact bundled elemental-reference manifest, and
the verified training snapshot. AL also checks that the emitted bundle's source
dataset hash matches the requested dataset before accepting the candidate.
Neural outputs are
total-cell formation energies. ASE-facing calculators add the recorded
elemental baseline **exactly once**, returning DFT-reference total energies;
existing evaluation subtracts its own model elemental references once.
Forces/stress and dropout access are preserved. New-head checkpoints are
auto-detected, and routed checkpoints retain their trained-head restriction
and frozen routing MLP. Unsupported elements, incompatible/tampered artifacts,
and fine-tuned checkpoints without a verified energy convention fail explicitly;
unmarked foundation checkpoints are used unchanged.

Checkpoint reconstruction materializes HydraGNN's lazy graph-conditioning
projectors from saved weight shapes before loading weights. Single-head resume
and inference use strict loading; declared routed/selected-head inference reloads
strictly after the upstream loader so lazy backbone weights cannot be dropped.

Both fine-tuners honor the training configuration's resolved autocast dtype
for dry-run, optimization, and validation losses. BF16 keeps FP32 master
parameters and uses autocast for compute; single-head checkpoint inference
also honors that setting. FP32/FP64 paths do not enable autocast.

The fine-tuning recipe belongs to the model: `mlip.fine_tuner` is
`finetune_hydragnn` (default; routed multi-head, needs `mlp_checkpoint`) or
`finetune_hydragnn_newhead` (drops all heads and trains one new head). The
fine-tuners are single-process, so configure one node/one rank for the
`training` pool.

### Approved elemental DFT prerequisite

Declared HydraGNN checkpoints with bundled routing artifacts can be used by
direct relaxation, chat, and the LangGraph executor without an external
BranchWeightMLP path. Artifact integrity is still verified when loading the
calculator. Branch-weight confidence is unavailable when fewer than two trained
branches are active; relaxation reports no `top_branch_weight` in that case.

`matsim_agents.workflows.active_learning.elemental_dft` calculates or reuses reference
energies for an approved phase list. Every phase explicitly declares whether its
input geometry is already verified as converged or requires relaxation. The
verified path remains a fixed-geometry single point; the other path performs a
DFT relaxation, requires the requested force threshold to be met, then performs
a production-protocol single point on the relaxed geometry. It does not search
for additional structures or magnetic states, apply fitted corrections, or
prove an elemental ground state: selection is the lowest energy per atom among
the declared phases at zero pressure.

The phase-plan YAML has this schema (paths are relative to the plan file):

```yaml
protocol: declared_phases_zero_pressure_v3
phases:
  - phase_id: nb-bcc
    element: Nb
    structure_path: references/Nb-bcc.extxyz
    kind: bulk
    geometry_status: requires_relaxation
    magnetic_state: nonmagnetic
    relaxation_force_tolerance_eV_per_A: 0.02
    relaxation_stress_tolerance_eV_per_A3: 0.01
    relaxation_max_steps: 200
  - phase_id: ta-bcc
    element: Ta
    structure_path: references/Ta-bcc.extxyz
    kind: bulk
    geometry_status: verified_converged
    magnetic_state: nonmagnetic
  - phase_id: oxygen-triplet
    element: O
    structure_path: references/O2-vacuum.extxyz
    kind: molecule
    geometry_status: requires_relaxation
    magnetic_state: triplet
    relaxation_force_tolerance_eV_per_A: 0.02
    relaxation_max_steps: 200
    qe_magnetic_settings:
      nspin: 2
      tot_magnetization: 2
    kpts: [1, 1, 1]
```

`geometry_status` is required: use `verified_converged` only when convergence
has been independently established and attested; otherwise use
`requires_relaxation`. Bulk relaxation defaults to relaxing atoms and cell
(`vc-relax`/VASP `ISIF=3`); set `relax_cell: false` to retain the input cell.
Bulk cell relaxation requires an explicit `relaxation_stress_tolerance_eV_per_A3`.
The maximum absolute final stress component must be within this threshold; VASP
and QE stress are checked in eV/Å³ and the measured stress is retained in the
relaxation cache record. The tolerance participates in the cache identity.
Molecules always retain their cell and vacuum while relaxing atomic positions.
The force tolerance and ionic-step limit are explicit per phase. The example is
a declaration format, not supplied physical reference data or a general
convergence recipe. Molecular cells need adequate vacuum. For VASP, replace
QE overrides with `vasp_magnetic_settings: {ISPIN: "2", NUPDOWN: "2"}` and
declare k-point settings in the shared VASP configuration instead of phase
`kpts`. Magnetic-state text documents intent; the actual DFT spin controls must
be supplied explicitly in the shared configuration or allowed phase overrides.
Wrong-backend overrides are errors.

Supply a standalone YAML `DFTConfig` block (the contents of AL `dft`, not a
whole AL configuration), with absolute executable, wrapper, pseudopotential,
and template paths. Generated QE inputs must explicitly pin both cutoffs,
k-points, and occupations to avoid composition-dependent automatic defaults.
QE templates cannot be combined with magnetic overrides because those overrides
would be ignored. Phase overrides are restricted to declared spin controls and
QE reference k-point sampling; they cannot change the functional or potentials.
Unverified QE geometries require generated QE inputs; template-based QE inputs
cannot be safely transformed into relaxations and are rejected before DFT work.
Trusted geometries can continue to use the fixed-geometry path with templates.
The common compound method signature and each reference's actual method
signature are both recorded, including the explicit state/sampling exceptions.

```bash
python -m matsim_agents.workflows.active_learning.elemental_dft \
  --phase-plan /path/to/approved-phases.yaml \
  --config /path/to/dft-and-resources.yaml \
  --elements Nb Ta O \
  --cache-dir /path/to/shared-reference-cache \
  --output-dir /path/to/new-reference-snapshot \
  --approve-reference-phases --approve-dft \
  --max-dft-calculations 5
```

The `--config` file holds a top-level `dft:` block and a `resources:` plan
with a `dft` pool ([resource dispatch](resource-dispatch.md)); the first DFT
job group of that pool runs every reference calculation.
Run on allocated compute resources with the backend's wrapper/environment,
not a login node. Phase-list approval is always required. Without
`--approve-dft`, all phases must already have valid compatible cached results.
The calculation cap counts missing unique DFT invocations and fails before
launching if the full requested reference set exceeds it. An uncached phase
requiring relaxation normally costs two invocations: the relaxation and its
final single point; a verified geometry costs one. Cache identities include
the input/relaxed geometry, method/executable/potential/template hashes, and
declared phase and relaxation controls; per-entry file locks serialize shared
cache calculations. Malformed cache entries fail explicitly rather than
silently recalculating. Only force-converged relaxations and successful finite
single-points are cached; failures expose their work directories. This stage's
DFT calls are separate from AL label budgets and must be budgeted with its own
cap.

The output directory must be new. Its `elemental-references.json` is compatible
with the formation-label preparation command and includes all tested phases,
selected phases, calculation locations, actual reference method signatures,
and both source and final geometry hashes, plus relaxation provenance.
Selected final geometries are copied into the snapshot.
All input formats supported by ASE, including CIF and POSCAR, are serialized
as extxyz before caching the single point. The source hash remains separate
from the hash of the serialized geometry used by the cache and snapshot.
Formation-label snapshots retain this selection audit. Cache reuse does not
change an already-created reference/training snapshot or choose new endpoints
silently. QE and VASP caches remain method-separated.

The implemented replacement workflow:

- Require an approved list of pure-element reference phases covering all
  training species, independently of comparison or promotion being enabled.
- Reuse verified compatible DFT calculations or, with DFT approval, calculate
  missing references before training. References must declare their structures,
  magnetic states, DFT method, and any corrections. Molecular references such
  as triplet O2 must be explicitly declared rather than treated as bulk crystals.
- Select the lowest DFT energy per atom among the declared compatible phases
  for each element under a zero-pressure energy protocol. This is the lowest
  among tested phases, not a claim of a global ground-state search. Other
  pressure or temperature conditions need an explicit corresponding protocol.
- Preserve raw DFT totals and derive total-cell formation-energy labels as
  `E_formation = E_DFT - sum(N_element * e_element_reference)`. Forces remain
  unchanged. Actual cell atom counts must be used, including supercells.
- Apply one reference set to all polymorphs and compatible formulas; do not
  independently zero each polymorph. Keep elemental reference selection fixed
  within a training snapshot and record its provenance.
- Remove mixture-fitted offsets from both HydraGNN trainers and record the new
  declared DFT formation-energy convention in each fine-tuned checkpoint.
  Adapting to this convention does not assert that the pretrained head used it.
- Make checkpoint reload, materials ranking, evaluation, and hull analysis
  honor that convention. Reconstruct DFT-reference total energies only where
  needed, and prevent a second elemental subtraction from formation outputs.

The formation-energy convention is HydraGNN's; UMA and MACE keep their
foundation's total-energy convention and align the labels to it instead
([Fine-tuning UMA and MACE](#fine-tuning-uma-and-mace)). Per-formula
cumulative collection and permanent train/held-out membership remain unchanged.
Validation must cover multiple polymorphs, cell multiplicity, reference reuse,
missing or incompatible references, approval gates, unchanged forces, and
checkpoint reload without double referencing. CPU contract tests cover the
reference/training/inference wiring; end-to-end real GPU model and DFT
qualification of this new workflow remains pending.

## Fine-tuning UMA and MACE

A foundation model's energies follow its pretraining DFT setup (UMA OMat and
MACE-MP: VASP PBE). Labels from Quantum ESPRESSO, or VASP with other settings,
differ from it by large per-element constants; training on raw totals would make
the model absorb them. The UMA and MACE fine-tuners
(`matsim_agents.backends.mlip.training.finetune_uma` / `finetune_mace`, both requiring
`--elemental-reference-manifest`) therefore shift every label by the elemental
reference offsets:

```text
E_label = E_DFT - sum over atoms i of (e_DFT[el_i] - e_model[el_i])
```

where `e_DFT` is the DFT energy per atom of each declared elemental reference
and `e_model` the base model's energy per atom on the same geometry. DFT
formation energies relative to the references are unchanged, and the fine-tuned
model stays on its foundation's scale. The alignment is written to
`energy-alignment.json` beside the checkpoint, with the hashes of the training
dataset and reference manifest; `retrain_uma` / `retrain_mace` reject a model
whose recorded hashes do not match what they asked for. Fine-tuning a model that
has an `energy-alignment.json` reuses its recorded model references, so repeated
fine-tunes keep one scale. Promotion comparisons are unaffected: each model's
formation energies use its own predicted elemental references.

Other details:

- **UMA**: the energy loss is per atom (`(E_err / N_atoms)^2`, so large cells do
  not dominate) plus `force_weight` times the force MSE. 10% of the frames (at
  least one) are held out, scored after every epoch, and the weights of the best
  epoch are kept.
- **MACE**: `mace_run_train` uses the foundation's atomic energies
  (`--E0s foundation`) instead of refitting them from the training set
  (`--E0s average`), which cannot separate the elements when the AL data cover
  few compositions; multihead replay of pretraining data is off (compute nodes
  are offline). When every frame carries a DFT stress, stress is trained too
  (`--loss stress`). The retrain fails if no `mace_finetuned.model` is written.
- **Adapters**: `trainer.finetune_mode: lora` (default) trains low-rank adapters
  and the heads, which keeps more of the pretraining on small AL datasets;
  `full` trains all weights. HydraGNN uses its fine-tuner's own strategy.
- **Start point**: by default each retrain starts from the current model (the
  last promoted one). `trainer.start_from: foundation` starts every retrain from
  the model the run was configured with, on the cumulative dataset, so
  successive fine-tunes do not compound.
- **Stress**: the UMA and HydraGNN fine-tuners train on energies and forces only.
- **Launchers**: every backend uses the same `trainer.train_launcher` contract.
  The trainer runs `bash <launcher> <backend> <nodes> <ranks_per_node> -m
  <module> <args...>`, where the MLIP configuration builds the fine-tuner
  module (`mlip.fine_tuner`) and its complete argument list. A launcher does not
  interpret those arguments: it prepares the site environment of `<backend>`,
  applies the training pool's placement (`MATSIM_ASSIGNED_NODES`,
  `MATSIM_DEVICE_MASK`, `MATSIM_NODE_DEVICES`), and runs `python "$@"`. The
  site launchers are `deployments/<facility>/launchers/_mlip-train-step-<facility>.sh`
  (Perlmutter and Aurora: hydragnn, uma, mace; Frontier: hydragnn). Without a
  launcher the fine-tuner runs in the driver's process environment on its node.

## Keeping the surrogate loaded across iterations

The AL loop loads the surrogate on the driver for scoring and in one worker per
mlip device for MD. When the resource plan gives the `mlip` pool devices that
neither the `dft` nor the `training` pool uses, these models and the MD workers
stay loaded across iterations and are reloaded only when a candidate is
promoted; the promotion comparison then reuses the loaded incumbent (unless it
was loaded with MC dropout for acquisition). When the devices are shared, the
models are released before DFT and reloaded each iteration, as before. The run
log states which applies. The retrained HydraGNN candidate is checked from its
recorded hashes rather than loaded; the promotion comparison loads it.

## Current promotion and evaluation behavior

Offline evaluation of fine-tuned models (which split, which partition selects,
and which is the untouched test) is described in
[MLIP evaluation](./mlip-evaluation.md). In the AL loop below, promotion uses
held-out validation labels only; those are held out frame by frame, so they are
correlated with training frames of the same trajectories.

Promotion requires both held-out formation-energy MAE (eV/atom) and
force-component MAE (eV/Å) to meet their absolute limits and not worsen relative
to the incumbent, apart from an absolute `1e-12` numerical tolerance in each
metric's units. At least one must also improve by
`promotion_min_relative_improvement` (default `0.05`, meaning a 5% relative
MAE reduction). Improvement in one metric cannot compensate for regression
in the other. Equal performance, two worse metrics, or improvements below the
configured threshold retain the incumbent, with explicit rejection reasons.
Finite, nonnegative MAEs and the existing minimum evaluated-frame counts are
still required.

Configure the threshold in AL `trainer` or campaign `retraining` YAML:

```yaml
promotion_min_relative_improvement: 0.05
```

The Perlmutter CLI exposes `--promotion-min-relative-improvement`; the campaign
launcher uses `MATSIM_CAMPAIGN_PROMOTION_MIN_RELATIVE_IMPROVEMENT` (default
`0.05`). Values must be finite and between zero and one. Zero still requires a
strict reduction larger than the numerical tolerance. An exact-zero incumbent
MAE cannot improve, but the other metric may qualify. This threshold is a policy
choice, not statistical significance or broad scientific qualification.

Promotion never accepts an MAE regression on either metric. Training,
cumulative collection, and permanent held-out partitions are unchanged;
rejected candidates do not replace the incumbent.

Scientific MLIP-versus-DFT energy comparisons use formation energies, not
raw energy zeros or offsets fitted to training/test compounds. Every energy
test must provide DFT calculations for its pure-element reference geometries
using the same DFT setup as its compound labels. Each MLIP evaluates those
exact fixed geometries first and subtracts its own elemental energies; DFT
subtracts its own references. Molecular references such as O2 are normalized
by their actual atom count. Missing elements, impure references, changed
geometry hashes, non-finite energies, and missing manifests are errors.

Supply a JSON manifest, with paths relative to the manifest:

```json
{
  "backend": "qe",
  "method_signature": "qe-pbe-reference-protocol-v1",
  "references": {
    "Nb": {
      "structure_path": "Nb.extxyz",
      "structure_sha256": "<SHA-256 of the fixed geometry file>",
      "energy_eV": -20.0
    },
    "O": {
      "structure_path": "O2.extxyz",
      "structure_sha256": "<SHA-256 of the fixed geometry file>",
      "energy_eV": -10.0
    }
  }
}
```

The energies above are illustrative total-cell values, not physical reference
data. Populate them from converged DFT calculations on the declared geometries,
with full coverage of the elements in the test. The manifest is not a request
to launch DFT: the caller must supply those calculations and ensure method
compatibility with the compound dataset. Use a separate manifest for each
DFT protocol being compared. Baseline errors remain available as diagnostics.
File-based energy evaluation, fine-tune/evaluation, and promotion require a
`*.extxyz.manifest.json` compound-dataset sidecar. They check its `sha256`,
`dft_backend`, and `method_signature` against the data and elemental manifest
before inference. Missing or incompatible metadata is an error, not a warning.
Campaign promotion checks supplied validation metadata before starting the
loop. Fine-tune/evaluation writes matching sidecars for newly generated splits;
fraction-based active learning also maintains separate training and held-out
sidecars on each iteration, checking the held-out hash and method before append.
New split sidecars explicitly record `split_role: training_pool` for training
and `split_role: validation` for held-out data, including AL-generated splits.
Training accumulates within each formula's AL dataset across iterations and
restarts; this does not pool labels across formulas. `validation_fraction`
targets the cumulative held-out fraction by assigning only newly accepted
frames. Existing assignments are permanent: training frames never move to
validation, held-out frames never enter training, and duplicate labels are
rejected against both partitions. The target fraction is approximate for
small batches and cannot retroactively repartition prior data.

Readiness is checked against the cumulative data, not the latest batch.
Training needs at least two accumulated training frames; energy comparison
also needs at least `promotion_min_evaluated_frames` energy-labelled and
force-labelled held-out frames. Valid small batches are saved with
`training_status: deferred` and explicit `training_deferred_reasons` until
those requirements are met. Iteration state records both new-frame counts and
`n_training_frames_total` / `n_validation_frames_total`. No candidate is trained
or promoted during deferral, and the incumbent remains active. The
campaign skips post-training reevaluation when training is deferred; this is
distinct from rejection of an evaluated candidate, which still follows the
configured promotion-rejection policy. A one-iteration
run acquiring two fresh labels with a 20% split therefore retains one training
and one validation frame and defers training; extend the iteration count or
resume the same formula dataset to accumulate more data. DFT failures and
duplicate rejection can delay readiness further. These are software minimums,
not evidence that such a small validation set scientifically qualifies a model.

Eval-only reuse requires existing, hash-matching split sidecars. Data without a
recorded DFT method signature is rejected; do not infer compatibility from the
elemental manifest alone. The caller
remains responsible for convergence of the supplied DFT calculations.

Energy evaluation and fine-tune/evaluation require
`--elemental-reference-manifest`; force-only evaluation does not require
elemental references or model energy predictions. Campaign promotion uses
`--promotion-validation-reference-set` / `trainer.validation_reference_set`
for this JSON manifest, not for training-partition offset fitting. Perlmutter
retraining submissions that request model promotion or enable
`trainer.compare_after_training` in the selected AL config require
`MATSIM_CAMPAIGN_PROMOTION_VALIDATION_REFERENCE_SET` before submission. The
all-model launcher checks this before starting model servers.
Fine-tune/evaluation launchers require `ELEMENTAL_REFERENCE_MANIFEST`, or an
existing `elemental_references.json` in each dataset's case directory.
Promotion energy thresholds now apply to
`formation_energy_mae_eV_per_atom`; raw total-energy errors remain diagnostic.
Evaluation outputs no longer contain `energy_*_per_atom_shifted` fields; use
`formation_energy_*_per_atom`. Reassess historical thresholds under this
convention. Force comparisons and within-method polymorph/hull
ranking are unchanged; final DFT ranking results still never enter training.

Nb-Ta-O reference preparation merges curated phases before unary structure
deduplication, retaining curated entries ahead of equivalent bootstrap,
AFLOW, or pyXtal candidates so duplicate unary phases are not relaxed twice.

Campaigns use this comparison for held-out incumbent/candidate evaluation and
promotion only. Active-learning acquisition, force metrics, same-composition
polymorph ranking, and method-specific hull construction retain their existing
semantics. Disabling energy comparison does not remove the training-reference
requirement: `trainer.enabled: true` always needs `trainer.training_references`
or `trainer.validation_reference_set`. Before training a candidate that will be compared, the active-learning
loop checks element coverage and requires the reference backend/method signature
to match the generated compound labels. Use the campaign's recorded DFT method
signature, not a descriptive label from an unrelated hull protocol.

For method `m`, the formation energy per atom is
`(E_compound_m - sum_i(n_i * mu_i_m)) / N`, where `mu_i_m` is that method's
elemental energy per atom on the declared geometry. Different MLIPs and DFT
protocols retain separate baselines; no cross-method totals are pooled.
This removes additive elemental energy zeros, not approximation errors.
If the fixed reference phases are not elemental ground states, describe the
result as formation energy relative to those declared phases rather than
claiming a ground-state thermodynamic formation energy.

Pure-element baseline errors remain visible because cancellation can yield
good formation energies even when individual elemental predictions are poor.
Formation-energy accuracy does not establish hull stability, force accuracy,
or accurate same-composition energy differences. Promotion retains force
and incumbent-regression gates rather than relying on formation energy alone.

For Codabench, the same convention applies to Task 1 and the energy inputs to
Task 5. See the [participant guide](../benchmarks/codabench/starting_kit/README.md#elemental-reference-manifest)
for the reference schema, inference commands, outputs, and release requirements.
