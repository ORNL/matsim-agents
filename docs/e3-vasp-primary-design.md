# E3 VASP-Primary Design Review
Status: design proposal for author review. No E3 production or pilot jobs are authorized. This document does not establish that any model-system cell is supported, any reference dataset is ready, or any VASP allocation is available.

## A. Unified E3 Protocol
E3 evaluates pretrained HydraGNN, UMA, and MACE on the six material systems using one newly frozen VASP reference methodology. Its maximum baseline matrix is 18 model-system cells; each cell first passes a model-support and independent-data qualification. A cell is classified as `unsupported`, `insufficient validation evidence`, `pass`, or `fail`. Only a supported cell that fails a frozen validation-only scientific accuracy gate enters the matched adaptive C-versus-D workflow. Insufficient evidence is not a failure; obtain independent VASP references or report the gap. A passing surrogate is retained, but does not waive DFT required by the scientific contract, including the Nb-Ta-O domain hull.

New E3 structures may be informed by reviewed historical geometries, but all E3 reference energies and forces must be newly calculated with the frozen VASP signature. QE energies and forces remain in a separate historical/QE namespace and must not enter E3 training, validation, test, accuracy thresholds, phase rankings, or hull construction. Do not mix VASP and QE totals or infer equivalence from a shared functional label. No cross-code energy normalization is authorized for E3.

For each material, define immutable structure identities, scientific groups, observables, VASP input signature, convergence rules, data lineage, and train/validation/test group assignments before evaluating models. Group related trajectory frames, polymorph variants, orderings, and duplicate structures together to prevent leakage. Validation chooses adaptation eligibility and checkpoints; the test partition is used once for final reporting. Preserve failed DFT attempts and retries.

## B. Full Baseline Matrix
The table enumerates all 18 candidate combinations. Every cell is pending system-specific runtime support, checkpoint/license/dependency review, and VASP-labeled independent validation data. Repository code or a YAML template is not evidence that a cell can run or that its reference set is ready. Run zero-shot evaluation only after those gates pass. The conditional C/D trigger applies per row, never based on test data.

| Material system | MLIP | VASP-reference baseline | C/D adaptation trigger |
|---|---|---|---|
| LiFePO4 | HydraGNN | Pending support and data qualification | Only if validation fails |
| LiFePO4 | UMA | Pending support and data qualification | Only if validation fails |
| LiFePO4 | MACE | Pending support and data qualification | Only if validation fails |
| Cantor FCC (CrMnFeCoNi) | HydraGNN | Pending support and data qualification | Only if validation fails |
| Cantor FCC (CrMnFeCoNi) | UMA | Pending support and data qualification | Only if validation fails |
| Cantor FCC (CrMnFeCoNi) | MACE | Pending support and data qualification | Only if validation fails |
| NbTaVHfZrTi BCC | HydraGNN | Pending support and data qualification | Only if validation fails |
| NbTaVHfZrTi BCC | UMA | Pending support and data qualification | Only if validation fails |
| NbTaVHfZrTi BCC | MACE | Pending support and data qualification | Only if validation fails |
| Phosphorene | HydraGNN | Pending support and data qualification | Only if validation fails |
| Phosphorene | UMA | Pending support and data qualification | Only if validation fails |
| Phosphorene | MACE | Pending support and data qualification | Only if validation fails |
| Zn(HCOO)2 MOF | HydraGNN | Pending support and data qualification | Only if validation fails |
| Zn(HCOO)2 MOF | UMA | Pending support and data qualification | Only if validation fails |
| Zn(HCOO)2 MOF | MACE | Pending support and data qualification | Only if validation fails |
| Nb-Ta-O | HydraGNN | Pending support and fresh VASP domain evidence | Only if validation fails; hull DFT is independent |
| Nb-Ta-O | UMA | Pending support and fresh VASP domain evidence | Only if validation fails; hull DFT is independent |
| Nb-Ta-O | MACE | Pending support and fresh VASP domain evidence | Only if validation fails; hull DFT is independent |

Record an explicit status and reason for every row, including model-load failures, unsupported chemistry, missing independent groups, and cells blocked by resource approval. Do not drop unsupported or blocked rows from the denominator without showing them.

## C. Nb-Ta-O Migration Plan
The current inventory contains one limited Perlmutter VASP UMA campaign YAML and seven Nb-Ta-O QE campaign YAMLs. The VASP recipe is not an E3 campaign: it is restricted to an NbO2 seed, one iteration, a small selection, and disabled training. The QE files and the recovered QE run are historical/QE-specific evidence, not templates to convert in place.

1. Preserve all existing QE campaign YAMLs, datasets, run records, and method signatures unchanged. Keep QE data in a distinct archive/catalog namespace and label it `QE historical; excluded from E3-VASP`.
2. Create new versioned E3-VASP manifests for the agreed Nb-Ta-O phase domain and model cells. Never overwrite the limited VASP YAML or change the backend field of a QE YAML to imply that its data have been migrated.
3. Review historical structure files only as candidate geometries/prototype leads. Verify composition, atom order, structure hash, source, and lineage; then prepare and converge the VASP geometry under the frozen E3 protocol. A QE-relaxed geometry is not automatically a VASP-converged reference.
4. Recalculate all labels needed by E3 with VASP. Exclude QE energies/forces from E3 training, validation, test, thresholds, candidate ranking, and hulls. If an independent future cross-code study is approved, store its results separately; it does not silently relax this rule.
5. Register backend, complete VASP method signature, code/version, POTCAR identities/hashes, structure hash, convergence status, observable units, provenance, and group ID in each dataset manifest. Enforce a single backend/signature per E3 dataset and hull.
6. Regenerate leakage-safe train/validation/test splits by scientific group. Do not treat the historical QE training/validation partition as an E3 split or the old validation subset as a test set.
7. Retain `prepare_nb_ta_o_references.py` as a generator interface only after explicitly selecting its VASP path and validating the emitted manifest. Its QE default must not be allowed to silently supply E3 references.

The five canonical paper-case templates for LiFePO4, Cantor FCC, NbTaVHfZrTi BCC, phosphorene, and Zn-formate default to VASP, but this is workflow configuration evidence only. Existing historical paper labels remain ineligible until provenance, method, convergence, and group audits pass; the E3 design does not assume reuse.

## D. Nb-Ta-O Reference-Domain Manifest Proposal
Freeze a versioned, bounded domain before structure generation. The following is a draft scope inherited from the run-plan audit, not a claim of complete Nb-Ta-O thermodynamics. Each logical phase ID must later resolve to a validated structure, hash, source, and group. Required polymorph counts are minima for this declared domain, not proof that all physical polymorphs have been found.

| Subsystem | Required compositions | Draft minimum structure coverage |
|---|---|---|
| Elements | Nb, Ta, O2 | At least one validated elemental reference structure per element; freeze molecular O2 spin and cell treatment |
| Nb-Ta binary | NbTa plus Nb and Ta endpoints | At least one validated NbTa ordering/prototype; endpoints link to elemental references |
| Nb-O binary | NbO, NbO2, Nb2O3, Nb2O5 | At least one validated phase per formula; at least two distinct polymorph families each for NbO2 and Nb2O5 |
| Ta-O binary | TaO, TaO2, Ta2O3, Ta2O5 | At least one validated phase per formula; at least two distinct polymorph families each for TaO2 and Ta2O5 |
| Nb-Ta-O ternary | NbTaO4 and exploratory NbTaO5 | At least two predeclared symmetry-distinct orderings/prototypes for NbTaO4; NbTaO5 is secondary and is not promoted to a required stable phase solely from an LLM proposal |

Proposed logical identifiers include `element-Nb`, `element-Ta`, `element-O2`, `binary-NbTa-*`, `binary-NbO-*`, `binary-NbO2-*`, `binary-Nb2O3-*`, `binary-Nb2O5-*`, `binary-TaO-*`, `binary-TaO2-*`, `binary-Ta2O3-*`, `binary-Ta2O5-*`, `ternary-NbTaO4-*`, and `ternary-NbTaO5-*`. The wildcard is a slot for a frozen prototype/polymorph/ordering ID, not a structure identity. Do not assign final IDs or counts until crystallographic sources and generator outputs are reviewed.

The manifest schema should record: schema and domain version; target question; included/excluded formulas and rationale; phase ID and polymorph family; declared and parsed formula; structure source, generator/version/seed, raw and canonical structure hashes; group ID; VASP version and full method signature; POTCAR symbols and hashes; relaxation and force/energy convergence criteria and outcomes; raw energy and forces with units; attempt/retry lineage; and domain qualification result. Use a single VASP signature across the hull, including consistent functional, PAW release, correction policy, spin rules, energy cutoff, k-point policy, smearing, electronic convergence, ionic/cell relaxation protocol, and force/stress tolerances. Freeze those settings before production.

The archived manuscript reports PBE/PAW, ENCUT=520 eV, EDIFF=1e-6 eV, no Hubbard U, and case-specific spin, smearing, and k-point density for the five earlier VASP systems. These values are a starting proposal, not a complete relaxation signature for Nb-Ta-O; confirm Nb/Ta/O POTCAR releases, magnetic/oxidation-state treatment, O2 setup, k-point/cell policy, and ionic convergence before approval. Use separate formula, polymorph, and VASP-convergence gates. Any failed gate leaves the hull provisional and reported only relative to the frozen domain. Explicitly exclude unlisted ternary compositions, unenumerated polymorphs/orderings, high-pressure phases, and finite-temperature effects from the claim.

## E. Fair C-versus-D Protocol
For every failed supported cell, compare:

- C: LLM-guided candidate selection and acquisition rationale under a fixed evidence contract.
- D: deterministic numerical uncertainty/diversity selection under the identical candidate universe and contract.

Both arms start from the same checkpoint and initial VASP-labeled train/validation evidence, receive the same candidate pool and information, and have the same attempted-label cap, VASP signature, maximum retry policy, adaptation algorithm, validation rule, and checkpoint-promotion approval gate. Only the acquisition policy differs. Freeze stopping and expansion rules before observing outcomes. Keep final test labels hidden from both policies and use them only for final evaluation.

Count every requested label toward that policy's attempt cap, including failed calculations and retries. Maintain two linked ledgers: actual physical DFT work and policy-counterfactual budget. A cache key includes canonical structure identity plus the full VASP method signature and relevant settings. Identical C/D requests may share one physical calculation; record both policy requests, timestamps, and one shared result. Charge the actual solver cost once to the study ledger, while each policy ledger records the counterfactual cost it would have incurred without sharing. This preserves an equal virtual budget and exposes the physical savings from deduplication. Attribute unique failures/retries to the requesting policy; never erase a failed attempt because another policy later succeeds.

Use separate predeclared gates for surrogate accuracy and evidence sufficiency. Validation failure can trigger adaptation. Missing phase references, weak independent groups, or incomplete hull coverage can require DFT regardless of MLIP performance. Fine-tune only after sufficient independent acquired labels and only under the same trainer, hyperparameter-selection, validation, and promotion rules in both arms. Report pass, fail, insufficient evidence, unsupported, blocked, adaptation attempted, adaptation completed, and adaptation unsuccessful as distinct outcomes. Report held-out accuracy, scientific decisions, attempt counts, retries, actual/counterfactual DFT costs, model-training resources, LLM inference costs, and scientific limitations per cell.

## F. Manuscript Consistency Report
The source of truth inspected is `docs/paper/new-manuscript.zip` containing `MATSIM_AGENTS/main.tex`; `docs/paper/main.tex` is absent, and the older tracked `research/paper/manuscript/main.tex` is not this uploaded revision. Keep the ZIP intact until the author chooses a canonical editable source.

The archived methods section “Materials systems, structures and first-principles settings” states that five systems use VASP and Nb-Ta-O uses a separate QE protocol excluded from VASP numerical totals. It describes VASP AL labels, gives a DFT settings table only for the five systems, and contains fine-tuning results over five AL-corrected cases. Those statements can remain historical descriptions only if clearly separated from the proposed E3. They do not establish a six-system VASP experiment, and QE energies cannot be relabeled as VASP evidence.

Required manuscript changes before presenting E3 as executed:

1. In the staged design subsection, state that all six E3 reference datasets use the same frozen VASP signature; QE evidence and any cross-code comparison are separate and excluded from E3.
2. In the materials/methods subsection and DFT table, add a complete Nb-Ta-O VASP relaxation/reference protocol only after it is frozen and measured. Identify the Nb-Ta-O reference-domain version and its explicit limits.
3. In active-learning and fine-tuning methods, define the 18-cell support/evidence matrix, validation-only adaptation trigger, independent group splits, equal C/D attempted-label budgets, shared calculation-cache accounting, and separately measured evidence trigger.
4. In results, distinguish historical five-system VASP and historical Nb-Ta-O QE results from new E3 outcomes. Replace “five VASP plus one QE” only when new VASP labels and results exist; do not rewrite old values as VASP results.
5. In limitations, restrict Nb-Ta-O claims to stability relative to the qualified declared domain; list missing phase families and unsupported cells.

Proposed methods wording (use only after the protocol is approved): “For E3, all six material systems were evaluated against newly generated VASP reference data under a frozen, system-documented PBE/PAW protocol. Historical QE calculations for Nb-Ta-O were retained as a separate dataset and were not pooled with, used to train against, or compared numerically as VASP labels. Nb-Ta-O stability is reported only relative to the versioned reference domain and convergence gates specified in the manifest.” Change “were evaluated” to future/planned tense until the experiment is completed.

## G. HPC Feasibility and Cost
Repository evidence shows VASP build/launcher recipes for Perlmutter, Aurora, and Frontier, but `docs/hpc-validation.md` marks on-system revalidation as required. This establishes recipe presence, not availability of a licensed VASP executable, access to all required POTCAR datasets, successful system-specific convergence, allocation balance, or permission to submit jobs. Perlmutter documentation records VASP build/launcher details and POTCAR resolution; Aurora and Frontier have their own environment and validation gates. No facility or allocation is selected here.

No defensible VASP E3 cost estimate is available. The repository has no approved six-system VASP workload, representative per-material relaxation timing distribution, verified allocation join, current account balance, billing rate, or approved concurrency. Historical QE Nb-Ta-O timings are not transferable to VASP. The previously noted small VASP Si/SiC qualification timers are not representative of alloy, oxide-domain, phosphorene, or MOF workloads and do not by themselves establish allocation-hours. Do not report invented node-hours, GPU-hours, or currency.

Cost qualification, after explicit approval, should use bounded representative structures and record VASP application walltime, ionic steps, convergence/failure and retry rates, scheduler allocation elapsed, node-hours, physical GPU-hours where applicable, and workflow overhead separately. Stratify by material, cell size, magnetic treatment, and relaxation class. Estimate the full workload bottom-up from the frozen structure manifest and measured distributions, include retry/setup headroom, apply stage-level hard caps, and obtain facility/account approval before expanding. Record billing only from verified facility rules. VASP license, POTCAR access, target-system model support, runtime environment, and resource ceilings are go/no-go gates.

## H. Ordered Implementation and Approval Checklist
1. Approve the six-system VASP-primary E3 scope, the prohibition on QE/VASP energy mixing, intended scientific questions, and the distinction between history and new E3 outputs.
2. Qualify VASP license/executable, PAW dataset access, and DFT launcher on the selected facility; choose a facility only after on-system validation and account approval.
3. Freeze one versioned method signature for each system under a consistent VASP functional/PAW policy, including relaxation, spin, smearing, k-points, and all convergence criteria.
4. Freeze structure sources, generators/seeds, domain manifest, phase IDs, structure hashing, scientific-group definitions, and formula/polymorph/convergence gates; separately qualify Nb-Ta-O's bounded hull domain.
5. Audit each of the 18 model-system cells for model/checkpoint support, environment, observable support, and training/inference feasibility. Mark unsupported cells explicitly.
6. Generate and review VASP reference datasets; preserve QE records separately. Validate provenance, method signatures, composition, hashes, convergence, group-aware train/validation/test partitions, and independent test isolation.
7. Freeze task-specific validation-only accuracy thresholds, evidence-sufficiency gates, metrics, adaptation eligibility, equal attempted-label caps, retry/stop/expand rules, and C/D cache/counterfactual accounting.
8. Obtain author approval for the bounded timing qualification, facility/account, destinations, resource caps, data handling, and stop conditions. No job submission before this approval.
9. Run the approved qualification only; reconcile native VASP timers with scheduler records and revise cost ceilings. Qualification is not scientific E3 output.
10. Obtain separate approval for reference generation and baseline evaluation; report the complete 18-cell status matrix and evidence gaps before any adaptive submissions.
11. Authorize C/D execution only for each supported cell that fails validation; apply matched rules, charge all failures/retries, share only identical physical VASP calculations, and keep the test partition sealed.
12. Complete final independent evaluation and domain qualification, reconcile all accounting, update manuscript language in the chosen canonical source, and report limitations. Do not commit scientific claims for unfinished or blocked cells.

This plan requests no allocation and authorizes no pilot, DFT, inference, or production job.