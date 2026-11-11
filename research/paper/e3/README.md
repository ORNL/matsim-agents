# E3 status: references, zero-shot evaluation, and first acquisition decisions

Strategies: **LLM-guided adaptation** and **Rule-based adaptation**. Protocol and
code are in [`benchmarks/e3/`](../../../benchmarks/e3/README.md).
The E3 Round 1 BCC DFT cases have since completed or reached terminal outcomes;
the large E3-MLIP campaign has not started. No campaign DFT is authorized by
this status document.

Mandatory exploration, learning, discovery, statistical, and reporting gates
are specified in
[`scientific-objectives-v1.yaml`](../../../benchmarks/e3/scientific-objectives-v1.yaml).
The MLIP generation and optimization protocol is specified separately in
[`e3-mlip-exploration-plan-v1.md`](e3-mlip-exploration-plan-v1.md).
The selected primary checkpoint is **MACE-OMAT-0 medium**, trained on OMat24;
its MACE-Torch loader alias is `medium-omat-0` and E3 CLI alias is
`mace-omat-0-medium`. Campaign start remains blocked pending the exploration
plan's species, class-feasibility, bounded-pilot, and resource gates. Its MOF
transfer applicability is unvalidated and must be reported as a limitation,
not silently generalized.
E3-MLIP covers all six material classes with a target of 5,000 unique valid
configurations per class per strategy (60,000 total logical optimizations,
before matched-size controls); the approval matrix is
[`mlip-suitability-v1.yaml`](mlip-suitability-v1.yaml). E3-DFT identifiers,
results, execution protocol, and acquisition budgets remain unchanged.
Perlmutter preparation and launch constraints are in the
[combined E3-MLIP/E4 runbook](../perlmutter-e3-e4-execution-runbook.md).

- [`inventory.json`](inventory.json): six-material inventory.
- [`round1-bcc-manifest.json`](round1-bcc-manifest.json) and
  [`round1-bcc.json`](round1-bcc.json): the frozen Round 1 paired BCC decision.
- [`replay-summary.superseded-composition-diversity.json`](replay-summary.superseded-composition-diversity.json):
  references, zero-shot metrics, splits, and superseded v1 Rule-based
  selections.
- Full outputs (pools, splits, MLIP predictions, SOAP inputs, ledgers, LLM
  prompt and reply) are in the untracked `runs/e3/replay-20261108/` and
  `runs/e3/round1-bcc-20261108/`.

## 1. Frozen DFT protocols (E4 production)

| | LiFePO4 (`lifepo4-pbe-verified-v1`) | BCC (`bcc-pbe-350eV-gamma222-nonspin-v1`) |
|---|---|---|
| Code | ALCF VASP 6.6.1 | ALCF VASP 6.6.1 |
| Functional | PBE, no +U | PBE |
| ENCUT / PREC | 520 eV / Accurate | 350 eV / Normal (after Low stage) |
| Smearing | ISMEAR 0, SIGMA 0.05 | ISMEAR 1, SIGMA 0.20 |
| Spin | ISPIN 2; FM start, Fe 5.0, others 0.6 | ISPIN 1 |
| k-points | MP 2x4x8 (112-atom supercell) | Γ 2x2x2 |
| EDIFF / EDIFFG | 1e-6 / -0.01 | 1e-6 / -0.01 |
| PAW | Fe_pv Li_sv O P | Nb_pv Ta_pv V_pv Hf_pv Zr_sv Ti_pv |

## 2. Qualified reference data

| Material | Admitted | Independent groups | Excluded sources (reason) |
|---|---:|---:|---|
| LiFePO4 | **0** | 0 | 3 historical unit-cell records (VASP 6.6.0; k-mesh 4x7x8 ≠ 4x8x8 equivalent); E4 LiFePO4 cases never ran |
| BCC | **54 frames** | **1** | 261 A/B PREC=Low frames; 280 custom-executable frames; 4 restart-test Low frames; 4 E2 cases (ENCUT 345/SIGMA 0.10); running E4 jobs |

The 54 BCC frames are the PREC=Normal ionic steps of the two ALCF A/B pilot
repeats (22 + 32) for Nb22Ta22V21Hf21Zr21Ti21. Every INCAR tag matches, VASP is
6.6.1, and every SCF converged (max 36 of NELM 170). Energies are in eV and
forces in eV/Å. The relaxations are ionically incomplete, so these are
per-frame labels, not relaxed endpoints. All 54 frames come from one starting
structure, which makes them **one** independent group. The group is assigned
to validation; test is empty.

## 3. Splits and overlap

| | LiFePO4 | BCC |
|---|---:|---:|
| Acquisition / validation / test pool structures | 84 / 18 / 18 | 84 / 18 / 18 |
| `make_split` checks | pass (near-duplicate warning: shared lattice) | pass |
| Reference structures in pool | 0 | 0 |
| Reference composition in pool | 0 | 0 (E4-BCC-1802–1804 removed) |
| E4 first-batch IDs in pool | 0 | 0 |

## 4. Zero-shot MLIPs

| Model | Configuration | BCC force MAE / RMSE (eV/Å) | BCC relative-energy MAE (meV/atom) | Status |
|---|---|---|---|---|
| UMA | `uma-s-1p1`, `omat` head, CPU, `.venv-uma` | 0.053 / 0.067 | 0.013 | insufficient validation evidence |
| MACE | MACE-MP-0 small, float64, CPU, `.venv-mace` | 0.157 / 0.205 | 0.040 | insufficient validation evidence |
| HydraGNN | frozen epoch-97 + branch MLP | not evaluated | — | XPU-only; strict-load rerun (job 9052180) hit the 840 s timeout (exit 124) with no results; units unverified |

The initial 54-frame metrics above use one correlated reference group; no cell
gets pass/fail. Relative energies within one near-relaxed trajectory span
little energy, so that metric does not discriminate between models. LiFePO4
has no references, so it has no zero-shot errors. MACE-MP and UMA-omat also
learn MP-style PBE+U energies for Fe oxides, a convention mismatch with the
plain-PBE protocol.

### Preliminary BCC model screen (2026-11-10)

The saved screening run contains 68 frames from four 128-atom BCC trajectories:
30 frames in two held-out E4 references (E4-BCC-0001 test and E4-BCC-0561
validation group 1) and 38 frames in the E3-R1-BCC-0394/0582 acquisition cases.
The metrics below are frame-weighted diagnostics, not independent samples:

| Model | Held-out force MAE / RMSE (eV/A) | Held-out within-trajectory relative-energy MAE (meV/atom) | Acquired force MAE / RMSE (eV/A) |
|---|---:|---:|---:|
| UMA `uma-s-1p1-omat` | 0.0507 / 0.0646 | 0.0316 | 0.0471 / 0.0595 |
| MACE `mace-mp-0-small` | 0.1392 / 0.1791 | 0.0360 | 0.1297 / 0.1705 |

UMA is the preliminary BCC force-screen leader. The result is not a final model
selection: there is only one held-out validation group (two are required),
frames within each relaxation are correlated, and within-trajectory relative
energies span a narrow range. Two held-out groups do not support a reliable
inferential confidence interval. The selection recommendation is to prioritize
UMA and retain frozen MACE as a committee comparator, pending explicit approval.
These are not materials-discovery results and provide no independent support
for a candidate's scientific utility. Source artifacts remain in the protected
sibling checkout at `../../../../matsim-agents/runs/e3/zero-shot-qualified-bcc-20261110/`;
`metrics.json` SHA-256 is
`2635930ececd5d112a1b6d62190f98a6fb3bef46813c9ca8146f0daa31f3335c` and
`frames-manifest.json` SHA-256 is
`0a23b2e29d7bfc5a65ad9071783467cf9f6b2001bd6f5ecef23a93c11f27f48a`.

### Codabench Reference Screen

The Codabench reference package contains 37 public labeled structures across
eight benchmark categories; the public split matches E3 classes only for one
LiFePO4 structure and one P monolayer. The other four E3 classes have no exact
public match. All 37 public structures were evaluated by frozen MACE-MP-0-small,
UMA-s-1p1-OMAT, and HydraGNN OMat24 head 7; all three completed 37/37 without
prediction failures. The 84 private leaderboard labels are excluded. VASP
minor-version differences are not a reason to exclude these labels.

An expanded screen subsequently evaluated 12 MACE foundation checkpoint/head
variants on the same public structures. The version-by-version model details,
matched metrics, paired comparisons, PBE/r2SCAN caveat, outlier analysis and
provisional OMAT-0 medium selection are in
[`mace-foundations-codabench-screen.md`](mace-foundations-codabench-screen.md)
and [`mace-foundations-codabench-screen.json`](mace-foundations-codabench-screen.json).

| Model | Public force-component MAE (eV/A) | Formation-energy MAE (eV/atom) | Inference + element-reference time (s, excluding model load) |
|---|---:|---:|---:|
| MACE-MP-0-small | 60.423 | 0.349 | 49.67 |
| UMA-s-1p1-OMAT | 60.351 | 0.677 | 433.79 |
| HydraGNN OMat24 (head 7, restored checkpoint) | 60.838 | 0.703 | 1.97 |

The previous HydraGNN result is superseded: its checkpoint loader matched no
trained tensors, so its reported energies and forces were not model predictions.
The corrected head-7 run matched all 200/200 model tensors (0 missing, 0 unused;
one wrapper prefix removed). Its formation-energy MAE is 0.3657 eV/atom over 36
structures excluding suspect Pd MATS-0039 (RMSE 0.4974; correlation 0.9618).
On the 30 structures with clean DFT force labels, force-component MAE is 1.642
eV/A (RMSE 7.431; median per-structure MAE 0.261). The all-37 force aggregate
remains dominated by bad DFT labels. Corrected provenance and per-structure
metrics are in
[`codabench-hydragnn-omat24-restored.json`](codabench-hydragnn-omat24-restored.json);
the comparison, label-quality exclusions, and model recommendation are in
[`mlip-zero-shot-comparison.md`](mlip-zero-shot-comparison.md).

These pooled force errors are dominated by shared outlier DFT force labels:
MATS-0030 Cu (max component 12,436 eV/A), MATS-0033 Fe (972 eV/A), and
MATS-0006 C (120 eV/A). Treat the pooled force metrics as a data-quality warning,
not a model ranking; inspect per-class and per-structure results.

The exact E3-class public cases are one LiFePO4 structure (MATS-0060) and one
P monolayer (MATS-0016). Their force-component MAE / absolute formation-energy
error are:

| Public case | MACE-MP-0-small | UMA-s-1p1-OMAT | HydraGNN OMat24 head 7 |
|---|---:|---:|---:|
| LiFePO4, MATS-0060 (eV/A; eV/atom) | 4.413; 0.356 | 4.237; 0.025 | 4.316; 0.265 |
| P monolayer, MATS-0016 (eV/A; eV/atom) | 2.547; 0.076 | 2.488; 0.010 | 2.357; 0.102 |

Each is a single structure, not an independent-group estimate or a class-level
ranking. HydraGNN's head predicts composition-baseline residual energies; its
formation energy, like MACE and UMA, is computed with same-model pure-element
subtraction:
$(E_{model}(compound)-\sum_i n_i E_{model}(element_i))/N$. The composition-linear
baseline cancels for HydraGNN, and no benchmark labels are used to fit a correction. The
results and logs are in [`runs/codabench/e3-public/`](../../../runs/codabench/e3-public/);
the 16-branch checkpoint is pinned to head 7 (`OMat24`) with no routing MLP.
These public screens are useful for model suitability, but not discovery or
E3 validation claims without sufficient independent groups.

## 5. Round 1 paired BCC decision (frozen; no DFT requested)

Frozen manifest: [`round1-bcc-manifest.json`](round1-bcc-manifest.json)
(SHA-256 `b2089e8f…`); full record in [`round1-bcc.json`](round1-bcc.json).
Both strategies received the same 84-structure acquisition pool, MACE/UMA
predictions, committee uncertainty, SOAP novelty, and five shared evidence
records, each in a separate ledger.

**LLM:** HV-Luna was unavailable. No direct HV-Luna endpoint or credential
exists here, and MAG lists no HV-Luna model; MAG's Luna routes need the
OpenAI/Amazon buckets that project `pipit` lacks. The documented fallback is
frozen: `gemini-4-argon@google` via MAG
(`https://mag.genesis.americansciencecloud.org/v1`), temperature 0,
max_tokens 8192, prompt template `e3-llm-guided-round-v2`. The response
confirmed the model name. MAG reported no API cost.

**Rule-based adaptation (v2, frozen before selection):**
- Uncertainty: `score_ensemble` force disagreement of {MACE-MP-0 small,
  UMA-s-1p1}.
- Eligibility: top 25% by uncertainty (21 of 84).
- Value: u/u_max + 0.5 × SOAP novelty/novelty_max, where novelty is the
  existing `soap_novelty` to the reference structures plus earlier picks.
- Near-duplicate exclusion: novelty ≤ 0.05 × the median nearest-neighbour
  novelty (tolerance 0.00195).
- Ties broken by candidate ID.

The composition-only v1 selections are kept as
[`replay-summary.superseded-composition-diversity.json`](replay-summary.superseded-composition-diversity.json).

| Strategy | Candidate | Composition | Committee disagreement (eV/Å), rank /84 | SOAP novelty | Rationale (abridged) |
|---|---|---|---|---|---|
| Rule-based adaptation | **E4-BCC-1077** | Ta46 Zr40 Ti32 Hf10 | 0.0973, 1 | 0.212 | highest disagreement; value 1.500 |
| Rule-based adaptation | E4-BCC-0394 | Nb46 Hf40 Ti32 Ta10 | 0.0957, 2 | 0.210 | value 1.484 |
| LLM-guided adaptation | **E4-BCC-1077** | Ta46 Zr40 Ti32 Hf10 | 0.0973, 1 | 0.212 | most stringent test of inter-model disagreement in a V- and Nb-free subsystem |
| LLM-guided adaptation | E4-BCC-0582 | Nb37 Ti36 V35 Hf20 | 0.0924, 4 | 0.177 | chemically complementary (Nb/V-rich, Ta/Zr-free); probes whether V/Hf size mismatch drives the MACE/UMA force-error gap |

The strategies differ in their second pick. Rule-based adaptation takes the
next most uncertain, structurally novel candidate. LLM-guided adaptation
trades a little uncertainty for complementary chemistry. Every selection
cites ledger evidence (4–5 records each). All four passed admissibility (pool
membership, held-out exclusion, citations, required fields, numerical-claim
check); no proposals were rejected.

**Overlap:** E4-BCC-1077 was chosen by both strategies, giving one shared
physical calculation. That makes 4 logical selections and 3 unique physical
calculations.

**State and budget:** all four selections are `frozen_awaiting_dft`. None has
been dispatched, so no logical attempt is consumed: 8 of 8 remain per
strategy.

**Decision-related resources (native units):**

| Item | Resource used |
|---|---|
| MLIP inference (84 candidates, CPU) | MACE 67.6 s, UMA 266.1 s |
| SOAP descriptors | 72.0 s |
| Uncertainty scoring | 0.02 s |
| LLM-guided adaptation | 19,876 prompt + 1,085 completion tokens, 26.7 s |
| DFT | 0 node-hours |

## 6. Proposed validation-only accuracy criteria

| Material | Criteria |
|---|---|
| BCC | force component MAE ≤ 0.10 eV/Å; group-relative energy MAE ≤ 5 meV/atom |
| LiFePO4 | force component MAE ≤ 0.05 eV/Å; antisite-pair energy (vs pristine) MAE ≤ 0.10 eV; Spearman ranking ≥ 0.8 |

Fine-tuning requires all of: a validation deficit; ionically converged
acquired labels from at least 2 independent groups disjoint from
validation/test; and coverage of the validation domain. An evidence-only arm
keeps the pretrained model fixed, separating the value of acquisition from the
value of adaptation.

## 7. Reference roles and Round 1 DFT plan

**Frozen reference roles** ([`reference-roles-bcc-v1.json`](reference-roles-bcc-v1.json),
content SHA-256 `aaac440b…`). These were assigned by calculation ID before any
E4 result was read.

| E4 calculation | E3 role | Composition |
|---|---|---|
| E4-BCC-0001 | test | Hf32 Nb32 Ta32 V32 |
| E4-BCC-0561 | validation group 1 | Hf32 Nb32 Ti32 V32 |
| E4-BCC-1121 | validation group 2 | Hf32 Ti32 V32 Zr32 |

All three are excluded from acquisition and fine-tuning. A structure that
fails to qualify is reported as a shortfall, never replaced. The 54 A/B-pilot
frames remain preliminary.

**Note:** validation group 1 shares the Nb–V–Hf–Ti element system with the
LLM-guided selection E4-BCC-0582 (Hf20 Nb37 Ti36 V35). The compositions and E3
groups differ, and the selections stay frozen. Any later adaptation result on
that validation group must report this subsystem proximity.

**Staged Round 1 batch** (`E3-round1-bcc-01`; see
[`deployments/aurora/jobs/e3-round1-bcc/`](../../../deployments/aurora/jobs/e3-round1-bcc/README.md)):
- E3-R1-BCC-1077, -0394, and -0582.
- Each runs one node for at most 24 h on HydraGNN in the `capacity` queue
  (72 node-hours total).
- They use the E4 production protocol and runner unchanged.
- Dry-run validation passed. Nothing has been submitted.
- MACE and UMA initial predictions for the three structures are retained in
  [`round1-bcc-initial-predictions.json`](round1-bcc-initial-predictions.json).

**Before Round 2:** run the three Round 1 calculations to qualified Normal-stage
convergence. Round 2 decisions use only acquired evidence; validation and test
labels are never shown to either strategy. LiFePO4 and HydraGNN limitations
are unchanged.

## Manuscript terminology

The local `research/paper/manuscript/main.tex` (last changed 2026-10-30,
commit `79d250c`) still uses "policy (C)/(D)" (10 occurrences). The revised
manuscript with LLM-guided adaptation and Rule-based adaptation is not in
this repository, so it is the authoritative source.

Reconciliation plan, after authorization:
1. Import the revised source on a branch.
2. Diff it against `main.tex`.
3. Keep E1/E2 identifiers.
4. Confirm only the two manuscript strategy names remain for E3.
5. Commit.

The local copy has not been edited.
