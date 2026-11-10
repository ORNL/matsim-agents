# E3 status: references, zero-shot evaluation, and first acquisition decisions

Strategies: **LLM-guided adaptation** and **Rule-based adaptation**. Protocol and
code are in [`benchmarks/e3/`](../../../benchmarks/e3/README.md).
No DFT or training was run.

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

These metrics are diagnostic. With one correlated reference group, no cell
gets pass/fail. Relative energies within one near-relaxed trajectory span
little energy, so that metric does not discriminate between models. LiFePO4
has no references, so it has no zero-shot errors. MACE-MP and UMA-omat also
learn MP-style PBE+U energies for Fe oxides, a convention mismatch with the
plain-PBE protocol.

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
