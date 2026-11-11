# Zero-shot Codabench screen of MACE foundation checkpoints

Recorded 2026-11-11. This is a MACE-focused extension to the initial three-model report [`mlip-zero-shot-comparison.md`](mlip-zero-shot-comparison.md). Aggregate metrics are also preserved in [`mace-foundations-codabench-screen.json`](mace-foundations-codabench-screen.json). Raw predictions and logs are in the Git-ignored `runs/codabench/e3-public/` directory.

## Executive result

Twelve checkpoint/head variants from [ACEsuit/mace-foundations](https://github.com/ACEsuit/mace-foundations) were evaluated zero-shot. The provisional user preference is **MACE-OMAT-0 medium** as a balanced general-purpose MACE checkpoint for E3 preparation. On this screen it has the lowest median per-structure force MAE among tested MACE variants (0.113 eV/A), competitive PBE formation-energy MAE (0.106 eV/atom), and no severe formation-energy outlier after excluding the suspect Pd label. It is not the best on every metric: MATPES-PBE-0 has the lowest formation-energy MAE (0.042 eV/atom), while MH-0 OMat-PBE has the lowest pooled/class-macro force MAE. MACE-OMAT-0 medium is a balanced provisional choice, not a statistically established universal winner.

## Protocol and scope

- MACE-Torch 0.3.16; models evaluated at float64 on CPU, one OpenMP thread.
- Frozen checkpoints; no fine-tuning, geometry relaxation, or model selection on these reference labels.
- The same 37 Codabench public structures and same DFT-relaxed elemental reference geometries were used for each checkpoint. All 12 completed 37/37 predictions with no missing elemental reference.
- Formation energy was computed consistently as `(E_model(compound) - sum_i n_i E_model(pure_element_i)) / N`, using each checkpoint and selected head for both compound and pure elements. No fit or correction used benchmark labels.
- The public set has 37 labels/1,314 atoms in eight categories. The 84 private leaderboard labels were not used.
- Seven structures with maximum DFT force above 10 eV/A were excluded from force-summary metrics only: BN (MATS-0002/0004), C (MATS-0006), MoSe2 (MATS-0013), Cu (MATS-0030), Fe (MATS-0033), and WC (MATS-0075). The cutoff is a diagnostic exclusion rule, not a validated physical threshold.
- Formation-energy metrics exclude elemental Pd MATS-0039, whose DFT label is +3.167 eV/atom and is suspect. The remaining 36 structures are used in energy summary columns.
- The Codabench labels are PBE; the VASP 6.6.0 vs E3/E4 6.6.1 version difference is not an exclusion criterion.
- Exact E3-class overlap is only one LiFePO4 structure (MATS-0060) and one P monolayer (MATS-0016). BCC refractory, FCC Cantor, Zn-formate MOF, and Nb-Ta-O have no exact public-set examples.

## Foundation versions and benchmark configurations

The training/domain descriptions below summarize the official MACE foundation-model catalogue. They are model provenance, not measured performance on this benchmark.

| Foundation version | mace-torch 0.3.16 alias | Head | Training-data / functional description | Intended domain |
|---|---|---|---|---|
| MACE-MP-0 small | `small` | default | MPTrj; PBE+U; original MP-0 family | Materials |
| MACE-MP-0 medium | `medium` | default | MPTrj; PBE+U; original MP-0 family | Materials |
| MACE-MP-0 large | `large` | default | MPTrj; PBE+U; original MP-0 family | Materials |
| MACE-MP-0b3 medium | `medium-0b3` | default | MPTrj; PBE+U; second-generation repulsion/high-pressure updates, phonon changes vs earlier variants | Materials |
| MACE-MPA-0 medium | `medium-mpa-0` | default | MPTrj+sAlex; PBE+U; expanded data and improved high-pressure stability | Materials |
| MACE-OMAT-0 small | `small-omat-0` | default | OMat24; PBE+U, VASP 5.4 settings | Inorganic materials |
| MACE-OMAT-0 medium | `medium-omat-0` | default | OMat24; PBE+U, VASP 5.4 settings | Inorganic materials |
| MACE-MH-0 | `mh-0` | `omat_pbe` | Multi-head OMAT/OMOL/OC20/MATPES model; PBE head | Inorganic materials |
| MACE-MH-1 | `mh-1` | `omat_pbe` | Newer multi-head OMAT/OMOL/OC20/MATPES model; PBE head | Inorganic materials |
| MACE-MH-1 | `mh-1` | `mp_pbe_refit_add` | Same MH-1 checkpoint, distinct MP PBE-refit head | Materials Project domain; not interchangeable with OMat head |
| MACE-MATPES-PBE-0 | `mace-matpes-pbe-0` | default | MATPES-PBE; PBE without +U | PBE materials |
| MACE-MATPES-r2SCAN-0 | `mace-matpes-r2scan-0` | default | MATPES-r2SCAN; r2SCAN functional | r2SCAN materials |

The repository lists MP/MPA models under MIT and OMAT/MH/MATPES under ASL. Check the exact release license before redistributing checkpoints. The MACE-OMOL family and MACE-MDP were not benchmarked: this Codabench public set has no suitable molecular/MOF cases, and MDP predicts dipoles/polarizabilities rather than energies/forces.

## Matched performance

Force metrics below are component errors on the 30 force-label-clean structures. `Force median` is the median of per-structure component MAEs; `Force class macro` first averages within each Codabench category, then averages categories. Energy metrics use 36 structures excluding suspect Pd MATS-0039. Runtime is inference plus pure-element reference evaluation, excluding model load.

| Checkpoint/head | Force pooled MAE | Force RMSE | Force median | Force class macro | Formation-energy MAE / median / max | Signed energy bias | Pearson r | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MP-0 small | 1.099 | 5.636 | 0.159 | 1.142 | 0.132 / 0.039 / 0.598 | +0.087 | 0.992 | 49.7 s |
| MP-0 medium | 15.080 | 133.527 | 0.128 | 15.335 | 0.158 / 0.062 / 0.779 | -0.027 | 0.986 | 28.2 s |
| MP-0 large | 60.141 | 443.062 | 0.126 | 61.117 | 0.133 / 0.057 / 1.085 | +0.039 | 0.990 | 59.6 s |
| MP-0b3 medium | 1.769 | 8.790 | 0.121 | 1.827 | 0.179 / 0.116 / 0.964 | -0.097 | 0.986 | 26.9 s |
| MPA-0 medium | 1.572 | 7.071 | 0.157 | 1.616 | 0.122 / 0.089 / 0.461 | -0.025 | 0.995 | 29.5 s |
| OMAT-0 small | 1.058 | 4.594 | 0.125 | 1.094 | 0.105 / 0.072 / 0.364 | -0.040 | 0.996 | 11.6 s |
| **OMAT-0 medium** | **0.981** | **4.401** | **0.113** | **1.016** | 0.106 / 0.040 / 0.412 | -0.028 | 0.996 | 25.5 s |
| MH-0 OMat-PBE | **0.918** | **4.171** | 0.131 | **0.952** | 0.080 / 0.012 / 0.521 | +0.055 | 0.995 | 26.9 s |
| MH-1 OMat-PBE | 1.171 | 5.405 | 0.131 | 1.209 | 0.066 / 0.019 / 0.520 | +0.048 | 0.996 | 40.0 s |
| MH-1 MP-PBE-refit-add | 1.175 | 5.326 | 0.134 | 1.215 | 0.071 / 0.029 / 0.569 | +0.027 | 0.996 | 41.7 s |
| MATPES-PBE-0 | 2.206 | 11.316 | 0.125 | 2.260 | **0.042 / 0.030 / 0.237** | -0.019 | **0.999** | 26.3 s |
| MATPES-r2SCAN-0 | 1.591 | 7.436 | 0.121 | 1.637 | **Not comparable to PBE labels** | N/A | Not ranked | 27.2 s |

Units: force columns eV/A; formation-energy columns eV/atom. Pearson r compares predicted and DFT formation energies on the 36-structure subset, except r2SCAN, which is not ranked against PBE labels.

### Paired bootstrap comparisons for the provisional OMAT-0 medium choice

Each value is `OMAT-0 medium MAE - comparator MAE`; negative favors OMAT-0 medium. Intervals bootstrap structures (20,000 resamples, seed 20261111) and are descriptive only.

| Comparator | Formation-energy difference, n=36 (95% interval) | Force difference, n=30 (95% interval) |
|---|---:|---:|
| MATPES-PBE-0 | +0.0647 [+0.0303, +0.1027] | -0.6586 [-2.0022, +0.0198] |
| MH-0 OMat-PBE | +0.0263 [-0.0141, +0.0663] | +0.0346 [-0.0008, +0.1006] |
| MH-1 OMat-PBE | +0.0404 [+0.0049, +0.0773] | -0.0955 [-0.3209, +0.0249] |
| MH-1 MP-PBE-refit-add | +0.0352 [+0.0004, +0.0726] | -0.0997 [-0.3279, +0.0211] |
| MPA-0 medium | -0.0153 [-0.0531, +0.0200] | -0.3205 [-0.9641, +0.0076] |
| MP-0 small | -0.0253 [-0.0830, +0.0303] | -0.0755 [-0.1975, -0.0075] |

OMAT-0 medium has the lowest force median among these MACE variants. Paired force intervals do not resolve a difference from MH-0/MH-1. MATPES-PBE-0 has significantly lower formation-energy MAE than OMAT-0 medium on this sample; MH-1 OMat-PBE is numerically better on formation-energy MAE but the paired interval against OMAT-0 medium includes zero. Thus OMAT-0 medium is a defensible balanced MACE choice, not the energy-accuracy winner.

### Force outlier interpretation

The seven DFT force outliers listed above generate very large cross-model pooled errors and are omitted from force-summary metrics. Separate from those reference-label outliers, the clean-label Pd structure MATS-0039 produces checkpoint-specific MACE force failures: maximum component prediction errors include about 2,574 eV/A for MP-0 medium and 7,119 eV/A for MP-0 large, while the DFT maximum atomic force is 0.86 eV/A. Such cases are real model/structure failures and remain in the force metrics. Report median/class-macro results and explicit outliers alongside the pooled averages.

### Only two public structures match E3 classes

| Checkpoint/head | LiFePO4 MATS-0060: formation error / force MAE | P monolayer MATS-0016: formation error / force MAE |
|---|---:|---:|
| MP-0 small | +0.356 / 4.413 | -0.076 / 2.547 |
| MP-0 medium | +0.194 / 4.134 | -0.202 / 2.340 |
| MP-0 large | +0.186 / 4.133 | -0.249 / 2.379 |
| MP-0b3 medium | +0.052 / 3.907 | -0.077 / 2.683 |
| MPA-0 medium | +0.210 / 4.117 | -0.079 / 2.475 |
| OMAT-0 small | +0.173 / 4.216 | -0.080 / 2.502 |
| **OMAT-0 medium** | **+0.165 / 4.218** | **-0.058 / 2.438** |
| MH-0 OMat-PBE | +0.310 / 4.204 | -0.084 / 2.460 |
| MH-1 OMat-PBE | +0.309 / 4.104 | -0.072 / 2.417 |
| MH-1 MP-PBE-refit-add | +0.229 / 4.074 | -0.094 / 2.415 |
| MATPES-PBE-0 | -0.031 / 4.126 | +0.020 / 2.443 |
| MATPES-r2SCAN-0 | Not comparable | Not comparable |

These are one-case diagnostics, not class-level estimates. There are no exact Codabench structures for BCC refractory, FCC Cantor, Zn-formate MOF or Nb-Ta-O; this Codabench screen therefore cannot select a MACE checkpoint for those classes. MACE-OMOL, the MACE family to consider for MOF-like structures, was not benchmarked here because no suitable molecule/MOF reference case is present.

## Model-selection interpretation

MACE-OMAT-0 medium is the user's provisional balanced MACE selection: it has the lowest median force error, low formation-energy error and no major energy outlier on the 36 PBE-comparable cases. It does not dominate all metrics. MATPES-PBE-0 is the clear formation-energy MAE leader; MH-0 OMat-PBE has better pooled force metrics; MH-1 OMat-PBE is a close alternative for formation energy. Selection should remain class-specific and provisional until the MACE variants are tested on relevant E3-class references and model-relaxed structures. The E3 DFT validation sets have to be sufficient and group-independent before scientific or discovery claims.

Do not rank MATPES-r2SCAN-0 formation energies against PBE labels: the functional differs. Its force comparison at PBE-relaxed geometries is a cross-functional diagnostic only. The results are zero-shot single points at DFT geometries, not relaxation/MD performance, and pretraining structure-level overlap with this benchmark has not been audited.
