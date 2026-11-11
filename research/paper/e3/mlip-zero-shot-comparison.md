# Zero-shot comparison of three pretrained interatomic potentials (MACE, UMA, HydraGNN) on the MATSIM-AGENTS Codabench reference set

Recorded 2026-11-10. Raw outputs: `runs/codabench/e3-public/` (not tracked);
aggregate summary: [`codabench-zero-shot-comparison.json`](codabench-zero-shot-comparison.json);
evaluation script: [`benchmarks/e3/codabench_public_benchmark.py`](../../../benchmarks/e3/codabench_public_benchmark.py).
The expanded 12-checkpoint MACE foundation screen is documented separately in
[`mace-foundations-codabench-screen.md`](mace-foundations-codabench-screen.md),
with structured results in
[`mace-foundations-codabench-screen.json`](mace-foundations-codabench-screen.json).

## 1. Purpose

Before launching the E3-MLIP campaign (adaptive crystallographic exploration driven by pretrained machine-learned interatomic potentials, MLIPs), the three candidate MLIPs were compared in zero-shot mode, i.e. with frozen pretrained weights and no fine-tuning, against independent DFT labels. The aim was to choose a primary model and a comparison model.

## 2. Models evaluated

| Label | Model | Checkpoint / configuration | Precision / device | How energies are produced |
|---|---|---|---|---|
| MACE | MACE-MP-0, "small" variant | Materials Project MACE foundation model (2023-12-10 release, 128-channel L=0, epoch 249), loaded via `mace.calculators.mace_mp(model="small")` | float64, CPU | Total cell energy (eV) and forces (eV/Å) from the ASE calculator |
| UMA | Meta FAIR UMA, small, v1.1 | `uma-s-1p1` via fairchem `pretrained_mlip.get_predict_unit`, task head `omat` (inorganic bulk materials, trained on OMat24) | CPU | Total cell energy (eV) and forces (eV/Å) from `FAIRChemCalculator` |
| HydraGNN | HydraGNN multi-dataset foundation model (SC26), 16 output heads | Checkpoint `multidataset_hpo-BEST6-fp64_epoch_97.pk` (SHA-256 8532b032…), config SHA-256 7c5962f0…; only output head 7 (OMat24 branch) used; the branch-weighting MLP that normally mixes heads was disabled | fp64, CPU | Head-7 energy and forces. The head was trained on OMat24 total energies minus a composition-only linear-regression baseline, so it predicts a residual energy, not a total energy |

Runtime environment: Aurora (ALCF) login node, CPU only, one OpenMP thread, each model in its own Python environment. Each model was loaded once and reused for every structure.

## 3. The independent test data: the Codabench reference set

### 3.1 What it is
The MATSIM-AGENTS Materials Discovery Challenge is a multi-task atomistic machine-learning benchmark hosted on Codabench (formation energy, forces, and other tasks). Its DFT reference labels were computed separately from all E3/E4 work, in a private repository (`matsim-agents-dft-reference`) whose DFT recipe is intentionally hidden from participants. None of these structures or labels were used to train, select, or tune any of the three models here, so this is an independent zero-shot test of the models as deployed in MATSIM-AGENTS. It does not rule out that public datasets used for pretraining (OMat24, MPtrj) contain similar structures.

### 3.2 How it was built
1. **Structure generation:** a design set of 159 structures was generated with `public_data/generate_structures.py` across material categories (2D monolayers, catalysis slabs, critical minerals, intermetallics, MAX phases, nuclear materials, perovskites, thermoelectrics, plus high-entropy alloys and ceramics). Ten high-entropy-ceramic structures with placeholder atoms were skipped.
2. **Anonymization:** structures were renamed MATS-0001 … MATS-0159; the mapping to real names is kept private.
3. **DFT calculations:** VASP 6.6.0 with PBE and PAW potentials (potpaw_PBE.64), using per-category recipes fixed by a convergence sweep:

| Category | ENCUT (eV) | k-spacing (Å⁻¹) | ALGO |
|---|---|---|---|
| 2D monolayer | 520 | 0.18 | Normal |
| Catalysis | 520 | 0.18 | Normal |
| Critical mineral | 520 | 0.16 | Normal |
| Intermetallic | 520 | 0.12 | Fast |
| MAX phase | 520 | 0.12 | Fast |
| Nuclear material | 600 | 0.144 | Normal |
| Perovskite | 520 | 0.16 | Normal |
| Thermoelectric | 520 | 0.16 | Normal |

4. **Elemental references:** 46 pure-element calculations at ENCUT 520 eV with ALGO=Normal.
   - Metals: Methfessel–Paxton smearing (ISMEAR=1, SIGMA=0.1).
   - Gases and insulators: Gaussian smearing (ISMEAR=0, SIGMA=0.01).
   - Fe, Co, Ni, Cr, Mn: spin-polarized with an initial magnetic moment.
   - Oxygen: an isolated O₂ molecule in a 15 × 16 × 17 Å box, spin-polarized and constrained to the triplet state (NUPDOWN=2).
   - The relaxed geometry of each element (CONTCAR) is stored.
5. **Labels for each structure:**
   - The DFT-relaxed geometry.
   - DFT forces at that geometry.
   - A DFT formation energy per atom:
     ΔH_f/N = E_compound/N − Σ_i x_i · E_ref,i
     where x_i is the atomic fraction of element i and E_ref,i is the DFT energy per atom of element i's reference.
6. **Release scope and split:** 121 structures completed DFT labelling. They were split, by formula and with random seed 42, into 37 public and 84 private leaderboard structures. Formulas that occur only once go to the private set.

### 3.3 What was used here
- **Public structures only:** the 37 labelled public structures (public-ID file SHA-256 166b7460…), 1,314 atoms in total.
- **Private set excluded:** none of the 84 private leaderboard labels were used for metrics, outputs or any correction.
- **VASP version:** the labels come from VASP 6.6.0, whereas the E3/E4 production protocol uses 6.6.1. As instructed, this difference was not treated as a reason to exclude labels.
- **Category composition of the public set:** 2D monolayer 7, catalysis 4, critical mineral 9, intermetallic 6, MAX phase 2, nuclear material 2, perovskite 5, thermoelectric 2.
- **Overlap with the six E3 material classes:** only one exact LiFePO₄ structure (MATS-0060) and one phosphorus monolayer (MATS-0016). There are no BCC refractory alloy, FCC Cantor alloy, Zn(HCOO)₂ MOF, or Nb–Ta–O oxide structures.

## 4. How predictions were made and scored

1. **No relaxation:** each structure was evaluated once at its DFT-relaxed geometry; the MLIP did not relax it. This tests the energy surface at the DFT minimum, not each model's own minimum.
2. **Forces:** errors are per Cartesian component, (F_model − F_DFT) in eV/Å.
3. **Formation energy, applied identically to all three models:**
   E_f,model = [E_model(compound) − Σ_i n_i · E_model(element_i)/N_i] / N
   - E_model(element_i) is the same model's single-point energy on the DFT-relaxed elemental reference geometry (the same CONTCARs used for the DFT labels), and N_i is the number of atoms in that reference cell.
   - The compound and the elemental references use the same model and the same output head.
   - No offset or correction is fitted to benchmark labels; the DFT labels are used only as the scoring target.
   - For HydraGNN, the composition-linear baseline cancels exactly in this difference, so in principle no baseline coefficients are needed.
4. **Error definition:** formation-energy error = E_f,model − E_f,DFT, in eV/atom.
5. **Statistics:**
   - Mean absolute error (MAE), median absolute error, and root-mean-square error (RMSE).
   - Class-balanced ("macro") averages: average within each category first, then across categories, so the largest category doesn't dominate.
   - Paired bootstrap of per-structure differences between models (20,000 resamples over structures, seed 20261110), reported as the mean difference with a 95% confidence interval.
   - Per-structure win counts.

## 5. Data-quality findings that change the analysis

### 5.1 Physically implausible DFT force labels
The largest DFT force on any atom, per structure, splits cleanly into two groups: 30 structures at ≤ 7.43 eV/Å and 7 structures at 39–12,435 eV/Å. Forces of that size are not physical for relaxed structures. Force metrics are therefore reported on the 30 clean structures; the 7 excluded are:

| ID | Category | Formula | Largest DFT force (eV/Å) |
|---|---|---|---|
| MATS-0002 | 2D monolayer | BN | 132.7 |
| MATS-0004 | 2D monolayer | BN | 130.9 |
| MATS-0006 | 2D monolayer | C | 138.2 |
| MATS-0013 | 2D monolayer | MoSe₂ | 109.6 |
| MATS-0030 | Catalysis | Cu | 12,435.5 |
| MATS-0033 | Catalysis | Fe | 1,156.7 |
| MATS-0075 | Critical mineral | WC | 39.2 |

All three models produce nearly identical force errors on these 7 structures (for example, 1,057 eV/Å per component for Cu, all three models). That is consistent with the labels, not the models, being wrong. Including them makes the pooled force MAE about 60 eV/Å for every model.

### 5.2 Suspect Pd formation-energy label (MATS-0039, catalysis, elemental Pd)
- The DFT label is +3.167 eV/atom for an elemental Pd structure.
- MACE predicts +11.33 and UMA +13.31 eV/atom (errors of +8.2 and +10.1).
- Either the label, the structure, or the Pd elemental reference is problematic. This case is excluded from the robust energy summaries and reported separately.

### 5.3 HydraGNN checkpoint-loading failure and correction
- The initial HydraGNN results are invalid: the frozen inference loader did not normalize the checkpoint's nested `module.model.` keys, so none of the 200 trained tensors matched the constructed model. The near-zero formation energies were predictions from effectively random initialization, not a HydraGNN energy-scale or normalization problem.
- The loader now strips recognized wrapper prefixes and fails closed if no model tensors match. The corrected public run reports 200/200 tensors matched, 0 missing, 0 unused, and one wrapper prefix removed. Head 7 (OMat24) and the frozen checkpoint are unchanged.
- Corrected formation-energy results: all 37 MAE 0.703 eV/atom and RMSE 2.168; excluding suspect Pd MATS-0039, MAE 0.3657, median absolute error 0.2581, RMSE 0.4974, maximum absolute error 1.7294, signed bias +0.0797, and Pearson correlation +0.9618.
- Corrected force-component results on the 30 structures with DFT maximum atomic force ≤10 eV/Å: MAE 1.642 eV/Å, RMSE 7.431, median per-structure MAE 0.261, and class-macro MAE 1.683. The all-37 force metrics remain dominated by the seven bad DFT labels.
- Corrected predictions still use same-head pure-element subtraction; no benchmark labels are fitted. The composition-linear training baseline cancels in this difference.
- The corrected checkpoint report and full per-structure results are in [`codabench-hydragnn-omat24-restored.json`](codabench-hydragnn-omat24-restored.json). Initial raw output files are retained for provenance but must not be used as model results.

### 5.4 UMA's systematic oxide shift
- On oxides without a strongly correlated transition metal, UMA's error is a near-constant −0.52 to −0.55 eV per oxygen atom: CeO₂, La₂O₃, MgAl₂O₄, TiO₂, UO₂, ZrO₂, BaTiO₃ (two structures), SrTiO₃. Per atom this is about −0.30 to −0.37 eV/atom.
- The Co and Fe oxides behave differently: LiCoO₂ +0.40, LaFeO₃ +0.39 and LiFePO₄ +0.04 eV per oxygen atom.
- This is consistent with a convention difference between UMA's training data (OMat24, a PBE/PBE+U convention) and the plain-PBE Codabench references (O₂ molecule, no Hubbard U), rather than random error.
- MACE shows no such shift (oxide median error 0.027 eV/atom).

## 6. Results

### 6.1 Raw aggregates, all 37 public structures (no exclusions)

| Model | Force MAE (eV/Å) | Force RMSE (eV/Å) | E_f MAE (eV/atom) | E_f RMSE (eV/atom) | CPU time for 37 structures + elemental refs (s, excl. model load) |
|---|---|---|---|---|---|
| MACE | 60.42 | 793.43 | 0.349 | 1.360 | 49.7 |
| UMA | 60.35 | 793.43 | 0.677 | 2.415 | 433.8 |
| HydraGNN (head 7, restored checkpoint) | 60.84 | 793.42 | 0.703 | 2.168 | 2.0 |

The force aggregates are dominated by the bad labels in §5.1.

### 6.2 Forces on the 30 structures with physical DFT forces

| Model | Pooled component MAE (eV/Å) | Median of per-structure MAE (eV/Å) | Class-balanced MAE (eV/Å) |
|---|---|---|---|
| MACE | 1.099 | 0.159 | 1.142 |
| UMA | 1.019 | **0.131** | 1.058 |
| HydraGNN (head 7, restored checkpoint) | 1.642 | 0.261 | 1.683 |

HydraGNN's previous force results were also invalid because its trained checkpoint was not loaded. The corrected values above replace them.

### 6.3 Formation energies, robust splits (absolute error, eV/atom; error = model − DFT)

| Subset | n | MACE MAE / median / max | UMA MAE / median / max | HydraGNN MAE / median / max |
|---|---|---|---|---|
| All, excluding Pd MATS-0039 | 36 | **0.132** / 0.039 / 0.60 | 0.414 / **0.033** / 10.57 | 0.366 / 0.258 / 1.729 |
| Non-oxides, excluding Pd | 24 | **0.127** / 0.048 / 0.52 | 0.477 / **0.012** / 10.57 | 0.320 / 0.152 / 1.729 |
| Oxides | 12 | **0.141** / **0.027** / 0.60 | 0.289 / 0.317 / 0.37 | 0.457 / 0.542 / 0.639 |

Per-structure wins (lower error):
- All except Pd: UMA beats MACE on 21/36. HydraGNN pairwise win counts are not reported here; the earlier comparisons used the unloaded checkpoint and have been withdrawn.
- Non-oxides: UMA beats MACE on 18/24.
- Oxides: UMA beats MACE on 3/12 (the −0.53 eV per oxygen atom shift).

Correlation between predicted and DFT formation energies (36 structures, Pd excluded): MACE +0.992, UMA +0.791 (lowered by the WS₂ failure), HydraGNN +0.962.

UMA's single catastrophic failure is WS₂ monolayer MATS-0019 (error +10.57 eV/atom). The W and S references are not at fault: WC, MoS₂ and CoS₂ errors are small for UMA. MACE's largest error excluding Pd is 0.60 eV/atom (LiCoO₂).

### 6.4 Paired bootstrap differences (A − B; negative favours A; mean and 95% CI over structures)

| Comparison | Formation energy, 37 structures | Formation energy, 30 clean-force structures | Force MAE, 30 clean structures |
|---|---|---|---|
| UMA − MACE | +0.329 [−0.032, +0.946]; UMA better on 21/37 | +0.442 [+0.006, +1.200]; UMA better on 16/30 | **−0.052 [−0.134, −0.004]**; UMA better on 15/30 |
| HydraGNN comparisons | Withdrawn; prior bootstrap used unloaded checkpoint | Withdrawn | Withdrawn |

The previously reported HydraGNN paired comparisons are invalid because they used an unloaded checkpoint. They must be recomputed from per-structure corrected predictions before comparative claims are made. The remaining clearly resolved results are:
- UMA has lower force errors than MACE; the interval excludes zero, but only just.
- The old HydraGNN comparison was based on invalid predictions; corrected paired comparisons have not yet been calculated.
- MACE beats UMA on formation-energy MAE on the 30-structure subset; the interval barely excludes zero.

### 6.5 Per category (formation-energy MAE over all structures in the category | force MAE over clean-force structures; eV/atom | eV/Å)

| Category | n (clean forces) | MACE | UMA | HydraGNN (restored) |
|---|---|---|---|---|
| 2D monolayer | 7 (3) | 0.254 / 0.924 | 1.591 / 0.895 | 0.496 / 0.884 |
| Catalysis | 4 (2) | 2.063 / 6.606 | 2.547 / 6.035 | 3.317 / 10.529 |
| Critical mineral | 9 (8) | 0.174 / 0.761 | 0.189 / 0.718 | 0.591 / 0.821 |
| Intermetallic | 6 (6) | 0.066 / 0.024 | **0.006** / 0.029 | 0.128 / 0.070 |
| MAX phase | 2 (2) | 0.081 / 0.435 | **0.005** / 0.393 | 0.220 / 0.499 |
| Nuclear material | 2 (2) | **0.065** / 0.125 | 0.361 / 0.160 | 0.323 / 0.339 |
| Perovskite | 5 (5) | **0.117** / 0.158 | 0.248 / 0.172 | 0.385 / 0.215 |
| Thermoelectric | 2 (2) | 0.019 / 0.100 | **0.012** / 0.063 | 0.085 / 0.106 |

UMA's 2D-monolayer and catalysis formation-energy numbers are dominated by WS₂ and Pd. Its nuclear-material and perovskite numbers reflect the oxide shift. HydraGNN entries are corrected; the catalysis energy error includes suspect Pd and should be read with §5.2.

### 6.6 The two structures that match E3 material classes (single structures, not class estimates)

| Structure | Model | Formation-energy error (eV/atom) | Force MAE (eV/Å) |
|---|---|---|---|
| LiFePO₄, MATS-0060 (28 atoms; DFT label −0.164 eV/atom; largest DFT force 2.80 eV/Å) | MACE | +0.356 | 4.413 |
| | UMA | **+0.025** | 4.237 |
| | HydraGNN (restored) | +0.265 | 4.316 |
| P monolayer, MATS-0016 (4 atoms; DFT label +0.332 eV/atom; largest DFT force 7.43 eV/Å) | MACE | −0.076 | 2.547 |
| | UMA | **−0.010** | 2.488 |
| | HydraGNN (restored) | −0.102 | 2.357 |

### 6.7 Per-structure results (all 37; error = model − DFT; forces are per-component MAE; † = force label excluded; ‡ = suspect energy label)

HydraGNN values below are overwritten with predictions from the corrected checkpoint-loaded run. Full precision and checkpoint provenance are in [`codabench-hydragnn-omat24-restored.json`](codabench-hydragnn-omat24-restored.json); the old raw output files remain only as provenance.

| ID | Category | Formula | N | DFT E_f (eV/atom) | Largest DFT force (eV/Å) | MACE ΔE_f | UMA ΔE_f | HydraGNN ΔE_f | MACE force MAE | UMA force MAE | HydraGNN force MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MATS-0002† | 2D monolayer | BN | 31 | −0.935 | 133 | +0.515 | −0.224 | +0.479 | 57.8 | 57.8 | 57.940 |
| MATS-0004† | 2D monolayer | BN | 32 | −0.493 | 131 | +0.432 | −0.238 | −0.464 | 59.6 | 59.6 | 59.616 |
| MATS-0006† | 2D monolayer | C | 32 | +1.810 | 138 | +0.154 | −0.031 | −1.729 | 62.9 | 62.9 | 62.956 |
| MATS-0011 | 2D monolayer | MoS₂ | 48 | −1.771 | 0.743 | +0.046 | +0.023 | +0.496 | 0.172 | 0.143 | 0.241 |
| MATS-0013† | 2D monolayer | MoSe₂ | 47 | +0.549 | 110 | −0.156 | +0.040 | −0.195 | 26.1 | 26.0 | 26.026 |
| MATS-0016 | 2D monolayer | P | 4 | +0.332 | 7.43 | −0.076 | −0.010 | −0.102 | 2.55 | 2.49 | 2.357 |
| MATS-0019 | 2D monolayer | WS₂ | 48 | +1.614 | 0.24 | +0.396 | +10.568 | −0.005 | 0.0534 | 0.0535 | 0.053 |
| MATS-0030† | Catalysis | Cu | 64 | +0.152 | 12,435 | +0.006 | +0.007 | −0.252 | 1,060 | 1,060 | 1057.438 |
| MATS-0033† | Catalysis | Fe | 31 | +0.485 | 1,157 | +0.032 | +0.035 | −0.034 | 116 | 116 | 116.340 |
| MATS-0035 | Catalysis | Ni | 63 | +0.348 | 0.571 | +0.049 | −0.004 | −0.137 | 0.162 | 0.111 | 0.111 |
| MATS-0039‡ | Catalysis | Pd | 63 | +3.167 | 0.86 | +8.164 | +10.140 | +12.845 | 13.0 | 12.0 | 20.948 |
| MATS-0047 | Critical mineral | CeO₂ | 95 | −3.483 | 2.06 | −0.002 | −0.363 | +0.576 | 0.120 | 0.121 | 0.287 |
| MATS-0051 | Critical mineral | CoS₂ | 11 | −1.070 | 0.793 | −0.019 | −0.017 | +0.999 | 0.258 | 0.206 | 0.338 |
| MATS-0055 | Critical mineral | La₂O₃ | 5 | −3.463 | 0.284 | +0.020 | −0.313 | +0.574 | 0.0632 | 0.0540 | 0.098 |
| MATS-0058 | Critical mineral | LiCoO₂ | 12 | −1.483 | 2.83 | +0.598 | +0.200 | +0.622 | 0.453 | 0.442 | 0.498 |
| MATS-0060 | Critical mineral | LiFePO₄ | 28 | −0.164 | 2.80 | +0.356 | +0.025 | +0.265 | 4.41 | 4.24 | 4.316 |
| MATS-0065 | Critical mineral | MgAl₂O₄ | 56 | −2.990 | 0.907 | −0.001 | −0.303 | +0.335 | 0.295 | 0.300 | 0.372 |
| MATS-0068 | Critical mineral | SiC | 64 | −0.219 | 2.42 | −0.041 | −0.010 | −0.585 | 0.200 | 0.143 | 0.364 |
| MATS-0072 | Critical mineral | TiO₂ | 71 | −2.869 | 2.09 | −0.034 | −0.357 | +0.563 | 0.290 | 0.246 | 0.290 |
| MATS-0075† | Critical mineral | WC | 53 | +2.285 | 39.2 | +0.493 | −0.115 | −0.803 | 0.840 | 0.688 | 0.672 |
| MATS-0097 | Intermetallic | Cu₃Au | 32 | −0.034 | 0.409 | −0.001 | +0.004 | −0.097 | 0.0563 | 0.0717 | 0.088 |
| MATS-0103 | Intermetallic | Fe₃Al | 16 | −0.195 | 0 | −0.038 | −0.004 | +0.250 | ~0 | ~0 | 0.130 |
| MATS-0107 | Intermetallic | FePt | 2 | −0.236 | 0 | +0.199 | −0.005 | +0.080 | ~0 | ~0 | 0.022 |
| MATS-0108 | Intermetallic | Ni₃Al | 32 | −0.460 | 1.20 | −0.049 | −0.006 | +0.167 | 0.0888 | 0.100 | 0.129 |
| MATS-0115 | Intermetallic | NiAl | 2 | −0.656 | 0 | −0.035 | −0.003 | +0.042 | ~0 | ~0 | 0.025 |
| MATS-0117 | Intermetallic | NiTi | 4 | +0.223 | 0 | −0.073 | −0.015 | −0.136 | ~0 | ~0 | 0.029 |
| MATS-0118 | MAX phase | Ti₂AlC | 31 | −0.315 | 3.40 | −0.130 | −0.005 | −0.309 | 0.629 | 0.558 | 0.644 |
| MATS-0124 | MAX phase | Ti₃AlC₂ | 12 | −0.811 | 1.94 | −0.031 | −0.006 | −0.131 | 0.240 | 0.228 | 0.355 |
| MATS-0128 | Nuclear material | UO₂ | 12 | −3.389 | 0 | −0.035 | −0.366 | +0.119 | ~0 | ~0 | 0.299 |
| MATS-0132 | Nuclear material | ZrO₂ | 95 | −3.318 | 2.01 | +0.095 | −0.357 | +0.526 | 0.250 | 0.320 | 0.380 |
| MATS-0137 | Perovskite | BaTiO₃ | 39 | −2.809 | 1.72 | −0.016 | −0.327 | +0.202 | 0.231 | 0.202 | 0.161 |
| MATS-0138 | Perovskite | BaTiO₃ | 5 | −3.077 | 0 | +0.000 | −0.317 | +0.505 | ~0 | ~0 | 0.171 |
| MATS-0142 | Perovskite | CsPbBr₃ | 39 | −1.332 | 0.288 | −0.033 | −0.051 | +0.021 | 0.0535 | 0.0570 | 0.087 |
| MATS-0146 | Perovskite | LaFeO₃ | 19 | −2.363 | 0.779 | +0.531 | +0.225 | +0.557 | 0.377 | 0.457 | 0.375 |
| MATS-0148 | Perovskite | SrTiO₃ | 39 | −3.101 | 1.17 | −0.006 | −0.318 | +0.639 | 0.129 | 0.141 | 0.281 |
| MATS-0153 | Thermoelectric | Bi₂Te₃ | 14 | −0.238 | 0.572 | −0.029 | +0.013 | −0.103 | 0.156 | 0.113 | 0.150 |
| MATS-0157 | Thermoelectric | PbTe | 63 | −0.453 | 0.272 | +0.008 | −0.011 | −0.067 | 0.0443 | 0.0129 | 0.062 |

## 7. Supporting evidence from a second, independent DFT dataset (BCC refractory alloys)

A separate zero-shot screen used qualified VASP 6.6.1 PBE relaxation trajectories of 128-atom Nb–Ta–V–Hf–Zr–Ti BCC supercells from the E3/E4 production protocol (ENCUT 350 eV, Γ-centred 2×2×2 k-mesh, non-spin-polarized, Normal-precision stage).
- **Held-out:** 30 frames from two reference structures (E4-BCC-0001, test role; E4-BCC-0561, validation role).
- **Acquired:** 38 frames from two structures chosen during E3 (E3-R1-BCC-0394 and -0582).
- HydraGNN was not evaluated on this set.

| Model | Held-out force MAE / RMSE (eV/Å) | Held-out within-trajectory relative-energy MAE (meV/atom) | Acquired force MAE / RMSE (eV/Å) |
|---|---|---|---|
| UMA | **0.0507 / 0.0646** | **0.0316** | **0.0471 / 0.0595** |
| MACE | 0.1392 / 0.1791 | 0.0360 | 0.1297 / 0.1705 |

UMA's force error is about 2.7× lower on both subsets. Frames within one relaxation are correlated, and only two held-out structure groups exist, so no confidence interval is reported.

## 8. Initial three-model recommendation and its limits

This recommendation predates the expanded MACE-family screen and the corrected
HydraGNN rerun. It applies only
to MACE-MP-0-small, UMA-s-1p1 and HydraGNN; the MACE checkpoint recommendation
is superseded by [`mace-foundations-codabench-screen.md`](mace-foundations-codabench-screen.md).

**Forces (what drives relaxation and molecular dynamics):**
- UMA has the lowest typical-case force error on Codabench: median per-structure 0.131 vs 0.159 eV/Å for MACE. The paired difference of −0.052 eV/Å has a 95% CI of [−0.134, −0.004], excluding zero.
- On BCC alloys, UMA is about 2.7× better than MACE.
- The corrected HydraGNN run has a force-component MAE of 1.642 eV/Å on the 30 clean-force structures and a formation-energy MAE of 0.366 eV/atom on 36 structures. Do not compare it with prior HydraGNN values or infer a model ranking until paired statistics are recomputed from the corrected predictions.

**Formation energies (what drives single-composition analysis and the convex hull):**
- Corrected HydraGNN formation energies are no longer excluded for a pipeline failure; class suitability remains uncertain because the public set is small and only two structures overlap E3 classes.
- Between MACE and UMA the evidence is split:
  - UMA is more accurate in the typical case: median error 0.033 vs 0.039 eV/atom overall and 0.012 vs 0.048 on non-oxides; it wins 21/36 structures.
  - MACE is more robust: MAE 0.132 vs 0.414, correlation 0.992 vs 0.791, no error above 0.60 eV/atom. UMA has one catastrophic failure (WS₂, +10.6 eV/atom) and a systematic −0.53 eV per oxygen shift on oxides.

**Overall zero-shot judgement:**
- **UMA (uma-s-1p1, OMat head):** recommended as the primary model, mainly for force accuracy (Codabench and BCC) and typical-case energy accuracy.
- **MACE-MP-0 small:** recommended as the comparison model and committee member. It is the more robust formation-energy reference and catches UMA's rare large failures.
- **HydraGNN (OMat24 head 7):** checkpoint loading is fixed and its corrected zero-shot metrics are reported above. The small public sample does not establish a primary-model recommendation; recompute paired comparisons before making ranking claims.
- UMA is **not** shown to be best on robust formation-energy accuracy. Formation-energy or hull conclusions should be checked against MACE, and UMA's oxide shift must be accounted for before comparing its oxide formation energies with plain-PBE references.

**Limits of the evidence:**
- Only 37 structures across 8 categories; only 2 match E3 classes. BCC, FCC Cantor, Zn-formate MOF and Nb–Ta–O have no matching Codabench structures, so conclusions for those classes rest on the BCC screen or are indirect.
- Most pairwise formation-energy intervals include or barely exclude zero.
- 7 of 37 DFT force labels and one formation-energy label are physically implausible.
- Single-point evaluation at DFT geometries only; no evaluation at each model's own minimum.
- Small model variants only (MACE "small", UMA "s"); larger checkpoints were not tested.
- Magnetism is not handled by any of the MLIPs, which carry no explicit spin. This affects LiFePO₄, Fe/Co/Ni-containing structures and the triplet-O₂ reference.
- The final model choice is to be made per E3 material class, following the E3-MLIP model suitability matrix.
