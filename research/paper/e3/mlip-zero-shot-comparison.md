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

### 5.3 HydraGNN head-7 formation energies are degenerate
- HydraGNN predicts a formation energy between −0.007 and −0.002 eV/atom for every one of the 37 structures (standard deviation 0.001, versus 1.676 for the DFT labels).
- Its correlation with the labels is −0.31, compared with +0.99 for MACE and +0.79 for UMA (Pd excluded).
- Each HydraGNN error is therefore almost exactly minus the DFT label (CeO₂: label −3.483, error +3.478; MoS₂: −1.771, +1.764; C: +1.810, −1.813).
- **Interpretation:** the energies coming out of this HydraGNN path are effectively additive in composition, which suggests a problem with energy scale, normalization or units (for example, a residual output not converted back to physical units). The HydraGNN energy pathway had not been independently checked for units before this benchmark.
- **Consequence:** the HydraGNN formation-energy numbers below are a pipeline diagnostic, not a measure of the model's physical accuracy. HydraGNN forces vary sensibly by structure and are assessed normally.

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
| HydraGNN (head 7) | 59.68 | 793.41 | 1.423* | 1.870* | 13.2 |

\*Degenerate energy pathway (§5.3). The force aggregates are dominated by the bad labels in §5.1.

### 6.2 Forces on the 30 structures with physical DFT forces

| Model | Pooled component MAE (eV/Å) | Median of per-structure MAE (eV/Å) | Class-balanced MAE (eV/Å) |
|---|---|---|---|
| MACE | 1.099 | 0.159 | 1.142 |
| UMA | 1.019 | **0.131** | 1.058 |
| HydraGNN | **0.180** | 0.133 | **0.254** |

HydraGNN's much lower pooled and class-balanced numbers come mostly from two structures where MACE and UMA fail and HydraGNN does not:
- Pd slab MATS-0039: per-component MAE 13.0 (MACE), 12.0 (UMA), 0.17 (HydraGNN).
- LiFePO₄ MATS-0060: 4.41, 4.24, 0.44.

On typical structures (the median), UMA and HydraGNN are tied and both beat MACE.

### 6.3 Formation energies, robust splits (absolute error, eV/atom; error = model − DFT)

| Subset | n | MACE MAE / median / max | UMA MAE / median / max | HydraGNN MAE / median / max* |
|---|---|---|---|---|
| All, excluding Pd MATS-0039 | 36 | **0.132** / 0.039 / 0.60 | 0.414 / **0.033** / 10.57 | 1.374 / 0.869 / 3.48 |
| Non-oxides, excluding Pd | 24 | **0.127** / 0.048 / 0.52 | 0.477 / **0.012** / 10.57 | 0.708 / 0.472 / 2.29 |
| Oxides | 12 | **0.141** / **0.027** / 0.60 | 0.289 / 0.317 / 0.37 | 2.706 / 3.031 / 3.48 |

Per-structure wins (lower error):
- All except Pd: UMA beats MACE on 21/36; HydraGNN beats MACE on 1/36 and UMA on 1/36.
- Non-oxides: UMA beats MACE on 18/24.
- Oxides: UMA beats MACE on 3/12 (the −0.53 eV per oxygen atom shift).

Correlation between predicted and DFT formation energies (36 structures, Pd excluded): MACE +0.992, UMA +0.791 (lowered by the WS₂ failure), HydraGNN −0.307.

UMA's single catastrophic failure is WS₂ monolayer MATS-0019 (error +10.57 eV/atom). The W and S references are not at fault: WC, MoS₂ and CoS₂ errors are small for UMA. MACE's largest error excluding Pd is 0.60 eV/atom (LiCoO₂).

\*Degenerate energy pathway (§5.3).

### 6.4 Paired bootstrap differences (A − B; negative favours A; mean and 95% CI over structures)

| Comparison | Formation energy, 37 structures | Formation energy, 30 clean-force structures | Force MAE, 30 clean structures |
|---|---|---|---|
| UMA − MACE | +0.329 [−0.032, +0.946]; UMA better on 21/37 | +0.442 [+0.006, +1.200]; UMA better on 16/30 | **−0.052 [−0.134, −0.004]**; UMA better on 15/30 |
| UMA − HydraGNN | −0.745 [−1.419, +0.085]; UMA better on 35/37 | −0.718 [−1.533, +0.285]; 28/30 | +0.526 [+0.000, +1.440]; UMA better on 18/30 |
| MACE − HydraGNN | −1.074 [−1.559, −0.549]; MACE better on 35/37 | −1.160 [−1.736, −0.513]; 28/30 | +0.578 [+0.012, +1.572]; MACE better on 14/30 |

The only clearly resolved results are:
- UMA has lower force errors than MACE; the interval excludes zero, but only just.
- MACE and UMA formation energies beat HydraGNN's degenerate output.
- MACE beats UMA on formation-energy MAE on the 30-structure subset; the interval barely excludes zero.

### 6.5 Per category (formation-energy MAE over all structures in the category | force MAE over clean-force structures; eV/atom | eV/Å)

| Category | n (clean forces) | MACE | UMA | HydraGNN* |
|---|---|---|---|---|
| 2D monolayer | 7 (3) | 0.254 / 0.924 | 1.591 / 0.895 | 1.072 / 0.901 |
| Catalysis | 4 (2) | 2.063 / 6.606 | 2.547 / 6.035 | 1.044 / 0.143 |
| Critical mineral | 9 (8) | 0.174 / 0.761 | 0.189 / 0.718 | 2.000 / 0.255 |
| Intermetallic | 6 (6) | 0.066 / 0.024 | **0.006** / 0.029 | 0.297 / 0.028 |
| MAX phase | 2 (2) | 0.081 / 0.435 | **0.005** / 0.393 | 0.560 / 0.397 |
| Nuclear material | 2 (2) | **0.065** / 0.125 | 0.361 / 0.160 | 3.350 / 0.112 |
| Perovskite | 5 (5) | **0.117** / 0.158 | 0.248 / 0.172 | 2.532 / 0.128 |
| Thermoelectric | 2 (2) | 0.019 / 0.100 | **0.012** / 0.063 | 0.342 / 0.064 |

UMA's 2D-monolayer and catalysis formation-energy numbers are dominated by WS₂ and Pd. Its nuclear-material and perovskite numbers reflect the oxide shift. \*HydraGNN energies degenerate (§5.3).

### 6.6 The two structures that match E3 material classes (single structures, not class estimates)

| Structure | Model | Formation-energy error (eV/atom) | Force MAE (eV/Å) |
|---|---|---|---|
| LiFePO₄, MATS-0060 (28 atoms; DFT label −0.164 eV/atom; largest DFT force 2.80 eV/Å) | MACE | +0.356 | 4.413 |
| | UMA | **+0.025** | 4.237 |
| | HydraGNN* | +0.160 | **0.442** |
| P monolayer, MATS-0016 (4 atoms; DFT label +0.332 eV/atom; largest DFT force 7.43 eV/Å) | MACE | −0.076 | 2.547 |
| | UMA | **−0.010** | 2.488 |
| | HydraGNN* | −0.336 | 2.486 |

### 6.7 Per-structure results (all 37; error = model − DFT; forces are per-component MAE; † = force label excluded; ‡ = suspect energy label)

| ID | Category | Formula | N | DFT E_f (eV/atom) | Largest DFT force (eV/Å) | MACE ΔE_f | UMA ΔE_f | HydraGNN ΔE_f* | MACE force MAE | UMA force MAE | HydraGNN force MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MATS-0002† | 2D monolayer | BN | 31 | −0.935 | 133 | +0.515 | −0.224 | +0.931 | 57.8 | 57.8 | 57.7 |
| MATS-0004† | 2D monolayer | BN | 32 | −0.493 | 131 | +0.432 | −0.238 | +0.489 | 59.6 | 59.6 | 59.6 |
| MATS-0006† | 2D monolayer | C | 32 | +1.810 | 138 | +0.154 | −0.031 | −1.813 | 62.9 | 62.9 | 62.9 |
| MATS-0011 | 2D monolayer | MoS₂ | 48 | −1.771 | 0.743 | +0.046 | +0.023 | +1.764 | 0.172 | 0.143 | 0.165 |
| MATS-0013† | 2D monolayer | MoSe₂ | 47 | +0.549 | 110 | −0.156 | +0.040 | −0.553 | 26.1 | 26.0 | 25.8 |
| MATS-0016 | 2D monolayer | P | 4 | +0.332 | 7.43 | −0.076 | −0.010 | −0.336 | 2.55 | 2.49 | 2.49 |
| MATS-0019 | 2D monolayer | WS₂ | 48 | +1.614 | 0.24 | +0.396 | +10.568 | −1.620 | 0.0534 | 0.0535 | 0.0534 |
| MATS-0030† | Catalysis | Cu | 64 | +0.152 | 12,435 | +0.006 | +0.007 | −0.158 | 1,060 | 1,060 | 1,060 |
| MATS-0033† | Catalysis | Fe | 31 | +0.485 | 1,157 | +0.032 | +0.035 | −0.491 | 116 | 116 | 116 |
| MATS-0035 | Catalysis | Ni | 63 | +0.348 | 0.571 | +0.049 | −0.004 | −0.353 | 0.162 | 0.111 | 0.113 |
| MATS-0039‡ | Catalysis | Pd | 63 | +3.167 | 0.86 | +8.164 | +10.140 | −3.174 | 13.0 | 12.0 | 0.174 |
| MATS-0047 | Critical mineral | CeO₂ | 95 | −3.483 | 2.06 | −0.002 | −0.363 | +3.478 | 0.120 | 0.121 | 0.129 |
| MATS-0051 | Critical mineral | CoS₂ | 11 | −1.070 | 0.793 | −0.019 | −0.017 | +1.066 | 0.258 | 0.206 | 0.251 |
| MATS-0055 | Critical mineral | La₂O₃ | 5 | −3.463 | 0.284 | +0.020 | −0.313 | +3.461 | 0.0632 | 0.0540 | 0.0523 |
| MATS-0058 | Critical mineral | LiCoO₂ | 12 | −1.483 | 2.83 | +0.598 | +0.200 | +1.480 | 0.453 | 0.442 | 0.471 |
| MATS-0060 | Critical mineral | LiFePO₄ | 28 | −0.164 | 2.80 | +0.356 | +0.025 | +0.160 | 4.41 | 4.24 | 0.442 |
| MATS-0065 | Critical mineral | MgAl₂O₄ | 56 | −2.990 | 0.907 | −0.001 | −0.303 | +2.987 | 0.295 | 0.300 | 0.299 |
| MATS-0068 | Critical mineral | SiC | 64 | −0.219 | 2.42 | −0.041 | −0.010 | +0.216 | 0.200 | 0.143 | 0.144 |
| MATS-0072 | Critical mineral | TiO₂ | 71 | −2.869 | 2.09 | −0.034 | −0.357 | +2.866 | 0.290 | 0.246 | 0.248 |
| MATS-0075† | Critical mineral | WC | 53 | +2.285 | 39.2 | +0.493 | −0.115 | −2.290 | 0.840 | 0.688 | 0.577 |
| MATS-0097 | Intermetallic | Cu₃Au | 32 | −0.034 | 0.409 | −0.001 | +0.004 | +0.028 | 0.0563 | 0.0717 | 0.0735 |
| MATS-0103 | Intermetallic | Fe₃Al | 16 | −0.195 | 0 | −0.038 | −0.004 | +0.190 | ~0 | ~0 | 0.0002 |
| MATS-0107 | Intermetallic | FePt | 2 | −0.236 | 0 | +0.199 | −0.005 | +0.233 | ~0 | ~0 | 0.0009 |
| MATS-0108 | Intermetallic | Ni₃Al | 32 | −0.460 | 1.20 | −0.049 | −0.006 | +0.455 | 0.0888 | 0.100 | 0.0958 |
| MATS-0115 | Intermetallic | NiAl | 2 | −0.656 | 0 | −0.035 | −0.003 | +0.653 | ~0 | ~0 | 0.0001 |
| MATS-0117 | Intermetallic | NiTi | 4 | +0.223 | 0 | −0.073 | −0.015 | −0.226 | ~0 | ~0 | 0.0001 |
| MATS-0118 | MAX phase | Ti₂AlC | 31 | −0.315 | 3.40 | −0.130 | −0.005 | +0.312 | 0.629 | 0.558 | 0.553 |
| MATS-0124 | MAX phase | Ti₃AlC₂ | 12 | −0.811 | 1.94 | −0.031 | −0.006 | +0.807 | 0.240 | 0.228 | 0.242 |
| MATS-0128 | Nuclear material | UO₂ | 12 | −3.389 | 0 | −0.035 | −0.366 | +3.385 | ~0 | ~0 | 0.0002 |
| MATS-0132 | Nuclear material | ZrO₂ | 95 | −3.318 | 2.01 | +0.095 | −0.357 | +3.315 | 0.250 | 0.320 | 0.224 |
| MATS-0137 | Perovskite | BaTiO₃ | 39 | −2.809 | 1.72 | −0.016 | −0.327 | +2.805 | 0.231 | 0.202 | 0.194 |
| MATS-0138 | Perovskite | BaTiO₃ | 5 | −3.077 | 0 | +0.000 | −0.317 | +3.075 | ~0 | ~0 | 0.0002 |
| MATS-0142 | Perovskite | CsPbBr₃ | 39 | −1.332 | 0.288 | −0.033 | −0.051 | +1.326 | 0.0535 | 0.0570 | 0.0597 |
| MATS-0146 | Perovskite | LaFeO₃ | 19 | −2.363 | 0.779 | +0.531 | +0.225 | +2.359 | 0.377 | 0.457 | 0.246 |
| MATS-0148 | Perovskite | SrTiO₃ | 39 | −3.101 | 1.17 | −0.006 | −0.318 | +3.097 | 0.129 | 0.141 | 0.138 |
| MATS-0153 | Thermoelectric | Bi₂Te₃ | 14 | −0.238 | 0.572 | −0.029 | +0.013 | +0.236 | 0.156 | 0.113 | 0.115 |
| MATS-0157 | Thermoelectric | PbTe | 63 | −0.453 | 0.272 | +0.008 | −0.011 | +0.448 | 0.0443 | 0.0129 | 0.0136 |

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

This recommendation predates the expanded MACE-family screen. It applies only
to MACE-MP-0-small, UMA-s-1p1 and HydraGNN; the MACE checkpoint recommendation
is superseded by [`mace-foundations-codabench-screen.md`](mace-foundations-codabench-screen.md).

**Forces (what drives relaxation and molecular dynamics):**
- UMA has the lowest typical-case force error on Codabench: median per-structure 0.131 vs 0.159 eV/Å for MACE. The paired difference of −0.052 eV/Å has a 95% CI of [−0.134, −0.004], excluding zero.
- On BCC alloys, UMA is about 2.7× better than MACE.
- HydraGNN ties UMA on typical structures and is the most robust on the two difficult structures (Pd slab, LiFePO₄).

**Formation energies (what drives single-composition analysis and the convex hull):**
- HydraGNN is excluded from this ranking until its energy pathway is fixed (§5.3).
- Between MACE and UMA the evidence is split:
  - UMA is more accurate in the typical case: median error 0.033 vs 0.039 eV/atom overall and 0.012 vs 0.048 on non-oxides; it wins 21/36 structures.
  - MACE is more robust: MAE 0.132 vs 0.414, correlation 0.992 vs 0.791, no error above 0.60 eV/atom. UMA has one catastrophic failure (WS₂, +10.6 eV/atom) and a systematic −0.53 eV per oxygen shift on oxides.

**Overall zero-shot judgement:**
- **UMA (uma-s-1p1, OMat head):** recommended as the primary model, mainly for force accuracy (Codabench and BCC) and typical-case energy accuracy.
- **MACE-MP-0 small:** recommended as the comparison model and committee member. It is the more robust formation-energy reference and catches UMA's rare large failures.
- **HydraGNN (OMat24 head 7):** not recommended for energies until the energy scale/normalization problem is diagnosed. Its forces are competitive, and it is about 33× faster than UMA and 4× faster than MACE on CPU.
- UMA is **not** shown to be best on robust formation-energy accuracy. Formation-energy or hull conclusions should be checked against MACE, and UMA's oxide shift must be accounted for before comparing its oxide formation energies with plain-PBE references.

**Limits of the evidence:**
- Only 37 structures across 8 categories; only 2 match E3 classes. BCC, FCC Cantor, Zn-formate MOF and Nb–Ta–O have no matching Codabench structures, so conclusions for those classes rest on the BCC screen or are indirect.
- Most pairwise formation-energy intervals include or barely exclude zero.
- 7 of 37 DFT force labels and one formation-energy label are physically implausible.
- Single-point evaluation at DFT geometries only; no evaluation at each model's own minimum.
- Small model variants only (MACE "small", UMA "s"); larger checkpoints were not tested.
- Magnetism is not handled by any of the MLIPs, which carry no explicit spin. This affects LiFePO₄, Fe/Co/Ni-containing structures and the triplet-O₂ reference.
- The final model choice is to be made per E3 material class, following the E3-MLIP model suitability matrix.
