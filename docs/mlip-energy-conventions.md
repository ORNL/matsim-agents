# MLIP and DFT energy conventions

This page states, for each MLIP backend, which energy labels it was pretrained
on, which labels it is fine-tuned on, which reference energies are used, and
what it returns at inference. The procedures are described in
[MLIP training references and model promotion](mlip-training-and-promotion.md);
this page explains the conventions behind them.

## Units

Every energy, force and stress inside matsim-agents is in eV, eV/Å and eV/Å³,
whichever DFT code produced it. Quantum ESPRESSO output (Ry, Ry/Bohr,
Ry/Bohr³) is converted once, when it is parsed, with the CODATA 2014 constants
of `ase.units` (the same values `ase.io` uses):

| Quantity | Factor |
|---|---|
| Ry → eV | 13.605693012183622 |
| Bohr → Å | 0.5291772105638411 |
| Ry/Bohr → eV/Å | 25.711033545240323 |
| Ry/Bohr³ → eV/Å³ | 91.81576765360094 |

Values that are already in eV are never converted again. The maximum atomic
force is computed from the per-atom force vectors; QE's `Total force` line is
the norm of all forces together and is kept only under its own name
(`final_total_force_ry_au`). VASP output is already in eV and Å.

## Reference energies: pure-element condensed phases

Unit conversion does not make total energies from different codes comparable.
A DFT total is measured from a zero set by the pseudopotential or PAW dataset.
VASP PAW totals are measured relative to the reference atoms built into each
POTCAR, while QE totals contain the full pseudo-atom energy. For
example, the 2-atom Si cell in `tests/data/dft_outputs/qe/relax_converged`
has a QE total of −16.73962331 Ry = −227.754 eV, i.e. −113.88 eV/atom, while a
VASP-PBE bulk Si total is about −5.4 eV/atom. Both values are in eV; they
differ in their zero.

matsim-agents removes this zero with **pure-element condensed-phase
references computed with the same DFT method as the labels**:

```text
E_formation = E_DFT - sum over elements (N_element * e_element_reference)
```

`e_element_reference` is the lowest DFT energy per atom among the declared
phases of that element (bulk crystals, or a declared molecule such as triplet
O2), computed by `matsim_agents.active_learning.elemental_dft` and frozen in a
hash-bound manifest. QE and VASP references are never mixed.

Condensed phases are used because they are the endpoints of convex-hull
analysis: the elements sit at 0 eV/atom, a compound is stable against its
elements when its formation energy is negative, and formation energies from
QE and VASP become comparable. Isolated atoms would give atomization energies
instead; they are used only for optional cohesive or atomization diagnostics.
A constant shift per element changes absolute formation energies but not the
energy above the hull or the decomposition products, as long as one reference
set is used throughout. Training labels, inference restoration, evaluation
and hull analysis must therefore use the same frozen manifest.

## Conventions by backend

| Backend | Pretrained energy label | Fine-tuning energy label | Model output after fine-tuning | Energy returned by the calculator |
|---|---|---|---|---|
| HydraGNN GFM, OMat24 branch (index 7) | `E_VASP − Σ c_Z`, with `c_Z` a per-element linear fit | total-cell formation energy against the DFT condensed references | total-cell formation energy | formation energy + Σ `e_DFT_ref`, added once (DFT-reference total) |
| UMA, `omat` task | VASP-PBE total energy (through the checkpoint's normalizer and element references) | `E_DFT − Σ (e_DFT_ref − e_UMA_ref)` | UMA-scale total energy | the same value; no offset is added back |
| MACE-MP | VASP-PBE total energy (foundation `E0s`) | `E_DFT − Σ (e_DFT_ref − e_MACE_ref)` | MACE-scale total energy | the same value; no offset is added back |

Forces are the same in every convention, because a constant per element does
not change with atomic positions. Forces are always trained on the raw DFT
forces in eV/Å.

### HydraGNN

The GFM branch order is fixed by the pretraining `multi_model_list`
(`datadir0` … `datadir15`): Alexandria, ANI1x, MPTrj, OC2020, OC2022, OC25,
ODAC23, **OMat24 (7)**, OMol25, OMol25-neutral, OMol25-non-neutral,
OPoly2026, Nabla2DFT, QCML, QM7X, transition1x.

The OMat24 dataset has no pure-element energies, so HydraGNN pretraining
subtracts a per-element linear fit instead. Each OMat24 frame's training
label is `E_VASP − Σ_i c_{Z_i}` for the whole cell, with the 118 coefficients
stored in the dataset's `energy_linear_regression_coeff` attribute (for
example Nb −9.3475, Ta −10.6846, O −7.2109 eV/atom). The energy is the
sum-pooled output of the branch, and forces are its negative position
gradient. This relation was checked on 200 frames of `OMat24-v2.bp`
(largest difference 3.6 × 10⁻⁶ eV).

The fitted `c_Z` approximate per-element reference energies, so the branch
already predicts a formation-like energy. Fine-tuning moves it from the fitted
references to the declared DFT condensed references; the two differ only by a
constant per element. This adaptation does not claim that the pretrained head
used the DFT references.

Consequences:

- An **unmarked foundation checkpoint** returns `E_VASP − Σ c_Z`, not a DFT
  total. Compare it with DFT only through formation energies, each method
  using its own elemental energies.
- A **fine-tuned checkpoint** has an `energy-convention.json` bound to the
  checkpoint hash and the reference manifest. The calculator adds the DFT
  elemental baseline exactly once and refuses a second restoration.
- The fine-tuners take the **raw DFT totals** and build the formation labels
  themselves. A dataset that has already been converted is rejected.
- `energy-convention.json` records the fine-tune protocol under `finetune`:
  the protocol name, `head_init` (`pretrained` or `random`), the pretrained
  heads that were kept (for example `["OMat24"]`), and the freezing mode. A
  kept pretrained head starts from the `E_VASP − Σ c_Z` scale and is adapted
  to the DFT formation scale; this record makes that visible to evaluation.

### UMA and MACE

UMA's `omat` task and MACE-MP were pretrained on VASP-PBE totals. Their
fine-tuners keep that scale: each label is shifted by
`Σ (e_DFT_ref − e_model_ref)`, where `e_model_ref` is the base model's energy
per atom on the same elemental reference geometries. Formation energies
relative to the references are unchanged by this shift. The offsets are
written to `energy-alignment.json` with the dataset and manifest hashes, and
later fine-tunes reuse them so the scale does not drift. Its `finetune` field
records the backend, base model, protocol and adapter settings (for UMA also
the task heads and that their output is denormalized and element-referenced).

The UMA fine-tuner trains on fairchem's physical output: the raw head output
is denormalized and element-referenced with the checkpoint's own task objects,
as `MLIPPredictUnit` does at inference. The calculator is built with the same
`task_name`.

## Comparing models with DFT

For held-out evaluation and promotion, every method uses its **own** elemental
energies on the same fixed reference geometries:

```text
formation_energy_m = (E_total_m - sum_i(n_i * e_ref_m[i])) / N
```

DFT totals subtract DFT references; MLIP totals subtract that MLIP's
predictions on the reference geometries. Never subtract DFT references from
raw MLIP totals, and never train one model on mixed QE and VASP labels.

## Verification status

| Item | Status |
|---|---|
| QE unit conversion and per-atom maximum force | software verified against `ase.io` on real QE outputs |
| HydraGNN OMat24 branch label (`E_VASP − Σ c_Z`, index 7) | numerically verified on the pretraining data |
| HydraGNN formation fine-tuning and single restoration | software verified (CPU contract tests); GPU qualification pending |
| UMA `omat` scale (normalizer and element references) | verified from fairchem 2.25 source; not checked numerically because no UMA checkpoint is available on Frontier |
| MACE foundation scale | software verified; not checked numerically here |
