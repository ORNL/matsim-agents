# lfp-ideal-force-criterion

```text
Claim (structure) about LiFePO4: The archived ideal olivine LiFePO4 relaxation (PBE without U) ended with every atomic force at or below the 0.01 eV/A criterion it was run with.
Decision criterion: {"quantity":"final_max_force","unit":"eV/A","target_structure":"LiFePO4-olivine-ideal (Li4Fe4P4O16","threshold":0.01,"comparison_structure":null}

Scope sufficient: None; scope: not specified
Available follow-up tests: {"nonconverged_calculation": "rerun-relaxation-tighter-ediffg"}

Evidence records:
[
 {
  "recorded_at": "2026-11-04T21:41:07.964394+00:00",
  "record_type": "evidence",
  "evidence_id": "ev-e218a23b8646cef426a1d61b",
  "kind": "relaxation",
  "subject": "LiFePO4",
  "campaign_id": "e2-lfp-phosphorene-v1",
  "characterization_contract_id": null,
  "calculation_id": null,
  "task_id": "lfp-ideal-force-criterion:ideal",
  "run_id": null,
  "method": {
   "backend": "vasp",
   "backend_version": "6.6.0",
   "model": null,
   "model_sha256": null,
   "method_signature": "vasp6.6.0-pbe-noU-spin-encut520-ediff1e-6-ediffg0.01-isif3",
   "settings": {
    "backend": "vasp",
    "backend_version": "6.6.0",
    "method_signature": "vasp6.6.0-pbe-noU-spin-encut520-ediff1e-6-ediffg0.01-isif3",
    "settings": {
     "functional": "PBE (no Hubbard U)",
     "potcars": [
      "PAW_PBE Li_sv 10Sep2004",
      "PAW_PBE Fe_pv 02Aug2007",
      "PAW_PBE P 06Sep2000",
      "PAW_PBE O 08Apr2002"
     ],
     "encut_eV": 520,
     "prec": "Accurate",
     "ispin": 2,
     "magmom_initial": "Fe 5.0 muB (all parallel), others 0.6 muB",
     "ismear": 0,
     "sigma_eV": 0.05,
     "ibrion": 2,
     "isif": 3,
     "ediffg_eV_per_A": -0.01,
     "archive_commit": "4e186975593e23ad5427aac0657ad4d46ee4bf20"
    },
    "calculation": "dft_reference/vasp/calculations/critical_minerals_LiFePO4_olivine_ideal",
    "outcar_sha256": "062d690a73085a6b5965c1ffb1a64d1c05013950f720c415be4e7122d962d036",
    "kpoints": "archived calculation KPOINTS",
    "ionic_accuracy_marker": "reached required accuracy"
   },
   "software_versions": {}
  },
  "evidence_level": "converged_dft",
  "outcome": "measured",
  "values": [
   {
    "name": "total_energy",
    "value": -195.77238993,
    "unit": "eV",
    "structure": "LiFePO4-olivine-ideal (Li4Fe4P4O16"
   },
   {
    "name": "final_max_force",
    "value": 0.009575,
    "unit": "eV/A",
    "structure": "LiFePO4-olivine-ideal (Li4Fe4P4O16"
   },
   {
    "name": "final_max_abs_principal_stress",
    "value": 1.22,
    "unit": "kbar",
    "structure": "LiFePO4-olivine-ideal (Li4Fe4P4O16"
   },
   {
    "name": "total_magnetization_abs",
    "value": 16.0,
    "unit": "muB",
    "structure": "LiFePO4-olivine-ideal (Li4Fe4P4O16"
   }
  ],
  "numerical_convergence": {},
  "reference_set_id": null,
  "reference_complete": null,
  "input_sha256": {},
  "output_artifact_sha256": {},
  "physical_assumptions": [],
  "uncertainties": {},
  "failure_reason": null,
  "parent_record_ids": [],
  "cost": null
 }
]

Rules for judging the claim:
- A failed_to_compute or not_converged record is not evidence for or against a material.
- A thermodynamic_stability claim is decided only by converged DFT stability (hull) evidence with
  complete reference phases; MLIP evidence supports at most an energy_ranking claim.
- An energy_ranking claim needs at least MLIP relaxation evidence of kind energy_ranking or
  stability.
- When admissible records disagree, the claim is unresolved.
- Evidence that does not cover the claim's stated scope cannot support a decisive verdict.
- Answer unresolved when the evidence does not decide the claim, and request DFT only when a DFT
    calculation could decide it. Select a follow-up test only when it addresses a visible evidence
    gap, and only from the provided options.

Does the evidence support, contradict, or leave the claim unresolved?
```
