# Scientific debate

Run `debate-2026-11-05T20-45-15Z_f33341a8`.

**Hypothesis.** Claim (property) about Zn(HCOO)2 guest-free anhydrous polymorphs: For the two recorded static calculations, the alpha COD 4515120 model has a lower sigma-zero energy per atom than the Pbcn COD 7205949 model.
Decision criterion: {"quantity":"static_energy_per_atom","unit":"eV/atom","target_structure":"Zn3(HCOO)6-alpha-COD4515120-static","threshold":null,"comparison_structure":"Zn(HCOO)2-Pbcn-COD7205949-static"}

Scope sufficient: True; scope: The claim concerns only the computed energies from these two finite meshes. The difference is 0.037 meV/atom and is not a robust stability or convergence-qualified ranking.
Available follow-up tests: {}

Evidence records:
[
 {
  "recorded_at": "2026-11-05T20:41:19.764147+00:00",
  "record_type": "evidence",
  "evidence_id": "ev-0170b0f63d871c8acab55668",
  "kind": "characterization",
  "subject": "Zn(HCOO)2 guest-free anhydrous polymorphs",
  "campaign_id": "e2-zn-formate-v1",
  "characterization_contract_id": null,
  "calculation_id": null,
  "task_id": "zn-alpha-pbcn-static-energy-order:comparison",
  "run_id": null,
  "method": {
   "backend": "arithmetic",
   "backend_version": null,
   "model": null,
   "model_sha256": null,
   "method_signature": "static-energy-per-atom-comparison-v1",
   "settings": {
    "calculation": "Static sigma-zero energies normalized by atom count",
    "functional": "PBE-D3(BJ)",
    "alpha_total_energy_eV": -670.89296251,
    "alpha_natoms": 108,
    "alpha_kpoints": "Gamma 3x4x3",
    "pbcn_total_energy_eV": -1341.77791177,
    "pbcn_natoms": 216,
    "pbcn_kpoints": "Gamma 4x2x2",
    "absolute_difference_eV_per_atom": 3.70989e-05,
    "qualification_records": [
     "e2-dft-znformate-anhydrous-v1-20261104/cases/zn_formate_alpha_Zn3HCOO6_COD4515120/verify-static.json",
     "e2-dft-znformate-anhydrous-v1-20261104/cases/zn_formate_lonsdaleite_Pbcn_COD7205949/verify-static.json"
    ],
    "interpretation_limit": "Computed finite-mesh ordering only; not a convergence-qualified polymorph stability ranking."
   },
   "software_versions": {}
  },
  "evidence_level": "converged_dft",
  "outcome": "measured",
  "values": [
   {
    "name": "static_energy_per_atom",
    "value": -6.2119718751,
    "unit": "eV/atom",
    "structure": "Zn3(HCOO)6-alpha-COD4515120-static"
   },
   {
    "name": "static_energy_per_atom",
    "value": -6.2119347767,
    "unit": "eV/atom",
    "structure": "Zn(HCOO)2-Pbcn-COD7205949-static"
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

## Argument · claude-sonnet · round 1

`round-001-turn-0001` · claude-sonnet-5-5@google · 916 completion tokens · 8.4 s

{"status":"supported","evidence_ids":["ev-0170b0f63d871c8acab55668"],"request_dft":false,"proposed_test_ids":[],"rationale":"The claim is limited to the two recorded static sigma-zero energies, and ev-0170b0f63d871c8acab55668 records both. The alpha COD 4515120 model has -6.2119718751 eV/atom (-670.89296251 eV over 108 atoms). The Pbcn COD 7205949 model has -6.2119347767 eV/atom (-1341.77791177 eV over 216 atoms). I checked the normalization and it is consistent. Alpha is lower by 3.71e-5 eV/atom, so the stated comparison holds. The record is a measured converged_dft record, not a failed or non-converged one. No peers have spoken, so I have no peer claims to support or dispute. The caveats are the very small 0.037 meV/atom gap and the different finite k-meshes (Gamma 3x4x3 for alpha, Gamma 4x2x2 for Pbcn). The numerical_convergence field is also empty. These caveats mean the result cannot be read as a polymorph stability ranking or a convergence-qualified ordering. They do not affect the narrower claim about the two recorded calculations, which the record's own interpretation_limit also restricts to a finite-mesh ordering. A falsification test for a broader claim would be k-mesh and cutoff convergence on both structures, or a common-mesh recalculation, which could flip a gap this small. No follow-up tests are offered, and none is needed for this scoped claim."}

