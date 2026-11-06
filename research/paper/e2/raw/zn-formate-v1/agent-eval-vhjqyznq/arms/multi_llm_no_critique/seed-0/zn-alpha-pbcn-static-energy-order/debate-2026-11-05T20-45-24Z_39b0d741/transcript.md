# Scientific debate

Run `debate-2026-11-05T20-45-24Z_39b0d741`.

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

`round-001-turn-0001` · claude-sonnet-5-5@google · 847 completion tokens · 7.4 s

{"status":"supported","evidence_ids":["ev-0170b0f63d871c8acab55668"],"request_dft":false,"proposed_test_ids":[],"rationale":"The claim is limited to the two recorded static sigma-zero energies, and ev-0170b0f63d871c8acab55668 records both. I rechecked the arithmetic. Alpha: -670.89296251 eV / 108 atoms = -6.2119718751 eV/atom. Pbcn: -1341.77791177 eV / 216 atoms = -6.2119347767 eV/atom. Alpha is lower by 3.71e-5 eV/atom, which is about 0.037 meV/atom, so the stated claim holds. This is a property claim about two recorded calculations, not a thermodynamic_stability or energy_ranking claim, so the hull and MLIP requirements do not apply. The scope statement itself says the result is not a convergence-qualified polymorph ranking. The two calculations use different k-meshes (Gamma 3x4x3 and Gamma 4x2x2) and the numerical_convergence field is empty. A 0.037 meV/atom gap is almost certainly smaller than the k-point and cutoff uncertainty, so I would not read it as physical stability ordering. Under the stated scope, that caveat does not change the verdict on what was computed. I see no admissible record that disagrees with the ordering. No follow-up tests are offered, and further DFT is not needed to decide this narrow claim, so I do not request it. No peers have spoken, so there are no peer claims to support or dispute. A useful falsification test for any broader ranking claim would be to converge k-mesh and cutoff for both cells and check whether the sign of the difference survives."}

