# Scientific debate

Run `debate-2026-11-05T20-45-37Z_06ac4d77`.

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

`round-001-turn-0001` · claude-sonnet-5-5@google · 1527 completion tokens · 12.8 s

**Position: the evidence supports the claim as literally scoped. It does not support any broader stability or ranking claim.**

**Arithmetic check.**
- Alpha: −670.89296251 eV / 108 atoms = −6.2119718751 eV/atom.
- Pbcn: −1341.77791177 eV / 216 atoms = −6.2119347767 eV/atom.
- Difference: alpha − Pbcn = −3.70984×10⁻⁵ eV/atom, which is −0.037 meV/atom. This matches the record's stated absolute difference of 3.70989×10⁻⁵ eV/atom, within rounding.
- The alpha model is therefore lower in sigma-zero energy per atom, so the claim holds.

**Why the claim is decidable.**
- The claim is explicitly restricted to "the two recorded static calculations."
- The scope statement says it concerns only the computed energies from these two finite meshes.
- This is a comparison of recorded numbers, not a thermodynamic_stability claim, so the hull and complete-reference requirement does not apply.
- It is also not a relaxed-structure energy_ranking claim, so I dispute reading the MLIP-evidence rule as a bar here. The record is DFT-level (PBE-D3(BJ), evidence_level converged_dft). Its own interpretation limit states "computed finite-mesh ordering only."
- Only one record exists, so there is no admissible disagreement that would force "unresolved."

**Assumptions and weaknesses I want on the table.**
1. The 0.037 meV/atom gap is orders of magnitude below typical DFT k-mesh convergence error (about 1 meV/atom) and below the accuracy of PBE-D3. The two calculations use different cell sizes (108 vs 216 atoms) and different Γ-centered meshes (3×4×3 vs 4×2×2). The sign of the difference could plausibly flip under a denser mesh or a common k-point density.
2. The `numerical_convergence` field is empty. Convergence is only indirectly attested by the referenced verify-static files, which I cannot inspect.
3. Static energies are at the given geometries. They are not necessarily relaxed to the same force or stress criteria, and they include no zero-point or thermal terms.

**Falsification tests.**
- Rerun both structures with matched k-point density and a k-mesh convergence series. If the sign of the difference changes, the finite-mesh ordering is an artifact. That would not falsify the narrow claim as written, but it would show the claim carries no physical content.
- Verify the total energies and atom counts from the output artifacts.

No follow-up tests are offered, so I select none.

**Verdict:** supported for the stated narrow scope. Any reading as "alpha is more stable than Pbcn" is unsupported.

## Argument · gemini-flash · round 1

`round-001-turn-0002` · gemini-2.5-flash@google · 2103 completion tokens · 10.8 s

**Position: The evidence supports the claim as literally stated.**

I concur with Claude-sonnet's arithmetic check and conclusion: the alpha model's static energy per atom (-6.2119718751 eV/atom) is indeed lower than the Pbcn model's (-6.2119347767 eV/atom) for the specific calculations recorded. The difference of approximately 0.037 meV/atom is correctly reported.

**Support for Peer Claims:**
I fully support Claude-sonnet's assertion that the claim, as explicitly scoped, is decidable based on the provided numerical evidence. The claim is a direct comparison of two recorded values, not a broader statement about thermodynamic stability or a convergence-qualified ranking. Claude-sonnet correctly highlights that the rules for judging thermodynamic stability or energy ranking claims (e.g., requiring complete reference phases or MLIP evidence) do not apply here, as this is a "converged_dft" level characterization of specific calculation outputs.

**Exposed Assumptions:**
1.  **Accuracy of Recorded Values:** I assume the `alpha_total_energy_eV`, `alpha_natoms`, `pbcn_total_energy_eV`, and `pbcn_natoms` values provided in the evidence record are accurate representations of the outputs from the underlying DFT calculations.
2.  **Validity of `converged_dft` label:** The `evidence_level` is stated as `converged_dft`. I assume this implies that each individual calculation (for alpha and Pbcn) was performed to a sufficient level of convergence for its *own* settings, even if the comparison between them might not be robustly converged for physical ranking. The referenced `qualification_records` are not directly inspectable, so I must trust this label.

**Distinguishing Evidence from Speculation:**
The numerical values for energy per atom are direct evidence. The conclusion that alpha has a lower energy per atom is a direct consequence of comparing these values. Speculation arises when attempting to infer broader physical meaning (e.g., thermodynamic stability) from this extremely narrow and potentially non-converged comparison.

**Falsification Tests:**
To falsify the *literal claim* as written, one would need to demonstrate that the recorded numerical values themselves are incorrect, or that the arithmetic performed on them is flawed.
1.  **Data Integrity Check:** Access the raw output files for the `alpha` and `Pbcn` calculations and verify that the `total_energy_eV` and `natoms` values match those reported in the evidence record. If a discrepancy is found that alters the energy per atom values such that Pbcn becomes lower or equal, the claim would be falsified. This aligns with Claude-sonnet's suggestion to "Verify the total energies and atom counts from the output artifacts."

While not falsifying the *literal claim*, I also agree with Claude-sonnet's proposed test to assess the *physical significance* of the difference:
2.  **Robust Convergence Study:** Perform a k-point mesh convergence study for both structures, ensuring a consistent k-point density or a common convergence criterion. If, under such a robust comparison, the sign of the energy difference flips or becomes negligible (within convergence tolerance), it would demonstrate that the current ordering is an artifact of the specific, disparate k-point meshes used, rendering the claim physically meaningless despite being numerically true for the recorded calculations.

No follow-up tests are offered, so I select none.

## Argument · gemini-flash · round 2

`round-002-turn-0003` · gemini-2.5-flash@google · 1951 completion tokens · 14.4 s

My position remains that the evidence **supports** the claim as literally stated, but I strongly reiterate that this claim carries no physical meaning regarding polymorph stability.

**Current Argument:**
The claim is a direct numerical comparison of two specific computed values from the provided evidence record.
1.  The static energy per atom for the alpha model is -6.2119718751 eV/atom.
2.  The static energy per atom for the Pbcn model is -6.2119347767 eV/atom.
Since -6.2119718751 eV/atom is indeed lower than -6.2119347767 eV/atom, the claim is numerically true for these two recorded calculations. The difference is approximately 0.037 meV/atom. The scope explicitly limits the claim to "the two recorded static calculations" and "computed energies from these two finite meshes," which is precisely what the evidence provides.

**Support for Peer Claims:**
I fully support Claude-sonnet's and Gemini-flash's arithmetic verification; their calculations are correct. I also agree with both peers that the claim is decidable *within its extremely narrow, literal scope*. Claude-sonnet correctly points out that the rules for `thermodynamic_stability` or `energy_ranking` claims, which would require more robust convergence or complete reference phases, do not apply here due to the precise wording of the hypothesis. Gemini-flash accurately distinguishes between the direct numerical evidence and the speculation that arises from inferring broader physical meaning.

**Exposed Assumptions:**
1.  **Data Integrity:** We assume the numerical values for total energies and atom counts reported in the `evidence` record are accurate reflections of the underlying DFT calculation outputs.
2.  **`converged_dft` Label:** As Gemini-flash noted, we assume that each individual calculation (alpha and Pbcn) was converged according to its own specified settings, even if the comparison between them is not robustly converged for physical ranking. The `qualification_records` are not inspectable, so we rely on this label.

**Distinguishing Evidence from Speculation:**
The numerical values presented in the `evidence` record and their direct arithmetic comparison constitute the evidence. The conclusion that alpha has a lower energy per atom *for these specific calculations* is a direct consequence of this evidence. Any interpretation that this 0.037 meV/atom difference implies actual thermodynamic stability or a robust energy ranking between the polymorphs is pure speculation, especially given the disparate k-point meshes and the magnitude of the difference being well below typical DFT convergence tolerances. The `interpretation_limit` in the evidence record itself explicitly cautions against such broader interpretations.

**Falsification Tests:**
1.  **Falsifying the Literal Claim:** As Gemini-flash suggested, the only way to falsify the *literal claim* is to demonstrate that the recorded numerical values for total energy or atom count in the evidence record are incorrect. This would involve inspecting the original calculation output files and showing that the reported values lead to a different ordering or equality.
2.  **Falsifying Physical Significance:** I reiterate Claude-sonnet's and Gemini-flash's proposal for a k-point convergence study. If, under a consistent and sufficiently dense k-point mesh (or a common k-point density), the sign of the energy difference flips, or the difference becomes negligible (e.g., within 0.1 meV/atom), it would demonstrate that the current ordering is an artifact of the specific, disparate computational parameters and lacks physical meaning. This would not falsify the *literal claim* about the recorded numbers, but it would invalidate any scientific utility of the claim.

## Argument · claude-sonnet · round 2

`round-002-turn-0004` · claude-sonnet-5-5@google · 1831 completion tokens · 15.8 s

**Position: the evidence supports the claim as literally scoped (alpha lower than Pbcn in recorded sigma-zero energy per atom). It gives no support to any physical stability or ranking claim.**

**Arithmetic check.** I recomputed both values.
- Alpha: −670.89296251 / 108 = −6.21197188 eV/atom.
- Pbcn: −1341.77791177 / 216 = −6.21193478 eV/atom.
- The difference is −3.71×10⁻⁵ eV/atom, which is 0.037 meV/atom. The sign is unambiguous, and the record's own values entries agree with the settings entries.

**Why I land on "supports" rather than "unresolved."**
- The hypothesis is about "the two recorded static calculations," and the scope field says it concerns only these finite-mesh energies. The record gives exactly those two numbers, with `evidence_level: converged_dft` and `outcome: measured`.
- It is not a thermodynamic_stability claim, so the hull and complete-reference rule does not apply. It is not a relaxed-structure energy_ranking claim either, so the MLIP-evidence requirement does not apply.
- There is a single admissible record, so there is no disagreement among records to force "unresolved."
- The `interpretation_limit` restricts how the result may be used. It does not change the truth of the numerical comparison.

**Disputes and refinements.**
- I partly dispute Claude-sonnet's wording that the gap is "orders of magnitude" below typical k-mesh error. At roughly 1 meV/atom versus 0.037 meV/atom it is about 1.4 orders. The qualitative point stands, but the mesh-sensitivity claim is speculation, since no convergence series is in the record.
- I agree with both peers that the sign could flip under a matched k-point density. That is a hypothesis, not evidence, and it cannot count against the literal claim.
- Gemini-flash's second-round text lists "Gemini-flash" among the peers it supports, which looks like a slip. It does not affect the substance, and I agree with the position it states.
- The one real fault line is the reading of the claim. If a grader treated "lower energy" as implying a physically meaningful ordering, the empty `numerical_convergence`, the mismatched meshes (Γ 3×4×3 versus 4×2×2), and the sub-meV gap would make the answer unresolved. The hypothesis text and scope statement explicitly disclaim that reading, so I do not adopt it.

**Assumptions.**
1. The totals and atom counts match the underlying outputs. The verify-static files are not inspectable here.
2. The `converged_dft` label means each calculation was individually qualified.
3. Both energies are for the same functional, PBE-D3(BJ), and a comparable pseudopotential and cutoff setup. The record states the functional but not the cutoff or the pseudopotentials.

**Falsification tests.**
- **Literal claim:** re-extract the final sigma-zero energies and NIONS from the two output files. The claim fails only if the per-atom values reorder or become equal.
- **Physical content (does not falsify the literal claim):** run a k-mesh convergence series for both cells at matched density, and check cutoff and pseudopotential equivalence. A sign flip or a difference within tolerance would show that the ordering is only a mesh artifact.

No follow-up tests are offered, so I select none.

**Verdict: supported, for the narrow literal scope only.**

## Verdict · claude-sonnet

`verdict-001-turn-0005` · claude-sonnet-5-5@google · 519 completion tokens · 4.9 s

{"status":"supported","evidence_ids":["ev-0170b0f63d871c8acab55668"],"request_dft":false,"proposed_test_ids":[],"rationale":"The claim is scoped to the two recorded static sigma-zero energies, and one converged_dft, measured record gives both values. I recomputed them. Alpha is -670.89296251/108 = -6.2119718751 eV/atom. Pbcn is -1341.77791177/216 = -6.2119347767 eV/atom. Alpha is lower by about 3.71e-5 eV/atom (0.037 meV/atom), which matches the record's stated difference. There is only one record, so no admissible records disagree. The claim is a comparison of recorded numbers, not a thermodynamic_stability or relaxed energy_ranking claim, so the hull and MLIP-evidence rules do not apply. The scope statement says the claim covers only these finite meshes, so it does not assert a stability ranking. I agree with Claude-sonnet and Gemini-flash that the comparison is decidable at this literal level. The limits are real. The gap is far below typical k-mesh error, the meshes differ (Γ 3x4x3 vs Γ 4x2x2), numerical_convergence is empty, and the qualification files cannot be inspected. These limits mean the ordering carries no physical stability meaning, since a matched-density k-mesh series might flip the sign. They do not change the numerical truth of the recorded comparison. No follow-up tests are offered and no DFT run is needed to decide the literal claim."}

## Verdict · gemini-flash

`verdict-002-turn-0006` · gemini-2.5-flash@google · 1951 completion tokens · 11.1 s

{"status": "supported", "evidence_ids": ["ev-0170b0f63d871c8acab55668"], "request_dft": false, "proposed_test_ids": [], "rationale": "The evidence record ev-0170b0f63d871c8acab55668 directly provides the static energy per atom for both specified models. For the alpha model (Zn3(HCOO)6-alpha-COD4515120-static), the value is -6.2119718751 eV/atom. For the Pbcn model (Zn(HCOO)2-Pbcn-COD7205949-static), the value is -6.2119347767 eV/atom. A direct numerical comparison confirms that -6.2119718751 eV/atom is lower than -6.2119347767 eV/atom. The claim is explicitly scoped to 'the two recorded static calculations' and 'computed energies from these two finite meshes,' which is precisely what the provided evidence addresses. Therefore, the claim is numerically supported. As Claude-sonnet and Gemini-flash correctly highlight, this support is strictly for the literal interpretation of the claim and does not imply a robust physical stability ranking, given the extremely small energy difference (0.037 meV/atom) and the explicit interpretation limit in the evidence record itself."}

