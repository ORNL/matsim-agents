# cantor-sampled-ranking-adequacy

```text
Claim (energy_ranking) about Cantor FCC sampled configurations: The available calculations establish a physically reliable relaxed-configuration energy ranking among the four sampled Cantor FCC arrangements.
Decision criterion: {"quantity":"energy_per_atom","unit":"eV/atom","target_structure":"cantor-ideal-static-MATS0080","threshold":null,"comparison_structure":"cantor-random1-static-MATS0081"}

Scope sufficient: False; scope: Only two converged static single points exist; both associated relaxations fail per-ionic-step SCF qualification, random1 static has 0.466 eV/A residual force, the static energy difference is only 0.192 meV/atom, one fresh static solution changes the total moment, and random2/random3 have no qualified static results. One 2x2x2 mesh and one initial magnetic configuration do not bound numerical or magnetic uncertainty. This is not a global stability claim.
Available follow-up tests: {"scope_gap": "If ranking is retained, qualify ionic relaxations for the compared arrangements, then perform matched denser-k-mesh statics on the closest qualified pair and one alternative collinear magnetic initialization. Do not broaden beyond the four sampled configurations without a separate scientific design."}

Evidence records:
[
 {
  "recorded_at": "2026-11-06T16:41:07.853096+00:00",
  "record_type": "evidence",
  "evidence_id": "ev-54ffb0964eab50c0a8461805",
  "kind": "characterization",
  "subject": "Cantor FCC 108-atom exact ideal configuration MATS-0080",
  "campaign_id": "e2-cantor-fcc-v1",
  "characterization_contract_id": null,
  "calculation_id": null,
  "task_id": "cantor-sampled-ranking-adequacy:ideal-static",
  "run_id": null,
  "method": {
   "backend": "vasp",
   "backend_version": "6.6.1",
   "model": null,
   "model_sha256": null,
   "method_signature": "vasp6.6.1-pbe-spin-encut480-smear0.10-gamma2x2x2-fcc-v1",
   "settings": {
    "functional": "PBE",
    "paw": [
     "Co",
     "Cr_pv",
     "Fe_pv",
     "Mn_pv",
     "Ni_pv"
    ],
    "encut_eV": 480,
    "prec": "Accurate",
    "ediff_eV": 1e-06,
    "nelm": 200,
    "ispin": 2,
    "magmom_initial": "22*2.0 22*3.0 22*3.0 22*3.0 20*1.0 muB",
    "ismear": 1,
    "sigma_eV": 0.1,
    "kpoints": "Gamma 2x2x2",
    "kpar": 8,
    "relaxation": {
     "ibrion": 2,
     "isif": 3,
     "nsw": 200,
     "ediffg_eV_A": -0.02
    },
    "static": {
     "ibrion": -1,
     "nsw": 0,
     "same_geometry_as_relax_output": true
    },
    "software_build": "VASP 6.6.1 17Jul2026 (build Oct 15 2026 18:33:52) complex",
    "source_bundle": "e2-dft-v1-20261104, PBS 9040206",
    "case_json_sha256": "198f80240ec4fe68145749d014f586252698a82387527c3c3eb2c0bbbe2a35ce",
    "verify_static_json_sha256": "8d47c9ba3f5c5ddc8c79f85ca9ba352914ff6ae53dc1020d5cb9b00c39bfc36c",
    "verify_relax_json_sha256": "a481b0500fb3f1012cb07f8a7b70991f29b964fad3577c25a9c6d462d9d70acb",
    "calculation": "cases/fcc_hea_CrMnFeCoNi_Cantor_108_ideal/static",
    "static_poscar_sha256": "b13cbaa6aaf6d3ef1aa6656720daf930cfc24bc67d6aad63d9d48efb5e6daf80",
    "static_outcar_sha256": "c2cef81a1887303980b2a129c4f1e86d9d9baf528b6fdf8f3f3fe105f7ef8011",
    "static_oszicar_sha256": "224711205f22b363107aa6cfd2396c481f39f9339f7f7c366390b774afb0f0de",
    "static_vasprun_sha256": "2d6c5e52cc8b0fafc4c08e60d858d36ee0707e2f1b7bc09680cfa18aed0feb91",
    "static_potcar_sha256": "c0e6f0e3856534892c239c7a8ec155102fdb9ee703a10e88b94d0c13f7ae4918",
    "matched_relax_contcar_sha256": "b13cbaa6aaf6d3ef1aa6656720daf930cfc24bc67d6aad63d9d48efb5e6daf80",
    "preceding_relaxation": "not_qualified (SCF not converged on ionic steps 2,3,4,5,12 of 25)",
    "static_wavecar_bytes": 0,
    "static_scf_iterations": 64
   },
   "software_versions": {}
  },
  "evidence_level": "converged_dft",
  "outcome": "measured",
  "values": [
   {
    "name": "max_force",
    "value": 0.012182336311233572,
    "unit": "eV/A",
    "structure": "cantor-ideal-static-MATS0080"
   },
   {
    "name": "energy_per_atom",
    "value": -7.828097256944444,
    "unit": "eV/atom",
    "structure": "cantor-ideal-static-MATS0080"
   },
   {
    "name": "total_magnetization",
    "value": 63.4263,
    "unit": "muB",
    "structure": "cantor-ideal-static-MATS0080"
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
 },
 {
  "recorded_at": "2026-11-06T16:41:07.898646+00:00",
  "record_type": "evidence",
  "evidence_id": "ev-47538e8e8ffb560efba09b7b",
  "kind": "relaxation",
  "subject": "Cantor FCC ideal MATS-0080 relaxation",
  "campaign_id": "e2-cantor-fcc-v1",
  "characterization_contract_id": null,
  "calculation_id": null,
  "task_id": "cantor-sampled-ranking-adequacy:ideal-relax",
  "run_id": null,
  "method": {
   "backend": "vasp",
   "backend_version": "6.6.1",
   "model": null,
   "model_sha256": null,
   "method_signature": "vasp6.6.1-pbe-spin-encut480-smear0.10-gamma2x2x2-fcc-v1",
   "settings": {
    "functional": "PBE",
    "paw": [
     "Co",
     "Cr_pv",
     "Fe_pv",
     "Mn_pv",
     "Ni_pv"
    ],
    "encut_eV": 480,
    "prec": "Accurate",
    "ediff_eV": 1e-06,
    "nelm": 200,
    "ispin": 2,
    "magmom_initial": "22*2.0 22*3.0 22*3.0 22*3.0 20*1.0 muB",
    "ismear": 1,
    "sigma_eV": 0.1,
    "kpoints": "Gamma 2x2x2",
    "kpar": 8,
    "relaxation": {
     "ibrion": 2,
     "isif": 3,
     "nsw": 200,
     "ediffg_eV_A": -0.02
    },
    "static": {
     "ibrion": -1,
     "nsw": 0,
     "same_geometry_as_relax_output": true
    },
    "software_build": "VASP 6.6.1 17Jul2026 (build Oct 15 2026 18:33:52) complex",
    "source_bundle": "e2-dft-v1-20261104, PBS 9040206",
    "calculation": "cases/fcc_hea_CrMnFeCoNi_Cantor_108_ideal/relax",
    "outcar_sha256": "b77dc45cf98dedd5d75ce63dd72514c79df6cdbb7e81f663fe42cf7eb9a47505",
    "unconverged_ionic_steps": [
     2,
     3,
     4,
     5,
     12
    ],
    "final_max_force_eV_A": 0.012477882953450077,
    "ionic_accuracy_marker": "present"
   },
   "software_versions": {}
  },
  "evidence_level": "converged_dft",
  "outcome": "not_converged",
  "values": [
   {
    "name": "max_force",
    "value": 0.012477882953450077,
    "unit": "eV/A",
    "structure": "cantor-ideal-relax-MATS0080"
   }
  ],
  "numerical_convergence": {},
  "reference_set_id": null,
  "reference_complete": null,
  "input_sha256": {},
  "output_artifact_sha256": {},
  "physical_assumptions": [],
  "uncertainties": {},
  "failure_reason": "Five ionic SCF cycles reached NELM=200; relaxation is not qualified despite normal exit, marker, and final force.",
  "parent_record_ids": [],
  "cost": null
 },
 {
  "recorded_at": "2026-11-06T16:41:07.960266+00:00",
  "record_type": "evidence",
  "evidence_id": "ev-9c17747679a6fc7e543fa258",
  "kind": "characterization",
  "subject": "Cantor FCC 108-atom exact random1 configuration MATS-0081",
  "campaign_id": "e2-cantor-fcc-v1",
  "characterization_contract_id": null,
  "calculation_id": null,
  "task_id": "cantor-sampled-ranking-adequacy:random1-static",
  "run_id": null,
  "method": {
   "backend": "vasp",
   "backend_version": "6.6.1",
   "model": null,
   "model_sha256": null,
   "method_signature": "vasp6.6.1-pbe-spin-encut480-smear0.10-gamma2x2x2-fcc-v1",
   "settings": {
    "functional": "PBE",
    "paw": [
     "Co",
     "Cr_pv",
     "Fe_pv",
     "Mn_pv",
     "Ni_pv"
    ],
    "encut_eV": 480,
    "prec": "Accurate",
    "ediff_eV": 1e-06,
    "nelm": 200,
    "ispin": 2,
    "magmom_initial": "22*2.0 22*3.0 22*3.0 22*3.0 20*1.0 muB",
    "ismear": 1,
    "sigma_eV": 0.1,
    "kpoints": "Gamma 2x2x2",
    "kpar": 8,
    "relaxation": {
     "ibrion": 2,
     "isif": 3,
     "nsw": 200,
     "ediffg_eV_A": -0.02
    },
    "static": {
     "ibrion": -1,
     "nsw": 0,
     "same_geometry_as_relax_output": true
    },
    "software_build": "VASP 6.6.1 17Jul2026 (build Oct 15 2026 18:33:52) complex",
    "source_bundle": "e2-dft-v1-20261104, PBS 9040206",
    "case_json_sha256": "fbcf16a937b8ff4f9d2228f99a027c13ba272c38d19c2a569285738103f53a31",
    "verify_static_json_sha256": "89de6186088995d2088a180a61c1e4af8ab757d132b8f2449049e77e4672baca",
    "verify_relax_json_sha256": "39e4dfc0770753f59d2b2e2ccdd309b59a7552b146945658aebd3ff060a804c4",
    "calculation": "cases/fcc_hea_CrMnFeCoNi_Cantor_108_random1/static",
    "static_poscar_sha256": "d4c95d796fb7a96c93559ebc538e2afcbaba1fd62f835895bcf0dc96f33c9320",
    "static_outcar_sha256": "3c3181cecbca9adb2af2630f936ebf6720290aa49b2b25ca1a55b705121c9013",
    "static_oszicar_sha256": "63e90931821b9afc052f8522b904a2dba4ae046efb7189c4db7ec2dba8a685d0",
    "static_vasprun_sha256": "31277b68807e51bfe336cf531399e6aa20c698439b09739bb0c8fbdb0481e924",
    "static_potcar_sha256": "c0e6f0e3856534892c239c7a8ec155102fdb9ee703a10e88b94d0c13f7ae4918",
    "matched_relax_contcar_sha256": "d4c95d796fb7a96c93559ebc538e2afcbaba1fd62f835895bcf0dc96f33c9320",
    "preceding_relaxation": "not_qualified (SCF not converged on ionic steps 4,5,6,7 of 37)",
    "static_wavecar_bytes": 0,
    "static_scf_iterations": 105,
    "magnetic_note": "static converged to 63.7847 muB vs relaxation final 62.2226 muB from the same MAGMOM start; separate electronic solution is possible"
   },
   "software_versions": {}
  },
  "evidence_level": "converged_dft",
  "outcome": "measured",
  "values": [
   {
    "name": "max_force",
    "value": 0.46587408264250973,
    "unit": "eV/A",
    "structure": "cantor-random1-static-MATS0081"
   },
   {
    "name": "energy_per_atom",
    "value": -7.827904987314815,
    "unit": "eV/atom",
    "structure": "cantor-random1-static-MATS0081"
   },
   {
    "name": "total_magnetization",
    "value": 63.7847,
    "unit": "muB",
    "structure": "cantor-random1-static-MATS0081"
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
 },
 {
  "recorded_at": "2026-11-06T16:41:08.032121+00:00",
  "record_type": "evidence",
  "evidence_id": "ev-a6ae516463066bcf748458cd",
  "kind": "relaxation",
  "subject": "Cantor FCC random1 MATS-0081 relaxation",
  "campaign_id": "e2-cantor-fcc-v1",
  "characterization_contract_id": null,
  "calculation_id": null,
  "task_id": "cantor-sampled-ranking-adequacy:random1-relax",
  "run_id": null,
  "method": {
   "backend": "vasp",
   "backend_version": "6.6.1",
   "model": null,
   "model_sha256": null,
   "method_signature": "vasp6.6.1-pbe-spin-encut480-smear0.10-gamma2x2x2-fcc-v1",
   "settings": {
    "functional": "PBE",
    "paw": [
     "Co",
     "Cr_pv",
     "Fe_pv",
     "Mn_pv",
     "Ni_pv"
    ],
    "encut_eV": 480,
    "prec": "Accurate",
    "ediff_eV": 1e-06,
    "nelm": 200,
    "ispin": 2,
    "magmom_initial": "22*2.0 22*3.0 22*3.0 22*3.0 20*1.0 muB",
    "ismear": 1,
    "sigma_eV": 0.1,
    "kpoints": "Gamma 2x2x2",
    "kpar": 8,
    "relaxation": {
     "ibrion": 2,
     "isif": 3,
     "nsw": 200,
     "ediffg_eV_A": -0.02
    },
    "static": {
     "ibrion": -1,
     "nsw": 0,
     "same_geometry_as_relax_output": true
    },
    "software_build": "VASP 6.6.1 17Jul2026 (build Oct 15 2026 18:33:52) complex",
    "source_bundle": "e2-dft-v1-20261104, PBS 9040206",
    "calculation": "cases/fcc_hea_CrMnFeCoNi_Cantor_108_random1/relax",
    "outcar_sha256": "9c1cae13a7586ff01d0da2d6b7b4efa16b927a26cc47d7a2e5f569cff2798993",
    "unconverged_ionic_steps": [
     4,
     5,
     6,
     7
    ],
    "final_max_force_eV_A": 0.01783516834795792,
    "ionic_accuracy_marker": "present"
   },
   "software_versions": {}
  },
  "evidence_level": "converged_dft",
  "outcome": "not_converged",
  "values": [
   {
    "name": "max_force",
    "value": 0.01783516834795792,
    "unit": "eV/A",
    "structure": "cantor-random1-relax-MATS0081"
   }
  ],
  "numerical_convergence": {},
  "reference_set_id": null,
  "reference_complete": null,
  "input_sha256": {},
  "output_artifact_sha256": {},
  "physical_assumptions": [],
  "uncertainties": {},
  "failure_reason": "Four ionic SCF cycles reached NELM=200; relaxation is not qualified despite normal exit, marker, and final force.",
  "parent_record_ids": [],
  "cost": null
 },
 {
  "recorded_at": "2026-11-06T16:41:08.102437+00:00",
  "record_type": "evidence",
  "evidence_id": "ev-b5f9de910d86bf461b725bcf",
  "kind": "relaxation",
  "subject": "Cantor FCC random2 MATS-0082 relaxation",
  "campaign_id": "e2-cantor-fcc-v1",
  "characterization_contract_id": null,
  "calculation_id": null,
  "task_id": "cantor-sampled-ranking-adequacy:random2-relax",
  "run_id": null,
  "method": {
   "backend": "vasp",
   "backend_version": "6.6.1",
   "model": null,
   "model_sha256": null,
   "method_signature": "vasp6.6.1-pbe-spin-encut480-smear0.10-gamma2x2x2-fcc-v1",
   "settings": {
    "functional": "PBE",
    "paw": [
     "Co",
     "Cr_pv",
     "Fe_pv",
     "Mn_pv",
     "Ni_pv"
    ],
    "encut_eV": 480,
    "prec": "Accurate",
    "ediff_eV": 1e-06,
    "nelm": 200,
    "ispin": 2,
    "magmom_initial": "22*2.0 22*3.0 22*3.0 22*3.0 20*1.0 muB",
    "ismear": 1,
    "sigma_eV": 0.1,
    "kpoints": "Gamma 2x2x2",
    "kpar": 8,
    "relaxation": {
     "ibrion": 2,
     "isif": 3,
     "nsw": 200,
     "ediffg_eV_A": -0.02
    },
    "static": {
     "ibrion": -1,
     "nsw": 0,
     "same_geometry_as_relax_output": true
    },
    "software_build": "VASP 6.6.1 17Jul2026 (build Oct 15 2026 18:33:52) complex",
    "source_bundle": "e2-dft-v1-20261104, PBS 9040206",
    "calculation": "cases/fcc_hea_CrMnFeCoNi_Cantor_108_random2/relax",
    "case_json_sha256": "4232d66a9717bcd2b3d22bc6e9428209ad6b70fac8a1d6fa19b330e3eba704c2",
    "incar_sha256": "e6d1647e5f947dd0aedcd7b8b53d25135fddf81229b6a8c0f98ae032f3264a52",
    "poscar_sha256": "351648a90e89e10975bc1e9763b3549191178b058b69abf1319a14c75da340f9",
    "outcar_sha256": "08f9ad147b042858fcae6dc33bee79cf70de755f9ef487ce4913d748c10197fd",
    "oszicar_sha256": "f749b296cf52202fe13e88d2ecd8a6568d007585bf32370a23392805739d942f",
    "ionic_steps_recorded": 21,
    "final_recorded_force_eV_A": 0.169,
    "scf_capped_ionic_steps": 9
   },
   "software_versions": {}
  },
  "evidence_level": "converged_dft",
  "outcome": "not_converged",
  "values": [],
  "numerical_convergence": {},
  "reference_set_id": null,
  "reference_complete": null,
  "input_sha256": {},
  "output_artifact_sha256": {},
  "physical_assumptions": [],
  "uncertainties": {},
  "failure_reason": "PBS walltime terminated the relaxation at 21 ionic steps; final recorded force was 0.169 eV/A and nine ionic steps reached NELM=200. No static exists.",
  "parent_record_ids": [],
  "cost": null
 },
 {
  "recorded_at": "2026-11-06T16:41:08.163707+00:00",
  "record_type": "evidence",
  "evidence_id": "ev-bea28ffbd28caf685397e4a9",
  "kind": "relaxation",
  "subject": "Cantor FCC random3 MATS-0083 relaxation",
  "campaign_id": "e2-cantor-fcc-v1",
  "characterization_contract_id": null,
  "calculation_id": null,
  "task_id": "cantor-sampled-ranking-adequacy:random3-relax",
  "run_id": null,
  "method": {
   "backend": "vasp",
   "backend_version": "6.6.1",
   "model": null,
   "model_sha256": null,
   "method_signature": "vasp6.6.1-pbe-spin-encut480-smear0.10-gamma2x2x2-fcc-v1",
   "settings": {
    "functional": "PBE",
    "paw": [
     "Co",
     "Cr_pv",
     "Fe_pv",
     "Mn_pv",
     "Ni_pv"
    ],
    "encut_eV": 480,
    "prec": "Accurate",
    "ediff_eV": 1e-06,
    "nelm": 200,
    "ispin": 2,
    "magmom_initial": "22*2.0 22*3.0 22*3.0 22*3.0 20*1.0 muB",
    "ismear": 1,
    "sigma_eV": 0.1,
    "kpoints": "Gamma 2x2x2",
    "kpar": 8,
    "relaxation": {
     "ibrion": 2,
     "isif": 3,
     "nsw": 200,
     "ediffg_eV_A": -0.02
    },
    "static": {
     "ibrion": -1,
     "nsw": 0,
     "same_geometry_as_relax_output": true
    },
    "software_build": "VASP 6.6.1 17Jul2026 (build Oct 15 2026 18:33:52) complex",
    "source_bundle": "e2-dft-v1-20261104, PBS 9040206",
    "calculation": "cases/fcc_hea_CrMnFeCoNi_Cantor_108_random3/relax",
    "case_json_sha256": "d963871ed97defedeb1417e664922ac7195477162e8ea08b810d042f13c5dcd2",
    "incar_sha256": "81e6496c4f0a8bb3c3f2a2c8e1b28045ebec6672109206b2dd018e09d7714fd3",
    "poscar_sha256": "bc68f007bdb17d4e2e16302d0f48f27ae1ba8d1e8c1fae26fc0f5c807e310320",
    "outcar_sha256": "cdd83892c4b8fae3dad77ecf54f6c36afa7247ed7f693e426356a4514066a901",
    "oszicar_sha256": "017e436e182ed4c30e1db5c4158b5166b0e06ecd35fd6558440d77b0a4738e63",
    "ionic_steps_recorded": 6,
    "final_recorded_force_eV_A": 0.314,
    "scf_capped_ionic_steps": 4
   },
   "software_versions": {}
  },
  "evidence_level": "converged_dft",
  "outcome": "failed_to_compute",
  "values": [],
  "numerical_convergence": {},
  "reference_set_id": null,
  "reference_complete": null,
  "input_sha256": {},
  "output_artifact_sha256": {},
  "physical_assumptions": [],
  "uncertainties": {},
  "failure_reason": "MPI_Reduce failed after MPIR_CVAR_PROGRESS_TIMEOUT=600 (progress timeout); process received SIGTERM and exited 143. No static exists.",
  "parent_record_ids": [],
  "cost": null
 },
 {
  "recorded_at": "2026-11-06T16:41:08.225190+00:00",
  "record_type": "evidence",
  "evidence_id": "ev-5ba7f0eadd904a0add85056c",
  "kind": "characterization",
  "subject": "Cantor FCC ideal and random1 exact configurations",
  "campaign_id": "e2-cantor-fcc-v1",
  "characterization_contract_id": null,
  "calculation_id": null,
  "task_id": "cantor-sampled-ranking-adequacy:ranking-evidence",
  "run_id": null,
  "method": {
   "backend": "arithmetic",
   "backend_version": null,
   "model": null,
   "model_sha256": null,
   "method_signature": "static-energy-per-atom-difference-from-verified-OUTCAR",
   "settings": {
    "formula": "E0_sigma0_per_atom(ideal) - E0_sigma0_per_atom(random1)",
    "ideal_eV_per_atom": -7.828097256944444,
    "random1_eV_per_atom": -7.827904987314815,
    "ideal_minus_random1_meV_per_atom": -0.192269629629,
    "static_outcar_sha256": [
     "c2cef81a1887303980b2a129c4f1e86d9d9baf528b6fdf8f3f3fe105f7ef8011",
     "3c3181cecbca9adb2af2630f936ebf6720290aa49b2b25ca1a55b705121c9013"
    ],
    "interpretation": "Exact finite-mesh static ordering only; not a qualified relaxed-configuration ranking."
   },
   "software_versions": {}
  },
  "evidence_level": "converged_dft",
  "outcome": "measured",
  "values": [
   {
    "name": "energy_per_atom",
    "value": -7.828097256944444,
    "unit": "eV/atom",
    "structure": "cantor-ideal-static-MATS0080"
   },
   {
    "name": "energy_per_atom",
    "value": -7.827904987314815,
    "unit": "eV/atom",
    "structure": "cantor-random1-static-MATS0081"
   },
   {
    "name": "final_max_force",
    "value": 0.012182336311233572,
    "unit": "eV/A",
    "structure": "cantor-ideal-static-MATS0080"
   },
   {
    "name": "final_max_force",
    "value": 0.46587408264250973,
    "unit": "eV/A",
    "structure": "cantor-random1-static-MATS0081"
   }
  ],
  "numerical_convergence": {},
  "reference_set_id": null,
  "reference_complete": false,
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
