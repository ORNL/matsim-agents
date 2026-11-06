# Scientific debate

Run `debate-2026-11-06T16-45-26Z_71565751`.

**Hypothesis.** Claim (energy_ranking) about Cantor FCC sampled configurations: The available calculations establish a physically reliable relaxed-configuration energy ranking among the four sampled Cantor FCC arrangements.
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

## Argument · claude-sonnet · round 1

`round-001-turn-0001` · claude-sonnet-5-5@google · 1550 completion tokens · 12.7 s

**Position: unresolved. The claim is not supported, and it is not contradicted either.**

**What the evidence establishes**
- Two static single points converged: ideal MATS-0080 at -7.828097 eV/atom and random1 MATS-0081 at -7.827905 eV/atom. The ideal structure is lower by 0.192 meV/atom.
- That is an exact finite-mesh static ordering of two geometries. It is not a ranking of relaxed configurations. The evidence record itself says so: "not a qualified relaxed-configuration ranking."

**Why the claim fails**
1. Both relaxations are not_converged. Ideal had SCF cap hits (NELM=200) on 5 of 25 ionic steps. Random1 had them on 4 of 37 steps. Under the rules, not_converged records are not evidence for or against anything. The static geometries are CONTCARs from these unqualified relaxations, so their provenance is compromised.
2. Random1's static max force is 0.466 eV/Å, far above the 0.02 eV/Å target. The relaxation reported a final force of 0.0178 eV/Å, but the static run on the same geometry gives 0.466 eV/Å. This is a large inconsistency. It suggests the static electronic solution differs from the relaxation's, or that the relaxation's forces were unreliable because of the SCF failures. Either way, random1 is not at a relaxed minimum in the static calculation.
3. The magnitude of the difference is too small. 0.19 meV/atom is about 21 meV per 108-atom cell. The comparison involves a structure with 0.47 eV/Å residual force, a magnetic moment that changed (63.78 muB in the static vs 62.22 muB at the end of the relaxation), a single 2x2x2 Γ mesh, and a single MAGMOM start. Any of these could plausibly shift the energy by more than 0.19 meV/atom. A 0.47 eV/Å force implies a relaxation energy that could be a few meV/atom or more. That is an energy error larger than the gap and, if random1 relaxed, it would likely lower random1's energy, possibly reversing the order.
4. Random2 has no static result, only a walltime-killed relaxation with a final force of 0.169 eV/Å. Random3 failed with an MPI error. Neither contributes evidence. The claim is about all four sampled arrangements, so at most 2 of 4 are covered.
5. The evidence level is DFT only, with no MLIP energy_ranking record. The rule that an energy_ranking claim needs "at least MLIP relaxation evidence of kind energy_ranking or stability" is not clearly met by the admissible records. The records are kind characterization and relaxation, and the relaxations are inadmissible.

**Why not "contradict"**
The static ordering (ideal lower) does not refute the claim. It simply cannot be promoted to a reliable relaxed ranking. A contradiction would require admissible records that disagree, and none exist.

**Assumptions I am exposing**
- I assume the static energy at a 0.47 eV/Å-force geometry is not close to the relaxed energy. That is plausible but not verified here, so I treat the sign of the shift as unknown. It is not necessarily large enough to flip the order.
- The ideal structure's energy is probably closer to its relaxed value, since its force is 0.012 eV/Å. The asymmetry therefore biases the comparison against random1 and would make the true gap smaller, not larger.

**Falsification and follow-up tests**
- Requalify the relaxations. Rerun ideal and random1 with a more robust electronic setup, such as ALGO=All or a smaller mixing parameter, a higher NELM, and WAVECAR restart. The criterion is zero SCF-capped ionic steps and a final force below 0.02 eV/Å in a matched static.
- Then run matched denser-k-mesh statics on the ideal/random1 pair, and one alternative collinear magnetic initialization such as an antiferromagnetic Cr/Mn start. The claim survives only if the sign of the ΔE persists and the magnitude exceeds the spread across k-mesh and magnetic start. If the order flips under either, the claim is falsified for that pair.
- For random2 and random3, complete the relaxations and statics before any four-way ranking is stated. The scope should not be broadened beyond these four configurations.

**Request for DFT:** Yes, because further DFT could decide the ranking for the pair. As things stand, that DFT has not been done.

I look forward to peer challenges, particularly on whether anyone argues that the 0.19 meV/atom ordering is robust.

## Argument · gemini-flash · round 1

`round-001-turn-0002` · gemini-2.5-flash@google · 3885 completion tokens · 23.8 s

**Position: Unresolved.** The available calculations do not establish a physically reliable relaxed-configuration energy ranking among the four sampled Cantor FCC arrangements.

**Evaluation of the Hypothesis**

The hypothesis claims that the available calculations establish a "physically reliable relaxed-configuration energy ranking" among four sampled arrangements. This claim is not supported by the evidence for several critical reasons:

1.  **Lack of Relaxed Configurations:** The core of the claim, "relaxed-configuration," is directly contradicted by the evidence. Both `ev-47538e8e8ffb560efba09b7b` (ideal-relax) and `ev-a6ae516463066bcf748458cd` (random1-relax) are explicitly marked as "not_converged" due to multiple ionic steps reaching the maximum number of SCF iterations (NELM=200). According to the rules, "A failed_to_compute or not_converged record is not evidence for or against a material." Therefore, the geometries used for the static energy calculations are not from qualified relaxed states.

2.  **Unacceptable Residual Forces:** Even if we consider the static calculations on their own, the `random1-static` calculation (`ev-9c17747679a6fc7e543fa258`) reports a `max_force` of 0.466 eV/A. This is an extremely high residual force, indicating that the structure is far from a local energy minimum. Comparing the energy of such a highly unrelaxed structure to another (even if the other, `ideal-static`, has a lower force of 0.012 eV/A) cannot yield a "physically reliable relaxed-configuration energy ranking."

3.  **Insufficient Scope:** The hypothesis refers to "the four sampled Cantor FCC arrangements." However, `random2` (`ev-b5f9de910d86bf461b725bcf`) and `random3` (`ev-bea28ffbd28caf685397e4a9`) relaxations were either terminated by walltime or failed due to an MPI error, respectively. No static energy calculations exist for these two configurations. Therefore, a ranking among all four arrangements is impossible with the current data.

4.  **Marginal Energy Difference and Numerical Uncertainty:** The energy difference between the `ideal` and `random1` static configurations is only 0.192 meV/atom (`ev-5ba7f00eadd904a0add85056c`). This value is well within the typical numerical uncertainty of DFT calculations, especially given the high residual force on `random1`, the limited k-point sampling (2x2x2 Gamma mesh), and the potential for different magnetic solutions. The `random1-static` calculation explicitly notes a change in total magnetization (63.78 muB) compared to the relaxation's final state (62.22 muB), suggesting a different electronic solution that could significantly impact the energy.

5.  **Explicit Disclaimer in Evidence:** The `interpretation` field of `ev-5ba7f0eadd904a0add85056c` explicitly states: "Exact finite-mesh static ordering only; not a qualified relaxed-configuration ranking." This directly undermines the claim.

**Support for Peer Claims**

I **support** `claude-sonnet`'s assessment that the claim is "unresolved." `claude-sonnet` correctly identifies the critical issues of non-converged relaxations, the high residual force on `random1`, the small energy difference being within numerical noise, and the incomplete data for `random2` and `random3`. The observation about the change in magnetic moment for `random1` between relaxation and static is also a crucial point that I agree further compromises the reliability of the energy comparison.

**Assumptions Exposed**

The hypothesis implicitly assumes that:
*   The geometries obtained from the "not_converged" relaxations are sufficiently close to true relaxed configurations for a reliable energy ranking. This is a false assumption given the explicit failure reasons and high residual forces.
*   The single 2x2x2 Gamma k-point mesh and initial magnetic configuration are sufficient to capture the true ground state energies and their relative ordering with meV/atom precision. This is a strong assumption, particularly for magnetic systems, and is not supported by the evidence.
*   The energy difference of 0.192 meV/atom is physically significant and not an artifact of numerical noise or incomplete convergence. This is highly questionable given the context.

**Falsification Tests**

To falsify the claim (or rather, to establish a reliable ranking), the following would be necessary:
1.  **Converged Relaxations:** All four configurations must undergo fully converged ionic relaxations, meaning no SCF failures on any ionic step and final forces below the target threshold (e.g., 0.02 eV/A).
2.  **Robust Static Calculations:** Static calculations on these truly relaxed geometries must be performed with sufficient numerical precision, including k-point convergence tests to ensure the energy differences are stable and significant.
3.  **Magnetic State Exploration:** For each configuration, alternative collinear magnetic initializations (e.g., ferromagnetic, antiferromagnetic, or different initial moments) should be explored to ensure the lowest energy magnetic state is found. If the energy ordering changes under different magnetic states, the claim of a single ranking is challenged.
4.  **Consistency Check:** If, after these steps, the energy ordering between `ideal` and `random1` reverses, or if the energy difference becomes smaller than the established numerical uncertainty, the current implied ranking is falsified. If `random2` or `random3` are found to have lower energies than `ideal`, the claim of a ranking among the four is also falsified.

The current evidence is insufficient to support the claim, leaving it unresolved. Further, more robust DFT calculations are required.

## Argument · gemini-flash · round 2

`round-002-turn-0003` · gemini-2.5-flash@google · 3381 completion tokens · 19.9 s

**Position: Unresolved.** The available calculations do not establish a physically reliable relaxed-configuration energy ranking among the four sampled Cantor FCC arrangements.

**Evaluation of the Hypothesis**

The hypothesis claims that the available calculations provide a "physically reliable relaxed-configuration energy ranking." This claim is not supported by the evidence for several fundamental reasons:

1.  **Lack of Qualified Relaxed Configurations:** The core premise of the claim, "relaxed-configuration," is directly undermined by the evidence. The relaxations for both `cantor-ideal-static-MATS0080` (`ev-47538e8e8ffb560efba09b7b`) and `cantor-random1-static-MATS0081` (`ev-a6ae516463066bcf748458cd`) are explicitly marked as "not_converged" due to multiple ionic steps failing to reach SCF convergence (NELM=200). Per the rules, "A failed_to_compute or not_converged record is not evidence for or against a material." Therefore, the geometries used for the subsequent static energy calculations are not derived from qualified relaxed states.

2.  **Unacceptable Residual Force on `random1`:** The static calculation for `random1` (`ev-9c17747679a6fc7e543fa258`) reports an extremely high `max_force` of 0.466 eV/Å. This value is far above any reasonable threshold for a relaxed structure (typically <0.02 eV/Å). Comparing the energy of such a highly unrelaxed structure to another (even if `ideal-static` has a lower force of 0.012 eV/Å) cannot yield a "physically reliable relaxed-configuration energy ranking." The energy of `random1` would almost certainly decrease significantly upon proper relaxation, potentially altering or even reversing the observed energy ordering.

3.  **Insufficient Scope and Incomplete Data:** The hypothesis refers to "the four sampled Cantor FCC arrangements." However, the relaxations for `random2` (`ev-b5f9de910d86bf461b725bcf`) and `random3` (`ev-bea28ffbd28caf685397e4a9`) were either terminated prematurely or failed entirely, and no static energy calculations exist for these two configurations. Thus, a ranking among all four arrangements is impossible with the current data.

4.  **Marginal Energy Difference and Numerical Uncertainty:** The calculated energy difference between `ideal` and `random1` is only 0.192 meV/atom (`ev-5ba7f0eadd904a0add85056c`). This value is well within the typical numerical uncertainty of DFT calculations, especially considering the high residual force on `random1`, the use of a single 2x2x2 Gamma k-point mesh, and the noted change in total magnetization for `random1` between its relaxation and static calculations (62.22 muB vs 63.78 muB). Such a small difference cannot be considered reliable given these significant computational shortcomings.

5.  **Explicit Disclaimer in Evidence:** The `interpretation` field of `ev-5ba7f0eadd904a0add85056c` explicitly states: "Exact finite-mesh static ordering only; not a qualified relaxed-configuration ranking." This direct statement from the evidence itself confirms that the claim is not met.

**Support for Peer Claims**

I **support** `claude-sonnet`'s and `gemini-flash`'s conclusions that the claim is "unresolved." Both peers accurately identify the critical issues of non-converged relaxations, the unacceptably high residual force on `random1`, the small and potentially unreliable energy difference, and the incomplete coverage of all four configurations. `claude-sonnet`'s point about the asymmetry in relaxation quality biasing the comparison against `random1` is particularly insightful, as a proper relaxation would likely lower `random1`'s energy. `gemini-flash` correctly highlights the implicit assumptions regarding the sufficiency of the current computational parameters for meV/atom precision.

**Assumptions Exposed**

The hypothesis implicitly assumes:
*   That geometries resulting from "not_converged" relaxations are sufficiently close to true relaxed configurations to form a reliable energy ranking. This is demonstrably false given the SCF failures and high residual forces.
*   That the single 2x2x2 Gamma k-point mesh and initial magnetic configuration are sufficient to capture the true ground state energies and their relative ordering with meV/atom precision, especially for magnetic systems. This is a strong and unsupported assumption.
*   That the observed 0.192 meV/atom energy difference is physically significant and not an artifact of numerical noise or incomplete convergence. Given the context, this is highly speculative.

**Falsification Tests**

To establish a physically reliable relaxed-configuration energy ranking, or to falsify the current implied ordering, the following steps are necessary:

1.  **Qualified Ionic Relaxations:** All four configurations must undergo fully converged ionic relaxations, ensuring zero SCF failures on any ionic step and final forces below a stringent threshold (e.g., 0.02 eV/Å or lower). This is the most critical first step.
2.  **Numerical Convergence Studies:** Perform k-point mesh convergence tests (e.g., 4x4x4, 6x6x6 Gamma meshes) on the truly relaxed structures to ensure the energy differences are stable and significant relative to numerical noise.
3.  **Magnetic State Exploration:** For each configuration, explore alternative collinear magnetic initializations (e.g., different initial moments, antiferromagnetic configurations) to confirm that the lowest energy magnetic state has been found. If the energy ordering changes with different magnetic states, the claim of a single ranking is challenged.
4.  **Re-evaluation of Ranking:** If, after these rigorous steps, the energy ordering between `ideal` and `random1` reverses, or if the energy difference becomes smaller than the established numerical uncertainty, the current implied ranking is falsified. Furthermore, if `random2` or `random3` are found to have lower energies than `ideal` or `random1`, the claim of a ranking among the four is also falsified.

The current evidence is insufficient to support the claim, leaving it unresolved. More robust and complete DFT calculations are required.

## Argument · claude-sonnet · round 2

`round-002-turn-0004` · claude-sonnet-5-5@google · 2019 completion tokens · 16.8 s

**Position: unresolved. I agree with both peers on the verdict, but I dispute some of their reasoning.**

**Evidence**
- Two static single points converged: ideal at -7.828097 eV/atom and random1 at -7.827905 eV/atom, a gap of -0.192 meV/atom.
- Both preceding relaxations are `not_converged`, with SCF caps hit on ionic steps (5 of 25 and 4 of 37).
- Random2 was killed by walltime, and random3 failed with an MPI error. Neither has a static result.
- The ranking record itself states "not a qualified relaxed-configuration ranking."
- No admissible energy_ranking or stability record exists. The only records that count are two static characterizations, and they cover two of four configurations.

**Disputes with peers**
1. **claude-sonnet on the size of the random1 force error.** claude-sonnet says a 0.47 eV/Å force implies a relaxation energy of "a few meV/atom or more." I think this is overstated. 0.466 eV/Å is the maximum force on one atom, not an rms force. With a stiffness of order 5–10 eV/Å², one atom at that force gives about 10–20 meV per cell, or roughly 0.1–0.2 meV/atom over 108 atoms. That is the same size as the 0.192 meV/atom gap, so a flip is neither shown nor excluded.
2. **Both peers on the 0.47 eV/Å force.** Both treat it as plain evidence that random1 is "far from a minimum." The CONTCAR hash is identical between relaxation and static, and the force is 0.0178 eV/Å in the relaxation but 0.466 eV/Å in the static. That points at least as much to a different electronic or magnetic solution (63.78 vs 62.22 μB) as to a poor geometry. It may also reflect SCF contamination of the relaxation's forces. The static energy may therefore belong to a different magnetic basin than the one the relaxation followed. This is a separate failure mode from poor relaxation, and it cannot be bounded with a relaxation-energy estimate.
3. **gemini-flash on numerical uncertainty.** gemini-flash says 0.192 meV/atom is "well within typical numerical uncertainty" and that random1 "would almost certainly decrease significantly." Neither statement is quantified in the evidence, and the `uncertainties` fields are empty. I would state it as "unbounded by the records," not "known to be smaller than noise."
4. **claude-sonnet on contradiction.** I agree it is not contradicted, because there are no admissible records that disagree.

**Assumptions and uncertainty**
- Ideal's 0.012 eV/Å force makes its static energy near-relaxed, but not qualified. Its relaxation also had five SCF-capped steps.
- Even if both were fully relaxed, one 2×2×2 Γ mesh with σ = 0.1 eV and a single MAGMOM start (Cr and Mn moments set parallel to Fe/Co) cannot establish a sub-meV ordering in a frustrated magnet.
- Any ordering between ideal and random1 is a statement about finite-mesh static points, not about relaxed configurations.

**Falsification path**
This follows the offered scope_gap test only.
1. Requalify the ideal and random1 relaxations, with no SCF-capped ionic steps and a matched static force below 0.02 eV/Å.
2. Run matched denser-k-mesh statics on that pair and one alternative collinear magnetic initialization.
3. The ordering is retained only if the sign persists and exceeds the spread across k-mesh and magnetic start.
4. Random2 and random3 need qualified relaxations and statics before any four-way ranking is stated.

**Request for DFT:** Yes. DFT could decide the pair ordering, but that DFT has not been done.

## Verdict · claude-sonnet

`verdict-001-turn-0005` · claude-sonnet-5-5@google · 1168 completion tokens · 9.6 s

{"status":"unresolved","evidence_ids":["ev-5ba7f0eadd904a0add85056c","ev-9c17747679a6fc7e543fa258","ev-54ffb0964eab50c0a8461805","ev-47538e8e8ffb560efba09b7b","ev-a6ae516463066bcf748458cd","ev-b5f9de910d86bf461b725bcf","ev-bea28ffbd28caf685397e4a9"],"request_dft":true,"proposed_test_ids":["scope_gap"],"rationale":"My verdict is unresolved. The claim is not supported, and the evidence does not contradict it either.\n\nSupported points:\n- Two static single points converged. Ideal MATS-0080 is at -7.828097 eV/atom and random1 MATS-0081 is at -7.827905 eV/atom, so ideal is lower by 0.192 meV/atom (ev-54ffb0964eab50c0a8461805, ev-9c17747679a6fc7e543fa258, ev-5ba7f0eadd904a0add85056c).\n- That is only a finite-mesh static ordering of two geometries. The ranking record itself says it is \"not a qualified relaxed-configuration ranking.\"\n\nWhy the claim is not supported:\n- Both preceding relaxations are not_converged because of SCF cap hits at NELM=200 (ev-47538e8e8ffb560efba09b7b, ev-a6ae516463066bcf748458cd). Under the rules these records are not evidence either way, so the static geometries have no qualified relaxed provenance.\n- Random2 was killed by walltime and random3 failed with an MPI error (ev-b5f9de910d86bf461b725bcf, ev-bea28ffbd28caf685397e4a9). Neither has a static result, so at most two of the four arrangements are covered.\n- Random1 static has a 0.466 eV/A max force. Its relaxation reported 0.0178 eV/A at an identical geometry hash, and its total moment shifted from 62.22 to 63.78 muB. Peer arguments that this is simply a poor geometry are only partly right. A different electronic or magnetic solution is at least as plausible, and the records do not distinguish the two.\n- The 0.192 meV/atom gap is about 21 meV per cell. That is the same order as plausible errors from the residual force, the single 2x2x2 Gamma mesh with sigma=0.1 eV, and the single MAGMOM start. No uncertainties are recorded.\n- On the dispute between peers over how large the force-related error is: I agree with claude-sonnet's second-round point that it is unbounded by the records. It is neither shown to be larger than the gap nor shown to be smaller. I would not accept gemini-flash's flat statement that the gap is \"well within typical uncertainty\" as established.\n\nWhy not contradicted: no admissible records disagree. The ideal-lower static ordering is not refuted, only unqualified.\n\nDecisive test: the scope_gap option. Requalify the ideal and random1 relaxations with no SCF-capped ionic steps. Then run matched denser-k-mesh statics on that pair plus one alternative collinear magnetic initialization. The pair ordering holds only if its sign persists and exceeds the spread across mesh and magnetic start. Random2 and random3 would need qualified relaxations and statics before any four-way ranking. DFT could decide this, so I request it."}

## Verdict · gemini-flash

`verdict-002-turn-0006` · gemini-2.5-flash@google · 2681 completion tokens · 15.7 s

{"status": "unresolved", "evidence_ids": ["ev-54ffb0964eab50c0a8461805", "ev-47538e8e8ffb560efba09b7b", "ev-9c17747679a6fc7e543fa258", "ev-a6ae516463066bcf748458cd", "ev-b5f9de910d86bf461b725bcf", "ev-bea28ffbd28caf685397e4a9", "ev-5ba7f0eadd904a0add85056c"], "request_dft": true, "proposed_test_ids": ["scope_gap"], "rationale": "The claim that the available calculations establish a 'physically reliable relaxed-configuration energy ranking among the four sampled Cantor FCC arrangements' is unresolved. The evidence is insufficient and contains significant computational flaws.\n\n**Supported Points:**\n*   Only two static single points (ideal MATS-0080 and random1 MATS-0081) have converged results, showing ideal to be lower in energy by 0.192 meV/atom (ev-5ba7f0eadd904a0add85056c). This is an 'exact finite-mesh static ordering only; not a qualified relaxed-configuration ranking' as explicitly stated in the evidence (ev-5ba7f0eadd904a0add85056c).\n*   The relaxations for both ideal (ev-47538e8e8ffb560efba09b7b) and random1 (ev-a6ae516463066bcf748458cd) are marked as 'not_converged' due to multiple ionic steps failing SCF convergence (NELM=200). Per the rules, these records are not admissible evidence for or against the material.\n*   The static calculation for random1 (ev-9c17747679a6fc7e543fa258) has a very high residual maximum force of 0.466 eV/A, indicating it is far from a relaxed state. Its total magnetization also changed significantly from the preceding relaxation (63.78 muB vs 62.22 muB).\n*   The other two configurations, random2 (ev-b5f9de910d86bf461b725bcf) and random3 (ev-bea28ffbd28caf685397e4a9), lack any static energy results due to failed or incomplete relaxations. Therefore, a ranking among all four configurations is impossible with the current data.\n\n**Unresolved Disputes:**\n*   The reliability of the 0.192 meV/atom energy difference is highly questionable given the high residual force on random1, the non-converged relaxations, the single k-point mesh (2x2x2 Gamma), and the change in magnetic moment. It is unclear if this difference is physically significant or within numerical noise.\n*   The impact of the non-converged relaxations on the final static energies and geometries is not quantified. The static energies are based on geometries from these unqualified relaxations.\n\n**Strongest Evidence:**\n*   The explicit statement in `ev-5ba7f0eadd904a0add85056c` that the ordering is "not a qualified relaxed-configuration ranking" directly undermines the claim.\n*   The 'not_converged' status of the relaxations (ev-47538e8e8ffb560efba09b7b, ev-a6ae516463066bcf748458cd) means the geometries are not reliably relaxed.\n*   The high maximum force on random1-static (0.466 eV/A in ev-9c17747679a6fc7e543fa258) indicates that this structure is not at a relaxed minimum, making its energy comparison unreliable for a 'relaxed-configuration' ranking.\n\n**Assumptions:**\n*   The hypothesis implicitly assumes that the geometries obtained from the 'not_converged' relaxations are sufficiently close to true relaxed configurations for a reliable energy ranking. This is not supported by the evidence.\n*   It assumes that the single 2x2x2 Gamma k-point mesh and initial magnetic configuration are sufficient to capture the true ground state energies and their relative ordering with meV/atom precision, which is a strong assumption for magnetic systems.\n\n**Decisive Tests:**\nTo address the identified gaps and potentially resolve the claim, the following tests are necessary:\n1.  Requalify ionic relaxations for the ideal and random1 arrangements, ensuring no SCF-capped ionic steps and final forces below the target threshold (e.g., 0.02 eV/A).\n2.  Perform matched denser-k-mesh static calculations on the closest qualified pair (ideal and random1).\n3.  Perform one alternative collinear magnetic initialization for the qualified pair.\n4.  Complete qualified relaxations and static calculations for random2 and random3.\n\n**Calibrated Verdict:** The claim is unresolved. The current evidence does not provide a physically reliable relaxed-configuration energy ranking for even two of the four sampled arrangements, let alone all four. The computational results are compromised by non-converged relaxations, high residual forces, and incomplete coverage."}

