# Radiation-damage extension: implementation status

This page separates what is implemented and unit-tested from what is scientifically established.
Nothing here is numerically validated: no production DFT or MLIP result, no validated SPECTRA-PKA
result (one software smoke run only), and no kinetic prediction exists. Unit tests establish software behaviour only.

## Qualification levels

| Level | Meaning |
| --- | --- |
| `IMPLEMENTED_TESTED` | Code exists and passes synthetic unit tests. |
| `INPUT_BLOCKED` | Code exists but required external inputs are missing. |
| `EXTERNAL_UNVERIFIED` | Interface to external software; no real run made. |
| `EXTERNAL_SMOKE_RUN` | External software installed and run once end to end; output not validated. |
| `NOT_ESTABLISHED` | No scientific claim is made. |

## Items

| # | Item | Module | Level | Caveat |
| --- | --- | --- | --- | --- |
| 1 | W to Re/Os substitutional transmutation | `discovery/transmutation.py` | `IMPLEMENTED_TESTED` | Counts are scenario inputs, not predicted compositions. |
| 2 | Time-dependent inventory ingestion | `discovery/transmutation_source.py` | `IMPLEMENTED_TESTED` | Reads external inventories; does not compute them. |
| 3 | `V_n He_m H_p` complexes | `discovery/gas_defects.py` | `IMPLEMENTED_TESTED` | Geometry only; nothing relaxed. |
| 4 | Species, spin and MLIP capability gate | `discovery/species_support.py` | `IMPLEMENTED_TESTED` | Re/Os/H/He are `MLIP_UNSUPPORTED` or undeclared. |
| 5 | Binding and insertion energy contracts | `discovery/gas_energetics.py` | `IMPLEMENTED_TESTED` | No energies computed. |
| 6 | Mechanism registry and NEB preparation | `kinetics/gas_mechanisms.py` | `IMPLEMENTED_TESTED` | No barriers computed. |
| 7 | Cluster-dynamics bridge | `kinetics/cluster_dynamics.py` | `IMPLEMENTED_TESTED` | Bubble physics rejected as future work; bubble kinetics `NOT_ESTABLISHED`. |
| 8 | Transfer-driven gas acquisition | `campaign/gas_transfer.py` | `IMPLEMENTED_TESTED` | Not shown to beat random selection on any real data. |
| 9 | 1,024-site QE campaign | `campaign/gas_qe_campaign.py` | `IMPLEMENTED_TESTED` | Re/Os/H/He PSlibrary 1.0.0 PAW files are now in the Frontier cache, checksummed, with headers inspected (see `SOURCE.txt` there). Cutoffs are not convergence-tested (the Os suggestion of 53.8 Ry exceeds the campaign `ecutwfc`). No QE run with them yet. Own dispatch, not `slurm_jobs`. |
| 10 | External SPECTRA-PKA adapter | `integrations/external/spectra_pka/` | `EXTERNAL_SMOKE_RUN` | Built on Frontier; PKA data archive (`pkas.zip`) cached, extracted per campaign. One real run (W-182, upstream PWR test spectrum) gave `STRUCTURALLY_VALID`, after the `columns=` line and parser layout were fixed. Not validated against a reference. |
| 11 | PKA-informed ranking | `campaign/pka_acquisition.py` | `IMPLEMENTED_TESTED` | Policy only; no per-PKA jobs. |
| 12 | Surviving-defect production | `irradiation/contracts.py` | `NOT_ESTABLISHED` | Default `NoCascadeModel`; a PKA is not a surviving defect. |
| 12a | Wigner-Seitz defect analysis, clustering, surviving-defect counts, weighted ensemble statistics (WP3.2, 3.5, 3.3 descriptors, 3.4 records, 3.6 bookkeeping, 3.9 source terms, 3.10 surrogate (synthetic data only), most of 3.7) | `defects/{wigner_seitz,survival,ensemble,contract_link,tracking,local_structure,records,source_terms}.py` | `IMPLEMENTED_TESTED` | Synthetic-structure and synthetic-sample tests only (counts agree with `defects/mapping.py`; interval coverage checked on simulated independent samples). No cascade data and no reference comparison, so the statistics describe nothing physical yet. Intervals assume independent realizations. `contract_link.py` maps an ensemble to `SurvivingDefects` and returns `ESTABLISHED` only with a stated cascade provenance and at least two realizations; it does not check that provenance, and `NoCascadeModel` stays the default. A known V2He2H1 complex and V1He_n cavities built by `defects/gas.py` are recovered on the host sublattice only: analysing the raw structure shows a gas atom in a vacancy as an antisite and no vacancy, so gas atoms must be removed before counting vacancies (not automated). That is the only WP3.11 known-complex check. `tracking.py` counts, peaks and persistence over a frame sequence of analyses (identity is the reference site or atom index, so a vacancy hop reads as one vanishing and one appearing; the persistence window has no default); it was tested only on constructed frame sequences, not on trajectory data, and does not decide which transients matter. `local_structure.py` gives per-atom coordination, centrosymmetry, fixed-cutoff common neighbour analysis (fcc, hcp, bcc, icosahedral, else `other`) and Voronoi volumes; it was tested on perfect fcc and bcc cells and a single bcc vacancy only. Cutoffs have no defaults, there is no polyhedral template matching or dislocation analysis, and it does not separate thermal disorder from real transformation. `records.py` emits one record per vacancy, interstitial atom and antisite (type, identities, position, species, environment within a stated cutoff, cluster membership, method, ambiguity counts, caller-supplied provenance) with geometric type labels; solute and gas tags come only from caller-supplied species lists, and transmutation-associated and extended defects are not assigned. Tested on constructed bcc W cells only. `source_terms.py` combines a `PKASpectrum` (per-target-atom-per-second basis required) with per-element, per-energy-bin yield estimates into a defects-per-atom-per-second rate with units, per-volume and per-dpa scalings, and a cluster-dynamics `SourceTerm`. Spectrum bins outside the yield bins are an error by default, or excluded or held at the nearest bin with the rate fraction reported; yield statistics are propagated, spectrum uncertainty only if the caller states one (independent per bin), and material or temperature agreement is recorded but not checked. Tested with constructed spectra and yields only. `workflows/radiation/cascade_surrogate.py` (3.10) fits a bootstrap ridge ensemble (existing `RidgeEnsemble` and `OODGate`) on cascade samples of one target, unit and potential, refuses out-of-distribution queries, reports error only on disjoint rows, and ranks candidate cascades by uncertainty per node-hour with a spread penalty. It is linear with no symmetry reduction, so it cannot represent channelling; the acquisition ranking is a policy not shown to beat random selection; it was tested on synthetic linear data only and no cascade data were used. Generation of cascade ensembles (3.8) and use on real cascade data are not done. Not validated. |
| 13 | MLIP NEB with uncertainty-gated QE NEB escalation | `workflows/active_learning/{neb_sampler,loop}.py`, `backends/dft/qe_neb_input.py` | `IMPLEMENTED_TESTED` | See the NEB escalation table below. `neb.x` and `pw.x` each ran once on a tiny Li cell (plumbing only); no physical barrier exists. |

## NEB escalation (WP2) tracker

| Task | Status | Evidence | Numerical validation |
| --- | --- | --- | --- |
| Escalation policy (`mlip_only`, `uncertainty_gated`, `always_dft_neb`) with DFT NEB budget ledger | Implemented | `tests/test_active_learning_neb_escalation.py` | None |
| Missing uncertainty never read as low | Implemented | Test with a huge limit and no uncertainty | None |
| Escalation record: policy, reason, MLIP id | Implemented | Unit tests | None |
| Restart refuses a path from a different MLIP (main and refinement stage) | Implemented | `tests/test_active_learning_neb_sampler.py` | None |
| NEB-image labels marked `dft_single_point`, `label_is_dft_mep=False` | Implemented | Unit test | None |
| QE `neb.x` launch and ingest on Frontier (cce 21.0.4, rocm/7.2.0) | Hardware smoke passed | Job 5998160 | Plumbing only; tiny Li cell, one image group |
| Al vacancy-hop NEB driver (`scripts/neb/physical_neb_vacancy.py`): vc-relax, endpoints, NEB | Ran to the NEB stage; no barrier | Job 5998331 | None. Endpoints relaxed; `neb.x` was killed by the 1500 s driver timeout in path iteration 4 (about 6 min per iteration, 7 images, 8 GPUs). Run state `failed`, no barrier. The intermediate estimates (1.16, 1.09, 0.77 eV) were still moving and are not a result. |

Limitations: a stored DFT NEB workflow directory is refused rather than reused, so no cross-model reuse
exists outside the sampler restart check. No converged physical barrier has been computed, nothing is
physically validated or scientifically qualified. WP1, WP3 and WP4 have software-level pieces only, all tested on constructed data and none run against reference data:
WP1 (`science/cascade/`, `workflows/radiation/cascade_candidates.py`) has recoil and ZBL helpers, evidence qualification, a candidate adapter with an out-of-distribution diagnostic, dynamics with drift measurement, boundary heuristics, a protocol with checkpoint and restart, and a LAMMPS availability stub (no cascade has been run). WP3 (`science/defects/cascade_ensemble.py`) has ensemble statistics and constructed-data verification (3.8, 3.11), plus the earlier items. WP4 has `science/kinetics/event_library.py` and interstitial and solute states in `ClusterState` (4.1, 4.2), `science/kinetics/rate_construction.py` with a detailed-balance check against supplied free energies only (4.3, which also includes `harmonic_tst.py`), `science/defects/cascade_injection.py` and `science/defects/exposure_history.py` (4.4, piecewise-constant flux and temperature, linear flux scaling, no cascade overlap), and `workflows/radiation/transmutation_inventory.py` (4.5, chain declaration, product coverage, gas rates; nuclear-data uncertainty reported as unavailable; no hard-spectrum chain or inventory has been run). WP4.6 has no specification in the available plan text and is not done. Not done: real cascades, threshold-displacement and cascade validation, size sensitivity with real MLIPs, an alloy demonstration, a physical NEB barrier, and OpenMC runs.

## Compute state

* Job 5986052 (128-atom k-point test, meshes 4/6/8, 4 nodes) reported Slurm `COMPLETED`, but all
  three `pw.x` runs ended with exit 139 (segmentation fault) after printing the FFT grid and
  before the first SCF iteration. No energies exist. Each run left a roughly 7 GB core file
  under `runs/qe-kconv-128atom/`. All shared libraries of `pw.x` resolve under the job's module
  stack (`ldd` shows none missing). A `gdb` backtrace of the `k4` core shows the fault in
  `buiol_write_record` (QE's in-memory buffer), called from `save_buffer` in `wfcinit`. The root
  cause is not established; no fix has been tried. The same binary and inputs need a one-node
  reproduction before any change to the sbatch scripts or the QE input.
* Reproduction on one node (128 atoms, 8 ranks, 36 k-points, `disk_io` default): job 5986978
  also exited 139 in `buiol_write_record` under `save_buffer` in `wfcinit`. The same input with
  `disk_io='medium'` (job 5987031) did not crash but made no progress: it printed the first
  SCF iteration header and was cancelled by the 20 minute limit (Slurm `TIMEOUT`), with no
  converged SCF. So `disk_io='medium'` avoids the crash in this test but is not shown to
  work; it is one variable changed, with an unproven outcome.
* Root cause and fix of the `pw.x` segfault (software-verified on one 128-atom debug job; no
  energies, nothing numerically validated). QE was rebuilt from scratch with the pinned
  toolchain `cpe/24.11`, `cce/18.0.1` (at the time, Frontier's default cpe/26.03 / cce 21.0.4
  was not used for QE; QE has since been rebuilt on cce 21.0.4 + rocm/7.2.0, see
  `docs/quantum-espresso-frontier.md`). A standalone reproducer showed that Cray `ftn` (cce 18, 19, 20) segfaults at run
  time on an array assignment inside an OpenMP `WORKSHARE` (orphaned; a rebuilt `pw.x` with
  `PARALLEL WORKSHARE` still crashed in the outlined region, found with a core dump taken in
  job 5988713). `PW/src/buffers.f90` uses that construct in `buiol_write_record` and
  `buiol_read_record`. Replacing both with an explicit-shape `PARALLEL DO` copy
  (`buiol_copy_z`; patch in `deployments/frontier/setup/patches/qe-cce-orphaned-workshare/`,
  applied by `build-qe-gpu-frontier.sh`) removes the crash: job 5988725 (same input, default
  `disk_io`) passed `wfcinit` and printed SCF iteration 1, then reached the 15 minute debug
  limit (`TIMEOUT`) with no converged SCF. No convergence or energy exists yet for the
  128-atom case; the wavefunction start-up alone takes several minutes.
* Slurm `COMPLETED` is not QE success. Launchers must propagate the `srun` exit code and
  check for `convergence has been achieved` and `JOB DONE`.
* See [multiproperty-selection.md](multiproperty-selection.md) for the property, selection,
  and acquisition workflow. All of it is software-verified only.
* The 2x2x2 k-mesh and the 51/650 Ry cutoffs used for the 1,024-atom input are unconverged.
* Debug QoS is not sufficient for production 1,024-atom QE.
* `pristine_ferromagnetic` stays `INPUT_BLOCKED`; gas examples are nonmagnetic (`nspin: 1`).

## Tests

`tests/test_radiation_e2e_synthetic.py` chains alloy, transmutation, gas complex, mock
SPECTRA-PKA, PKA sampling and ranking, and checks composition, determinism and labels only.
Mock-executable tests are not nuclear-physics verification.
