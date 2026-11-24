# Radiation-damage science: implementation progress

Authoritative task checklist for the Radiation-damage science capability. The source specification was provided externally and is not stored in this workspace; this file is the record of it.

Status: `[ ]` not started, `[~]` in progress, `[x]` complete.

## Mandatory execution rules

- Implement code immediately after the minimal inspection; no further architecture proposals.
- Complete 2-3 microtasks per interaction; resume from the first incomplete task.
- Run focused tests, not the full suite, after each small change.
- At most two targeted corrections for the same unresolved test failure; then record a blocker.
- No HPC jobs, large MLIP models, QE, or NEB runs.
- Do not modify unrelated modules, frozen benchmarks, or archived E3 outputs.
- Keep the existing E3 allowed-element restrictions unchanged.

## Phase A1: Generalized BCC alloy generation

- [x] A1.1 Inspect existing interfaces
  - Files: `src/matsim_agents/discovery/composition.py`, `src/matsim_agents/discovery/seeds.py`, `benchmarks/e3/structure_generation.py`, `tests/test_e3_mlip_structure_generation.py`.
  - Acceptance: identify a minimal approach that preserves E3 behavior.
  - Finding: E3 alloy generation (`_alloy`, `_ALLOY_ELEMENTS`) is self-contained. No `src/` module imports it, except `cli/workflows.py`, which runs the E3 production script as a subprocess. The new generator is a separate module, so E3 is unchanged. `composition.py` supplies `Composition` parsing and `seeds.py` supplies prototypes; neither is needed for the BCC builder.
- [x] A1.2 Implement `AlloySpec`
  - File: `src/matsim_agents/discovery/alloys.py`.
  - Typed Pydantic model: crystal structure, element counts, supercell size, cell parameter, random seed.
  - Validate element symbols, nonnegative integer counts, nonempty composition, exact total atom count.
  - Acceptance: unit tests cover valid and invalid specifications.
- [x] A1.3 Implement BCC supercell generation
  - Periodic conventional cubic BCC cell built with ASE. Default is an 8 x 8 x 8 replication of the two-atom cell (1,024 sites).
  - Acceptance: expected geometry, topology (8 first neighbours), periodicity, and atom count.
- [x] A1.4 Validate composition
  - Assign species to sites with exact integer counts. Reference composition: W 640, Ta 128, Cr 64, V 96, Hf 32, Mo 64.
  - Reject inconsistent compositions.
  - Acceptance: generated structures contain exactly the requested counts.
- [x] A1.5 Add the tungsten example
  - Files under `examples/fusion_damage/` using the 1,024-atom reference composition. No DFT, NEB, or MLIP relaxation.
  - Acceptance: the example generates the correct structure locally.
- [x] A1.6 Implement deterministic random occupations
  - Explicitly seeded RNG assigns species to sites while preserving composition.
  - Acceptance: same seed gives identical occupations and structure data; different seeds give distinct orderings.
- [x] A1.7 Add regression tests
  - Focused tests for the new generator, plus the relevant existing E3 regression tests.
  - Acceptance: new tests and relevant E3 tests pass.

## Phase A2: Chemical short-range order

- [x] A2.1 First-neighbor-shell identification
  - File: `src/matsim_agents/discovery/alloys.py` (`first_neighbor_list`).
  - Periodic directed pairs from ASE `neighbor_list` with a cutoff at the midpoint of the first and second shells; no all-pairs matrix.
  - Acceptance: eight first neighbours per site for small supercells and the 1,024-atom reference; periodic image vectors are body diagonals of length sqrt(3)/2 a.
- [x] A2.2 Second-neighbor-shell identification
  - File: `src/matsim_agents/discovery/alloys.py` (`second_neighbor_list`).
  - Reuses the A2.1 `neighbor_list` call with an outer cutoff at the midpoint of the second and third shells, then keeps pairs at or beyond the first/second midpoint.
  - Acceptance: six second neighbours per site (axis-aligned, length a) for small supercells and the 1,024-atom reference; shell disjoint from A2.1; directed pairs reciprocal.
- [x] A2.3 Warren-Cowley SRO descriptors
  - File: `src/matsim_agents/discovery/alloys.py` (`warren_cowley_sro`, `SROResult`, `ShellSRO`).
  - alpha_ij = 1 - P(j|i)/c_j from the A2.1/A2.2 neighbour lists, with c_j the actual atomic fraction. One species-indexed matrix per shell; species ordering is sorted symbols unless given explicitly.
  - Checks: directed pair counts must be symmetric (reciprocity); absent, unlisted, or duplicate species, non-finite lattice parameter, empty or non-periodic structures raise `ValueError`.
  - Acceptance: ordered B2 binary gives alpha = [[1,-1],[-1,1]] (first shell) and [[-1,1],[1,-1]] (second shell); six-element reference gives two 6 x 6 matrices obeying sum_j c_j alpha_ij = 0 and pair-count row sums z * N_i; random alloys report measured nonzero SRO.
- [x] A2.4 Independent analytical validation of SRO descriptors
  - File: `tests/test_alloys_sro_validation.py` (no change to `alloys.py`).
  - Reference: independent integer half-lattice grid with modular wrap, using no ASE neighbour search; pair counts and alpha must match exactly for four compositions (including the 1,024-atom reference) and three seeds.
  - Layered (001) block-of-two ordering: closed-form first-shell alpha = 0 and second-shell alpha = [[-1/3, 1/3], [1/3, -1/3]] on three cell shapes.
  - Random alloys: exhaustive average over all arrangements on a 16-site cell equals the finite-composition expectation E[alpha_ii] = (N - N_i)/((N - 1) N_i), E[alpha_ij] = -1/(N - 1) for a binary and a ternary composition. Nonzero finite-size bias is asserted to vanish only as the cell grows.
- [x] A2.5 Targeted chemical short-range-order generation
  - Files: `src/matsim_agents/discovery/alloys.py` (`SROTarget`, `generate_targeted_sro`, `TargetedSROResult`), `tests/test_alloys_targeted_sro.py`.
  - Bounded greedy search: swap two sites of different species, keep only strict decreases of L = sum w (alpha - alpha_target)^2 over the explicitly targeted (shell, pair) entries. Sites and exact species counts are fixed; the input structure is not modified.
  - Stops when L <= tolerance or after `max_attempts` swaps. Returns the structure, full measured SRO, objective, initial objective, attempts, accepted swaps, seed, tolerance, and convergence flag.
  - Pair targets are unordered (alpha_ij = alpha_ji); duplicates are merged and conflicts rejected. Rejected: unknown species, alpha outside [1 - 1/max(c_i, c_j), 1], full target rows violating sum_j c_j alpha_ij = 0, non-BCC coordination, non-periodic or single-species structures, invalid seed, attempt limit, or tolerance.
  - Limitation: a finite periodic cell admits only discrete alpha values, so an in-range target may be missed; the result is a bounded-search configuration, not an equilibrium state. Greedy acceptance can stall in a local minimum.
- [x] A2.6 Target SRO feasibility and achieved-ordering validation
  - Files: `src/matsim_agents/discovery/alloys.py` (`InvalidSROTargetError`, `validate_targeted_sro`, `TargetedSROValidation`, `TargetComparison`, `TargetedSROResult.status`), `tests/test_alloys_sro_feasibility.py` (18 tests).
  - Target checks are cheap necessary conditions, not a feasibility proof: finite alpha, supported shell, known species, alpha_ij = alpha_ji when both directions are given, alpha within [1 - 1/max(c_i, c_j), 1] (off-diagonal) or the diagonal bound from finite counts, complete rows obey sum_j c_j alpha_ij = 0 (tolerance 1e-6), partial rows may not already imply conditional probabilities above 1. Unspecified pairs are unconstrained.
  - Violations raise `InvalidSROTargetError` (a `ValueError` subclass) before any search. Status reuses the A2.5 result: `converged` (objective <= tolerance) or `not_converged`; a valid target missed within the budget is never reported as invalid.
  - `validate_targeted_sro` re-measures the returned structure and raises `ValueError` unless counts, site positions, cell and pbc match the input; reported alpha, objective, initial objective and convergence flag match the recomputed values; and a replay from the same seed, tolerance and attempt count reproduces the structure.
  - Limitation: bounds are necessary only, so a target can pass and still be unreachable; reproducibility is checked by replaying the search, so it costs one more run.
- [x] A2.7 Independent configurational ensemble generation
  - Files: `src/matsim_agents/discovery/alloy_ensemble.py` (`AlloyEnsembleSpec`, `generate_alloy_ensemble`, `AlloyEnsemble`, `ConfigurationRecord`, `occupation_fingerprint`), `tests/test_alloy_ensemble.py` (30 tests). `alloys.py` gains `BCCTopology`/`bcc_topology` and an optional `topology` argument on `warren_cowley_sro` and `generate_targeted_sro` so the lattice neighbour shells are built once per ensemble.
  - Modes: `random` (shuffled exact counts) and `targeted_sro` (random start, then the A2.5 optimizer for at most `max_swap_attempts` swaps). Ensemble size is configurable with no upper limit; five is only the intended pilot size.
  - Seeds: `matsim_agents.seeding.stream_seed(seed, "alloy-ensemble", "occupation"|"optimizer", k)` gives each configuration its own 32-bit occupation seed and optimizer seed; global NumPy state is untouched. Same specification gives identical structures and metadata.
  - Per configuration: ID (`<mode>-<index>`), parent `AlloySpec`, mode, seeds, species counts, SHA-256 occupation fingerprint, first- and second-shell pair counts and alpha, and for targeted mode the convergence flag, objective and attempts. `metadata()` is JSON-serializable and holds no coordinates; structures stay in memory.
  - Exact duplicate occupations (equal fingerprints) are reported in `duplicate_groups` and `n_unique`; nothing is retried.
  - Independence diagnostic: `overlap_matrix()` gives the fraction of sites with equal species for every pair, and `overlap_summary()` (also in `metadata()`) reports its min, mean and max against the random expectation sum_i c_i^2. This is descriptive only: it is consistent with independence at the pair level, not proof of it, and it is not expected to match for targeted-SRO members.
  - Limitations: independent seeds do not imply statistically uncorrelated or structurally unique configurations; symmetry equivalence is handled separately in A2.8; targeted-SRO members are bounded-search configurations, not equilibrium structures, and some may not converge. No 1,024-atom targeted-SRO campaign was run; the 1,024-atom test is random mode only.
- [x] A2.8 Duplicate and symmetry-equivalence detection, ensemble SRO statistics
  - Files: `src/matsim_agents/discovery/alloy_ensemble.py`, `tests/test_alloy_ensemble.py`. All functions only report; nothing in the ensemble is modified or regenerated.
  - A2.8a `find_exact_duplicates`: fingerprints filter candidates and occupation arrays confirm each match, so a hash collision cannot merge distinct occupations. The first occurrence is kept.
  - A2.8b `ensemble_sro_statistics`: per-shell mean and sample standard deviation (ddof = 1) of alpha over exact-unique configurations; the standard deviation is `None` for a single configuration.
  - A2.8c `find_translation_equivalents` and `find_symmetry_equivalents`: on the parent BCC supercell, the canonical form of an occupation is its lexicographically smallest image under all periodic translations (including the body-centre shift) combined with every cubic point operation that preserves the supercell (48 for a cube, fewer otherwise). Equal forms mean exact equivalence. Results separate exact, translation and point-group equivalents. Integer first- and second-shell pair counts filter candidates; at most `max_canonicalizations` (default 256) are canonicalized, and any rest are reported in `unchecked` with status `incomplete`. Structures not on the parent supercell raise `ValueError`.
  - Cross-cell `find_cross_cell_equivalents`: crystals in any integer BCC supercell (non-orthogonal included), with any site order, origin or rigid orientation. Each structure is rotated onto the cubic axes from its second-shell vectors and reduced exactly with the integer adjugate of its cell. A symmetry-invariant local-environment signature (species with first- and second-shell neighbour counts per site, O(n_sites)) buckets structures, so only signature matches are compared; distinct configurations cost no pairwise work. Each comparison is exact (at most 48 n_sites^2 lookups) and bounded by `max_comparisons` (default 1,024) with the same `unchecked`/`incomplete` reporting.
  - Acceptance: translated, mirrored and rotated copies are merged while different occupations stay distinct; the 1,024-atom reference has 48 point operations and a rotated plus body-centre-shifted copy is detected in about 0.3 s; sheared, shifted, reordered and arbitrarily tilted copies are matched across cells; six distinct random configurations need zero comparisons.
  - Limitations: chemical species are never permuted; equivalence requires the same lattice parameter and an undistorted lattice (tolerance 1e-4 in units of a/2); a nominated pair still costs up to 48 n_sites^2 lookups.
- [x] A2.8d BCC point-group symmetry equivalence
  - No new code: `find_symmetry_equivalents` (A2.8c) already combines the cached translation permutations with cached site permutations of the 48 signed coordinate permutations, keeps only those valid for the actual supercell, keeps the first configuration as representative, preserves exact and translation classifications, and reports `status`/`unchecked` when its budget stops the check.
  - Acceptance (`tests/test_alloy_ensemble.py`, cubic 2x2x2 Mo8Ta4W4 cell): 90-degree, 180-degree and 120-degree [111] rotations are merged (48 operations, status complete); a mirror, inversion and the x = y mirror are merged; a Mo-Ta site swap and a Ta/W relabelling stay distinct.
  - Scope: ideal BCC supercells only; distorted geometries are not supported.
- [x] A2.9 Phase A2 integration test
  - File: `tests/test_phase_a2_integration.py` (2 tests).
  - Small Mo12Ta8W12 cell: A1 generation with exact composition; first- and second-shell SRO; targeted-SRO optimization lowers the objective and the reported alpha matches re-measurement; a four-member random ensemble is deduplicated (exact and symmetry, status complete); ensemble SRO mean and sample standard deviation match an independent recomputation from re-measured alphas; the whole workflow repeats identically with the same seeds.
  - 1,024-atom W640Ta128Cr64V96Hf32Mo64 reference: exact atom count and composition, eight first and six second neighbours per atom, two finite 6 x 6 SRO matrices, deterministic structure and SRO. No targeted optimization or symmetry search at this size.

Phase A2 is complete.

## Phase A3: Radiation-defect generation

All generators live in `src/matsim_agents/discovery/radiation_defects.py`, with tests in `tests/test_radiation_defects.py`. Every generated structure is an unrelaxed initial candidate: nothing is relaxed, no energies are computed, and no stability is claimed.

- [x] A3.1 Defect data contracts
  - `DefectSpec`, `DefectSite`, `DefectConfiguration`, `DefectOutcome`, `make_defect_config_id`, `geometry_fingerprint`. Parent atoms keep `atom_id` = parent index; inserted atoms get ids >= `n_parent_atoms`; `DefectSite.initial_position` is optional.
- [x] A3.2 Vacancy generation (`generate_vacancy`).
- [x] A3.3 Dumbbell interstitials (`generate_dumbbell_interstitial`)
  - Orientations 111, 110 and 100 taken along the cell vectors; configurable separation; minimum-image overlap rejection.
- [x] A3.4 Additional interstitial candidates
  - A3.4a `generate_bcc_interstitial`: tetrahedral (24 variants) and octahedral (6 variants) sites from crystallographic offsets through the cell vectors; the ideal nearest-atom shell must match or a non-BCC error is raised.
  - A3.4b `generate_crowdion`: periodic <111> row found with minimum-image distances; the inserted atom sits between two row atoms and `n_participating` row atoms move away with a linear taper (default amplitude 0.25 of the row spacing). Participant ids, initial and final coordinates, direction, amplitudes and crowdion length are recorded.
  - Available classes: <111>, <110> and <100> dumbbells, tetrahedral, octahedral and collective <111> crowdion candidates.
  - Limitation: the crowdion pattern is a geometric starting point, not a validated or universally stable structure in the six-component alloy.
- [x] A3.5 Frenkel-pair generation (`generate_frenkel_pair`)
  - The atom at a lattice site moves to the BCC tetrahedral or octahedral position next to an anchor atom (reusing `generate_bcc_interstitial`). It keeps its species, index and `atom_id`; atom count and composition are unchanged. The vacancy and interstitial sites share that `atom_id`; the interstitial records its `initial_position`.
  - The minimum-image vacancy-interstitial separation is recorded and must be at least the configurable `min_separation`; overlaps are rejected using the A3.4a distance check against the unmodified parent (conservative: the moving atom's old position still counts).
- [x] A3.6 Defect environment metadata (`radiation_defect_environments.py`)
  - Vacancy first/second-shell indices and composition from the parent lattice (A2 neighbour lists; other vacated sites are listed separately). Interstitial nearest atoms, distances and coordination from the defective structure with minimum-image distances. Smallest periodic vacancy-interstitial separation. Parent-versus-defective composition change, with inputs checked against the configuration.
  - Limitation: geometric metadata only; no energies, stability or qualification.
- [x] A3.7a Lattice-site assignment (`radiation_defect_mapping.assign_lattice_sites`)
  - Each relaxed atom maps to its nearest parent site by minimum-image distance. Statuses: `on_site`, `off_lattice` (nearest site beyond `site_tolerance`), `ambiguous` (second-nearest site within `ambiguity_margin`), `conflict` (several atoms claim one site; none is merged or dropped). Vacant sites exclude sites made uncertain by ambiguous atoms. Fixed-cell only; mismatched cells, non-periodic or non-finite input are rejected.
  - Limitation: geometric only; thresholds are fractions of the lattice parameter and have not been calibrated against relaxed MLIP or DFT structures.
- [x] A3.7b Defect-site identification (`radiation_defect_mapping.identify_defect_sites`)
  - Reads vacancies (parent position and species), interstitials (relaxed position) and species antisites from a `SiteAssignment`. Ambiguous and conflicting atoms are reported as `unresolved_atoms` with `uncertain_sites`, never as vacancy, interstitial or antisite; `resolved` is False in that case.
- [x] A3.8 Outcome classification (`radiation_defect_classification.classify_defect_outcome`)
  - Emits the existing `DefectOutcome` classes from A3.7 results: `recombined_restored`, `recombined_antisite`, `persisted` (same vacancy sites, same interstitial count, no antisite), `transformed`. Unconverged or unknown relaxation, atom-count or composition mismatch, non-finite coordinates, a changed cell, and ambiguous or conflicting atoms all give `unresolved` with a reason in `notes`.
  - Split and collective defects (configurations with `displaced` sites) that would be `transformed` are `unresolved` instead, because one-atom-per-site mapping cannot represent them.
  - Limitation: a geometric label for one structure, not evidence of stability or of DFT/MLIP qualification; tested on synthetic fixtures only.
- [x] A3.9 Divacancy generation (`generate_divacancy`)
  - Removes two sites, chosen as an explicit `partner` or as the `variant`-th first- or second-shell neighbour (A2 neighbour lists, ordered by index). Records removed species, the minimum-image separation and the shell it falls in (0 for neither). Sites are stored in index order, so swapping the two sites gives the same configuration. Invalid or ambiguous requests raise `ValueError`.
- [x] A3.10 Defect-energy interface (`radiation_defect_energies.radiation_defect_formation_energy`)
  - Thin wrapper over the existing `discovery.defects.defect_formation_energies` (E_f = E_defect - E_host - sum dn_i mu_i). Chemical potentials are supplied explicitly per vertex. Rejected with `ValueError`: unconverged or unknown relaxation, `unresolved` or `recombined_restored` outcomes, an outcome from another configuration or structure, differing host/defect methods, missing or non-finite potentials or energies, and a composition that does not match the configuration. Reports min/max over vertices and whether the defect conserves stoichiometry (vertex-independent).
  - Limitation: tested with synthetic numeric fixtures only. Neutral defects only; magnetism is unhandled (`defects` limitation texts are attached). The numbers carry no DFT/MLIP qualification and no hull-derived reference is computed here.
- [x] A3 integration tests (`tests/test_radiation_defect_pipeline.py`): accepted as complete by the maintainer with partial verification (54-atom cases pass; 1,024-atom cases not run to completion)
  - Runs every generator (vacancy, both divacancy shells, tetrahedral and octahedral interstitial, Frenkel pair, dumbbell, crowdion) on the 54-atom cell and the 1,024-atom reference cell through `defect_environment`, site mapping and `classify_defect_outcome`, plus the energy interface with synthetic energies. Also checks that a dumbbell with a foreign species is `unresolved` and never scored.
  - Status: `-k "54-atom or foreign"` gives 9 passed, 0 failed, 12 deselected (the 8 generator cases on the 54-atom cell and the foreign-species dumbbell test). The 12 deselected tests are the 8 generator cases on the 1,024-atom cell and the 4 energy-interface tests, which also use the 1,024-atom cell. No complete run of the file yet. Parametrized cases are slow on the 1,024-atom cell (about 3 minutes). One run stopped on a wrong test expectation (dumbbell and crowdion are `persisted` for a host-species atom, not `unresolved`), which was then corrected; the following runs were interrupted after 13 passes. The A1 to A3 regression run has not been done.
  - 1,024-atom verification (each test run alone, 180 s timeout): 12 passed, 0 failed, 0 timed out (8 generator cases: vacancy, divacancy shell 1 and 2, tetrahedral, octahedral, Frenkel, dumbbell, crowdion; 4 energy-interface cases: vacancy, divacancy-1, tetrahedral, Frenkel). Wall time 15 to 88 s per test, peak memory about 2.26 GB. The cost is dominated by site mapping and environment building on 1,024 atoms (the first test also pays import and cell construction); no bottleneck was fixed. Not yet done: the A1 to A3 regression run and the Step 3 invariant audit (the 1,023, 1,025 and 1,022 atom counts are asserted directly only for vacancy and interstitial on the 1,024-atom parent; the divacancy 1,022 count and the Frenkel 1,024-atom composition are asserted only on 54- and 16-atom cells and via `n_atoms == len(defective)` at 1,024).
  - Phase A3 is complete as a software deliverable (A3.1 to A3.10 and the integration tests are written and committed). Marked complete at the maintainer's request without a full run of the integration file and without the A1 to A3 regression run; both remain open verification items. The 1,024-atom integration cases are untested. Nothing here establishes DFT or MLIP qualification.

## Phase B: migration pathways and Quantum ESPRESSO NEB

Task definitions are in `radiation-damage-phase-b-spec.md`. Software verification only: no DFT, MLIP, NEB, VASP or HPC execution, and nothing here is a physical qualification.

- [x] B1.1 NEB data contracts (`discovery/neb_contracts.py`)
  - IMPLEMENTED and VERIFIED (synthetic fixtures). Typed records for endpoints (`generated`, `relaxed`, `numerically_qualified`, each needing the evidence the next level implies), identity-based atom correspondence, migration mechanism (always `requires_validation`), image sequence, engine-independent calculation spec, NEB result (`failed`, `incomplete`, `unconverged`, `converged_unqualified`, `qualified`) and the full transition record. Endpoint compatibility checks atom count, identity set, species per identity, parent, ordering, defect type, full periodicity and fixed cell. At least 4 images (QE).
  - Limitation: contracts only; no geometry is generated here.
- [x] B1.2 Migration endpoint generation (`discovery/neb_endpoints.py`)
  - IMPLEMENTED and VERIFIED on 16-atom BCC cells. `generate_vacancy_hop` (nearest-neighbour mover enters the vacancy; minimum-image displacement; `occupant_exchange` recorded from species; new vacancy at the mover's former parent site; hops chain), `generate_interstitial_migration` (explicit caller-given final positions, overlap and no-op rejected, mechanism unvalidated), `generate_frenkel_recombination` (direct return, or exchange with a vacancy neighbour; `restored` vs `exchanged` ordering is computed from parent-site species, never assumed; vacancy and interstitial `atom_id` recorded). Identities preserved, initial structure never modified, endpoints `generated` only.
  - Limitation: candidates only. No relaxation, no saddle knowledge, no interstitial path search.
- [x] B1.3 Correspondence and periodic displacements (`discovery/neb_paths.py`)
  - IMPLEMENTED and VERIFIED on small synthetic cells. `correspond_atoms` matches by `atom_id` (reordered arrays work; species, identity set, count, cell, periodicity, duplicate ids, non-finite coordinates rejected). `periodic_displacements` returns minimum-image vectors, lengths and unwrapped final positions; exact ties (for example half a cell vector) raise unless `allow_ambiguous`, then they are listed. `minimum_image` rounds in the Minkowski-reduced cell and searches 27 images; it matches brute force on a skewed cell.
  - Finding: ASE 3.28 `find_mic` called on single vectors returned non-shortest images on the skewed test cell (86 of 343 vectors) but was exact for the batch, so it is not relied on. `neb_endpoints.py` now uses `minimum_image` too.
- [x] B1.4 Linear periodic image generation (`discovery/neb_images.py`)
  - IMPLEMENTED and VERIFIED on small cells. `generate_linear_images(initial, final, n_images)` interpolates in identity-matched minimum-image displacement space (cell-boundary crossings take the short route; reordered final arrays give identical images), keeps both endpoints fixed (image 0 is the initial structure; the last image equals the final structure modulo lattice vectors), preserves order, `atom_id` and species, and is deterministic with per-image fingerprints. Rejected: fewer than 4 images or non-integer counts, species/identity/cell/periodicity errors, non-finite coordinates, ambiguous displacement (unless allowed), an overlap below `min_distance` in any image, and a per-image step above `max_step`.
  - Limitation: a linear path is only an initial guess; it is not a barrier or a mechanism. The default `min_distance` (1.0 A) and `max_step` (1.0 A) are guards, not physical criteria.

## Progress log

- A1.1: inspection complete; no E3 change required.
- A1.2 to A1.4, A1.6: implemented in `src/matsim_agents/discovery/alloys.py`.
- A1.7: `tests/test_alloys.py` (20 tests) and `tests/test_e3_mlip_structure_generation.py` pass together (39 passed). Ruff check and format pass for the new files.
- A1.5: `examples/fusion_damage/` runs locally and writes 1,024 atoms with the exact reference counts.
- Scope check: `benchmarks/`, `runs/`, and existing `src/` and `tests/` files are unchanged (git status).
- Not done: targeted SRO, SQS-like and local-clustering orderings (out of scope for A1; `OrderingMode` only admits `random`).
- A2.1: added `first_neighbor_list` in `alloys.py` (ASE `neighbor_list`, cutoff between shells). `tests/test_alloys.py`: 24 passed, including 3 new first-shell tests (1x1x1, 2x2x2, 3x2x4 supercells, and the 1,024-atom reference). Ruff check and format pass.
- A2.2: added `second_neighbor_list` in `alloys.py`. `tests/test_alloys.py`: 35 passed, including 3 new second-shell tests (1x1x1, 2x2x2, 3x2x4 supercells) and a reference-structure test for disjointness from the first shell and reciprocity. Ruff check and format pass.
- A2.3: added `warren_cowley_sro` in `alloys.py`. `tests/test_alloys.py` (45 tests) and `tests/test_e3_mlip_structure_generation.py` pass together (64 passed). Ruff check and format pass.
- A2.4: added `tests/test_alloys_sro_validation.py` (18 tests). With `tests/test_alloys.py`: 63 passed in about 18 s; the two exhaustive-average cases take about 16 s. Ruff check and format pass.
- A2.4 fix: the exhaustive-average compositions are now `{Mo: 14, W: 2}` and `{Mo: 1, Ta: 1, W: 14}` (120 and 240 arrangements instead of 560 and 1,680). With `tests/test_alloys.py`: 63 passed in 4.2 s. Ruff check and format pass.
- A2.5: added `generate_targeted_sro` in `alloys.py` and `tests/test_alloys_targeted_sro.py` (21 tests). One test-only correction (a 2x2x1 cell has 8 sites, not 16). With `tests/test_alloys.py`, `tests/test_alloys_sro_validation.py` and `tests/test_e3_mlip_structure_generation.py`: 103 passed. Ruff check and format pass on the changed files.
- CI fix: the full-suite job failed on `tests/test_seeding.py::test_no_random_generator_or_seed_outside_the_seeding_module`, which forbids generators outside `matsim_agents.seeding`. `alloys.py` now uses `numpy_rng` and `alloy_ensemble.py` uses `stream_seed` instead of `np.random.default_rng` and `SeedSequence`. Seeded outputs differ from earlier commits.
- Seed consistency: `AlloySpec.seed` is now required (`StrictInt`, >= 0; no default of 0), matching `AlloyEnsembleSpec.seed` and `generate_targeted_sro(seed=...)`. Each function draws from a named stream: `generate_alloy` uses `(seed, "alloy-occupation")`, `generate_targeted_sro` uses `(seed, "targeted-sro")`, and ensembles derive per-configuration seeds with `stream_seed(seed, "alloy-ensemble", "occupation"|"optimizer", k)`. A record's `seed` and `optimizer_seed` replay the configuration through the public `generate_alloy` and `generate_targeted_sro` (tested). Equal seeds given to different functions no longer share draws. Outputs from earlier commits are not reproduced; no stored results depended on them.
- A2.6: added target feasibility checks and `validate_targeted_sro` in `alloys.py`, with `tests/test_alloys_sro_feasibility.py` (18 tests). Two test-only correction rounds (the chosen swap left alpha unchanged; an objective override equal to the true value). With the other alloy tests and `tests/test_e3_mlip_structure_generation.py`: 121 passed; structural tests: 59 passed. Ruff check and format pass.
- A2.7: added `alloy_ensemble.py` and `tests/test_alloy_ensemble.py` (30 tests); topology reuse added to `alloys.py`. Alloy, ensemble and E3 tests together: 151 passed, 0 failed, 0 skipped. Structural tests: 59 passed, 0 failed, 0 skipped. Ruff check and format pass. No debugging corrections were needed.
- A2.7 follow-up: added the pairwise occupation-overlap diagnostic and 3 tests (`tests/test_alloy_ensemble.py`: 33 passed, 0 failed, 0 skipped). Ruff check and format pass.
- A2.8: exact duplicates (`e3841b8`), SRO statistics (`8ac8d70`), translation, point-group and cross-cell equivalence (`d287ae1`, `ed060d8`). `tests/test_alloy_ensemble.py` with the seeding-policy test: 55 passed, 0 failed, 0 skipped. Ruff check and format pass.
- A2.8d and A2.9: three point-group tests in `tests/test_alloy_ensemble.py` and `tests/test_phase_a2_integration.py`. With the seeding-policy test: 60 passed, 0 failed, 0 skipped. Ruff check and format pass. No debugging corrections were needed.
- CI fix: `tests/test_dependency_contract.py` expected `ase==3.26.0` after `pyproject.toml` moved to 3.29.0; the test now expects 3.29.0 (`60240c4`).
- Not done: SQS-like and local-clustering orderings; `OrderingMode` still admits only `random`.
- A3.1 to A3.4: `tests/test_radiation_defects.py` 71 passed, 0 failed, 0 skipped. Ruff check and format pass. A3.4b needed three test-side correction rounds (fixture size, species validated before ASE, minimum-image vectors in a test).
- A3.5: `generate_frenkel_pair`. `tests/test_radiation_defects.py` 82 passed, 0 failed, 0 skipped. Ruff check and format pass. No debugging corrections were needed.
- A3.6: `radiation_defect_environments.py`. `tests/test_radiation_defect_environments.py` 6 passed, 0 failed, 0 skipped. Ruff check and format pass. No debugging corrections were needed.
- A3.7a: `radiation_defect_mapping.py`. `tests/test_radiation_defect_mapping.py` 8 passed, 0 failed, 0 skipped. Ruff check and format pass. No debugging corrections were needed.
- A3.7b: `identify_defect_sites`. `tests/test_radiation_defect_mapping.py` 13 passed (A3.7a and A3.7b), 0 failed, 0 skipped. Ruff check and format pass. No debugging corrections were needed.
- A3.8: `radiation_defect_classification.py`. `tests/test_radiation_defect_classification.py` 11 passed, 0 failed, 0 skipped. Ruff check and format pass. One debugging correction round (expected composition for relocated interstitials; split defects reported as unresolved).
- A3.9: `generate_divacancy`. `tests/test_radiation_defects.py` 99 passed, 0 failed, 0 skipped. Ruff check and format pass. No debugging corrections beyond one test-helper fix before the first run.
- A3.10: `radiation_defect_energies.py`. `tests/test_radiation_defect_energies.py` 7 passed, 0 failed, 0 skipped. Ruff check and format pass. No debugging corrections were needed.
- A3 integration: added `tests/test_radiation_defect_pipeline.py`. Ruff check and format pass. Test status: PARTIAL. `-k "54-atom or foreign"`: 9 passed, 0 failed, 12 deselected (1,024-atom cases). The full-file runs were interrupted (13 passed, then `KeyboardInterrupt`), so no full-file result is recorded. One test-expectation correction was made after the first failure.
- A3 marked complete at the maintainer's request. Open verification items: full run of `tests/test_radiation_defect_pipeline.py` (1,024-atom cases) and the A1 to A3 regression run.
- A3 verification gate (partial): the 12 1,024-atom integration tests each passed alone (see the A3 integration entry). The A1 to A3 regression run is still pending, so A3 is not yet marked software-verified.
- B1.1: `neb_contracts.py`. `tests/test_neb_contracts.py` 14 passed, 0 failed, 0 skipped. Ruff check and format pass. No debugging corrections (two test-side edits before the first run: no random generator, simpler validation call).
- B1.2: `neb_endpoints.py`. `tests/test_neb_endpoints.py` 10 passed, 0 failed, 0 skipped. Ruff check and format pass. One correction: a test used an atom that really is a neighbour as the "far" atom; the test was fixed, not the code.
- B1.3: `neb_paths.py`. `tests/test_neb_paths.py` 14 passed; B1.1-B1.3 together 38 passed, 0 failed, 0 skipped. Ruff check and format pass. Two corrections: the first draft used ASE `find_mic`, a test against brute force showed it can miss the shortest image on a skewed cell, so an exact search replaced it (code fix, test not weakened); a test claim about ASE was narrowed to the single-vector case after measuring.
- B1.4: `neb_images.py`. `tests/test_neb_images.py` 16 passed; B1.1-B1.4 together 54 passed, 0 failed, 0 skipped. Ruff check and format pass. No code corrections (one test line-length fix).
- PR #77 Copilot review (A1 code): fixed in `alloys.py`. `counts` is a read-only mapping and serializes to plain JSON; `lattice_parameter` rejects non-finite values; counts must be strict integers (booleans and floats rejected). Regression tests added. `tests/test_alloys.py`: 31 passed. Ruff check and format pass.
