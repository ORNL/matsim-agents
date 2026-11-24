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
  - Limitations: independent seeds do not imply statistically uncorrelated or structurally unique configurations; no symmetry-equivalence detection (A2.8); targeted-SRO members are bounded-search configurations, not equilibrium structures, and some may not converge. No 1,024-atom targeted-SRO campaign was run; the 1,024-atom test is random mode only.

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
- A2.6: added target feasibility checks and `validate_targeted_sro` in `alloys.py`, with `tests/test_alloys_sro_feasibility.py` (18 tests). Two test-only correction rounds (the chosen swap left alpha unchanged; an objective override equal to the true value). With the other alloy tests and `tests/test_e3_mlip_structure_generation.py`: 121 passed; structural tests: 59 passed. Ruff check and format pass.
- A2.7: added `alloy_ensemble.py` and `tests/test_alloy_ensemble.py` (30 tests); topology reuse added to `alloys.py`. Alloy, ensemble and E3 tests together: 151 passed, 0 failed, 0 skipped. Structural tests: 59 passed, 0 failed, 0 skipped. Ruff check and format pass. No debugging corrections were needed.
- A2.7 follow-up: added the pairwise occupation-overlap diagnostic and 3 tests (`tests/test_alloy_ensemble.py`: 33 passed, 0 failed, 0 skipped). Ruff check and format pass.
- Not done: SQS-like and local-clustering orderings; `OrderingMode` still admits only `random`.
- PR #77 Copilot review (A1 code): fixed in `alloys.py`. `counts` is a read-only mapping and serializes to plain JSON; `lattice_parameter` rejects non-finite values; counts must be strict integers (booleans and floats rejected). Regression tests added. `tests/test_alloys.py`: 31 passed. Ruff check and format pass.
