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
- [ ] A2.2 Second-neighbor-shell identification
  - Acceptance: six second neighbours per site; consistent with A2.1 shell labels.
- [ ] A2.3 Warren-Cowley SRO descriptors
  - Acceptance: first- and second-shell parameters for all species pairs; pair-count reciprocity; measured (not assumed) SRO for random occupations.

## Progress log

- A1.1: inspection complete; no E3 change required.
- A1.2 to A1.4, A1.6: implemented in `src/matsim_agents/discovery/alloys.py`.
- A1.7: `tests/test_alloys.py` (20 tests) and `tests/test_e3_mlip_structure_generation.py` pass together (39 passed). Ruff check and format pass for the new files.
- A1.5: `examples/fusion_damage/` runs locally and writes 1,024 atoms with the exact reference counts.
- Scope check: `benchmarks/`, `runs/`, and existing `src/` and `tests/` files are unchanged (git status).
- Not done: targeted SRO, SQS-like and local-clustering orderings (out of scope for A1; `OrderingMode` only admits `random`).
- A2.1: added `first_neighbor_list` in `alloys.py` (ASE `neighbor_list`, cutoff between shells). `tests/test_alloys.py`: 24 passed, including 3 new first-shell tests (1x1x1, 2x2x2, 3x2x4 supercells, and the 1,024-atom reference). Ruff check and format pass.
- A2.2 and A2.3: not started.
- PR #77 Copilot review (A1 code): fixed in `alloys.py`. `counts` is a read-only mapping and serializes to plain JSON; `lattice_parameter` rejects non-finite values; counts must be strict integers (booleans and floats rejected). Regression tests added. `tests/test_alloys.py`: 31 passed. Ruff check and format pass.
