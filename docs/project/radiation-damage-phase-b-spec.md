# Radiation damage, Phase B: migration pathways and Quantum ESPRESSO NEB (specification)

Condensed from the maintainer's task statement. Authoritative task list for B1.1 to B2.8.
Software verification only; no DFT, MLIP, NEB, VASP or HPC execution. Do not claim physical qualification.

## Execution rules

- Work on `feat/radiation-damage-phase-a`. No force-push, no reset. Stage only the task's files.
- Per task: inspect only directly relevant code, implement, write focused tests, run them, run Ruff on changed files,
  at most two targeted fixes, update `radiation-damage-implementation-progress.md`, commit
  (`radiation-damage: <what> (B<n>.<m>)`), push, continue. On push failure keep the commit and record `PUSH_PENDING`.
- Blocked after two corrections: record the blocker, mark `BLOCKED`, skip dependents, continue independent tasks.
- Never disable, weaken or delete a test to get a pass. Never fabricate results.
- Reuse `discovery/defects.py`, `discovery/point_defects.py`, `discovery/bcc_defects.py`, `backends/dft/qe.py`, `qe_relax.py`, `process.py`,
  existing ledgers, artifact directories and CLI conventions. No second DFT execution framework.
- New modules (where appropriate): `discovery/neb_paths.py`, `backends/dft/qe_neb.py`.
- Final report: table `Task | Status | Tests | Commit`, final branch and hash, counts (passed, failed, skipped, not run),
  blocked items, unpushed commits, software-completion status, outstanding physical qualification. No Phase C.

## B1 migration paths and endpoints

- B1.1 NEB data contracts: records for initial and final defect state, stable atom identity correspondence, migration
  mechanism, ordered images, NEB calculation spec, NEB result with convergence status. Metadata: endpoint structure ids,
  parent configuration id, defect type, mechanism, identities, species and atom counts, cell, image count, generation
  parameters, endpoint relaxation status, numerical method identity, provenance. Distinguish generated, relaxed,
  numerically qualified endpoints, completed-but-unconverged calculations, qualified results. Endpoints must have equal
  atom counts, identity sets, species per identity, compatible cell and periodicity, finite coordinates, one-to-one
  correspondence. Different vacant sites are allowed.
- B1.2 Endpoint generation: vacancy hop (nearest neighbour; moving atom enters the vacancy; record species and
  displacement; keep ids), interstitial migration (explicit endpoint interface, mechanism recorded, needs later
  validation), Frenkel recombination (atom returns to a vacancy site; distinguish restored ordering from exchange of
  occupants; record vacancy/interstitial correspondence).
- B1.3 Correspondence and periodic displacements: identity-based mapping (not ASE indices), species validation, true
  minimum-image displacement (including skewed cells), unwrapped trajectories, ambiguity detection.
- B1.4 Image generation: linear interpolation in periodic displacement, configurable image count (QE requires more than
  three), endpoints fixed, deterministic, identity-preserving, collision checks (overlap, large jumps, non-finite,
  broken correspondence, invalid cell). A linear path is a guess, not a barrier or mechanism.
- B1.5 Pathway validation and qualification states: candidate, geometry_validated, endpoint_relaxation_pending,
  numerically_qualified, rejected. Checks: counts, identities, species, cells, continuity, image count, overlaps,
  endpoint relaxation metadata, method consistency, provenance. Unrelaxed endpoints are never numerically qualified.
- B1.6 Serialization: JSON metadata plus ASE-compatible geometry files; keep endpoint ids, correspondence, mechanism,
  defect metadata, cell, parameters, validation outcome, hashes; no duplicated coordinates in metadata; repeatable
  fingerprints.

## B2 Quantum ESPRESSO NEB

- B2.1 Config contracts (neb.x path, images, `nstep_path`, `path_thr` in eV/Angstrom, `opt_scheme`, `CI_scheme`,
  `restart_mode`, minimum image, PW settings, pseudopotentials for W, Ta, Cr, V, Hf, Mo, k-points, cutoffs, smearing,
  spin, parallel launch, directories). Reject invalid images, schemes, thresholds, missing pseudopotentials,
  inconsistent PW settings. No universal magnetic treatment; record how Cr magnetism is handled.
- B2.2 Input writer: single-file format `BEGIN`, `BEGIN_PATH_INPUT`, `&PATH`, `END_PATH_INPUT`, `BEGIN_ENGINE_INPUT`,
  PW namelists and cards, `BEGIN_POSITIONS`, `FIRST_IMAGE`, `INTERMEDIATE_IMAGE`, `LAST_IMAGE`, `END_POSITIONS`,
  `END_ENGINE_INPUT`, `END`. Same species and atom order across images; support no-CI, auto and manual climbing
  images; no climbing image from the first iteration; reuse `write_pw_input` helpers where practical.
- B2.3 Execution: `neb.x -inp neb.in` (never `< neb.in`); reuse `run_launcher`; capture exit code, stdout, stderr,
  timestamps, input hash, working directory, arguments; exit code 0 is not proof of convergence; no site-specific
  commands in generic logic. Tests use a mocked executable.
- B2.4 Parser: `prefix.dat` columns are reaction coordinate, energy (eV, relative to first image) and residual error
  (eV/Bohr; `path_thr` is eV/Angstrom). Outputs: image indices, relative energies, forward and reverse barriers
  (`E_saddle - E_initial`, `E_saddle - E_final`), endpoint difference, residuals, convergence, climbing image used,
  warnings. Sampled maximum is an estimate, not a validated saddle. Distinguish converged, iteration limit, electronic
  failure, process failure, incomplete output, missing images, invalid numbers, unqualified barrier. Never fabricate
  missing values.
- B2.5 Restarts and staged CI: stage 1 `CI_scheme='no-CI'`; stage 2 `auto` or manual only after sufficient stage 1
  convergence; keep previous path, outputs, hashes and stage history; reject incompatible restart metadata; never delete
  earlier output before the replacement succeeds.
- B2.6 Workflow integration: parent, defect, mechanism, endpoints, validation, images, QE inputs, record plan, optional
  dispatch, ingest, qualification, barrier with provenance. Default is dry run; real dispatch needs explicit config.
  Reuse existing objective-graph, execution and evidence-ledger abstractions.
- B2.7 CLI and config: prepare pathway, validate endpoints, generate images, write QE input, inspect results, resume
  explicitly; existing CLI and registration mechanism; YAML or existing config; help text and a minimal example.
- B2.8 Integration verification: vacancy migration, interstitial migration, Frenkel recombination, synthetic QE parser
  cases, mocked process and restart. At most light geometry checks on the 1,024-atom reference.

## Software completion is not scientific qualification

Real qualification needs relaxed endpoints, consistent DFT settings, converged NEB, saddle refinement, image-count and
supercell sensitivity, magnetic-state checks and reference comparison. None of that is done here.
