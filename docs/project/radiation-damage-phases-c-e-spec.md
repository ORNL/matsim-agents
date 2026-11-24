# Radiation damage: Phases C, D, E (condensed authoritative spec)

Branch: `feat/radiation-damage-phase-a`. Cycle per task: inspect, code, test, fix (max 2 targeted
corrections), Ruff, update `radiation-damage-implementation-progress.md`, commit, push, next.
Never force-push or merge. Never fabricate GPU or DFT evidence. Never weaken tests.
Commit style: `radiation-damage: <what> (<task>)`. Missing evidence is not a zero barrier.
Synthetic fixtures must be labelled synthetic. Kinetic outputs are conditional on the supplied network.

## Phase C: transition networks and kinetics (regime A: recovery of a generated damage state)
- C1.1 contracts: defect state, transition event (source, destination, mechanism, barriers, attempt
  frequency, multiplicity, evidence, qualification, temperature-dependent rate), absorbing states, provenance.
- C1.2 directed graph: deterministic state and transition IDs, duplicate detection, atom-identity
  conservation, consistent parent/cell context, no merging across chemical environments.
- C1.3 elementary transitions from B mechanisms (vacancy hop, interstitial migration, Frenkel
  recombination, trapping/detrapping with evidence); no guessed barriers.
- C1.4 NEB evidence ingestion: endpoint match, forward/reverse convention, units, method, convergence,
  reject unconverged/mismatched evidence for qualified rates.
- C1.5 network validation and completeness (fraction of candidates with usable evidence).
- C2.1 HTST rates `g * nu * exp(-Ea / kB T)`; explicit prefactors; underflow safe; detailed-balance check only on request.
- C2.2 Gillespie CT-KMC: seeded, horizon, max events, absorbing and zero-rate states, trace.
- C2.3 ensembles: independent streams, first passage, censoring, recombination probability.
- C2.4 uncertainty and sensitivity (barriers, prefactors, T, missing channels); no assumed independence.
- C2.5 analytical benchmarks and A3/B/C integration with synthetic fixtures.

## Phase D: workflow integration
- D1 capability registration with readiness levels. D2 end-to-end workflow reusing existing orchestration,
  dry run, checkpoints, resume. D3 hypothesis revision with deterministic gates and budgets, optional LLM.
- D4 non-blocking Slurm integration (`sbatch -q debug -A LRN110`), persistent manifest, mocked tests, no duplicate submissions.
- D5 CLI, config, reports (dry-run default). D6 end-to-end integration and regression.

## Phase E: qualification
- E1 qualification matrix and status transitions (NOT_TESTED, SUBMITTED, RUNNING, PASSED_SOFTWARE_TESTS,
  NUMERICALLY_VALIDATED, SCIENTIFICALLY_QUALIFIED, FAILED, BLOCKED); no promotion without evidence.
- E2 MLIP validation protocol (reference ingestion, errors, uncertainty); GPU smoke job submitted non-blocking.
- E3 DFT/NEB qualification workflow (inputs and convergence checks only; no large DFT campaigns).
- E4 kinetic-model qualification (analytical benchmarks). E5 scalability and reproducibility.
- E6 material readiness (W-rich, Fe-Cr RAFM, V-4Cr-4Ti in wt%, ODS, SiC/SiC): lightweight applicability checks only.
- E7 final evidence-grounded report separating software completeness from scientific qualification.

## Slurm rules
Never block or poll in loops. Record job IDs in `outputs/radiation_damage/validation_jobs.json`.
Max 2 outstanding GPU jobs, one node one GPU, finite time, no duplicates, one retry for diagnosed failures.
Do not commit Slurm logs or large outputs.

## Final gate
Targeted tests per phase, Ruff, one final regression, final report table (Task, Status, Tests, Commit,
Outstanding) with job counts, unverified evidence and unpushed commits.
