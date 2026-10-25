# Characterization contracts

A characterization answers one question about given structures of one
composition: what is the relaxed structure, which polymorph is lowest in
energy, or is the material thermodynamically stable. Unlike active learning,
phase exploration, or a campaign, it has no optimization objective. The
question, the evidence that answers it, and its limits are declared up front
in a **contract**; a deterministic **recipe** plans and runs the calculations;
the **report** gives a conclusion only when the evidence supports it, and
otherwise says what is missing.

```bash
matsim-agents characterize CONTRACT.yaml
```

The command prints the report as JSON and exits non-zero for
`invalid_protocol` and `failed_infrastructure`. Running the same contract
again resumes it: completed calculations are never repeated.
Programmatically, use `matsim_agents.campaign.characterization`
(`load_contract`, `plan_recipe`, `run_characterization`, `outstanding_steps`).
Examples are in [examples/characterization](../examples/characterization).

## Recipes

| Recipe | Question | Observables (unit, default minimum fidelity) | Calculations |
| --- | --- | --- | --- |
| `relaxation` | Relaxed geometry and energy | `relaxed_structure` (structure), `energy_per_atom` (eV/atom), `residual_max_force` (eV/A), all `mlip_relaxation` | One relaxation per structure |
| `energy_ranking` | Lowest-energy structure among same-composition candidates (at least two) | `relative_energy_per_atom` (eV/atom, `mlip_relaxation`) | One relaxation per structure, then a ranking |
| `thermodynamic_stability` | Energy above the convex hull of the declared reference phases | `formation_energy_per_atom`, `energy_above_hull` (eV/atom, `converged_dft`) | One DFT relaxation per structure, then the hull |

An energy ranking says which structure is lowest; it never claims that the
composition is thermodynamically stable. Only the `thermodynamic_stability`
recipe does, and only from converged DFT.

## The contract

| Field | Meaning |
| --- | --- |
| `contract_id` | Identifies the run directory `<output_root>/characterization-<contract_id>/`. A changed question needs a new id (see [Resuming](#resuming)). |
| `objective` | The question, in words. |
| `formula`, `structures` | The composition and its structure files; every structure must have this composition. |
| `recipe` | One of the three recipes. |
| `observables` | Optional. Overrides the recipe's defaults (for example a higher `minimum_evidence_level`). Units are fixed per observable and validated; an observable the recipe does not produce, or a level above `converged_dft` (the most a recipe produces), is rejected. |
| `methods.mlip` | MLIP settings, as for `matsim-agents relax`. |
| `methods.dft`, `methods.dft_method_signature` | The DFT backend, and the scientific signature of its method. `methods.dft.settings` keys are checked against the backend's settings (`QESettings`, `VASPSettings`), excluding those the relaxation sets itself; raw overrides (`extra_control`, `extra_system`, `extra_electrons`, `extra_cell`, `extra_incar`) may not set what the geometry, convergence, or a settings field controls (for example `calculation`, `cell_dofree`, `ISIF`, `ENCUT`). |
| `geometry` | Degrees of freedom of every relaxation (`GeometryControls`); atoms and cell by default. Cell-only relaxation (`relax_atoms: false`) is rejected, because the relaxations cannot yet hold the atoms fixed. A nonzero `pressure_GPa` is also rejected, because the relaxations run at zero pressure. |
| `tolerances` | Residual force, relaxation steps, degeneracy tolerance for rankings, and hull tolerance. |
| `reference_energies` | A `ReferenceEnergySet` JSON (`thermodynamic_stability`). |
| `budget` | `max_mlip_relaxations` and `max_dft_calculations`: calculations attempted over the whole contract, failed ones and resumed runs included. |
| `approvals`, `dft_approved` | DFT waits for `dft_approved: true` when `approvals.before_dft` is set (the default). |
| `output_root` | Where the run directory is created (default `runs/` next to the contract file). |

Relative paths in a contract file (structures, `reference_energies`,
`output_root`, `methods.dft.pseudo_dir`, `methods.dft.potcar_dir`) are relative
to that file.

DFT is planned only when a required observable needs at least
`low_fidelity_dft`. Raising the `relaxation` or `energy_ranking` observables to
`converged_dft` relaxes the structures with DFT instead of an MLIP.

### Reference energies for stability

The `reference_energies` file is a `ReferenceEnergySet`:

```json
{
  "identifier": "mg-o-qe-pbe-v1",
  "method_signature": "qe-pbe-sssp-efficiency-ecut60",
  "elemental_energies_eV_per_atom": {"Mg": "<DFT energy per atom of Mg>", "O": "<...>"},
  "competing_phases": {"MgO2": "<formation energy per atom, eV/atom>"}
}
```

These energies must be computed with the same DFT method as the
characterization relaxations, fully relaxed (atoms and cell). The contract declares that method's signature
in `methods.dft_method_signature`, and it must equal the reference set's
`method_signature`. Any backend the reference set or its entries declare must
also equal `methods.dft.backend`. The reference set must be complete for the composition:
every element, at least one competing phase in each binary subsystem, and for
ternary and larger compositions a ternary competitor. The example ships no
reference energies; they are yours to compute, for example with `relaxation`
contracts whose observables require `converged_dft` and that use the same
`methods.dft`.

The DFT backends otherwise choose some settings per composition, which would
give the references and the material different methods. A stability contract
must therefore fix them in `methods.dft.settings`:

| Backend | Settings that must be fixed |
| --- | --- |
| `qe` | `ecutwfc_ry`, `ecutrho_ry`, `occupations`, `pseudopotentials` |
| `vasp` | `encut_ev`, `ismear`, `sigma`, `kspacing` |

The k-point grid of Quantum ESPRESSO still follows each cell's size (VASP's
`kspacing` fixes the density), and the stability recipe requires relaxing
atoms and cell.

## Statuses

| Status | Meaning |
| --- | --- |
| `sufficiently_characterized` | Every observable meets its fidelity, and the conclusion follows from cited evidence. |
| `inconclusive` | Evidence is missing or insufficient. `missing` lists what and `recommended_next_test` says what to run. No conclusion is given. |
| `invalid_protocol` | The contract cannot be answered as given: an unreadable structure or one of another composition, a reference-energy file that is not a valid `ReferenceEnergySet`, a competing phase without elemental energies for all of its elements, or references computed with another DFT method or backend. Nothing is calculated. |
| `failed_infrastructure` | A calculation could not run (missing executable, crashed process). This is not evidence about the material; running the contract again retries it. |
| `planned`, `in_progress`, `cancelled` | Reserved for orchestrators that track contracts. |

The report (`characterization-report.json` in the run directory) lists each
observable with its value, unit, evidence level, and the evidence it cites.
Each evidence item names its relaxation run, artifact, and the artifact's
SHA-256. A report only concludes when sufficiently characterized, a
characterized report must cite evidence, and an observable may only cite
evidence the report lists.

## Guarantees

- **No stability claim from MLIP evidence.** A `thermodynamic_stability`
  contract that lowers its observables below `converged_dft` is rejected when
  it is validated. Unconverged DFT is low-fidelity evidence and gives
  `inconclusive`, never "unstable". A VASP relaxation counts as converged only
  when it reports "reached required accuracy"; one that stops at `NSW` is not.
- **Non-finite outputs are failures.** A relaxation whose energy or residual
  force is NaN or infinite is recorded as failed and retried within the
  budget, never used as evidence.
- **No stability claim from incomplete references.** A provisional reference
  set gives `inconclusive` with the missing elements and phases listed, before
  any DFT is spent.
- **No unnecessary DFT.** MLIP-level contracts never call DFT; DFT waits for
  approval and stops at the budget, and the report says which steps are
  blocked.
- **Fidelity is checked against every observable.** If the evidence is below
  an observable's `minimum_evidence_level`, the report is `inconclusive`.
- **Resumable.** See [Resuming](#resuming).
- **Near-degenerate rankings are inconclusive.** When the two lowest
  structures are within `degeneracy_tolerance_eV_per_atom`, the ranking
  recommends a higher-fidelity comparison instead of naming a winner.

## Resuming

The state of every step is saved as it finishes (`characterization-state.json`).
Running the contract again skips completed steps, retries failed ones within
the budget, and returns the saved assessment once the ranking or hull has been
computed from complete inputs (`outstanding_steps` lists what remains).

The run is bound to a fingerprint of the question: every contract field except
`budget`, `approvals`, `dft_approved`, and `output_root`, plus the SHA-256 of
each structure file and the reference-energy file. Approving DFT or raising a
budget therefore continues the same run; changing anything else, including
editing an input file in place, is refused and needs a new `contract_id`.
Paths are resolved to absolute paths first, so loading the contract by a
relative or an absolute filename resumes the same run. Each completed
relaxation also records the SHA-256 of its optimized structure; if that file
has since been changed or removed, the run is refused rather than reusing it.

Only one execution of a contract runs at a time: it holds a lock on the run
directory while it reads, executes, and saves, and a concurrent execution is
refused before it calculates anything.

Every relaxation attempt, failed ones included, and every concluded answer is
also appended to `evidence-ledger.jsonl` in the run directory; the report's
evidence identifiers are ledger identifiers. See
[Evidence ledger](evidence-ledger.md).

## Stability fidelity in campaigns

`StabilityReport.evidence_level` records the fidelity of every stability
report. Campaigns only admit convex-hull reports from DFT into their hull and
reference set (`CampaignState.record_stability`), and hull recalibration only
reuses DFT reports. Campaign states saved before this field existed (schema 1)
are migrated explicitly when loaded: their convex-hull reports came from DFT
refinement and become `converged_dft`, and relative rankings came from MLIP
relaxations and become `mlip_relaxation`.

## Failure recovery

A failed or unconverged DFT relaxation is classified
(`matsim_agents.execution.recovery.classify_failure`) and handled by the
versioned policy `recovery-v1`:

| Failure class | Detected by | Recovery |
| --- | --- | --- |
| `scf_not_converged` | QE "convergence NOT achieved"; VASP at `NELM` | Up to 2 retries: QE halves `mixing_beta` and doubles `electron_maxstep`; VASP doubles `NELM` and sets `ALGO = All`. |
| `ionic_step_limit` | Finished at `max_relaxation_steps` above the force tolerance | Up to 2 retries from the last structure. |
| `walltime` | Return code 124 | 1 retry with twice the `timeout_sec`, when one is set. |
| `non_finite_output`, `no_result` | NaN/inf, or no energy or structure | 1 retry, unchanged. |
| `launcher`, `approval_or_input`, `unknown` | Missing executable, refused approval, invalid input | None. |

Every retry counts against `max_dft_calculations`. Each attempt is a ledger
record whose `numerical_convergence.failure_class` names the class. The
choice to retry or give up is a `DecisionRecord`: action
`recover:<class>`, `defer:<class>` (a retry the budget stopped), or
`give_up:<class>`, actor `recovery-policy:recovery-v1`, `policy_version`,
and `thresholds` (retries allowed and used, budget). A retry, with its
structure, settings, and counters, is saved in the run state before it
starts and kept until an attempt with no further retry is saved, so an
interrupted or deferred retry runs on the next execution; a deferred retry
first records its `recover:<class>` decision. The
adjusted settings are part of the retry's method record and do not change the
contract. MLIP relaxations are not retried: a rerun repeats the failure.

## Limitations

- Stability is relative to the declared reference phases at 0 K, without
  zero-point or finite-temperature contributions; it is not a global
  ground-state search.
- An exception inside a relaxation is reported as `failed_infrastructure`;
  its failure class (see Failure recovery) is in the ledger record.
- Only DFT attempts are recovered, and only by the fixed rules below; a
  failure the policy does not recognise is not retried.
- The energy is the final energy of the relaxation; there is no separate
  static calculation at the relaxed cell (Pulay stress is not corrected).
- Recipes cover relaxation, ranking, and hull stability. Phonons, elastic
  constants, electronic structure, and defects are not implemented.

## Status

Implemented and tested with deterministic stand-ins for the relaxation
operator (`tests/test_characterization.py`). Contracts have not yet been
qualified with real MLIP or DFT calculations on HPC systems.
