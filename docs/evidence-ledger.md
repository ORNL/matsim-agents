# Evidence ledger

Every scientific statement matsim-agents makes should be traceable to the
calculations behind it. The evidence ledger is an append-only file of three
kinds of records:

| Record | What it is |
| --- | --- |
| `EvidenceRecord` | One numerical result, or a failure to compute it. Includes method (backend, model, method signature, settings, software versions), evidence level, outcome, values with units, numerical convergence, reference set and its completeness, input and output file SHA-256, physical assumptions, uncertainties, and parent records. |
| `HypothesisRecord` | One revision of a claim: claim type, claim, falsifiable prediction, status (`proposed`, `supported`, `contradicted`, `unresolved`, `superseded`), cited evidence, rationale, proposed tests, the agent and model that made it, and its prior revision. |
| `DecisionRecord` | A workflow action (for example deactivating a formula) with the evidence it rests on and who decided it. |

The implementation is in `matsim_agents.execution.ledger`.

## Guarantees

- **Content identifiers.** A record's identifier is the hash of its content,
  excluding the time it was recorded. Replaying a workflow with the same
  outputs produces the same identifiers, and appending an existing record does
  nothing. Fields added later (`cost`, `policy_version`, `thresholds`,
  `expected_cost`, `observed_cost`) are left out of the hash when unset, so
  older ledgers keep their identifiers.
- **Tamper evidence.** The file is a JSON-lines hash chain. Editing, deleting,
  or reordering a line is detected when the ledger is opened
  (`LedgerIntegrityError`). `verify_artifacts()` re-hashes the recorded input
  and output files.
- **Truncation evidence.** Each append also rewrites a checkpoint beside the
  ledger, `<ledger>.head`, with the sequence number and hash of the last
  entry. Opening a ledger that no longer reaches its checkpoint is an
  integrity error. This covers a dropped suffix and an emptied or deleted
  file. Copy the checkpoint along with the ledger. A ledger without a
  checkpoint is read without this check. Deleting both files is not detected.
- **Concurrent writers.** An append holds `<ledger>.lock`. It is refused if the
  file changed after this process read it, including an append that raced the
  read: reopen the ledger and retry.
- **Immutable history.** A revision of a hypothesis names the revision it
  follows, which must be the latest one. Status changes add revisions; earlier
  revisions are never rewritten (`history(hypothesis_id)` returns all of them).
- **Deterministic checks before appending.** A hypothesis revision or decision
  is rejected (`LedgerValidationError`) when:
  - it cites evidence that is not in the ledger;
  - a `supported` or `contradicted` claim, or a decision, cites no evidence;
  - it cites a failed or unconverged calculation as evidence for or against a
    material;
  - a `thermodynamic_stability` claim rests on anything weaker than converged
    DFT, or on a hull whose reference phases are incomplete;
  - an `energy_ranking` claim cites no ranking or stability evidence.
- **No silent mixing of fidelities.** A hypothesis's `support_level` is the
  lowest evidence level among what it cites; it rests on all of it.
- **Failure is not evidence.** A calculation that could not run is recorded as
  `failed_to_compute`, and one that did not converge as `not_converged`, each
  with its reason.

Appends take a lock file next to the ledger (`fcntl.flock` is not supported on
NERSC CFS compute nodes), and a writer whose view of the file is outdated is
refused rather than forking the chain.

## Where ledgers are written

| Workflow | Ledger | Contents |
| --- | --- | --- |
| Campaign (`run_campaign`) | `<output_dir>/evidence-ledger.jsonl` | One record per formula run: its stability report (a new record, with the previous one as parent, whenever hull recalibration changes it) or its failure. Review decisions and hypothesis revisions. |
| Characterization (`matsim-agents characterize`) | `<run directory>/evidence-ledger.jsonl` | Every relaxation attempt, failed ones included, and every sufficiently characterized answer citing the relaxations it rests on. The report's evidence identifiers are ledger identifiers. |

## Campaign reviews

The evidence shown to reviewing LLMs lists each formula's ledger records (kind,
evidence level, outcome, method, reference completeness, convergence, values
with units) and the current revision of every hypothesis. Reviewers must cite
evidence identifiers:

- a deactivation or reactivation counts only with a measured result for that
  formula among its citations;
- each hypothesis revision is checked as above before it is appended; a
  rejected revision is kept in the review notes with the reason;
- a decision that reaches agreement is appended with the union of the
  agreeing reviewers' citations, and only then applied.

Reviewers revise an existing hypothesis by its `hypothesis_id`; a new claim gets
an identifier derived from the campaign and the claim. The LLM text is
recorded as given and never regenerated on replay.

## Migration

Campaign state schema 3 adds `evidence_ledger_file`. A campaign saved before it
(schema 2) is migrated explicitly the first time its ledger is opened
(`open_campaign_ledger`):

- stability reports become evidence records, and failed formula runs become
  `failed_to_compute` records;
- review deactivations and reactivations become decisions, and review
  hypothesis revisions become hypothesis revisions; both carry
  `imported_from: campaign_state.review_history`. The old revisions cited
  free text rather than evidence identifiers, so that text is kept in the
  rationale and they have no evidence identifiers.

The backfill is idempotent and deterministic: replaying it gives the same
identifiers.

## Cost accounting

`CostRecord` splits resources into LLM tokens and seconds, MLIP relaxations
and seconds, DFT calculations and seconds, fine-tuning seconds, and failed
jobs and seconds. Characterization stores each attempt's cost as
`EvidenceRecord.cost`; a failed attempt counts only under failed jobs. A debate
review stores the debate's token usage as the `observed_cost` of its first
decision, along with `policy_version` and the agreement `thresholds`. A
recovery decision to retry has `expected_cost` set to one DFT calculation.
`ledger show` reports the totals under `cost`.

## Command line

```bash
matsim-agents ledger verify runs/campaign/evidence-ledger.jsonl   # chain and artifacts
matsim-agents ledger show runs/campaign/evidence-ledger.jsonl     # counts and current hypotheses
matsim-agents ledger show LEDGER --hypothesis hyp-0123abcd        # every revision of one claim
```

`verify` exits non-zero when the hash chain is broken or a recorded file
changed (`--no-artifacts` checks the chain only).

## Limitations

- Identifiers include software versions, so the same calculation recorded by
  a different installation gets a different identifier.
- The ledger does not store artifacts, only their hashes; moving or deleting a
  run directory is reported by `verify` but the files cannot be recovered.
- Hull recalibration records the new values of earlier reports, not the
  recalibration itself as a separate decision.
- Only campaigns and characterizations write ledgers; standalone `relax`,
  active learning, and phase exploration do not yet.
