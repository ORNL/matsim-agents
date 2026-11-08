# E4 Aurora DFT Stage 1 readiness

> Superseded by [`stage1-blocker-resolution-report.md`](./stage1-blocker-resolution-report.md)
> on 2026-11-08. This document is retained as a historical snapshot; its
> inventory, PAW, duplicate-screening, and allocation statements are not current.

## Decision and execution boundary

**NO-GO for PBS submission.** This report prepares the E4-A VASP campaign; it
does not authorize or submit jobs. No new VASP calculation was launched for
this preparation. The four historical E4 BCC A/B jobs listed below did run and
are archived; they are not evidence that the planned 3,060-case campaign is
underway.

The current PBS files are dry-run templates only. They print metadata and exit
without invoking VASP. No production execution script with validated
checkpoint/restart behavior is ready. The existing
`deployments/aurora/launchers/run-e2-dft-case-aurora.sh` is not a validated
replacement for the E4 two-precision workflow.

## Structures and provenance

| Family | Available candidate inventory | Provenance and suitability | Remaining qualification |
|---|---:|---|---|
| Refractory BCC | 2,040 planned candidates; 8 selected for the 16-case Stage 1 subset | The E4 A/B 128-site parent inputs are in `/lus/flare/projects/HydraGNN/mlupopa/AgenticAI4Materials/e4-vasp-ab-pilot-inputs/Nb22Ta22V21Hf21Zr21Ti21/case-1/`. Candidate occupations cover quaternary, quinary, and six-component Nb-Ta-V-Hf-Zr-Ti systems; the Stage 1 subset has three quaternaries, three quinaries, and two six-component cases. These are new seeded occupations on the staged BCC parent. Suitable for a controlled computational workload after the archived A/B input gaps and PAW provenance are resolved. | The candidate structures are not certified unique against public or private archives. The full public OSTI payloads were not enumerated. The exact ionic force stopping criterion is absent from the staged `START`. |
| Cr-Mn-Fe-Co-Ni FCC | 510 planned candidates; 4 selected for Stage 1 | The available E2 Cantor source is a 108-atom FCC cell at `/lus/flare/projects/HydraGNN/mlupopa/AgenticAI4Materials/e2-dft-v1-20261104/cases/fcc_hea_CrMnFeCoNi_Cantor_108_ideal/`. The proposed 128-atom geometry is generated as a 4x4x2 conventional FCC supercell from the source lattice parameter; it is not an archived 128-atom E4 structure. Suitable as an explicitly derived throughput candidate, not as a previously qualified E4 structure. | No 128-atom SCF/relaxation qualification exists. The archived 108-atom relaxation cases had SCF problems. |
| LiFePO4 | Up to 510 planned candidates; 4 selected for Stage 1 | The 28-atom source is available at `/lus/flare/projects/HydraGNN/mlupopa/AgenticAI4Materials/matsim-agents-dft-reference/dft_reference/vasp/calculations/critical_minerals_LiFePO4_olivine_ideal/`. A 2x2x1 repeat gives 112 atoms. Symmetry-based enumeration produces 510 distinct occupations for this parent: 1 pristine, 8 with one antisite exchange pair, 496 with two pairs, and 5 with three pairs. This is a reproducible candidate inventory, not proof of uniqueness against the inaccessible reference archive. | VASP 6.6.1 compatibility with the archived 6.6.0 input has not been validated. External structure-duplicate review remains incomplete. |

The existing `stage1-inputs/` directory is a first-pass 16-case bundle and is
**stale** relative to the current generator: its LiFePO4 case IDs and manifest
metadata predate the latest symmetry/defect-site changes. Do not treat it as a
frozen or submission-ready input bundle. A regenerated bundle also requires
the approved PAW component root; POTCAR bytes must not be copied into the Git
worktree.

## Recovered methods and outstanding input details

### BCC E4 A/B inputs

The recovered input uses PBE, `ENCUT=350 eV`, `ISMEAR=1`, `SIGMA=0.2 eV`,
`ISPIN=1`, `EDIFF=1e-6`, `ISIF=3`, `IBRION=2`, `NSW=240`, `POTIM=0.1`,
`LREAL=F`, `ISYM=0`, and `KPAR=8`, with Low then Normal precision. Restart
settings switch to `ISTART=1` and `ICHARG=1`. `EDIFFG` is not specified; the
ionic force criterion therefore was not frozen in that deck. The archived
combined PAW hash is
`4c2efb63638e6f89bd80129da1c38cf658ccce4a5c6da45b5e18c760659c11e2`.
The shared A/B input hashes are POSCAR
`76307c91038dc45ba84c0bfe8a77fd5d4cd404d8e7fd32c5894432f46b7e3170`,
KPOINTS
`153f5a32fe014286ff99b50343adaba85ccc699c88bbdaff4d2bacd148e75b47`,
and START
`ece3dfec13b93270b873903f765eca52af2ed1d320d148f0edd0bee175980eb1`.
Published Nb-Ta-V methods corroborate the broad cutoff and mesh but do not
verify all settings or the Hf/Ti PAW releases for this E4 chemistry.

### FCC and LiFePO4 candidate inputs

The Cantor E2 input used PBE, `ENCUT=480 eV`, spin polarization,
`ISMEAR=1`, `SIGMA=0.10 eV`, Gamma 2x2x2, `EDIFF=1e-6`, `NELM=200`,
`ISIF=3`, and `EDIFFG=-0.02 eV/A`. The archived combined PAW hash is
`c0e6f0e3856534892c239c7a8ec155102fdb9ee703a10e88b94d0c13f7ae4918`.
The 128-atom cell and its SCF behavior remain unqualified.

The LiFePO4 source uses PBE without Hubbard U, `ENCUT=520 eV`,
`PREC=Accurate`, `EDIFF=1e-6`, `ISMEAR=0`, `SIGMA=0.05 eV`, `ISPIN=2`,
`IBRION=2`, `ISIF=3`, `NSW=200`, `EDIFFG=-0.01 eV/A`, and archived
Fe_pv/Li_sv/O/P PAWs. The derived 2x2x1 supercell uses a candidate 2x4x8
mesh. The archived run was VASP 6.6.0; 6.6.1 and derived-input compatibility
remain to be checked.

## Archived E4 BCC A/B execution evidence

Jobs 9047960-9047963 all completed the Low-precision ionic relaxation and
started Normal precision, then hit their 8-hour PBS limit (exit `-29`).
Normal-stage values below are partial, not converged results.

| Job | Build | Elapsed | Low ionic steps | Normal partial steps | Low mean LOOP time | Low wall seconds/step |
|---:|---|---:|---:|---:|---:|---:|
| 9047960 | Custom | 08:01:04 | 121 | 19 | 9.78 s | 212.5 s |
| 9047961 | ALCF | 08:01:22 | 134 | 22 | 8.91 s | 188.4 s |
| 9047962 | ALCF | 08:01:23 | 127 | 32 | 8.87 s | 192.4 s |
| 9047963 | Custom | 08:01:30 | 137 | 3 | 9.72 s | 203.6 s |

The Low-stage `reached required accuracy` marker appeared in all four logs;
Normal did not complete in any. The two ALCF runs had lower average LOOP times
and more Low-stage work per elapsed time, but two uncontrolled repeats do not
establish a definitive binary winner. No GPU utilization telemetry was found.
The custom and ALCF executable SHA-256 values begin `ebd2840277` and
`3354871ac0`, respectively; both are VASP 6.6.1. Historical nonempty
WAVECAR/CHGCAR files are present, but trusted restart manifests for those old
attempts are absent, so their restartability is not established.

## Allocations, queue limits, and cost envelope

The read-only Aurora snapshot is dated 2026-11-06. It reports positive
balances and user access for all requested projects:

| Family | Allocation | Balance (node-hours) | Expiration |
|---|---|---:|---|
| BCC | `HydraGNN` | 36,707.0 | 2027-01-10 |
| FCC | `GenesisHackathonOct26` | 14,824.7 | 2027-02-19 |
| LiFePO4 | `GridAI` | 34,029.1 | 2027-01-10 |

These are historical read-only balances, not a current pre-submission check.
Project-specific queue acceptance was not exposed and no submission probe was
made.

The `capacity` queue supports 1-16 nodes and up to 168 hours; `debug-scaling`
supports up to 256 nodes but only a 1-hour walltime in the checked snapshot.
The `prod` route begins at 256 nodes. Thus 32-, 64-, and 128-node tiers do not
have a verified long-walltime route. The existing proposed schedule is
task-parallel weak scaling across disjoint structures, not strong scaling.

For the 16-case Stage 1 subset, the prior planning cap was one node for up to
8 hours per case: **128 node-hours maximum** (64 BCC, 32 FCC, 32 LiFePO4).
This is a ceiling, not a runtime estimate; the FCC and LiFePO4 runtimes are
unknown. Archived BCC A/B jobs consumed about eight node-hours each and still
did not complete Normal precision. The full 3,060-candidate target cannot be
costed credibly from these data. Any revised budget requires approved pilot
measurements and must retain failed-attempt charges.

## PBS and qualification status

The three PBS files under `deployments/aurora/e4-templates/` and the generated
`pbs-dryrun-*.pbs` files are planning-only: they contain placeholders or print
metadata and exit. They are not validated VASP scripts. The E2 runner
`deployments/aurora/launchers/run-e2-dft-case-aurora.sh` has segmented restart
logic, but it has not been validated for E4 precision switching, PAW identity,
or safe walltime checkpointing. Do not submit it for this campaign.

Before any later submission, the following are blocking:

1. Reconcile and approve the BCC ionic stopping criterion and exact PAW
   component releases/checksums, including Hf and Ti.
2. Enumerate the public structure payloads and available local archives for
   duplicate screening; keep candidate IDs blocked until completed.
3. Qualify the 128-atom FCC SCF and verify the LiFePO4 derived input and PAW
   compatibility.
4. Generate a fresh frozen manifest and input bundle from verified parent and
   PAW sources; confirm POSCAR hashes, canonical IDs, species order, and all
   input hashes.
5. Implement and dry-run a restart-safe execution script with explicit signal
   handling and output validation; keep each attempt isolated by case.
6. Revise the scaling plan to tiers supported by verified walltime and queue
   access. In particular, do not assume 32-128-node long jobs are available.
7. Recheck live balances and allocation eligibility immediately before a
   separately approved submission.

## Validation performed for this preparation

- `.venv/bin/ruff check benchmarks/e4/aurora_dft/prepare_stage1.py benchmarks/e4/aurora_dft/planning.py tests/test_e4_stage1.py` — passed.
- `.venv/bin/pytest -q tests/test_e4_stage1.py tests/test_e4_planning.py tests/test_e4_aurora_dft.py` — 18 passed.
- The LiFePO4 source-derived 112-atom parent generated 510 distinct canonical
  occupations with the category counts stated above.
- No PBS job was submitted and no VASP calculation was run.
