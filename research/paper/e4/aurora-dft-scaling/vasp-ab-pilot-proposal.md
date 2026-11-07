# VASP Binary A/B Pilot Proposal

## Decision and boundary

Status: **SUBMITTED with explicit user approval (2026-11-06).** Four
independent 1-node, 8-hour `capacity` jobs on `HydraGNN`, using
`deployments/aurora/jobs/job-e4-vasp-ab-aurora.pbs`: custom/repeat-01
9047960, alcf/repeat-01 9047961, alcf/repeat-02 9047962, custom/repeat-02
9047963. Do not resubmit, restart, or cancel without user approval.

The comparison target is the same 128-atom, Hf/Ti-containing BCC case with the
custom repository build and the ALCF-provided binary. The recovered E2 ideal
POSCAR is an eligible comparison structure after the E4 input protocol is
verified, but remains excluded from E4 production reuse. Its source and E2
provenance are documented in `existing-data-audit.md`.

## Input and binary identity

| Item | Custom build | ALCF-provided build |
|---|---|---|
| Path | `external/vasp6/src/vasp.6.6.1/bin/vasp_std` | `/soft/applications/vasp/vasp.6.6.1/bin/vasp_std` |
| SHA-256 | `ebd284027717ff564abf7ceaf93c3aca3261b129d7b60b3f2e3e2aa82d04c552` | `3354871ac0b7a812ebe3444dbd3d5bb4176995a5c37c3859978b58b5c7b55eb6` |
| VASP version | 6.6.1 | 6.6.1 |

Both binaries must run in the same explicitly captured Aurora module environment,
including `mpich/prd/5.0.0.aurora_test.51a9474`. ALCF documents that this module's
`LD_LIBRARY_PATH` takes precedence over binary RUNPATH and recommends it to avoid
the GPU-aware MPI performance regression in the default `87e2045` build. Verify
the effective MPI library resolution for both binaries without starting VASP.
Do not rebuild either binary or modify the existing launchers for this pilot.

Use identical per-stage `POSCAR`, `INCAR`, `KPOINTS`, and ordered PAW
concatenation across the two binaries. Record SHA-256 values for all inputs and
PAW components; never publish POTCAR bytes. Each binary gets its own outputs,
and stage 2 consumes only the corresponding binary's stage-1 outputs. Use the
same rank/thread count, GPU mapping, CPU affinity, filesystems, staging policy,
VASP environment variables, and workflow for both binaries. Capture full module
and environment snapshots.

Run layout (user decision): each VASP run gets an exclusive node and uses 8
of its 12 Intel GPU tiles, via the repository's single-node launcher
(`deployments/aurora/launchers/_vasp-step-aurora.sh`): 8 MPI ranks, one per
FLAT tile (`ZE_FLAT_DEVICE_HIERARCHY=FLAT`,
`--gpu-bind=list:0:1:2:3:4:5:6:7`), 8 OpenMP threads per rank
(`--depth=8 --cpu-bind=depth`), `OMP_STACKSIZE=512m`,
`MPICH_GPU_SUPPORT_ENABLED=1`, and `KPAR=8`. The 2x2x2 mesh with `ISYM=0`
has 8 k-points; Davidson with several ranks per k-point hangs on Aurora
(`docs/vasp-aurora.md`), and this layout already ran this structure in E2.
Only the VASP binary path differs between the two variants.

### Run and checkpoint naming

Follow the user's composition/case hierarchy. Each atomistic structure owns a
separate case directory, containing its immutable starting inputs. All generated
files for that structure stay below that case directory; no run may read,
write, restart from, or archive files belonging to another case.

```text
<campaign>/<composition>/case-1/POSCAR
<campaign>/<composition>/case-1/KPOINTS
<campaign>/<composition>/case-1/POTCAR
<campaign>/<composition>/case-1/START
<campaign>/<composition>/case-1/custom/repeat-01/0.*
<campaign>/<composition>/case-1/custom/repeat-01/attempts/attempt-01/
<campaign>/<composition>/case-1/custom/repeat-02/0.*
<campaign>/<composition>/case-1/custom/repeat-02/attempts/attempt-01/
<campaign>/<composition>/case-1/alcf/repeat-01/0.*
<campaign>/<composition>/case-1/alcf/repeat-01/attempts/attempt-01/
<campaign>/<composition>/case-1/alcf/repeat-02/0.*
<campaign>/<composition>/case-1/alcf/repeat-02/attempts/attempt-01/
<campaign>/<composition>/case-2/...
```

For example, a composition directory can be named `Ta16Nb88Zr24`, with
distinct atomistic structures in `case-1`, `case-2`, and so on. `POSCAR`,
`KPOINTS`, `POTCAR`, and `START` belong to that case only and are treated as
read-only source inputs during the pilot. Copy them into each isolated
binary/repetition attempt directory and verify their hashes. Do not point VASP
at another case's inputs, checkpoints, or outputs. Case directories must not
share mutable files or a writable working directory.

Within each binary/repetition root, store the numbered snapshots directly at
that root and give every invocation a fresh work directory under `attempts/`.
Retain snapshots using the user's precision prefixes, incrementing the number
for each same-precision restart. `N` explicitly marks the switch to Normal
precision; its sequence starts at zero independently:

```text
Low:    0.INCAR  0.POSCAR  0.CONTCAR  0.OUTCAR  0.XDATCAR  0.WAVECAR  0.CHGCAR
  1.INCAR  1.POSCAR  1.CONTCAR  1.OUTCAR  1.XDATCAR  1.WAVECAR  1.CHGCAR
  2.INCAR  2.POSCAR  2.CONTCAR  2.OUTCAR  2.XDATCAR  2.WAVECAR  2.CHGCAR ...
Normal: N0.INCAR N0.POSCAR N0.CONTCAR N0.OUTCAR N0.XDATCAR N0.WAVECAR N0.CHGCAR
  N1.INCAR N1.POSCAR N1.CONTCAR N1.OUTCAR N1.XDATCAR N1.WAVECAR N1.CHGCAR
  N2.INCAR N2.POSCAR N2.CONTCAR N2.OUTCAR N2.XDATCAR N2.WAVECAR N2.CHGCAR ...
Logs:   rlx1.0.out rlx2.0.out ...; rlx1.out rlx2.out ...
```

These are archived snapshots at the binary/repetition root, not VASP's live
working files. Use VASP's standard mutable filenames only inside the current
attempt work directory. Before a same-precision restart, seed a fresh work
directory from the latest completed snapshot of that precision, then archive
the new result under the next unused number (`0.*` to `1.*` to `2.*`, or `N0.*`
to `N1.*` to `N2.*`).
For the precision switch, seed Normal `N0.*` from the selected Low snapshot,
including its paired checkpoint files. A checkpoint is always paired with its
case, binary, repetition, precision, and snapshot number. Preserve logs in the
attempt directory using the user's log names.

Copy snapshots with no-clobber semantics: stop if a numbered destination
already exists, and verify hashes after copying. The attached
SQLite/ThreadPoolExecutor runner claims distinct database rows but does not set
a per-command working directory, so each command must explicitly enter its
assigned attempt work directory. Keep this pilot sequential (`-j 1`); any
concurrent workers must have distinct case/binary/repetition/attempt
directories. Database row locking does not prevent file collisions. The
related NbTaV archive documents `0.CONTCAR`, `0.OUTCAR`, and `rlx1.out` for Low;
the `N0.*` convention is from the user's established workflow.

At the user's direction, the pilot case uses the E2 ideal BCC structure and
PAW components reordered to the species order `Nb Ta V Hf Zr Ti`. The case
inputs are staged privately outside the repository at
`/lus/flare/projects/HydraGNN/mlupopa/AgenticAI4Materials/e4-vasp-ab-pilot-inputs/Nb22Ta22V21Hf21Zr21Ti21/case-1/`:

| File | SHA-256 | Content |
|---|---|---|
| `POSCAR` | `76307c91038dc45ba84c0bfe8a77fd5d4cd404d8e7fd32c5894432f46b7e3170` | Nb 22, Ta 22, V 21, Hf 21, Zr 21, Ti 21 (128); Direct; a = 13.3667 A |
| `POTCAR` | `4c2efb63638e6f89bd80129da1c38cf658ccce4a5c6da45b5e18c760659c11e2` | Nb_pv 08Apr2002, Ta_pv 07Sep2000, V_pv 07Sep2000, Hf_pv 06Sep2000, Zr_sv 04Jan2005, Ti_pv 07Sep2000 |
| `KPOINTS` | identical to E2 source | Gamma-centred 2x2x2 |
| `START` | `ece3dfec13b93270b873903f765eca52af2ed1d320d148f0edd0bee175980eb1` | Supplied INCAR template with the user's decisions: `ISTART=0`, `ICHARG=2`, `ISMEAR=1`, `SIGMA=0.2`, `KPAR=8`; `PREC = PRECISION` placeholder |

They were derived from the E2 case
`/lus/flare/projects/HydraGNN/mlupopa/AgenticAI4Materials/e2-dft-v1-20261104/cases/bcc_hea_NbTaVHfZrTi_equiatomic_128_ideal/`
(POSCAR `798e572ee9dd7c5a6a40a0a1c80989a0e1c259548f22d503a69ca1a1440044d7`,
POTCAR `40c87ed0dc4b827194210fd185433b977e945a39544e0e3dd0f0905491151e2d`,
order Hf Nb Ta Ti V Zr), which was not modified. Verified: lattice and
per-species site sets are identical to the source; each POTCAR component is
byte-identical to its source component; POTCAR order matches the POSCAR
species line. The source's all-zero velocity block was dropped. `START` is
the INCAR template; each stage's INCAR is `START` with `PRECISION` replaced by
`Low` or `Normal`. The largest component ENMAX is 263.673 eV, so
350 eV exceeds each component's ENMAX. This records the selected pilot input;
it does not establish that this was the original E4 production PAW set or
independently validate Hf/Ti/Zr suitability for a production-equivalence
claim. Keep the POTCAR itself private.

The Nature Scientific Data article for 128-site Nb-Ta-V BCC alloys
(https://doi.org/10.1038/s41597-024-03720-3) reports PBE, PAW, 350 eV,
2x2x2, normal precision, Methfessel-Paxton smearing with 0.2 eV broadening,
and a 0.01 eV/A force threshold. It describes low- then normal-precision
relaxation and gives an energy-change threshold of 1e-4 eV. Its linked OSTI
record (10.13139/OLCF/2217644) and Figshare README support those details for
Nb-Ta-V, but the README gives a different energy-change threshold (1e-6 eV).
The paper does not specify numeric `ISMEAR`, the complete relaxation tags, or
PAW release identifiers/checksums. This is related protocol evidence, not the
production input provenance for the six-element E4 case.

The exact E4 `ISMEAR`, `SIGMA`, remaining relaxation tags, and original
production PAW provenance remain unresolved. The selected comparison PAW set
is from E2, whose input deck uses a different 345 eV / 0.10 eV-smearing
protocol; do not copy those E2 INCAR settings into this comparison. The
user-selected PAW set permits planning with that explicit choice, but it does
not support a claim that the pilot reproduces the original E4 production
method.

## Supplied INCAR template

### Checkpoint policy for future submissions (2026-11-07)

The job now overrides only generated attempt INCAR copies with `NSW=30`,
`LWAVE=.TRUE.`, `LCHARG=.TRUE.`, and `LH5=.FALSE.`. Immutable case inputs,
their hashes, and the submitted jobs above are unchanged. VASP writes usable
WAVECAR/CHGCAR outputs at normal segment termination, not after every ionic
step. Each 30-step segment that exhausts NSW continues at the same precision
from its own CONTCAR/WAVECAR/CHGCAR; Low must converge before Normal starts.
`E4_NSW` can override the segment length (positive integer), and the same
setting must be used for both variants. Segment restarts reset some
ionic-optimizer history, so timings from this policy must not be pooled
unqualified with uninterrupted 240-step attempts.

In addition to the step limit, each segment requests an ionic-boundary STOPCAR
after 3,600 seconds (`E4_CHECKPOINT_SECONDS`, positive integer). An accepted
time-triggered checkpoint is classified as `checkpoint:time` and continues at
the same precision within the allocation. The timer requests a stop, not a
hard interruption: completion can exceed one hour by the remaining ionic-step
and output time. If the job-wide deadline comes first, its stop takes priority.
Use identical segment timing and step limits for both A/B variants.

The job requests `LSTOP=.TRUE.` in the active attempt's STOPCAR after 27,000
seconds from script startup (7.5 hours), reserving 30 minutes of the requested
8-hour walltime for an ionic-step boundary, restart output, and archiving.
`E4_SOFT_STOP_SECONDS` may override that elapsed-time threshold, but must be
positive and below 28,800 seconds. It is a job-wide deadline, not a timer reset
for each segment. TERM/USR1 also request this graceful stop. No new segment or
precision stage starts after a job-wide stop request. Adjust the threshold
earlier if one ionic step plus output can exceed the reserved time; this is
not a guarantee against scheduler termination.

Normally terminated attempts must have at least one completed ionic step and
nonempty CONTCAR, WAVECAR, and CHGCAR
before being accepted as checkpoints. Each accepted snapshot has a
`<tag>.checkpoint.sha256` manifest, and the run record identifies its tag and
precision. STOPCAR is retained with the snapshot when present. A graceful
STOPCAR stop without ionic convergence is recorded as `interrupted:stopcar`
and exits with status 75, not as a physical convergence failure. Abnormal
termination, electronic nonconvergence, and empty restart files remain
failures. A stop between segments also exits 75 and leaves the preceding
accepted snapshot intact. There is no automatic cross-allocation resubmission;
resuming an interrupted job still requires user approval and its paired,
same-case/binary/repetition checkpoint. Do not reuse a STOPCAR when restarting.

#### Explicit recovery across allocations

For repetitions created with this revised policy, submit the same job with
the same `REPO`, `CASE_DIR`, `VARIANT`, and `REPEAT`, adding `E4_RESUME=1`.
Submission remains a manual, user-approved action; the job never calls qsub.
Fresh mode (`E4_RESUME=0`, default) refuses an existing repetition.

An exclusive `flock` lock prevents concurrent allocations for one repetition.
Allocation-specific environment snapshots and stop markers are retained under
`allocations/allocation-*`; old stop requests do not affect a resumed allocation.
The protocol identity binds the canonical case path, binary checksum, variant,
repetition, source input hashes, segment length, and segment timer. Resume
rejects identity changes, missing metadata, or altered checkpoint files.
The soft-stop threshold may be adjusted between allocations without changing
the physical inputs or segment policy.

Each accepted manifest hashes INCAR, OUTCAR, OSZICAR, CONTCAR, WAVECAR, and
CHGCAR. Only after manifest creation does an atomic rename publish
`latest-checkpoint`, recording the tag, precision, status, and manifest hash.
Resume verifies that exact file set against the recorded manifest and restarts
from the latest accepted snapshot, never from an abandoned attempt's files.
Attempt directories and snapshot tags are reserved before launch; numbering
skips abandoned attempts and never overwrites prior snapshots.

An interrupted Low checkpoint resumes Low; a converged Low checkpoint begins
Normal; an interrupted Normal checkpoint resumes Normal. An already converged
Normal checkpoint exits successfully without running VASP. Geometry,
wavefunctions, and charge density are always copied from the same snapshot,
without its STOPCAR. Hash verification detects changes but is not a full VASP
restart-readability test.

Legacy jobs, including the originally submitted repeat-01/repeat-02 jobs,
lack the revised trusted metadata and cannot use this resume mode directly.
Their outputs require separate validation and controlled migration; the
runner does not automatically bless old or partially written files.

The user supplied the following INCAR text for the comparison. It is recorded
verbatim as the common template. Substitute `PREC=Low` for stage 1 and
`PREC=Normal` for stage 2 before hashing each stage's INCAR.

```text
#ISTART =             0     #  startjob: no WAVECAR file
ISTART =             1     #  startjob: no WAVECAR file
ICHARG =             1     #  charge: from atoms
#ICHARG = 12                # playing with convergence
#ICHARG = 2                 # playing with convergence
#AMIX = 0.1
#BMIX = 0.01
#INIWAV =             1     #  random initialization for wf.
NELM   =             170    #  maximum of 40 electronic steps
NELMIN =              2    #  minimum of two steps
#NELMDL =             -5    #  no update of charge for 3 steps
PREC   =             PRECISION
ENCUT  =             350.0
EDIFF  =             1.0E-6 # accuracy for electronic minimization
ISPIN  =             1
NSW    = 240  !   allow 10 steps
#ISIF   = 2   !   relax ions only
#ISIF   = 7   !   relax volume only
ISIF   = 3   !   relax volume ions shape
IBRION = 2   !   use CG algorithm
POTIM  = 0.1 !   reduce trial step
#LREAL  = Auto
LREAL  = F
ISYM = 0
```

The requested relaxation sequence is:

1. Stage 1: start from this case's `POSCAR` and run `PREC=Low` in a fresh
  work directory. Archive the first Low attempt as `0.*`; if it is resumed at
  the same precision, archive subsequent attempts as `1.*`, `2.*`, etc.
2. After the selected Low result meets the agreed ionic convergence criterion,
  seed a fresh Normal work directory from that Low snapshot and its paired
  checkpoint files. Begin the Normal sequence at `N0.*`.
3. Stage 2: run `PREC=Normal`. Same-precision Normal restarts advance to
  `N1.*`, `N2.*`, etc. If using `ISTART=1` and `ICHARG=1`, restore only the
  matching case/binary/repetition checkpoint files to VASP's live filenames.
  Never share restart state across cases or binary runs.
4. Record the two stages independently using the `0.*` and `N0.*` artifact
  prefixes above. Each workflow ends with the converged Normal stage; there is
  no final static calculation (user decision).

Before use, resolve these input issues explicitly:

- Resolved (user decision): the first Low run uses `ISTART=0` and `ICHARG=2`
  (as staged in `START`); every later Low restart and every Normal run uses
  `ISTART=1` and `ICHARG=1` with its own paired `WAVECAR` and `CHGCAR`.
- Resolved (user decision): `ISMEAR=1`, `SIGMA=0.2`.
- Resolved (user decision): `EDIFFG` is not set, so VASP's default applies
  (10 x `EDIFF` = 1e-5 eV energy-change criterion for ionic relaxation);
  `NSW=240` is the maximum ionic-step count per attempt.
- The comments describing 40 electronic steps and 10 ionic steps do not match
  `NELM=170` and `NSW=240`. Confirm whether those comments are stale.

The proposal remains planning-only until these choices, the stage-specific
`PREC` values, `KPOINTS`, selected starting structure,
and staged restart state are frozen identically for both binaries. See the VASP wiki entries for
[`ISTART`](https://www.vasp.at/wiki/index.php/ISTART) and
[`ICHARG`](https://www.vasp.at/wiki/index.php/ICHARG).

## Resource-capped sequence

Proposed allocation: `HydraGNN`; one exclusive Aurora node in `capacity` for a
single allocation of at most 32 hours. The 8-hour per-workflow cap is a
resource ceiling, not an estimate or guarantee of completion. `capacity` is the documented route for
one node and walltime above one hour. Project-specific queue acceptance remains
unverified. Do not use `debug` or `debug-scaling`: both are limited to one hour
in the checked snapshot, while the archived E2 relaxation alone took 21,157
seconds (5.88 hours). This E2 runtime is planning context, not a forecast for
the E4 protocol.

Run four workflows sequentially on the same node, in counterbalanced order
`custom/repeat-01, ALCF/repeat-01, ALCF/repeat-02, custom/repeat-02`. Each
workflow includes the `PREC=Low` relaxation and the `PREC=Normal`
continuation from its converged geometry, and ends there. This gives two
repetitions per binary and limits
order and node variation. Each complete workflow is capped at 8 hours.

Automatic restart (user decision): if a Low or Normal run exhausts
`NSW=240` ionic steps without reaching the ionic convergence criterion, the
workflow automatically starts the next same-precision attempt (`1.*`, `2.*`,
... for Low; `N1.*`, `N2.*`, ... for Normal) from that attempt's `CONTCAR`,
`WAVECAR`, and `CHGCAR` with `ISTART=1` and `ICHARG=1`. NSW exhaustion is
identified from the OUTCAR: 240 completed ionic steps and no
`reached required accuracy` marker. Any other termination (VASP error,
non-zero exit, missing outputs, electronic non-convergence) stops the
workflow without restart. Restarts count against the same 8-hour cap. Stop a
workflow at its cap; do not continue into an uncapped job.
Maximum charge is therefore 32 node-hours, including incomplete attempts.
Record actual allocated walltime and charge even when a workflow fails. If
either binary fails to complete within its cap, report that result and do not
extend the cap without separate approval.

The structure pair is an intentional comparison control, not two new unique
production structures. Do not include these four runs in E4 unique-calculation
counts. Preserve independent input, stdout/stderr, and output directories for
all four runs; never let one binary consume another binary's output. Any
separately approved retry must use a new attempt directory, not overwrite a
prior run's checkpoint files.

## Measurements and qualification

For every run, record binary and input hashes, module list, MPI library
resolution, node identity, rank/thread and affinity layout, start/end times,
allocated node-hours, VASP-reported elapsed and CPU time, SCF iteration counts
per ionic step, ionic step count and stop reason, final energy and forces,
stress and cell, peak memory, exit status, warnings, and hashes of retained
outputs. Distinguish scheduler wait from allocated runtime.

Scientific qualification is evaluated before comparing speed:

- Every ionic step has a converged electronic calculation; both precision
  stages end with VASP's `reached required accuracy` marker under the default
  `EDIFFG`, not interruption. Stage 2 starts from stage 1's converged
  `CONTCAR` and matching restart files.
- Report final maximum forces. Aligned
  relaxed structures must have matching atom mapping, fractional positions,
  and lattice within preregistered tolerances.
- Proposed cross-binary acceptance thresholds, to be confirmed before
  execution: final energy difference no more than 1e-4 eV/atom, maximum aligned
  force-component difference no more than 5e-3 eV/A, and relative lattice
  difference no more than 1e-4. Also compare the energy, forces, and stress of
  the first ionic step of `0.*`, which both binaries compute at the identical
  starting geometry.
- Report convergence warnings and any differing ionic paths or iteration
  counts. Bitwise identity is not required. A failed numerical gate blocks a
  performance-winner claim even if one binary is faster.

Report per-workflow walltime and node-hours, medians and spread across the two
repetitions, and speedup only if all four workflows qualify. With two
repetitions, treat the result as a bounded pilot, not a broad benchmark.

## Release gates

Before any separately approved execution:

1. Freeze the comparison `INCAR`, `KPOINTS`, POSCAR, and the selected POTCAR;
  verify hashes and all required tags. Do not claim reproduction of the
  original E4 production method unless its source inputs are independently
  recovered.
2. Keep the selected POTCAR private and retain its source and checksum in the
  execution record.
3. Confirm `HydraGNN` is accepted on `capacity`, and confirm the 1-node / 32-hour
   resource request is allowed by current queue and project policy.
4. Freeze the common inputs, runtime environment, rank/thread/GPU mapping,
   numerical thresholds, output paths, and stop behavior in a reviewed
   execution plan.
5. Obtain explicit user approval for that exact execution plan.

Until all gates are closed, this document is not an execution plan and the
existing invalid-placeholder PBS templates remain inert.