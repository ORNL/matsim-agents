# BCC restart qualification proposal

**Status: executed.** The two authorized jobs passed the BCC restart-mechanics
qualification. This is not a convergence, materials-validation, or production
result; full BCC production remains NO-GO.

## Selected case and protocol

Use the existing 128-atom archived E4 BCC A/B case at
`e4-vasp-ab-pilot-inputs/Nb22Ta22V21Hf21Zr21Ti21/case-1`, with the original
calculation/input identity preserved and new output under the unused
`alcf/repeat-03/`. The new repeat starts from the original POSCAR and verified
inputs. It does **not** load or trust files from `repeat-01` or `repeat-02`;
those historical outputs remain unchanged. The source POSCAR composition is
Nb22-Ta22-V21-Hf21-Zr21-Ti21; its SHA-256 is
`76307c91038dc45ba84c0bfe8a77fd5d4cd404d8e7fd32c5894432f46b7e3170`.

The archived deck is PBE, `ENCUT=350 eV`, `PREC=Low` then `Normal`,
`EDIFF=1e-6`, `ISPIN=1`, `ISMEAR=1`, `SIGMA=0.2`, Gamma 2x2x2,
`IBRION=2`, `ISIF=3`, and original `NSW=240`, with `POTIM=0.1`, `LREAL=F`,
`ISYM=0`, and `KPAR=8`. The existing segmented runner sets `NSW=30` per
checkpoint segment, then advances the same structure and precision. The
archived START contains **no explicit `EDIFFG`**. Both qualification OUTCARs
report VASP's positive default `EDIFFG=1e-5 eV`, which is an energy-change
criterion, not a force threshold. It was preserved for this restart-mechanics
test; no replacement force criterion is inferred.

The related published Nb-Ta-V BCC dataset documents a maximum-force target of
`0.01 eV/Å` in its [dataset README](https://ndownloader.figshare.com/files/46336081)
and [Scientific Data article](https://doi.org/10.1038/s41597-024-03720-3).
That is published-method evidence for a force threshold, but it does not prove
that the exact six-element E4 calculation used an explicit `EDIFFG` tag.
VASP's [EDIFFG definition](https://www.vasp.at/wiki/index.php/EDIFFG) says a
negative value terminates ionic relaxation when forces satisfy the magnitude
threshold. Therefore `EDIFFG=-0.01` is the evidence-aligned proposed production
setting, **not a recovered historical tag**. It must be approved before
production. The alternative `-0.02` is twice as loose as the related
published-method threshold and may stop at a different geometry; it is not
recommended. Relative to the exact archived input's energy-change default,
adding any negative `EDIFFG` is a real change in ionic stopping behavior.

### Proposed BCC production protocol (not authorized)

| Item | Proposed value or policy | Evidence/status |
|---|---|---|
| Functional and cutoff | PBE; `ENCUT=350 eV` | Present in the archived BCC inputs and consistent with the related published dataset. |
| PAWs and order | `Nb_pv` (08Apr2002), `Ta_pv` (07Sep2000), `V_pv` (07Sep2000), `Hf_pv` (06Sep2000), `Zr_sv` (04Jan2005), `Ti_pv` (07Sep2000); order Nb-Ta-V-Hf-Zr-Ti | Verified from the private archived BCC input/OUTCAR; retain restricted PAW bytes outside Git. |
| Electronic settings | `ISPIN=1`, `ISMEAR=1`, `SIGMA=0.20 eV`, `EDIFF=1e-6 eV`, `NELM=170`, `LREAL=F`, `ISYM=0`, `KPAR=8`; Gamma-centered 2x2x2 mesh | Recovered from the archived BCC case. |
| Precision | `PREC=Low`, then `PREC=Normal` | Recovered sequence and related published workflow. |
| Ionic relaxation | `IBRION=2`, `ISIF=3`, `POTIM=0.1 Å`, proposed `EDIFFG=-0.01 eV/Å` | First three tags are archived inputs. Negative force threshold is a proposed explicit production criterion; not in the archived case. |
| Step limit | Up to 240 ionic steps per precision stage, with at most 30 steps per checkpoint segment | `NSW=240` is in the archived START; 30-step segmentation is the tested runner policy. A production runner must enforce the cumulative 240-step ceiling, which the current segmented runner does not yet enforce. |
| Restart initialization | First Low segment: `ISTART=0`, `ICHARG=2`; verified same-protocol continuation: `ISTART=1`, `ICHARG=1`; `LWAVE=T`, `LCHARG=T`, `LH5=F` | Restart path was exercised successfully. Hash-check inputs, executable/runtime, and checkpoint files; refuse overwrite; retain one calculation ID and append each allocation's accounting. |
| Executable and placement | `/soft/applications/vasp/vasp.6.6.1/bin/vasp_std`; 8 MPI ranks x 8 threads on one node, ranks bound to tiles 0-7 | Executed and verified in the qualification, SHA-256 `3354871ac0b7a812ebe3444dbd3d5bb4176995a5c37c3859978b58b5c7b55eb6`. Proposed candidate, not an overall performance winner. |

The numerical settings other than the proposed `EDIFFG` are traceable to the
archived case or the tested restart runner; the force threshold is the only
proposed scientific stopping-setting change. The 240-step-per-stage ceiling
is a proposed operational enforcement of the archived `NSW=240` limit across
30-step segments, not a limit currently enforced by the runner. The historical
input remains untouched. Before production, the runner must enforce this
cumulative ceiling and production-specific force/convergence checks.

The refreshed BCC Stage 1 inputs match the recovered cutoff, smearing,
spin, k-mesh, relaxation algorithm, cell/ion relaxation mode, and two precision
stages. Their `EDIFFG` is likewise absent. No POTCAR bytes are added to the
repository. The Stage 1 candidate `E4-BCC-1801` is locally distinct from the
archived A/B occupation, but the restart qualification replays the archived
representative because that gives an exact input baseline.

## VASP and Aurora launch

Executed executable: `/soft/applications/vasp/vasp.6.6.1/bin/vasp_std`,
SHA-256
`3354871ac0b7a812ebe3444dbd3d5bb4176995a5c37c3859978b58b5c7b55eb6`.
This is the ALCF VASP 6.6.1 build; prior Low-stage wall time was about 8.5%
lower on average than the custom build, but that is not an overall performance
selection.

The PBS wrapper loads `oneapi/release/2026.1.0`,
`intel_gpu_umd_aicoe/2026.06.19`, and
`mpich/prd/5.0.0.aurora_test.51a9474`. Login-node module loading and binary
linkage were checked; `mpiexec` is PALS 1.8.0 and `ldd` found no missing
libraries. The existing launcher assigns 8 MPI ranks x 8 OpenMP threads on one
node, one flat XPU tile per rank (tiles 0–7), `--cpu-bind=depth`,
`MPICH_GPU_SUPPORT_ENABLED=1`, and Level Zero GPU selection. The authorized
continuation confirmed 8 MPI ranks bound to XPU tiles 0–7 and VASP reported
8 GPUs detected. PBS accounting reported `resources_used.ngpus=0`; retain that
discrepancy alongside the VASP offload evidence rather than treating the PBS
field alone as proof that no accelerators were used.

The job records show a full-node allocation (`select=1`, 208 CPUs in
`exec_vnode`) and do not request a per-GPU PBS resource. The inspected
[Aurora machine overview](https://docs.alcf.anl.gov/aurora/) and
[Aurora PBS guide](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/)
describe node-level scheduling and GPU-enabled applications, but do not
explicitly define the meaning of `resources_used.ngpus=0`. Runtime evidence
(PALS rank-to-tile binding and VASP's eight-GPU detection) supports that GPUs
were available to and detected by VASP; the PBS field alone is not evidence
of CPU fallback. The accounting convention is the more likely explanation,
but the public documentation inspected does not settle that field's precise
semantics.

## PBS script and run sequence

Exact script:
[`job-e4-bcc-restart-qualification.pbs`](../../../../deployments/aurora/jobs/job-e4-bcc-restart-qualification.pbs)

It requests one node, `capacity`, `HydraGNN`, and 15 minutes. Job 1 used
`E4_RESTART_PHASE=first` and was PBS job
`9051522.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`. After independent
checkpoint validation, the single authorized continuation used
`E4_RESTART_PHASE=resume` as PBS job
`9051551.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`. No other jobs were
submitted. Each PBS output is unique by job ID in the archived case directory;
the wrapper does not submit jobs itself.

The BCC runner verifies the input and executable hashes, rejects a pre-existing
repeat for a fresh run, and persists the calculation identity, checkpoint
manifest, attempts, and run record under the original case ID. Resume identity
now also covers the VASP step-launcher hash, MPI module, rank/thread layout,
device mask, and stack size. Each attempt has a separate directory; only
runner-created, hashed `CONTCAR`, `WAVECAR`, and `CHGCAR` checkpoints can be
consumed. Legacy A/B files are never imported.

## Duration, cap, and acceptance

The four archived Low stages took 24,431–27,893 seconds and averaged about
188–213 wall seconds per ionic step. Neither 15-minute allocation could
complete Low or reach Normal precision. Job 1's checkpoint request at 480
seconds took 695 seconds from launch to VASP return. For Job 2, the internal
soft-stop deadline was therefore set to 680 seconds (without changing VASP
inputs or the checkpoint protocol), leaving time for VASP to write a new
checkpoint before the scheduler limit.

| Jobs | Nodes/job | Walltime/job | Maximum node-hours |
|---:|---:|---:|---:|
| 2 sequential | 1 | 15 minutes | **0.50** |

Observed acceptance:

1. Job 1 verified the frozen input and executable hashes, completed two ionic
   steps (43 and 24 SCF iterations), and wrote nonempty `CONTCAR`, `WAVECAR`,
   and `CHGCAR`. VASP returned normally after the requested soft stop. The
   runner incorrectly classified the stop as `failed:unrecognized-stop`
   because VASP had removed `STOPCAR`; its PBS exit status was 1. Independent
   file, geometry, force, SCF, and VASP-footer checks passed, and the checkpoint
   was registered with a SHA-256 manifest for the conditional continuation.
2. Job 2 verified the same protocol identity and checkpoint manifest, logged
   source tag 0, and created an isolated attempt and a new valid tag-1
   checkpoint. Its wrapper passed with PBS exit status 0.
3. The resumed INCAR used `ISTART=1` and `ICHARG=1`; VASP reported reading PAW
   grids/wavefunctions and the supplied charge density. The resumed POSCAR is
   byte-identical to Job 1's CONTCAR. The cumulative calculation ID and run
   record were preserved.
4. Both jobs completed two ionic steps; neither reached relaxation
   convergence. No historical `repeat-01`/`repeat-02` output was used or
   modified, and no further jobs were submitted.

The measured restart-boundary force difference was at most
`8.08e-4 eV/Å` componentwise (RMS `2.04e-4 eV/Å`); the maximum force magnitude
changed from `1.323919` to `1.323934 eV/Å`. At the identical boundary
coordinates, the first resumed OSZICAR `E0` was `0.0207 eV` above the previous
final `E0`. No numerical acceptance tolerance was specified for this
mechanics test, so this is reported, not silently judged against a threshold.
The relaxation remains scientifically unqualified and incomplete.

### Restart-boundary numerical interpretation

The compared records are at identical coordinates: Job 1's checkpoint
CONTCAR and Job 2's resumed POSCAR are byte-identical, and the parsed
coordinates have zero displacement. Both records use the same Low-precision
input, cell, cutoff, k-mesh, smearing, spin, and electronic convergence
settings. Initialization differs as intended: the initial calculation used
`ISTART=0`, `ICHARG=2`; the resumed attempt used `ISTART=1`, `ICHARG=1`.
The compared SCF calculations took 24 and 35 iterations, respectively, and
both OUTCAR records report that `EDIFF=1e-6` was reached.

Recomputed from the matching records, `E0` changed from `-1198.07846318 eV`
to `-1198.05781549 eV` (`+0.02064769 eV`). The maximum absolute force
component difference is `0.00080791 eV/Å`; RMS over all components is
`0.00020376 eV/Å`. Maximum atomic force magnitude changed from
`1.32391921` to `1.32393378 eV/Å`. The values are not explained by coordinate
rounding or differing INCAR settings. Different electronic initialization
and SCF trajectories are a possible source, but `EDIFF` convergence within
each run does not bound the difference between converged states. The records
alone cannot establish that the shift is normal reconvergence, nor establish
numerical equivalence. No retrospective pass/fail threshold is applied.

### Smallest proposed numerical diagnostic (not authorized)

If numerical follow-up is desired, use two independent cold-start single-step
calculations at the exact shared boundary geometry in separate, new output
directories. Use the same VASP binary, modules, MPI/tile placement, PAW,
KPOINTS, and Low-precision numerical settings; start each with
`ISTART=0`, `ICHARG=2`, and `NSW=1`. Keep all historical inputs and attempts
read-only. Each job would request one node for at most 15 minutes; maximum
cost is two jobs and 0.50 node-hours. Do not retry automatically.

For each job require a normal VASP exit, a completed ionic/electronic record,
the `EDIFF` reached marker, finite energy and forces, and exact matching
starting coordinates and input/executable hashes. Record the energy and force
differences between both cold starts and the existing cold-start and resumed
records. These conditions qualify execution and produce comparable data, but
do not themselves establish equivalence. A numerical-equivalence decision
requires explicit, pre-registered energy and force tolerances; none are
retrospectively assigned here. A timeout or non-converged SCF is incomplete,
not evidence of numerical failure. This proposal is diagnostic only and
requires separate authorization before any submission.

## Queue and unresolved risks

`qstat -Qf capacity` reported the queue enabled with 1–16 nodes and 5-minute
to 168-hour walltimes. Both authorized jobs were accepted under `HydraGNN` and
`capacity`. `sbank` reported 36,674.7 node-hours before the test and 36,674.2
after it. Job 1 used 705 seconds (0.1958 elapsed node-hours) and Job 2 used
829 seconds (0.2303 elapsed node-hours), for 0.4261 elapsed node-hours total.
Scheduler accounting deducted 0.25 node-hours for each one-node 15-minute
allocation, 0.50 node-hours total, the full two-job requested cap.

The runner now requires a nonempty `E4_CALCULATION_ID` and appends one
`allocation_accounting` record per PBS allocation, including PBS job ID,
calculation ID, unique node count from `PBS_NODEFILE`, requested node-hours,
and measured elapsed node-hours. The run record is cumulative across resumes.
Elapsed use and requested allocation are reported separately; the scheduler's
accounting balance remains authoritative for charged node-hours. This runner
change postdates the completed qualification jobs; their per-job elapsed and
charged totals are reported above and in the scheduler records, not
retroactively inserted into their historical run record. Focused regressions
cover missing IDs, no-overwrite behavior, CHGCAR integrity, and accounting
across two allocations.

Full BCC production remains blocked pending approval of the proposed
`EDIFFG=-0.01 eV/Å`, implementation of the cumulative 240-step limit, and
numerical qualification. Public BCC structure archives remain incompletely
enumerated; this does not make the restart test a new scientific claim. Before
production, document residual external-duplicate risk and exclude the
confirmed LiFePO4 pristine supercell reuse `E4-LFP-0001`; retain the frozen
510-structure inventory and all remaining candidate provenance.

The FCC and LiFePO4 staged protocols have explicit `EDIFFG`, `IBRION=2`,
`ISIF=3`, and `NSW=200` (`-0.02` and `-0.01 eV/A`, respectively) and their
derived inputs match the recovered source settings, including their stated
meshes and PAW orders. Their capacity-queue project ACLs and restart behavior
have not been independently demonstrated. No FCC/LiFePO4 job is proposed or
authorized by this report.

## Execution outcome

- **Technical restart reliability: GO for this controlled BCC case.** Both
  checkpoints passed SHA-256 verification; Job 2 demonstrably read and
  continued from Job 1's saved wavefunctions/charge and produced the next
  isolated checkpoint under the unchanged protocol identity.
- **Numerical continuity: reported, not qualified.** The restart-boundary
  force differences and `0.0207 eV` `E0` shift are recorded above; no
  pre-approved acceptance tolerance exists.
- **Scientific interpretation and production readiness: NO-GO.** The test
  preserved the archived deck, which has no `EDIFFG`, and stopped after two
  ionic steps per job. It establishes restart mechanics only, not relaxation
  convergence or scientific validity.
- **Accounting:** PBS jobs 9051522 and 9051551 used 11:45 and 13:49 of their
  15-minute requests; `sbank` charged 0.50 node-hours in total. Both scheduler
  records report `resources_used.ngpus=0`, while VASP reported eight detected
  GPUs and the launcher bound ranks to tiles 0–7.

The runner now recognizes a valid `checkpoint-interval` stop even when VASP
deletes `STOPCAR` during normal shutdown. A focused regression test covers this
behavior. Pytest collection timed out in this environment; the focused test
function was executed directly, and shell syntax, lint, and direct checkpoint
regressions passed.
