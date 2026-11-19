# Nb-Ta-O Qualification Authorization Request

**Status:** fixed inputs are prepared but not numerically qualified. PBS job `9037133` used the superseded input deck and failed with exit status 127 before entering the runner; VASP did not execute. Its external case, POTCAR, and PBS log are preserved unchanged. No job has been submitted for the fixed protocol. The prior Q1 authorization does not cover changed numerical inputs; a separate authorization is required before a new submission.

## PAW Provenance

The user approved use of the exact `Nb_pv`, `Ta_pv`, and `O` component files from `external/vasp6/potcar`. Their local hashes and headers identify these files; the available evidence still does not establish an official homogeneous release identifier. No POTCAR payload is reproduced here. The selected components' maximum ENMAX is 400.000 eV, giving `ENCUT=max(600,1.3*400)=600 eV`.

| Component | Local `TITEL` header | `ENMAX` (eV) | Local SHA-256 |
|---|---|---:|---|
| `Nb_pv` | `PAW_PBE Nb_pv 08Apr2002` | 208.608 | `086061c276946e9fc55bb7633d9ce78b32a420dec762592491c07c1c47ded114` |
| `Ta_pv` | `PAW_PBE Ta_pv 07Sep2000` | 223.667 | `ffb05e92ac3f68ca84c26af10a80f2fbb0cf79e16fcc96fcae98089455bbbaaa` |
| `O` | `PAW_PBE O 08Apr2002` | 400.000 | `8a74b9a1f5fdb3d0c3e0183c7873177abdbef07d407b310b7edcd9ed0a3eea64` |

These verify local file identity and the declared PBE component labels, not the parent archive, release date, homogeneous provenance, or right to use the files. The local library has 314 components, mixed filesystem timestamps, and no verified release marker. Component dates in `TITEL` are not treated as a release identifier.

**Recorded approvals and remaining evidence:**

- The official VASP PAW release/catalog identifier, release date, source/install path, and authoritative checksum manifest; whether all three selected components belong to that same authorized release; and whether these exact hashes are approved or must be replaced.
- That the VASP license and PAW access cover the named user, Aurora project/account, compute facility, and proposed use, including approved storage locations and handling restrictions.
- The user approved the facility/account limit and the original Q1 request; PBS accepted account `HydraGNN`. The revised fixed protocol requires a new authorization. Its four-workflow qualification ceiling is **8.0 node-hours**, with Q1 capped at **2.0 node-hours**.
- The PBS walltime is 120 minutes per workflow on the `capacity` queue, with a soft stop at 115 minutes. It was raised from 30 minutes after job `9039526` ran 18 ionic steps (~80 s each) without converging before its 25-minute soft stop.

## Fixed Numerical Setup

- `PREC=Accurate`, `ENCUT=600 eV`, `EDIFF=1E-6`, and `EDIFFG=-0.01` for relaxations.
- Periodic structures use only explicit Gamma-centered regular-mesh KPOINTS at target reciprocal spacing 0.15 A^-1. For the five pilot POSCARs, the mesh is `4 8 9`; their actual cell reciprocal lengths are identical.
- Mesh rule: `N_i=max(1,ceil(|b_i|*2*pi/0.15))`, with reciprocal vectors `b_i` excluding 2*pi. The generator's Pymatgen reciprocal lengths include 2*pi and therefore directly divides those lengths by 0.15.
- Relaxation KPOINTS is computed from its POSCAR. Static KPOINTS is recomputed from the relaxed CONTCAR and both per-calculation mesh dimensions and KPOINTS hashes are recorded in `KPOINTS.json` and `provenance.json`.
- O2 uses a separate Gamma-only `1x1x1` mesh in a fixed 15 A cell. `KSPACING` and `KGAMMA` are forbidden in both INCAR files.
- These fixed values have not been independently cutoff/k-point convergence-tested. Cutoff and k-point sweeps were removed from the workflow.

## Dependent Qualification Stages

Every workflow is one PBS allocation on one node with a **120-minute total walltime**, including its relaxation and static VASP invocations. It is not 120 minutes per VASP invocation. Workflows are serial; each therefore has a ceiling of 2.0 node-hours. Progression requires review of the preceding stage and explicit authorization for the next stage.

The runner enforces this dependency: Q1 requires explicit Q1 qualification/resource/PAW-use approvals; Q2 and Q3 additionally require `NBTAO5_STAGE_GATE_RECORD`. Each reviewed record must identify `protocol_id`, `status=stage_passed_and_next_stage_authorized`, `authorized_stage_id`, `completed_stage_ids`, the current `protocol_manifest_sha256`, VASP executable hash, PAW component hashes, `reviewer`, and `reviewed_at_utc`. It must list every prerequisite stage as completed and authorize exactly the stage being submitted. A Q1 pass record authorizing Q2 is not reusable to submit Q3.

| Stage | Exact workflows | Cap | Progression criteria |
|---|---|---:|---|
| Q1 runtime/workflow smoke | `NbTaO5-ordering-01-baseline` | 2.0 node-hours | Fixed-protocol executable/PAW/input hashes match; electronic and ionic convergence pass; max force <=0.01 eV/A and bulk max absolute principal stress <=2 kbar; extraction/provenance are consistent. The prior job is a failed submission, not a Q1 result. |
| Q2 elemental baselines | `Nb-bcc-baseline`, `Ta-bcc-baseline` | 4.0 node-hours | Both runs converge electronically and ionically; forces and residual stress pass the same criteria. |
| Q3 oxygen reference | `O2-triplet-baseline` | 2.0 node-hours | Triplet, SCF, ionic, and force checks pass in the fixed cell; Gamma-only KPOINTS is recorded. |
| **Total** | **4 workflows / 8 VASP invocations** | **8.0 node-hours** | A pass creates a reviewable numerical record; it does not authorize production. |

The full numerical criteria remain in [the pilot proposal](nb-ta-o-pilot-proposal.md#qualification-gates). Any failure, hash mismatch, unexpected structural change, or interrupted run stops progression. No automatic retry or setting expansion is allowed. After Q3, an independent review compiles the workflow summaries and stage decisions into the final `status=qualified` numerical record consumed by the production guard; the runner never creates that status automatically. The five-ordering production run still requires separate explicit production and resource authorization.

## First-Stage Request

The revised first-stage case is `NbTaO5-ordering-01-baseline` at `research/nb-ta-o/pilot/vasp_inputs/qualification/NbTaO5-ordering-01/baseline`: `ENCUT=600 eV`, initial Gamma-centered KPOINTS `4 8 9`, and static mesh regenerated from the relaxed cell. The old external case used the superseded 520 eV/0.20 A^-1 deck and remains unchanged. Stage the revised case to a new external attempt directory only after separate authorization; never overwrite the prior attempt. Assemble the approved PAW source components only in an external directory.

The user previously approved source-tree PAW use, account/resource limits, and Q1 for the superseded protocol. Obtain separate explicit **Q1 authorization for `nbtao5-fixed-kmesh-v1`** before submission. The execution guard additionally requires `NBTAO5_PAW_RELEASE_APPROVED=YES`, `NBTAO5_RESOURCE_APPROVED=YES`, and `NBTAO5_QUALIFICATION_AUTHORIZED=YES`. Authorization for Q1 is not blanket authorization for Q2/Q3 or production.

Submission template for a future, separately authorized fixed-protocol attempt (not submitted):

```bash
cd <absolute-matsim-agents-checkout>
qsub -A HydraGNN -v NBTAO5_REPO_ROOT="$PWD",NBTAO5_CASE_DIR=<new-external-flare-case>,NBTAO5_LOCK_ROOT=<external-flare-lock-dir>,NBTAO5_PAW_ROOT=<approved-source-paw-root>,NBTAO5_PAW_RELEASE_APPROVED=YES,NBTAO5_RESOURCE_APPROVED=YES,NBTAO5_QUALIFICATION_AUTHORIZED=YES deployments/aurora/jobs/job-nbtao5-qualification-aurora.pbs
```

The approved local-source PAW root must reproduce the manifest component hashes. The lock directory and case directory must be outside the Git worktree; the case directory must be on the facility-confirmed persistent filesystem. The user-approved source use does not verify an official release identifier; record that limitation and stop if any component hash differs. Do not submit the revised protocol without its separate explicit authorization.

## Walltime and Artifacts

The PBS scripts request 120 minutes and retain scheduler stdout/stderr. At 115 minutes, the runner writes a soft-stop marker and `STOPCAR` files; a received `SIGTERM`/`SIGUSR1` does the same. The run-directory INCAR copies enable `LWAVE` for a restart WAVECAR, while staged inputs remain unchanged. The runner records interruption separately from physical convergence failure. The runtime provenance stores both relaxation and static KPOINTS dimensions, source-structure hashes, and file hashes; the static mesh is based on the relaxed CONTCAR.

Expected Q1 artifacts under `<external-flare-case>/runs/<PBS_JOBID>/`:

- `provenance.json`: case/settings, source input hashes, executable and PAW component hashes, assembled POTCAR hash, runtime/module context, job identity, and effective runtime INCAR hashes.
- `workflow_status.json`: running, complete, interrupted, or failed state with timestamps and reason.
- `relax/` and `static/`: `INCAR`, `POSCAR`, `KPOINTS`, `POTCAR`, launcher logs, VASP `OUTCAR`/`OSZICAR`/`vasprun.xml`, plus `CONTCAR` and `WAVECAR` when written.
- `relax/summary.json` and `static/summary.json`: parsed convergence flags, energies, ionic-step count, final force summary, and stress when parseable.
- `relax/KPOINTS.json` and `static/KPOINTS.json`: exact mesh, sampling policy, structure hash, and KPOINTS SHA-256 for each VASP invocation.
- `relax/summary.json` and `static/summary.json`: parsed SCF/ionic flags, energy, final force, stress, and explicit force/stress acceptance results.
- `walltime_stop_requested` when the soft limit or a scheduler signal requested termination; PBS stdout/stderr and accounting record retained by the facility.

A complete Q1 output bundle is reviewed and signed into the Q1 stage-gate record before Q2 can be authorized. The final numerical qualification record is created only after Q4 review. Neither record automatically authorizes the five production ordering relaxations; production requires its own explicit authorization.
