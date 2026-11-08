# E4 Stage 1 blocker-resolution report

**Refreshed:** 2026-11-08  
**Decision:** NO-GO for production submission. This report does not authorize or
submit jobs and does not authorize a VASP run.

## 1. Frozen inventory and Stage 1 bundle

The LiFePO4 inventory is preserved at `lifepo4-inventory-v1/`. It contains 510
112-atom structures with composition Li16Fe16P16O64: 1 pristine, 8
single-exchange-pair, 496 double-exchange-pair, and 5 triple-exchange-pair
configurations. All 510 canonical identities are unique; the minimum periodic
interatomic distance is 1.5362 Å. The frozen manifest SHA-256 is
`c8c6fb83ebd59388e8283d87a9670b2e2b326b8df88b2ff8c5415c3cccedda02`.
The generator configuration records the 28-atom parent path and its SHA-256,
the 2x2x1 supercell, symmetry settings, generator hashes, and software versions.

The current 16-case bundle is `stage1-inputs-v2/`: 8 BCC, 4 FCC, and 4 LiFePO4
cases. It has POSCAR hashes, canonical structure IDs, input checksums,
protocol/provenance fields, species order, and private PAW component references.
It contains no POTCAR bytes. Its summary explicitly marks execution
unauthorized and the external archive duplicate audit incomplete. The older
`stage1-inputs/` bundle is stale and must not be used.

## 2. PAW component audit

The approved local component root is the private, git-ignored directory
`external/vasp6/potcar/` (directory mode 2700; component files mode 0400).
Required components were compared with the archived combined POTCARs by
component payload; each matched after trimming only exterior whitespace. No
alternative PAW was substituted. No restricted POTCAR bytes were added to the
repository or generated bundles.

The selected-case element order is the POSCAR order. BCC order by case is:
`0001 Hf,Nb,Ta,V`; `0561 Hf,Nb,Ti,V`; `1121 Hf,Ti,V,Zr`;
`1201 Hf,Nb,Ta,V,Zr`; `1501 Hf,Nb,Ta,Ti,Zr`; `1701 Hf,Ta,Ti,V,Zr`;
`1801/1805 Hf,Nb,Ta,Ti,V,Zr`. FCC order is Co, Cr, Fe, Mn, Ni. LiFePO4 order
is Fe, Li, O, P.

| Family | Component | PAW title/release | Private component SHA-256 |
|---|---|---|---|
| BCC | Hf_pv | PAW_PBE Hf_pv 06Sep2000 | `db42cd3e4a48e81eaff18895282e2f5a37f2c8c6a4b9ac45e357affc13c83b84` |
| BCC | Nb_pv | PAW_PBE Nb_pv 08Apr2002 | `086061c276946e9fc55bb7633d9ce78b32a420dec762592491c07c1c47ded114` |
| BCC | Ta_pv | PAW_PBE Ta_pv 07Sep2000 | `ffb05e92ac3f68ca84c26af10a80f2fbb0cf79e16fcc96fcae98089455bbbaaa` |
| BCC | Ti_pv | PAW_PBE Ti_pv 07Sep2000 | `1a7a8efddf2a5b6c8861aa48543d0531abc8e9b6e54500458b415b06354c25ba` |
| BCC | V_pv | PAW_PBE V_pv 07Sep2000 | `7e6e3948baf479b1e61e3653bf3429049e07351b243ece4a7a45499cd5dea1a0` |
| BCC | Zr_sv | PAW_PBE Zr_sv 04Jan2005 | `e3d75732e64ed47e733ef12202d7082321f6b043dd450e16fe03b686b180cdc0` |
| FCC | Co | PAW_PBE Co 02Aug2007 | `3ae14252e5e27e2e5a5730c8d8131c6a7ea573dcc9d4d3d392959f07a3f519ee` |
| FCC | Cr_pv | PAW_PBE Cr_pv 02Aug2007 | `836672959fc86f3b167531577dbf63d7fb0b8d96aaf8b40fb3c4265879bd744b` |
| FCC | Fe_pv | PAW_PBE Fe_pv 02Aug2007 | `e96179f13e87664e48aed485c2113db4b894f4702180a1840e8206974d0cd1f5` |
| FCC | Mn_pv | PAW_PBE Mn_pv 02Aug2007 | `5da6f925d338804d4244b231b43927f3fccb021320fc01db0f865b4eda5a756d` |
| FCC | Ni_pv | PAW_PBE Ni_pv 06Sep2000 | `298ae0c75c9017ac7c19a9201530db0621deb288a6cb951496859d69da6505eb` |
| LiFePO4 | Fe_pv | PAW_PBE Fe_pv 02Aug2007 | `e96179f13e87664e48aed485c2113db4b894f4702180a1840e8206974d0cd1f5` |
| LiFePO4 | Li_sv | PAW_PBE Li_sv 10Sep2004 | `201875120238865c2f235e24081bce20639c4ae21bc4e97e31f9e3b7cc8fb95b` |
| LiFePO4 | O | PAW_PBE O 08Apr2002 | `8a74b9a1f5fdb0d3c3e0183c7873177abdbef07d407b310b7edcd9ed0a3eea64` |
| LiFePO4 | P | PAW_PBE P 06Sep2000 | `ff56d25c245497eab82fbf156c8a80195e6c8278b657680eda7251824aa36cf8` |

Archived combined-source POTCAR SHA-256 values are BCC
`4c2efb63638e6f89bd80129da1c38cf658ccce4a5c6da45b5e18c760659c11e2`, FCC
`c0e6f0e3856534892c239c7a8ec155102fdb9ee703a10e88b94d0c13f7ae4918`, and
LiFePO4 `f2a05d8ed7e9bfe3976233c793251a2ab96a49a35bc4f349babdc9f708ea4232`.
No required component is missing or known to be incompatible with its archived
source payload. The archived LiFePO4 run used VASP 6.6.0; compatibility with
the available VASP 6.6.1 binaries remains an execution qualification.

## 3. Duplicate screening

The local comparison covered accessible E2 records, the archived E4 BCC A/B
input and outputs, and mounted LiFePO4 reference structures. Public OSTI/
Globus payloads and other inaccessible private archives were not enumerated.
Thus local distinction is not a global uniqueness claim.

| Case ID | Local classification and evidence | External archive status / action |
|---|---|---|
| E4-BCC-0001 | Distinct from accessible E2 BCC and E4 A/B records; composition Nb32-Ta32-V32-Hf32. | Unresolved; keep blocked. |
| E4-BCC-0561 | Distinct from accessible E2 BCC and E4 A/B records; composition Nb32-V32-Hf32-Ti32. | Unresolved; keep blocked. |
| E4-BCC-1121 | Distinct from accessible E2 BCC and E4 A/B records; composition V32-Hf32-Zr32-Ti32. | Unresolved; keep blocked. |
| E4-BCC-1201 | Distinct from accessible E2 BCC and E4 A/B records; composition Nb26-Ta26-V26-Hf25-Zr25. | Unresolved; keep blocked. |
| E4-BCC-1501 | Distinct from accessible E2 BCC and E4 A/B records; composition Nb26-Ta26-Hf26-Zr25-Ti25. | Unresolved; keep blocked. |
| E4-BCC-1701 | Distinct from accessible E2 BCC and E4 A/B records; composition Ta26-V26-Hf26-Zr25-Ti25. | Unresolved; keep blocked. |
| E4-BCC-1801 | Same cell/composition as E4 A/B parent, but a different periodic occupation at 115/128 matched sites; not the prior incomplete structure. | Unresolved; keep blocked. |
| E4-BCC-1805 | Distinct from E4 A/B by composition; Nb46-Ta17-V17-Hf16-Zr16-Ti16. | Unresolved; keep blocked. |
| E4-FCC-0001 | Distinct from accessible 108-atom E2 Cantor cases; derived 128-atom Cr26-Mn26-Fe26-Co25-Ni25 structure. | Unresolved; keep blocked. |
| E4-FCC-0006 | Distinct from accessible 108-atom E2 Cantor cases; derived 128-atom Cr48-Mn20-Fe20-Co20-Ni20 structure. | Unresolved; keep blocked. |
| E4-FCC-0011 | Distinct from accessible 108-atom E2 Cantor cases; derived 128-atom Mn48-Cr20-Fe20-Co20-Ni20 structure. | Unresolved; keep blocked. |
| E4-FCC-0016 | Distinct from accessible 108-atom E2 Cantor cases; derived 128-atom Fe48-Cr20-Mn20-Co20-Ni20 structure. | Unresolved; keep blocked. |
| E4-LFP-0001 | Exact 2x2x1 periodic supercell reuse of the archived 28-atom ideal structure; not a same-size calculation duplicate. | Exclude from new pristine production unless a 112-atom size-effect calculation is explicitly intended. Keep in frozen inventory. |
| E4-LFP-0002 | Distinct from accessible archived 28-atom antisite/vacancy inputs under 2x2x1 cell-aware comparison. | Unresolved against other private/public archives; keep blocked. |
| E4-LFP-0003 | Distinct from accessible archived 28-atom antisite/vacancy inputs and E4-LFP-0002. | Unresolved against other private/public archives; keep blocked. |
| E4-LFP-0004 | Distinct from accessible 28-atom antisite/vacancy inputs; composition differs from the vacancy case. | Unresolved against other private/public archives; keep blocked. |

No exact same-size qualified or prior-incomplete duplicate was identified among
the BCC/FCC cases or the three LiFePO4 antisite cases. The LiFePO4 pristine
supercell is the one confirmed structural reuse.

## 4. Archived BCC A/B pilot

The four one-node jobs requested 28,800 seconds (8 hours) each and reached
Normal precision before walltime termination. The Low-stage VASP `Elapsed time`
values agree with the runner's rounded stage seconds; Normal did not complete.
There is no trustworthy Normal VASP elapsed-time record, so the separate
Normal-start-to-exit spans are not added to Low-stage elapsed time.

| Job | Build | Low VASP elapsed (s) | Low ionic steps | Available `0.OSZICAR` DAV records | Normal partial ionic steps | Last archived Low max force (eV/A) |
|---|---|---:|---:|---:|---:|---:|
| 9047960 custom/repeat-01 | Custom | 25,713.702 | 121 | 2,274 | 19 | 0.0154681 |
| 9047963 custom/repeat-02 | Custom | 27,892.577 | 137 | 2,461 | 3 | 0.00441301 |
| 9047961 alcf/repeat-01 | ALCF | 25,243.776 | 134 | 2,418 | 22 | 0.00403983 |
| 9047962 alcf/repeat-02 | ALCF | 24,430.779 | 127 | 2,365 | 32 | 0.00832713 |

All Low stages reported `reached required accuracy`; no Normal stage completed.
The SCF counts are the archived `0.OSZICAR` records (Low-stage output), not a
combined Low+Normal total; exact Normal-stage SCF totals and VASP elapsed times
are not recoverable from the archived records. All available SCF records are
DAV iterations. Final Low-stage ionic-step iterations were DAV 13, DAV 2,
DAV 2, and DAV 14 in table order. Mean Low
`LOOP` times were 9.78/9.72 seconds for Custom and 8.91/8.87 seconds for ALCF.
Recorded Low wall seconds per ionic step were 212.5/203.6 (Custom) and
188.4/192.4 (ALCF), an approximately 8.5% lower two-run mean for ALCF. This is
not enough to select a production binary: there were only two uncontrolled
repeats per build, no Normal completion, and no GPU utilization telemetry.

The four attempts have nonempty `CONTCAR` (15,323 bytes), `WAVECAR`
(1,895,063,808 bytes), and `CHGCAR` (9,595,537 bytes) files. They lack the
current runner's protocol/checkpoint manifests, so these files are not
validated safe restart points. The existing BCC runner rejects a resume without
trusted metadata; do not resume these archived attempts through that runner.

## 5. Restart-readiness assessment

The BCC A/B runner has explicit protocol identity, checkpoint-file SHA-256
manifests, isolated attempt directories, soft-stop handling, and resume checks
for checkpoints it created. This does not retroactively validate the four
legacy attempts.

The generic E2 runner is not restart-safe enough for production use as-is. It
copies an interrupted `CONTCAR` into the next input, removes WAVECAR when the
previous OUTCAR lacks a normal timing marker, and isolates segments, but it does
not bind restart files to a protocol/binary/input identity or verify a
checkpoint hash manifest before resuming. It also hardcodes the custom VASP
build. The Stage 1 bundle references private PAWs but current PBS templates do
not assemble and verify them at runtime.

Minimum restart acceptance checks are: exact input/protocol/binary identity;
POSCAR/CONTCAR atom count, cell, species order and finite coordinates; exact
checkpoint file set and SHA-256 manifest; compatible `ISTART`/`ICHARG` settings;
fresh attempt output isolation; verified executable hash; and a controlled
stop/resume sequence with failure status preserved. That last behavior cannot
be established without a compute test.

If separately authorized, the smallest useful test is one interrupted and one
resumed 15-minute one-node job for each of the three distinct execution paths
(BCC two-precision runner, FCC workflow, LiFePO4 workflow): 0.50 node-hours per
family, 1.50 node-hours total, charged to HydraGNN, GenesisHackathonOct26, and
GridAI respectively. This is a proposed cap only; no such test was submitted
or run.

## 6. PBS, executable, and allocation readiness

The three files under `deployments/aurora/e4-templates/` remain dry-run
templates: each contains unresolved queue/node/walltime placeholders, prints
metadata, and exits without VASP. They were not converted to submission-ready
scripts because the common E2 restart behavior is unverified, the runtime PAW
assembly is absent, and no production binary has been selected. Calling them
submission-ready would be unsafe.

The installed VASP 6.6.1 binaries are:

| Build | Path | SHA-256 |
|---|---|---|
| Custom | `external/vasp6/src/vasp.6.6.1/bin/vasp_std` | `ebd284027717ff564abf7ceaf93c3aca3261b129d7b60b3f2e3e2aa82d04c552` |
| ALCF | `/soft/applications/vasp/vasp.6.6.1/bin/vasp_std` | `3354871ac0b7a812ebe3444dbd3d5bb4176995a5c37c3859978b58b5c7b55eb6` |

The archived LFP calculation used VASP 6.6.0. The installed trees provide
executables, not the approved PAW component library. The historical BCC A/B
workflow used 8 MPI ranks and 8 OpenMP threads on one node; the generic E2
launcher path and its placement have not been qualified for these E4 inputs.

The live `capacity` queue snapshot shows 1-16 nodes and up to 168 hours; it is
globally enabled. `sbank` access and balances were refreshed at
2026-11-08 08:03:23 UTC:

| Family | Allocation | Balance (node-hours) | Expiration |
|---|---|---:|---|
| BCC | HydraGNN | 36,674.7 | 2027-01-10 |
| FCC | GenesisHackathonOct26 | 12,996.1 | 2027-02-19 |
| LiFePO4 | GridAI | 34,029.1 | 2027-01-10 |

User membership and positive balances were observed for all three projects.
The historical BCC A/B jobs demonstrate prior `capacity` use under HydraGNN.
The current queue query does not expose project-specific ACLs, so current
HydraGNN eligibility must still be rechecked before a later submission;
project-specific `capacity` acceptance for GenesisHackathonOct26 and GridAI
has not been demonstrated. No submission probe was made.

The existing BCC pilot PBS file requests `-A HydraGNN`, `capacity`, one node,
and 8 hours; it pins the MPICH module and uses 8 MPI ranks x 8 threads, checks
the selected executable SHA-256, isolates output by variant/repetition, records
the run and failure status, and requests a soft VASP stop on scheduler signals.
This is not a production PBS script: its input deck has no explicit `EDIFFG`,
its historical resumes are rejected without runner-generated manifests, and
the script is not qualified for FCC or LiFePO4. The generic E2 runner uses a
case lock and status file, but its restart and binary verification do not meet
the checks above. The three Stage 1 templates do not yet define node counts,
walltime, MPI/GPU placement, output directories, PAW assembly, or a VASP
executable.

## 7. Family decisions

| Family | Readiness |
|---|---|
| BCC refractory | **NO-GO for production.** PAWs and local input provenance are recovered, and all eight selected structures are distinct from accessible local BCC records. Still blocked by external archive screening, unspecified `EDIFFG`, unvalidated production restart, and no defensible binary selection. |
| FCC Cantor-family | **NO-GO for production.** PAWs and derived 128-atom inputs are staged; local 108-atom E2 cases are not duplicates. The 128-atom SCF/relaxation and restart path are unqualified, and external archive screening remains incomplete. |
| LiFePO4 | **NO-GO for production.** The 510-structure inventory is frozen and PAWs are verified. Exclude pristine E4-LFP-0001 from new pristine production as an exact periodic supercell reuse; retain the inventory. Antisite candidates are locally distinct, but external screening, VASP 6.6.1 compatibility, and restart qualification remain open. |

No PBS job or VASP run was launched, and no commit, push, or manuscript change
was made.
