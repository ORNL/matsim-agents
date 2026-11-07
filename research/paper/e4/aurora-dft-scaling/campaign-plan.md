# E4 Aurora DFT scaling plan

## Purpose and boundary

Measure task-parallel weak scaling, aggregate throughput, reliability, node-hour
consumption, and output qualification for three DFT workloads. The calculations
are planned as reusable scientific data, but E4 does not perform phase stability,
magnetic-ordering, or adaptive discovery studies. No PBS job, pilot, or DFT
calculation has been launched or is authorized by this plan.

The published E2 baseline is commit
`bce1ea0b52ad39f300d508244b9dd792efa259b7`. E2 files and results are read-only
historical provenance for this work.

## Inventory and allocation targets

| Family | Dedicated Aurora project | Candidate target | Structure size | Sweeps |
|---|---|---:|---:|---:|
| Refractory BCC, 4/5/6 elements from Nb-Ta-V-Hf-Zr-Ti | `HydraGNN` | 2,040 | 128 | 4 x 510 |
| Cr-Mn-Fe-Co-Ni FCC | `GenesisHackathonOct26` | 510 | 128 | 2 x 255 |
| LiFePO4 olivine | `GridAI` | Up to 510 | 112, proposed 2x2x1 | 2 x up to 255 |

These are not yet certified production counts. `manifests/candidate-inventory.jsonl`
contains 3,060 deterministic candidate IDs and plans 1, 1, and 2-3 Li/Fe
antisite exchanges. Structure hashes, occupations, LiFePO4 defect sites, and
canonical symmetry IDs are deliberately empty: public archive overlap and parent
geometries are not yet resolved. `run-manifests.jsonl` assigns every candidate
ID once across 64 disjoint schedule slots, but those IDs are not approved
production structure assignments.

## Proposed numerical protocols

### Refractory BCC

Use VASP 6.6.1, PBE, non-spin-polarized (`ISPIN=1`), `ENCUT=350 eV`,
Gamma-centered 2x2x2, `EDIFF=1e-6 eV`, and 128-atom disordered BCC cells.
Published records describe `PREC=Normal`, energy tolerance `1e-6 eV`, and
maximum force `0.01 eV/A`; exact staged relaxation inputs, `ISMEAR`, `SIGMA`,
and PAW release/hash still require recovery. The 350 eV cutoff and
semi-core-valence choices must match the published archive. Hf and Ti PAW
compatibility is an explicit preproduction gate. Do not substitute the distinct
345 eV E2 protocol.

### Cr-Mn-Fe-Co-Ni FCC

Use VASP 6.6.1, spin-polarized PBE, `ENCUT=480 eV`, Gamma-centered 2x2x2,
`EDIFF=1e-6 eV`, and a fixed initialization recovered from Cantor E2. Keep the
archived `ISMEAR=1`, `SIGMA=0.10 eV`, PAW set, and relaxation conventions only
after matching the actual input bytes. Do not call off-equiatomic candidates
equiatomic Cantor. The E2 relaxation failures prohibit scaling until a
separately approved qualification calculation demonstrates SCF convergence at
every ionic step and acceptable ionic termination.

### LiFePO4

Use VASP 6.6.0-compatible PBE without U, `ENCUT=520 eV`, `PREC=Accurate`,
`EDIFF=1e-6 eV`, spin-polarized with one fixed archived initialization, and
112 atoms in a 2x2x1 olivine supercell. The mounted E2 extraction documents
`ISMEAR=0`, `SIGMA=0.05 eV`, `EDIFFG=-0.01 eV/A`, `IBRION=2`, `ISIF=3`,
`NSW=200`, Fe_pv/Li_sv/O/P PAWs, and a primitive-cell 4x7x8 mesh. A
reciprocal-density-preserving 2x4x8 mesh is a candidate for 2x2x1, pending
verification against the original KPOINTS and source geometry. Exact atomic
ordering of `MAGMOM`, archived input hashes, and the parent cell are not mounted.

## Scaling schedule

Use task-parallel weak scaling. Do not describe this as strong scaling: all
production runs use disjoint structures. The requested tiers are 1, 2, 4, 8,
16, 32, 64, and 128 nodes.

- BCC: four sweeps, two independent calculations per node, sequential on a node
  unless a separately approved concurrency qualification supports otherwise.
  Per sweep: 2, 4, 8, 16, 32, 64, 128, 256 structures (510 total).
- FCC: two sweeps, one calculation per node. Per sweep: 1, 2, 4, 8, 16, 32,
  64, 128 structures (255 total).
- LiFePO4: same tier/task schedule as FCC, capped by the number of symmetry-
  distinct configurations confirmed from the parent geometry.

The 64 run rows are a capacity plan only. Read-only `qstat -Qf` confirmed on
2026-11-06 that `debug` supports 1-2 nodes up to one hour, `debug-scaling`
supports 2-256 nodes up to one hour, `capacity` supports 1-16 nodes up to 168
hours, and `prod` is a routing queue whose execution destinations start at 256
nodes. Thus tiers 32-128 have no long-walltime route in the current snapshot;
they fit `debug-scaling` only if a complete tier run fits one hour. The capacity
queue allows at most two running jobs per user and 512 assigned nodes total.
Current queue output does not establish project-specific authorization. No
submission probe was made.

| Node tier | Documented queue options | Walltime constraint | Planning status |
|---:|---|---|---|
| 1 | `debug`, `capacity` | 1 hour / 168 hours | Globally feasible; project access still needs queue confirmation. |
| 2 | `debug`, `debug-scaling`, `capacity` | 1 hour / 1 hour / 168 hours | Globally feasible. |
| 4, 8, 16 | `debug-scaling`, `capacity` | 1 hour / 168 hours | Globally feasible. |
| 32, 64, 128 | `debug-scaling` | 1 hour | Only viable if full run walltime fits; otherwise revise tiers or break into shorter authorized restartable runs. |

Live `sbank` records (2026-11-06) show active Aurora allocations and
`mlupopa` user rows for all three projects. Balances are 36,707 node-hours for
HydraGNN, 14,824.7 for GenesisHackathonOct26, and 34,029.1 for GridAI; their
expiration dates are 2027-01-10, 2027-02-19, and 2027-01-10 respectively. The
idealized balance ceilings are about 17.99, 29.07, and 66.72 node-hours per
planned calculation, before failures and restart overhead. These are arithmetic
ceilings, not runtime forecasts; sufficiency cannot be decided without a pilot.
Project IDs and PBS `-A` syntax are confirmed; project-specific queue acceptance
is not.

There is a minimum-node discrepancy for `debug-scaling`: the live queue record
reports 1 node, while the current user guide documents 2. Use 2 as the minimum
until ALCF clarifies the difference. If one-hour tiers above 16 nodes fail an
approved runtime pilot, an alternative for review is to cap the primary sweep
at 16 nodes under `capacity` and split each 510/255-structure workload into
repeated disjoint batches, with a final smaller batch. This would add runs and
remove 32/64/128-node evidence; it is not the same 64-run campaign and is not
adopted automatically.

## Pilot gate (not executed)

The detailed planning-only VASP binary A/B design, resource cap, and numerical
qualification criteria are in `vasp-ab-pilot-proposal.md`. That comparison is
NO-GO until the original BCC production inputs and PAW identity are verified;
this plan does not authorize execution.

After explicit separate approval, propose one representative relaxation per
family on one Aurora node. The pilot record must capture module list, VASP hash,
PAW component checksums (not POTCAR bytes), ranks/tiles, CPU/GPU affinity,
SCF status at every ionic step, ionic stop reason, force, peak memory, walltime,
restart behavior, output hashes, and node-hours. BCC should include an Hf/Ti
case; FCC should include a representative off-equiatomic arrangement; LiFePO4
should include pristine and single-antisite cases if the authorized pilot cap
allows both. Stop FCC expansion on repeated SCF failure. The pilot is not
authorized in this planning task.

## Resource accounting

Live balances are recorded in `allocation-status.json`, but numeric node-hour
forecasts are withheld because per-structure runtimes are unknown. After an
approved pilot, keep independent account ledgers. For one-node tasks and
measured per-structure runtime `t` hours,
the idealized compute-only estimates are `2040*t_BCC`, `510*t_FCC`, and
`N_LFP*t_LFP` node-hours. These exclude reservation gaps, staging, failures, and
restarts; actual accounting must use allocated walltime and retain failed
attempt costs. Planned, submitted, completed, verified, failed, restarted, and
remaining units must remain separated by `HydraGNN`,
`GenesisHackathonOct26`, and `GridAI`.

## Output qualification and restart

The SQLite registry in `benchmarks/e4/aurora_dft/registry.py` serializes
reservations with `BEGIN IMMEDIATE`, enforces unique calculation IDs and
structure/method pairs, rejects family/allocation mismatches, and keeps each
restart attempt and node-hour total under the same calculation ID. It contains
no scheduler or VASP submission path. Relaxations qualify only when ionic
termination, SCF convergence at every ionic step, and force criteria all pass.
Static SCF completion is recorded separately and is never called a qualified
relaxation.

## PBS and execution status

Three PBS templates carry the required project IDs and standard `select`,
`place=scatter`, and `filesystems=home:flare` fields. Their queue and walltime
are explicit invalid placeholders, and each body only prints metadata then
exits. They cannot run DFT. Do not render or submit until current queue/account
eligibility and walltime are verified independently.

The local VASP 6.6.1 executable and repository build are accessible; no VASP
module is listed by `module avail vasp`. The current ALCF VASP guide reports
that the default September 2026 MPICH build can be 20-25x slower per SCF step
and recommends `mpich/prd/5.0.0.aurora_test.51a9474`. The existing shared native
module helper loads oneAPI and the GPU UMD but does not select that MPICH
version. Before any authorized pilot, choose and record an explicit compatible
MPICH module without changing the existing launcher draft, then validate
rank/tile mapping. Existing one-node launcher evidence is not qualification for
32-128 node task dispatch.

## Readiness

| Project / family | Decision | Blocking evidence |
|---|---|---|
| `HydraGNN` / BCC | NO-GO | Allocation/user access/balance/expiration verified, but 32-128 tiers have a one-hour cap, published PAWs and inputs and duplicate audit are incomplete, Hf/Ti compatibility is unverified, and pilot cost is unknown. |
| `GenesisHackathonOct26` / FCC | NO-GO | Allocation/user access/balance/expiration verified, but 32-128 tiers have a one-hour cap and Cantor E2 relaxations repeatedly hit `NELM=200`; qualification gate is unmet. |
| `GridAI` / LiFePO4 | NO-GO | Allocation/user access/balance/expiration verified, but 32-128 tiers have a one-hour cap and archived POSCAR/KPOINTS/MAGMOM plus a symmetry-deduplicated 112-atom inventory are unavailable. |

All three projects remain assigned exactly as requested. No charge is redirected
to another allocation. The provisional 3,060 target remains unverified; the
number of genuinely unique, archive-audited structures is currently zero.