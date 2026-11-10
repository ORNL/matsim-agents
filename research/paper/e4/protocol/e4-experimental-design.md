# E4 experimental design (planning draft; no jobs authorized)

## Existing functionality audit

The shared harness is `benchmarks/portability/`. It already separates `config/science.yaml` from three facility overlays, writes commit/input/config/allocation provenance, runs smoke/adapter contracts, validates results, and compares summaries. Its `--qualification compute` path invokes the production relaxation workflow with real MLIP and QE configurations, requires both MLIP and QE, and emits `scientific_summary.json`. Adapter/contract values are explicitly not numerical backend evidence.

The execution layer already discovers PBS/Slurm allocations, resolves disjoint LLM/MLIP/DFT pools, maps device masks through facility presets, and records logical devices and physical GPU packages separately. DFT allocation helpers support concurrent per-job node groups; recovery is versioned. Current portability PBS/Slurm wrappers are one-node, 20-minute contract jobs, not E4 scaling launchers. `compare.py` checks input commit/structure hash and energy/force tolerances but does not yet compare all required geometry, workflow conclusions, and evidence provenance.

Missing: E4 run matrix and workload identity; fixed/weak workload assignment; scaling measurement schema and analysis; scheduler allocation accounting integrated with per-run outputs; 8-256-node resource-plan preflight; repeat-aware strong/weak analysis; E4-specific cross-facility contract comparison. No E4 production runs are represented by current scripts or deployment files.

## Candidate scientific workload

Candidate: Si relaxation and characterization based on `benchmarks/portability/structures/Si.vasp`, with a fresh MACE relaxation, a QE reference relaxation, registered provenance/evidence, and a fixed evidence decision/report. This is a candidate only: the existing Si portability contract is an engineering fixture, not E4 scientific evidence. E4 would create fresh per-task numerical results.

The current portability compute configs propose MACE (`small`) and QE (`Si_r.upf`, 4x4x4 Gamma, Gaussian smearing, 0.007 Ry). These do not yet define a publication-ready common method: model/checkpoint hash is absent; the pseudopotential is not present at the config's referenced path in this checkout; its hash and QE build versions are unrecorded; and the existing contract and compute configs disagree on force tolerance (0.05 versus 0.02 eV/A). MACE XPU qualification is explicitly provisional. Freeze none of these choices until Gates 1-3 pass.

Performance task bank: repeat the same frozen small Si task with unique task IDs, isolated output roots and cache/result reuse disabled. This deliberately measures compute throughput; repeats are not independent scientific discoveries. Strong workload W and weak workload w are null until measured task duration, steady concurrency and budget data exist. A task counts as successful only if both real numerical stages converge and evidence/report validation passes.

For strong scaling, keep the same W task IDs and hashes at all N. Report `S(N)=T_8(W)/T_N(W)` and `E_s(N)=S(N)/(N/8)`. For weak scaling, set `W_N=wN` from the measured per-node workload and report `Q(N)=C_N/T_N` and `E_w(N)=(Q(N)/N)/(Q(8)/8)`. The analyzer suppresses strong speedup when any task is invalid, rejects changed strong workloads and non-proportional weak task counts, and reports failures and allocation node-hours separately. No values are inferred or interpolated.

## Scaling and resource placement

Aurora node points are fixed at 8, 16, 32, 64, 128, 256, with three independent allocations per experiment. The planning-only resource profile currently uses exclusive node pools: LLM service nodes `max(1, ceil(N/32))`, MLIP nodes `max(1, ceil(N/16))`, and DFT workers on the remaining nodes. Orchestration runs on an LLM service node. Each DFT worker is provisionally one node. The exact model replica size, QE rank/thread layout, and pool ratios must be validated on the facility before freezing. This profile is a placement preflight, not a measured optimum.

Aurora geometry: 12 logical XPU tiles per node, two tiles per physical GPU package; totals are 96 tiles/48 packages at 8 nodes and 3,072 tiles/1,536 packages at 256 nodes. Frontier: 8 logical GCDs/4 MI250X packages per node. Perlmutter: 4 A100 GPUs/packages per node. These units are reported separately and never equated.

Strong scaling: same fixed W task IDs and input hashes at every Aurora size; report makespan, speedup against 8 nodes, efficiency, valid/failed tasks, stage times, scheduler allocation time, node/package-hours and utilization. Weak scaling: W_N=wN from a deterministic task bank with the same task-type distribution; report valid tasks/sec and efficiency relative to 8 nodes. Every run includes the shared LLM/MLIP/DFT service policy. A fixed-capacity central service may be a bottleneck and must be measured, not silently compensated.

Cross-platform consistency: one bounded, identical scientific contract, three repetitions each on Aurora, Frontier and Perlmutter, initially planned at three nodes per facility (one node for each service pool). This is a provisional minimal placement, not an authorized request. Candidate acceptance tolerances may start from the existing portability manifest (energy 0.05 eV/atom, force 0.05 eV/A, position RMSD 0.05 A), but they must be independently reviewed and frozen before results are examined. The same MLIP checkpoint and DFT pseudopotential/method must be used for a numerical-consistency claim. If only workflow-compatible, different backends/models can be run, report that separately as workflow portability, not numerical equivalence.

## Gates

0. Audit current configs, models, pseudo assets, placement and launchers (completed in this draft).
1. Software gate: run preflight and local synthetic tests only; validate input hashes, resource partitions, run IDs and analysis. No scheduler contact.
2. After authorization, small Aurora qualification to measure one real task, concurrency, LLM/MLIP/QE capacity and overhead; use measurements to choose W, w and pool profile.
3. After separate authorization, small same-contract Aurora/Frontier/Perlmutter compute qualification; establish method compatibility and scientific tolerances.
4. Only after owner approval of protocol and node-hour ceilings, run the 36 Aurora scaling allocations and 9 consistency allocations. Failed points remain in the report; no automatic retry of the campaign.

## Cost and unresolved inputs

No dollar or node-hour estimate is made: per-task allocation duration, service startup, queue overhead and project billing caps are unknown. Required before Gate 2: approved facility/account, project balance and ceiling; MACE/HydraGNN checkpoint and cross-platform support; local LLM choice if included; QE build/version and verified `Si_r.upf` hash on all sites; frozen force threshold and evidence decision rule; measured single-task duration and sustainable concurrency; task-count W and w; cache policy; service replica and DFT node/rank layout; allocation walltime and stop conditions. No production calculation or scheduler job is authorized by this design.
