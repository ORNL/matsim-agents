# Cross-facility portability benchmark

This directory defines one scientific problem and three deployment overlays so
Frontier, Aurora, and Perlmutter results can be compared without conflating a
machine-specific launch choice with a change to the science.

## Benchmark ladder

1. **Smoke:** imports the installed package and validates the selected DFT step
   launcher. It is dependency-light and is the first release gate.
2. **Relaxation:** the fixed Si structure supports MLIP, QE, VASP, MLIP→QE, and
   MLIP→VASP comparisons. VASP remains optional because its binary and POTCARs
   are licensed resources.
3. **Active learning:** the fixed four-candidate pool selects two structures,
   labels them with one DFT backend, records stable candidate IDs, and leaves
   retraining and model promotion disabled.
4. **Phase exploration:** a miniature Si exploration uses four candidates. It
   checks orchestration and artifact contracts, not scientific convergence of a
   production search.

The latter three suites are expressed in the canonical manifest and science
configuration. Their numerical backends remain the production workflows in
`matsim_agents`; this benchmark does not fork those implementations. Run them
with the corresponding workflow command after the smoke gate and place a
`scientific_summary.json` in the same result directory for cross-site numeric
comparison. This staged design prevents missing proprietary binaries or model
weights from making the basic portability check unusable.

## Inputs and acceptance

- `manifest.yaml` identifies the immutable benchmark and tolerances.
- `structures/Si.vasp` is owned by this benchmark rather than borrowed from a
  test fixture.
- `config/science.yaml` contains every machine-independent scientific choice.
- `config/{frontier,aurora,perlmutter}.yaml` may contain only scheduler,
  accelerator, rank, and launcher settings. `run.py` rejects scientific keys in
  an overlay.
- Validation compares invariants and tolerances, never bitwise-identical
  floating-point results.

Every run records the Git commit, exact structure digest, resolved
configuration, scheduler allocation metadata, executable discovery, plan, and
status. A comparison fails when runs use different source commits or structure
bytes. Optional numerical summaries use the tolerances in `manifest.yaml`.

## Running

Plan locally without a scheduler:

```bash
python benchmarks/portability/run.py \
  --facility frontier --suite active-learning --backend qe \
  --output runs/portability-plan
python benchmarks/portability/validate.py runs/portability-plan
```

Submit the smoke gate (allocation is supplied at submission, never embedded):

```bash
sbatch -A <allocation> --export=ALL,PROJECT_ROOT="$PWD" \
  deployments/frontier/jobs/job-portability-benchmark-frontier.sh
sbatch -A <allocation> --export=ALL,PROJECT_ROOT="$PWD" \
  deployments/perlmutter/jobs/job-portability-benchmark-perlmutter.sh
qsub -A <allocation> -v PROJECT_ROOT="$PWD" \
  deployments/aurora/jobs/job-portability-benchmark-aurora.sh
```

Compare completed directories:

```bash
python benchmarks/portability/compare.py \
  runs/portability/frontier-* runs/portability/aurora-* \
  runs/portability/perlmutter-*
```

Paper cases, scaling sweeps, model catalog benchmarks, and warm-start studies
remain specialized benchmarks. They are intentionally not deleted or silently
redirected to this small portability gate.

