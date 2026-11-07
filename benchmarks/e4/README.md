# E4 planning and dry-run preflight

This folder prepares the E4 campaign only. It has no scheduler-submit code. `prepare.py` validates synthetic allocation layouts through the existing `ResourcePlan` resolver and writes a run matrix whose every row is `not_authorized`.

From the repository root:

```bash
python benchmarks/e4/prepare.py
pytest -q tests/test_e4_planning.py
```

The default outputs are `research/paper/e4/manifests/e4-workload-manifest.json` and `e4-run-matrix.csv`. Workload sizes intentionally remain blank until a bounded real qualification measures task duration, stage capacity and sustainable concurrency.

After authorized runs, store one provenance-complete JSON object per line and analyze it with:

```bash
python benchmarks/e4/analyze.py research/paper/e4/results/raw-measurements.jsonl \
  --output research/paper/e4/results/e4-scaling-results.csv
```

Cross-platform consistency records use the same raw JSONL input with the
consistency record schema implemented in `benchmarks/e4/consistency.py`:

```bash
python benchmarks/e4/analyze.py research/paper/e4/results/consistency-measurements.jsonl \
  --kind consistency --output research/paper/e4/results/e4-consistency-results.csv
```

Contract/adapter outputs from `benchmarks/portability` are software checks only; they are not production numerical portability results. No result from this planner qualifies scientific consistency or scaling.
