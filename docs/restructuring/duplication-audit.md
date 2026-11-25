# Duplication audit (acceptance criterion 5)

The audit scanned `src/`, `benchmarks/` and `scripts/` for repeated public and private helper
names and for repeated hashing, launcher and host-name logic. Each finding is merged or kept
with a reason.

## Merged

| Duplicate | Single implementation | Notes |
| --- | --- | --- |
| `_child_environment` in `qe_relax.py` and `vasp_relax.py` | `backends/dft/process.child_environment` | Identical bodies. |
| Default-launcher lookup in `qe_relax.find_default_launcher` and `vasp_relax.find_default_launcher` | `backends/dft/process.launcher_from_environment(variable)` | Each backend keeps its own variable (`MATSIM_QE_LAUNCHER`, `MATSIM_VASP_LAUNCHER`). |
| `_short_host` in `backends/llm/servers.py` and `backends/mlip/pool.py`; inline `split(".", 1)[0]` in `workflows/relaxation.py` and `workflows/active_learning/trainer.py` | `execution/allocation.short_host` | |
| `workflows/scalability.canonical_json`/`digest` and `backends/mcp/records.sha256_json` | `core/hashing.sha256_json(value, default=...)` | MCP callers pass `default=str` and `scalability` passes its pydantic encoder, so the digests are unchanged. |
| Perovskite wrapper `examples/perovskite_oxides/run.py` | `matsim-agents run examples/perovskite_oxides/application.yaml` | The wrapper repeated the runner's generation and relaxation logic. |

## Kept, with reason

| Modules | Reason |
| --- | --- |
| `execution/ledger._canonical` | Uses `ensure_ascii=False`. Switching to `core.hashing` would change every stored ledger hash. |
| `neb/serialization._canonical` | An indented on-disk file format, not a hash input. |
| `benchmarks/e1/protocol._canonical_json`, `benchmarks/e3/production._canonical_json` | Same bytes as `core.hashing.canonical_json`, but they pin protocol and evidence hashes of frozen artifacts and live outside the package. |
| `benchmarks/e4/framework/workload.canonical_json` | `allow_nan=False` and not ASCII-escaped; the output differs from `core.hashing`. |
| `_reference_energy`/`_reference_forces` in `finetune_hydragnn.py`, `finetune_uma.py` and `workflows/active_learning/evaluate.py` | Label keys and constraint handling differ: UMA also accepts `REF_*`, HydraGNN reads forces with `apply_constraint=False`, `evaluate` prefers the matsim `info['energy']` schema. A merge would change training labels. |
| `_apply_model_override` in `active_learning/evaluate.py` and `multi_formula/formula_execution.py` | One line each (`cfg.mlip.with_checkpoint`), different signatures. |
| `PacketCandidate` in `science/acquisition/selection_revision.py` and `science/decision/revision.py` | Different packet schemas (`selection-packet-v1`, `objective-packet-v1`) with different fields. They share the number/evidence-id guards, which `decision/revision.py` already imports from the selection module. |
| `coverage_report` in `workflows/exploration/coverage.py` and `workflows/hierarchy/coverage.py` | Different domains: exploration stratum and convergence statistics versus required-interaction coverage for transfer. |
| `_now` in `ledger`, `task_board`, `qe_neb_run`, `qe_neb_stages`, `spectra_pka/runner` | One-line UTC timestamps. A shared helper would add a dependency between layers for no behavioral gain. |
| `_pair_key` in `science/acquisition/selection.py` and `workflows/hierarchy/coverage.py` | Different keys: a sorted candidate-id pair versus a shell/element-pair string. |
| `_read_structure` in `science/structures/screening.py` and `workflows/active_learning/md_sampler.py` | Different return types (pymatgen `Structure` versus ASE `Atoms`). |

## Method

Re-run the scan after structural changes: list top-level function and class names that occur in
more than one module under `src/matsim_agents`, then check each for identical bodies.
