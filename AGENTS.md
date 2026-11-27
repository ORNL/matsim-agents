# Agent instructions for matsim-agents

## HydraGNN: branch-7 (OMat24) ONLY — non-negotiable

Whenever matsim-agents uses HydraGNN (inference, relaxation, MD, NEB, LAMMPS,
active learning, fine-tuning, evaluation, benchmarks, deployments):

- Use only branch-7, the OMat24 head (0-based head index 7). Discard all other heads.
- Never use, load, train, or reintroduce a BranchWeightMLP, branch routing,
  fused/mixture multi-head inference, or `mlp_checkpoint`/`mlp_precision`.
  The MLP was trained so that all branches contribute energy but only one
  contributes forces; it is not used.
- Never select another head (`inference_head` must be `OMat24`, `7`, or `branch-7`).
- Fine-tune HydraGNN only with `finetune_hydragnn_newhead` starting from branch-7
  (`newhead.json` must record `head_index: 7`).
- Enforcement: `backends/mlip/config.py` (`HydraGNNConfig`), `backends/mlip/calculator.py`,
  `backends/mlip/hydragnn_energy.py`, and `tests/test_hydragnn_branch7_only.py`.
  Do not weaken or bypass these checks; if a task seems to need another head or the
  MLP, stop and ask the user.

## HydraGNN source code: the pinned checkout ONLY

- The HydraGNN code is the commit pinned in `benchmarks/e4_mlip/frozen-source.json`
  (see `docs/hydragnn-frozen-source.md`). Never use whatever `hydragnn` is on `PYTHONPATH`, in a
  venv, or in a sibling directory such as `$PROJ/HydraGNN`; that tree can differ and fail to load
  BEST6.
- Job and launcher scripts must `source deployments/common/hydragnn-env.sh` and call
  `hydragnn_export_pythonpath`. Do not set `HYDRAGNN_ROOT`, `HYDRAGNN_DIR`, `HYDRAGNN_EXAMPLE` or a
  HydraGNN `PYTHONPATH` anywhere else (`tests/test_hydragnn_env_single_source.py`).
- Loading or fine-tuning from a logdir checks the imported HydraGNN commit against the logdir's
  `hydragnn-source.json` or the checkpoint hash in
  `backends/mlip/hydragnn_sources.json` (`backends/mlip/hydragnn_source.py`). A logdir with
  neither is refused. Do not add registry entries without verifying the commit that loads them.

## Verify before claiming

- Read a file or run a command before describing it. Do not cite files or line numbers that were
  not read, or take them from a truncated search result.
- State plainly what was not run (for example the full test suite or CI).
