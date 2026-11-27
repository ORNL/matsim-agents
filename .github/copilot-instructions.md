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
