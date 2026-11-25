"""Standalone MLIP fine-tuning programs (UMA, MACE, HydraGNN) and their cost tracking.

Each ``finetune_*`` module is both an importable function and a CLI
(``python -m matsim_agents.backends.mlip.training.finetune_<model>``) invoked by
:mod:`matsim_agents.workflows.active_learning.trainer` or the facility launchers.
"""
