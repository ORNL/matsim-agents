"""Active-learning loop: HydraGNN MLFF surrogate <-> VASP ground-truth labeller.

High-level architecture
-----------------------
Each AL iteration:

1.  ``candidates``  — Generate candidate structures (HydraGNN-driven MD by
    default; pluggable).
2.  ``uncertainty`` — Score candidates and pick the top-K to label.
3.  ``vasp_runner`` — Submit K single-point VASP jobs concurrently across
    Frontier nodes via inner ``srun`` steps.
4.  ``vasp_io``     — Parse OUTCAR/vasprun.xml -> labelled (Atoms, energy,
    forces, stress) records.
5.  ``trainer``     — Append to dataset and (optionally) retrain HydraGNN.
6.  ``loop``        — Orchestrate the above, persist state for resume.

Entry points:
    * Python:  ``matsim_agents.active_learning.loop.run_active_learning``
    * CLI:     ``matsim-agents al run --config al.yaml``
"""

from matsim_agents.active_learning.config import ALConfig, DFTConfig, MLPConfig, UMAConfig

__all__ = ["ALConfig", "DFTConfig", "MLPConfig", "UMAConfig"]
