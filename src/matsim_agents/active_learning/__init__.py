"""Active-learning loop: MLIP surrogate <-> DFT ground-truth labeller.

High-level architecture
-----------------------
Each AL iteration:

1.  ``candidates``  — Generate candidate structures with the configured MLIP
    backend (HydraGNN or UMA).
2.  ``uncertainty`` — Score candidates and pick the top-K to label.
3.  ``backends.dft.runner`` — Submit K single-point DFT jobs concurrently across
    nodes; each ``backends.dft`` backend parses its outputs into labelled
    (Atoms, energy, forces, stress) records.
4.  ``dft_protocol`` — Record method and execution signatures of the DFT settings
    so labels from incompatible setups are never mixed.
5.  ``trainer``     — Append to dataset and optionally retrain the MLIP.
6.  ``loop``        — Orchestrate the above, persist state for resume.

Entry points:
    * Python:  ``matsim_agents.active_learning.loop.run_active_learning``
    * CLI:     ``matsim-agents al run --config al.yaml``
"""

from matsim_agents.active_learning.config import ALConfig

__all__ = ["ALConfig"]
