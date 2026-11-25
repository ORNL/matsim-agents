"""Large candidate-pool exploration: track, reduce, sample, promote, dispatch.

Where to start, in pipeline order:

- ``candidates``: ``CandidateManifest``, an append-only, resumable record of every candidate.
- ``reduction``: ``deduplicate_candidates`` removes exact and symmetry duplicates.
- ``sampling``: stratified and adaptive selection under an evaluation budget.
- ``magnetic_surrogate``: optional learned ranking with held-out evaluation.
- ``multifidelity``: evidence rules for promoting candidates F0 -> F3.
- ``dispatch``: non-blocking Slurm submission and result recording.
- ``coverage``: what was and was not explored.
- ``demo``: a synthetic end-to-end run used for scale and recovery tests.
"""
