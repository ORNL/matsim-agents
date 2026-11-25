"""Collinear magnetic configurations: define, generate, parse, compare.

Where to start:

- ``config``: ``MagneticConfiguration``, the validated per-atom starting moments.
- ``generators``: FM, AFM, ferrimagnetic and random-collinear starting states.
- ``policies``: material-specific starting-state rules that pick generators.
- ``results``: parse QE magnetic output and classify converged moments.
- ``competition``: rank converged states of one structure by energy.
- ``consistency``: compare magnetic outcomes across codes or settings.
- ``defects``: moments on defect cells and along migration paths.

Results are evaluated starting states, never a claim of the global magnetic ground state.
"""

from matsim_agents.discovery.magnetism.config import (
    MAGNETIC_MODES,
    UNSUPPORTED_MAGNETISM,
    MagneticConfiguration,
    MagneticMode,
    QualificationStatus,
    qe_polarization_from_moment,
    reject_unsupported,
)

__all__ = [
    "MAGNETIC_MODES",
    "UNSUPPORTED_MAGNETISM",
    "MagneticConfiguration",
    "MagneticMode",
    "QualificationStatus",
    "qe_polarization_from_moment",
    "reject_unsupported",
]
