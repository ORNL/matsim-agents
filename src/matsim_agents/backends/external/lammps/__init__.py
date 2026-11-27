"""Optional interface to an externally installed LAMMPS executable.

MATSIM-AGENTS does not bundle, build, or download LAMMPS or any interatomic potential. The
user supplies the executable (``LAMMPS_EXECUTABLE`` or ``lmp`` on ``PATH``) and the explicit
``pair_style``/``pair_coeff`` lines. A successful process is not a validated result.
"""
