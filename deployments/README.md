# Deployments

Machine-specific setup, launch, job, smoke-test, and configuration assets live
under one directory per supported leadership-class system.  Runtime Python code
must not import from this tree; platform selection belongs at deployment time.

- `frontier/`: OLCF Frontier (Slurm)
- `aurora/`: ALCF Aurora (PBS Pro)
- `perlmutter/`: NERSC Perlmutter (Slurm)
