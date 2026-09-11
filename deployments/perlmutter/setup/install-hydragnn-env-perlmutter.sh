#!/bin/bash
#SBATCH -J matsim-hydragnn-install
#SBATCH -A m5216_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -o runs/matsim-hydragnn-install-%j.out
#SBATCH -e runs/matsim-hydragnn-install-%j.err

# =============================================================================
# Slurm batch wrapper for install.sh (full matsim-agents + HydraGNN environment
# build), so the build runs on a GPU compute node instead of a login node.
#
# Why this exists:
#   Running install.sh directly on a login node (even backgrounded with
#   `setsid nohup ... & disown`) is fragile: heavy, long CPU-bound builds can
#   be silently killed by NERSC's login-node watchdog with no error/signal
#   logged. Submitting as a proper Slurm batch job is the robust alternative.
#
# NERSC QOS-naming gotcha (non-obvious, easy to misdiagnose as a GPU-hours
# allocation problem):
#   Submitting with `-q gpu_premium`/`-q gpu_shared`/`-q gpu_regular`/
#   `-q gpu_debug` together with `-C gpu` is REJECTED by NERSC's Lua submit
#   filter ("Job request does not match any supported policy" /
#   "Batch job submission failed: Unspecified error"), regardless of
#   resources/time/account balance. Use the generic QOS name WITHOUT the
#   `gpu_` prefix instead (`-q premium`, `-q shared`, `-q regular`, `-q debug`)
#   together with `-C gpu`; the site's submit filter auto-maps it internally
#   to the correct `gpu_*` QOS. `-q premium -C gpu -N 1` lands on the
#   `gpu_ss11` partition and gets a whole node (4x A100) with no need for
#   explicit `--gpus`/`-c`.
#
# Time budget:
#   HydraGNN's own Perlmutter installer unconditionally wipes and recreates
#   its conda venv on every run (see HydraGNN's
#   scripts/hpc/nersc/perlmutter/installation/install.sh), so every attempt
#   is a full rebuild from scratch -- there is no incremental resume. Full
#   builds have taken anywhere from ~3.5h to 8h+ depending on the specific
#   compute node assigned (node-to-node performance has varied noticeably;
#   if a build seems abnormally slow, it may be worth cancelling and
#   resubmitting to land on a different node). premium QOS allows up to a
#   2-day walltime if more headroom is needed.
#
# Usage:
#   sbatch deployments/perlmutter/setup/install-hydragnn-env-perlmutter.sh
#
#   # Check status:
#   squeue -j <jobid>            # while queued/running
#   sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,Start,End
#                                 # after it leaves the queue (squeue then
#                                 # returns "Invalid job id specified")
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
mkdir -p runs

export PERLMUTTER_LMOD_RESTORE_PATH=/opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
export PERLMUTTER_CPE_MODULE=cpe/26.03
export PERLMUTTER_PRGENV_MODULE=PrgEnv-gnu/8.7.0
# cray-mpich/8.1.30's GTL library (libmpi_gtl_cuda.so) links against
# libcudart.so.12, which is incompatible with cudatoolkit/13.0 (needed for
# torch 2.14/cu130) and breaks the mpi4py build. cray-mpich/9.1.0's GTL
# library links against libcudart.so.13.
export PERLMUTTER_MPICH_MODULE=cray-mpich/9.1.0
export EXPECTED_CUDA_MM=13.0
export INSTALL_UMA=1
export INSTALL_MACE=1

bash deployments/perlmutter/setup/install.sh
