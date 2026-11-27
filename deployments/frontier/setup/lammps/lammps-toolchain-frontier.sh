#!/bin/bash
# Frontier toolchain for the self-built LAMMPS (sourced by the build script and activate.sh).
module swap PrgEnv-cray PrgEnv-gnu >/dev/null 2>&1 || module load PrgEnv-gnu >/dev/null 2>&1
export LAMMPS_GCC_MODULE="${LAMMPS_GCC_MODULE:-gcc-native/14.2}"
export LAMMPS_MPICH_MODULE="${LAMMPS_MPICH_MODULE:-cray-mpich/8.1.32}"
export LAMMPS_CMAKE_MODULE="${LAMMPS_CMAKE_MODULE:-cmake/3.31.11}"
module load "${LAMMPS_GCC_MODULE}" "${LAMMPS_MPICH_MODULE}" "${LAMMPS_CMAKE_MODULE}"
module unload darshan-runtime >/dev/null 2>&1 || true
# Without this the loader takes the default MPICH from /opt/cray/pe/lib64, not the loaded one.
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
