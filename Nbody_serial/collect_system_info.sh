#!/usr/bin/env bash
set -euo pipefail

# Collect reproducibility metadata for the node that actually runs a Slurm job.
# This script is intentionally read-only: it records hardware, compiler, MPI,
# memory, NUMA, and OpenMP/MPI environment details without changing the system.
out="${1:-docs/system_info.txt}"
mkdir -p "$(dirname "$out")"

{
  # Timestamp and kernel/OS information make it possible to identify the exact
  # platform used for the final figures.
  echo "# system information"
  date -Is
  echo
  echo "## uname"
  uname -a || true
  # CPU topology is needed to justify the chosen rank/thread counts and to
  # explain why EPYC/GENOA nodes have different useful scaling ranges.
  echo
  echo "## cpu"
  lscpu || true

  # NUMA layout matters for memory bandwidth and first-touch effects.
  echo
  echo "## numa"
  if command -v numactl >/dev/null 2>&1; then
    numactl -H
  else
    echo "numactl not available"
  fi
  # Memory capacity is recorded to prove that the chosen N values fit in-core.
  echo
  echo "## memory"
  free -h || true

  # Compiler and MPI versions are part of the experimental environment.
  echo
  echo "## compiler"
  cc --version || true
  echo
  echo "## mpi compiler"
  mpicc --version || true
  echo
  echo "## mpi runtime"
  mpirun --version || true
  # Runtime environment variables document binding/threading choices inherited
  # from the batch script or cluster modules.
  echo
  echo "## openmp environment"
  env | sort | grep -E '^(OMP_|GOMP_|KMP_|I_MPI_|OMPI_|MPICH_)' || true
} > "$out"

echo "wrote $out"
