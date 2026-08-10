#!/usr/bin/env bash
set -euo pipefail

out="${1:-system_info.txt}"

{
  echo "# system information"
  date -Is
  echo
  echo "## uname"
  uname -a || true
  echo
  echo "## cpu"
  lscpu || true
  echo
  echo "## numa"
  if command -v numactl >/dev/null 2>&1; then
    numactl -H
  else
    echo "numactl not available"
  fi
  echo
  echo "## memory"
  free -h || true
  echo
  echo "## compiler"
  cc --version || true
  echo
  echo "## mpi compiler"
  mpicc --version || true
  echo
  echo "## mpi runtime"
  mpirun --version || true
  echo
  echo "## openmp environment"
  env | sort | grep -E '^(OMP_|GOMP_|KMP_|I_MPI_|OMPI_|MPICH_)' || true
} > "$out"

echo "wrote $out"
