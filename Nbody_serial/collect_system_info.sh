#!/usr/bin/env bash
set -euo pipefail

# Run this mode under the SAME srun placement options as the solver.
# It checks the current rank, without generating particles or running a benchmark.
# Usage: srun ... bash collect_system_info.sh --verify-binding numa|socket|core
if [[ "${1:-}" == "--verify-binding" ]]; then
  python3 - "${2:?expected numa, socket or core}" <<'PY'
import json
import os
import subprocess
import sys

mode = sys.argv[1]
if mode not in ('numa', 'socket', 'core'):
    raise SystemExit('Expected numa, socket or core')
topology = subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET,NODE'], text=True)
rows = [tuple(map(int, line.split(','))) for line in topology.splitlines()
        if line and not line.startswith('#')]
allowed = set(os.sched_getaffinity(0))
selected = [row for row in rows if row[0] in allowed]
cores = {(row[2], row[1]) for row in selected}
sockets = {row[2] for row in selected}
nodes = {row[3] for row in selected}
threads = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
valid = len(cores) == threads and len(selected) == len(allowed)
if mode == 'numa':
    valid = valid and len(nodes) == 1 and -1 not in nodes
    domain = {(r[2], r[1]) for r in rows if r[3] in nodes}
    valid = valid and cores == domain
elif mode == 'socket':
    valid = valid and len(sockets) == 1
    domain = {(r[2], r[1]) for r in rows if r[2] in sockets}
    valid = valid and cores == domain
else:
    valid = valid and len(cores) == 1 and threads == 1
print(json.dumps(dict(check='PASS' if valid else 'FAIL', mapping=mode,
                      host=os.uname().nodename, rank=os.environ.get('SLURM_PROCID'),
                      cpus=sorted(allowed), cores=sorted(cores),
                      sockets=sorted(sockets), numa=sorted(nodes),
                      cpus_per_task=threads)), flush=True)
sys.exit(0 if valid else 1)
PY
  shift 2
  if [[ $# -gt 0 ]]; then exec "$@"; fi
  exit 0
fi

# Collect reproducibility metadata for the node that actually runs a Slurm job.
# This script is intentionally read-only: it records hardware, compiler, MPI,
# memory, NUMA, and OpenMP/MPI environment details without changing the system.
out="${1:-results_final/system_info.txt}"
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
  echo "## linked MPI and OpenMP runtimes"
  ldd ./nbody_direct_hybrid || true
  # Runtime environment variables document binding/threading choices inherited
  # from the batch script or cluster modules.
  echo
  echo "## openmp environment"
  env | sort | grep -E '^(OMP_|GOMP_|KMP_|I_MPI_|OMPI_|MPICH_)' || true
} > "$out"

echo "wrote $out"
