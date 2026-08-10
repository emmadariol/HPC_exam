#!/usr/bin/env bash
set -euo pipefail

IMAGE=${1:-nbody.sif}

echo "Checking MPI linkage inside Singularity image: $IMAGE"
singularity exec "$IMAGE" bash -c 'which mpirun || true; ldd ./nbody_direct_hybrid | head -n 40'

echo "On the host, mpirun:"
which mpirun || true
ldd ./nbody_direct_hybrid | head -n 40 || true

echo "If the container binary links to libmpi inside the image, rebuild or use singularity --nv/--mpi flags to bind host MPI libs."
