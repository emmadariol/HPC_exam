#!/usr/bin/env bash
set -euo pipefail

image="${IMAGE:-hpc-nbody}"
input="${INPUT:-docker_plummer_128.bin}"

docker build -t "$image" . >/dev/null

docker run --rm \
  -v "$PWD":/data -w /data \
  "$image" \
  /opt/nbody/generate_ic --model 0 --n 128 --seed 42 --output "$input"

docker run --rm \
  -e OMPI_ALLOW_RUN_AS_ROOT=1 \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  -e OMP_NUM_THREADS=2 \
  -v "$PWD":/data -w /data \
  "$image" \
  mpirun --allow-run-as-root -np 2 /opt/nbody/nbody_direct_hybrid \
    --input "$input" \
    --nsteps 5 \
    --dt 1e-4 \
    --eps 0.05 \
    --energy-every 1 \
    --integrator kdk \
    --comm sendrecv \
    --kernel direct \
    --rsqrt exact \
    --quiet

rm -f "$input"
echo "Docker smoke test completed"
