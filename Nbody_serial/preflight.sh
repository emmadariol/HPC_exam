#!/usr/bin/env bash
set -euo pipefail

mpi_run="${MPI_RUN:-mpirun -np 2}"
use_srun="${USE_SRUN:-0}"
image="${IMAGE:-nbody.sif}"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "missing required command: $cmd" >&2
    return 1
  fi
}

run_mpi() {
  if [[ "$use_srun" == "1" ]]; then
    OMP_NUM_THREADS=2 srun --cpu-bind=verbose,cores --ntasks=2 --cpus-per-task=2 "$@"
  else
    OMP_NUM_THREADS=2 $mpi_run "$@"
  fi
}

require_cmd make
require_cmd python3
if [[ "$use_srun" == "1" ]]; then
  require_cmd srun
else
  require_cmd "${mpi_run%% *}"
fi

make clean
make all

./generate_ic --model 0 --n 128 --seed 42 --output preflight_plummer_128.bin
./nbody_direct_serial --input preflight_plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --quiet
run_mpi ./nbody_direct_hybrid --input preflight_plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --integrator kdk --comm sendrecv --quiet
run_mpi ./nbody_direct_hybrid --input preflight_plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --integrator dkd --comm overlap --quiet
OMP_NUM_THREADS=2 ./nbody_layout_benchmark --input preflight_plummer_128.bin --layout both --warmups 1 --inner-repeats 1 --header > preflight_layout.csv

cat > preflight_benchmark_results.csv <<'CSV'
kind,N,nsteps,ranks,threads,repeat,integrator,comm,kernel,rsqrt,dtype,total,io,drift,force,comm_wait,kick,energy,gpairs,status,max_rel_drift
strong,128,5,1,1,1,kdk,sendrecv,direct,exact,double,0.10,0.01,0.01,0.06,0.00,0.01,0.01,0.01,OK,1e-9
strong,128,5,2,1,1,kdk,sendrecv,direct,exact,double,0.06,0.01,0.01,0.03,0.01,0.01,0.01,0.02,OK,1e-9
weak,128,5,1,1,1,kdk,sendrecv,direct,exact,double,0.10,0.01,0.01,0.06,0.00,0.01,0.01,0.01,OK,1e-9
weak,256,5,2,1,1,kdk,sendrecv,direct,exact,double,0.14,0.01,0.01,0.09,0.01,0.01,0.01,0.02,OK,1e-9
CSV
python3 analyze_benchmark.py preflight_benchmark_results.csv preflight_benchmark_summary.csv
python3 plot_scaling.py preflight_benchmark_summary.csv preflight_scaling

cat > preflight_container_overhead.csv <<'CSV'
kind,mode,N,ranks,threads,repeat,total,status,max_rel_drift
strong,native,128,1,1,1,0.10,OK,1e-9
strong,container,128,1,1,1,0.11,OK,1e-9
weak,native,128,1,1,1,0.10,OK,1e-9
weak,container,128,1,1,1,0.12,OK,1e-9
CSV
python3 analyze_container_overhead.py preflight_container_overhead.csv preflight_container_summary.csv
python3 plot_container_overhead.py preflight_container_summary.csv preflight_container

python3 analyze_layout.py preflight_layout.csv preflight_layout_summary.csv

cat > preflight_energy_overhead.csv <<'CSV'
N,nsteps,ranks,threads,repeat,energy_every,total,force,energy,status,max_rel_drift
128,5,1,1,1,1,0.10,0.06,0.02,OK,1e-9
128,5,1,1,1,5,0.08,0.06,0.005,OK,1e-9
CSV
python3 analyze_energy.py preflight_energy_overhead.csv preflight_energy_summary.csv

if [[ -f "$image" ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    apptainer exec "$image" /opt/nbody/nbody_direct_hybrid --help >/dev/null
  elif command -v singularity >/dev/null 2>&1; then
    singularity exec "$image" /opt/nbody/nbody_direct_hybrid --help >/dev/null
  else
    echo "container runtime missing; skipping image smoke"
  fi
else
  echo "container image $image not present; skipping image smoke"
fi

rm -f preflight_plummer_128.bin plummer_128.bin plummer_128_final.bin plummer_128_hybrid_final.bin
echo "preflight ok"
