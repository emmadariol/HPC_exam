# Direct N-body simulation with MPI + OpenMP

Report: [`FINAL_REPORT.md`](FINAL_REPORT.md).

## Build 

```sh
module purge
module load openMPI/4.1.6

make                                   # native build (-march=native): all programs
make PRECISION=float                   # single-precision build
make clean && make nbody_direct_hybrid CFLAGS="-O3 -march=x86-64-v3 -Wall -Wextra -Wpedantic"   # portable build
make vec-report                        # compiler vectorisation reports
```

## Run

```sh
./generate_ic --model 0 --n 10000 --seed 123 --output plummer_10000.bin

export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=spread
srun -n 8 -c 8 --cpu-bind=verbose,cores ./nbody_direct_hybrid \
     --input plummer_10000.bin --nsteps 100 --dt 1e-4 --eps 0.05 --energy-every 10 \
     --comm sendrecv --kernel direct --rsqrt exact --accumulators 4
```

Options: `--comm sendrecv|overlap`, `--kernel direct|newton`, `--rsqrt exact|approx1|approx2`, `--accumulators 1|2|4|8`.

## Benchmarks (Slurm)

```sh
chmod +x submit.sh run_benchmarks.sh

# strong scaling, N = 100000, 100 steps
./submit.sh --cluster orfeo --bench scaling -- \
  SCALING_KINDS=strong STRONG_N=100000 NSTEPS=100 RANKS="8 16 32" THREADS=1 COMM=sendrecv ENERGY_EVERY=100 REPEATS=5 WARMUPS=0

# weak scaling, 10000 particles per rank
./submit.sh --cluster orfeo --bench scaling -- \
  SCALING_KINDS=weak WEAK_PER_RANK=10000 NSTEPS=100 RANKS="1 2 4 8 16" THREADS=1 COMM=sendrecv ENERGY_EVERY=100 REPEATS=5 WARMUPS=0

# the same inside the container: add  USE_CONTAINER=1 IMAGE=$PWD/nbody.sif

./submit.sh --cluster orfeo --bench hybrid       # rank x thread mappings
./submit.sh --cluster orfeo --bench ablation     # Newton, rsqrt, communication, accumulators
./submit.sh --cluster orfeo --bench evidence     # AoS/SoA layout and energy-check cost
./submit.sh --cluster orfeo --bench arch         # -march=native vs -march=x86-64-v3
./submit.sh --cluster orfeo --bench container    # container start-up and overhead
./submit.sh --cluster orfeo --bench osu --nodes 2   # OSU latency/bandwidth, native and container

squeue -u $USER
```

Results are written to `runs/<cluster>_<bench>_<date>/`.

## Container

```sh
# on a computer with Docker
docker build -t memid01/nbody-hpc:latest .
docker push memid01/nbody-hpc:latest

# on Orfeo
module load singularity/4.3.1
singularity pull --force nbody.sif docker://memid01/nbody-hpc:latest

export SINGULARITY_BINDPATH=/opt/programs/openMPI/4.1.6:/opt/programs/openMPI/4.1.6,/opt/programs/hwloc/2.12.0:/opt/programs/hwloc/2.12.0
export SINGULARITYENV_LD_LIBRARY_PATH=/opt/programs/openMPI/4.1.6/lib:/opt/programs/hwloc/2.12.0/lib
srun -n 8 singularity exec nbody.sif /opt/nbody/nbody_direct_hybrid --input plummer_10000.bin --nsteps 100
singularity exec nbody.sif ldd /opt/nbody/nbody_direct_hybrid    # check that the cluster MPI is used
```
