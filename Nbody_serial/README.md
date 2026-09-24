# Direct N-body simulation with MPI + OpenMP

HPC exam project, Exercise 1 (direct gravitational N-body). The full description of
the method, the experiments and the results is in [`FINAL_REPORT.md`](FINAL_REPORT.md).

## What is in this folder

```text
Nbody_serial/
├── nbody_direct_hybrid.c     main solver: MPI ring + OpenMP, KDK leapfrog, energy check, phase timers
├── nbody_direct_serial.c     serial reference solver (used for the vectorisation report)
├── nbody_layout_benchmark.c  force-only AoS vs SoA benchmark (OpenMP, no MPI)
├── memory_bandwidth.c        STREAM-like memory-bandwidth test (OpenMP)
├── generate_ic.c             initial conditions: Plummer sphere (--model 0) or uniform ball (--model 1)
├── nbody_common.h            shared types, precision switch, binary file format
├── Makefile                  builds all programs; `make vec-report` writes the compiler reports
├── run_benchmarks.sh         benchmark driver: repeats runs and writes one CSV line per run
├── benchmark_common.sh       helpers used by run_benchmarks.sh (launcher, binding, solver call)
├── submit.sh                 Slurm wrapper for Orfeo (and LEONARDO) around run_benchmarks.sh
├── analyze.py                turns raw CSV files into summaries (median, s, speedup) and SVG plots
├── plot_required_native_strong.py  plots used for the main strong-scaling campaign
├── collect_system_info.sh    records lscpu, numactl, memory, compiler and MPI versions
├── Dockerfile                container image (Ubuntu 24.04, -march=x86-64-v3, OSU benchmarks)
├── Singularity.def           equivalent Singularity recipe
├── jobs/                     extra Slurm scripts (rank/thread binding check)
├── results_final/            all data, summaries and figures used by the report (see its README)
├── docs/                     study notes and guides (in Italian)
└── FINAL_REPORT.md           the report
```

## Build

On an Orfeo GENOA compute node (so that `-march=native` sees the real processor):

```sh
module purge
module load openMPI/4.1.6
make                 # nbody_direct_hybrid, nbody_direct_serial, nbody_layout_benchmark, generate_ic, memory_bandwidth
make vec-report      # compiler vectorisation reports into results_final/
```

The main solver is compiled as

```sh
mpicc -std=c11 -DNBODY_USE_DOUBLE -O3 -march=native -Wall -Wextra -Wpedantic -fopenmp \
      -o nbody_direct_hybrid nbody_direct_hybrid.c -lm
```

`make PRECISION=float` builds single-precision arithmetic instead of double.

## Run

```sh
./generate_ic --model 0 --n 10000 --seed 123 --output plummer_10000.bin

export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=spread
srun -n 8 -c 8 --cpu-bind=verbose,cores ./nbody_direct_hybrid \
     --input plummer_10000.bin --nsteps 100 --dt 1e-4 --eps 0.05 \
     --energy-every 10 --comm sendrecv --kernel direct --rsqrt exact --accumulators 4
```

Main options: `--comm sendrecv|overlap` (blocking or overlapped ring),
`--kernel direct|newton` (Newton's third law, one rank only),
`--rsqrt exact|approx1|approx2`, `--accumulators 1|2|4|8`.
At the end the program prints the largest relative energy drift with an OK/FAIL
status (tolerance 1e-4), the time of each phase (largest value over the ranks)
and the pair throughput in Gpairs/s.

## Benchmarks on Orfeo

```sh
./submit.sh --cluster orfeo --bench scaling --name strong --result-dir $PWD/runs/strong_$(date +%Y%m%d_%H%M%S) -- \
  SCALING_KINDS=strong STRONG_N=100000 NSTEPS=20 RANKS="1 2 4 8 16 32 64" THREADS=1 \
  COMM=sendrecv RSQRT=exact ACCUMULATORS=4 ENERGY_EVERY=20 REPEATS=5 WARMUPS=1
```

Other benchmark families: `hybrid`, `ablation`, `layout`, `energy`, `memory`,
`container`, `osu`, `arch`. Parameters after `--` are passed to
`run_benchmarks.sh` as environment variables. Raw output goes to `runs/`
(ignored by git); curated results are copied to `results_final/`.

## Container

```sh
# on a computer with Docker
docker build -t memid01/nbody-hpc:latest .
docker push memid01/nbody-hpc:latest

# on Orfeo (building from Singularity.def needs fakeroot, not available to users)
module load singularity/4.3.1
singularity pull --force nbody.sif docker://memid01/nbody-hpc:latest

export SINGULARITY_BINDPATH=/opt/programs/openMPI/4.1.6:/opt/programs/openMPI/4.1.6,/opt/programs/hwloc/2.12.0:/opt/programs/hwloc/2.12.0
export SINGULARITYENV_LD_LIBRARY_PATH=/opt/programs/openMPI/4.1.6/lib:/opt/programs/hwloc/2.12.0/lib
srun -n 8 singularity exec nbody.sif /opt/nbody/nbody_direct_hybrid --input plummer_10000.bin --nsteps 100
```

The bind paths make the program inside the container use the cluster's own
Open MPI library; check it with `singularity exec nbody.sif ldd /opt/nbody/nbody_direct_hybrid`.

## Binary file format

All programs share one native-endian format: an 8-byte magic string `NBODYF1\0`,
a `uint64_t` particle count N, then N records of six `float` values
(x y z vx vy vz). All particles have the same mass, given with `--mass`
(default 1).
