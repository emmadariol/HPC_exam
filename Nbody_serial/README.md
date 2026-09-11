# Serial C11 direct N-body baseline

This directory contains the stand-alone programs for the direct gravitational N-body exercise:

- `nbody_direct_serial.c`: serial softened direct solver using a DKD leapfrog
  step and a relative energy-drift verifier.
- `nbody_direct_hybrid.c`: MPI + OpenMP direct solver for Exercise 1. Each MPI
  rank owns one permanent SoA chunk, rotates source chunks with a ring-shift
  communication pattern, parallelises the home-particle force loop with OpenMP,
  and reports per-section timings plus a pair-interaction rate.
- `nbody_layout_benchmark.c`: force-only AoS-vs-SoA microbenchmark using the
  same binary input format and force law as the solver.
- `collect_system_info.sh`: writes the hardware/software stack requested in the
  report.
- `run_benchmarks.sh scaling`: runs repeated strong/weak scaling experiments and
  writes a CSV that can be plotted for speedup and efficiency.
- `analyze.py summarize scaling`: reduces the raw benchmark CSV to medians, standard
  deviations, outlier counts, speedup, efficiency, and estimated ring
  communication bandwidth.
- `run_benchmarks.sh layout`, `analyze.py summarize layout`: generate and summarize the
  AoS-vs-SoA evidence table.
- `run_benchmarks.sh energy`, `analyze.py summarize energy`: quantify the cost of different
  `--energy-every` diagnostic periods.
- `run_benchmarks.sh memory`, `analyze.py summarize memory`: measure standalone
  STREAM-style RAM bandwidth for the hardware section of the report.
- `run_benchmarks.sh osu`: collect OSU latency/bandwidth microbenchmarks for the
  MPI stack used by the production runs.
- `run_benchmarks.sh arch`: compare native `-march=native` and portable
  `-march=x86-64-v3` builds on the same host, isolating code-generation effects
  from Singularity runtime overhead.
- `Dockerfile` and `Singularity.def`: starter container recipes for the
  required native-vs-container comparison. Both images also build OSU
  Micro-Benchmarks so `osu_latency` and `osu_bw` are available inside the
  container.
- `run_benchmarks.sh container`: repeated native-vs-Singularity timing table helper.
- `analyze.py summarize container`: reduces Singularity/Apptainer overhead CSVs to
  medians, standard deviations, and overhead percentages.
- `docs/REPORT_TEMPLATE.md`: checklist-style report skeleton matching the exam
  deliverables.
- `generate_ic.c`: initial-condition generator used by the project for Plummer
  sphere initial conditions (`--model 0`).

## Project layout

The repository is intentionally kept mostly flat for the executable sources and
benchmark drivers, because the `Makefile`, Slurm jobs, Dockerfile and
Singularity recipe all compile or call these files directly from the project
root.

```text
Nbody_serial/
├── *.c, *.h                  source code, benchmarks, and shared definitions
├── Makefile                  native build, smoke target, vectorization target
├── run_benchmarks.sh         unified benchmark driver used by jobs and local runs
├── analyze.py                unified CSV post-processing and SVG plotting CLI
├── Dockerfile                Docker image recipe
├── Singularity.def           Singularity/Apptainer recipe
├── FINAL_REPORT.md           final report with accepted figures and tables
├── README.md                 quick project overview
├── docs/                     extended guides, template, static info files
├── jobs/submit.sh            unified Slurm submission wrapper
├── results_final/            curated CSV/SVG/TXT files used by the report
└── runs/                     ignored local run history and Slurm scratch output
```

The codes are intended as an explicit, inspectable implementation of the N-body
exercise.  The scalable production path is still the required direct O(N^2)
all-pairs algorithm, but the hybrid solver now exposes the optimisation choices
measured in the report: MPI ring decomposition, coordinated MPI-IO input,
OpenMP/SIMD force loops, 1/2/4/8 accumulator chains, approximate reciprocal
square root, and the single-rank Newton-third-law ablation.

## Arithmetic type

All physical quantities in the solver and generators use the typedef `dtype`, defined in `nbody_common.h`.

Default build, double-precision arithmetic:

```sh
make
```

Single-precision arithmetic:

```sh
make clean
make PRECISION=float
```

The equivalent manual switches are:

```sh
-DNBODY_USE_DOUBLE
-DNBODY_USE_FLOAT
```

Only one of the two should be defined. If neither is defined, the header falls back to double precision.

## Binary file format

All programs use the same native-endian binary format. Particle data are stored in single precision, independently of the selected `dtype` used for arithmetic:

```text
byte 0..7       magic: "NBODYF1\0"
next 8 bytes    uint64_t particle count N
then N records  x y z vx vy vz, six float values per particle
```

The solver assigns one mass to every particle through `--mass`; mass is not stored per particle in the file. This keeps the initial-condition file compact and makes the equal-mass assumption explicit in the command line.

Because the format is deliberately minimal and native-endian, it is intended for same-machine teaching runs and benchmarks, not for long-term archival exchange between heterogeneous systems.

## Build

```sh
make
```

or explicitly:

```sh
cc -std=c11 -DNBODY_USE_DOUBLE -O2 -Wall -Wextra -Wpedantic nbody_direct_serial.c -lm -o nbody_direct_serial
cc -std=c11 -DNBODY_USE_DOUBLE -O2 -Wall -Wextra -Wpedantic generate_ic.c -lm -o generate_ic
```


Part of the assignment is to determine the best compiler’s flags and options, and the CPU bindings. List them in the final report.

## Example runs

Generate a small Plummer sphere and evolve it:

```sh
./generate_ic --model 0 --n 1000 --seed 123 --scale 1.0 --mass 1.0 --output plummer_1000.bin
./nbody_direct_serial --input plummer_1000.bin --nsteps 100 --dt 1e-4 --eps 0.05 --mass 1.0 --energy-every 10 --output final_state.bin
```

Run both smoke tests:

```sh
make run-smoke
```

Run the hybrid solver directly:

```sh
make nbody_direct_hybrid
OMP_NUM_THREADS=4 mpirun -np 2 ./nbody_direct_hybrid \
  --input plummer_1000.bin --nsteps 100 --dt 1e-4 --eps 0.05 \
  --mass 1.0 --energy-every 10 --comm sendrecv \
  --output final_state_hybrid.bin
```

The hybrid solver uses the KDK leapfrog scheme. `--comm sendrecv` uses a simple
blocking ring exchange, while
`--comm overlap` posts `MPI_Irecv`/`MPI_Isend` for the next ring chunk before
computing the current chunk and waits afterwards, so communication can overlap
with the OpenMP force loop when the MPI implementation and problem size allow it.

For scaling runs, keep `OMP_NUM_THREADS`, `OMP_PLACES`, `OMP_PROC_BIND`, and
the MPI binding policy in the benchmark log/report together with the exact
`make` flags. The hybrid executable prints max-rank section timings for I/O,
drift, force, kick, and energy diagnostics, so strong/weak scaling plots can
separate force-kernel scalability from diagnostic and communication overhead.

Example benchmark helpers:

```sh
bash ./collect_system_info.sh docs/system_info.txt
RANKS="1 2 4" THREADS="1 2" REPEATS=5 STRONG_N=4000 \
  WEAK_PER_RANK=1000 NSTEPS=50 ENERGY_EVERY=10 \
  bash ./run_benchmarks.sh scaling
python3 analyze.py summarize scaling benchmark_results.csv benchmark_summary.csv
python3 analyze.py plot scaling benchmark_summary.csv scaling
python3 analyze.py summarize gflops benchmark_summary.csv benchmark_summary_gflops.csv
make vec-report
```

The serial solver keeps checked I/O as the default. `--io-mode fast` skips
per-value finite/range validation for controlled production inputs, while
`--io-profile` reports separate bulk-read, bulk-write, and conversion/checking
times. Comparing otherwise identical `checked` and `fast` runs quantifies the
cost of conversion validation. `make vec-report` also writes
`docs/vectorization_serial_report.txt` for the serial force kernel.

The generated `benchmark_results.csv` contains one line per run, including
status, energy drift, section timings, and kernel rate. Use the median (or
trimmed mean and standard deviation) across the repeated rows in the report.
`WARMUPS` controls unrecorded warmup repetitions before each measured point.
`analyze.py summarize scaling` reports MAD-based outlier counts and estimates
communication bandwidth from the ring traffic and `comm_wait`. `analyze.py plot scaling`
writes SVG plots for strong/weak speedup, efficiency, and estimated
communication bandwidth. `analyze.py summarize gflops` converts the reported
pair-interaction rate to an estimated GFLOP/s column using an explicit
FLOP-per-pair model, so the report can state both the algorithmic rate
(`Gpairs/s`) and the derived floating-point rate.

Evidence helpers for the optimization discussion:

```sh
THREADS="1 2 4" REPEATS=5 WARMUPS=2 N=50000 bash ./run_benchmarks.sh layout
python3 analyze.py summarize layout layout_results.csv layout_summary.csv

RANKS=8 THREADS=1 REPEATS=5 WARMUPS=2 N=50000 NSTEPS=50 \
  ENERGY_LIST="1 5 10 50" bash ./run_benchmarks.sh energy
python3 analyze.py summarize energy energy_overhead.csv energy_overhead_summary.csv

THREADS=64 REPEATS=5 WARMUPS=1 bash ./run_benchmarks.sh memory
python3 analyze.py summarize memory memory_bandwidth.csv memory_bandwidth_summary.csv
python3 analyze.py plot memory memory_bandwidth_summary.csv memory_bandwidth

bash ./run_benchmarks.sh osu --mode native
bash ./run_benchmarks.sh osu --mode container --image nbody.sif
OUT=osu_microbench_native_vs_container.csv \
  bash ./run_benchmarks.sh osu --mode both --image nbody.sif

OUT=perf_counters.txt RANKS=1 THREADS=8 N=20000 NSTEPS=20 \
  bash ./run_benchmarks.sh perf
```

`run_benchmarks.sh osu` expects `osu_latency` and `osu_bw` in `PATH`, or explicit
`OSU_LATENCY=/path/to/osu_latency` and `OSU_BW=/path/to/osu_bw`, for native
runs. Container runs use the OSU binaries installed by `Dockerfile` and
`Singularity.def`.

Optional kernel experiments:

```sh
OMP_NUM_THREADS=4 mpirun -np 1 ./nbody_direct_hybrid \
  --input plummer_1000.bin --nsteps 20 --dt 1e-4 --eps 0.05 \
  --kernel newton --rsqrt exact --quiet

OMP_NUM_THREADS=4 mpirun -np 2 ./nbody_direct_hybrid \
  --input plummer_1000.bin --nsteps 20 --dt 1e-4 --eps 0.05 \
  --kernel direct --rsqrt approx --quiet

OMP_NUM_THREADS=4 mpirun -np 2 ./nbody_direct_hybrid \
  --input plummer_1000.bin --nsteps 20 --dt 1e-4 --eps 0.05 \
  --kernel direct --accumulators 1 --quiet
```

`--kernel newton` is intentionally single-rank only: in a distributed ring,
Newton-third-law reuse also requires returning the opposite force contribution
to the remote owner rank. The direct MPI ring path is the main scalable solver.
The single-rank Newton path uses a pre-allocated thread workspace and clears it
with `memset` at each force evaluation, so dynamic allocation is not inside the
timed force kernel.
`--accumulators 1|2|4|8` isolates the critical-path experiment requested in the
optimisation discussion; the default remains `4`, matching the production
kernel, while the wider sweep can identify the saturation point.

Singularity/Apptainer workflow:

```sh
singularity build nbody.sif Singularity.def
singularity run nbody.sif --help
RANKS="1 2 4" THREADS=1 REPEATS=5 bash ./run_benchmarks.sh container
python3 analyze.py summarize container container_overhead.csv container_overhead_summary.csv
```

On Orfeo, building a SIF directly from `Singularity.def` is not available to
normal users because fakeroot/proot support is missing. The reproducible path
used for the final results is therefore:

```sh
# Local workstation / WSL with Docker
docker build -t memid01/nbody-hpc:latest .
docker push memid01/nbody-hpc:latest

# Orfeo login node
module purge
module load singularity/4.3.1
singularity pull --force nbody.sif docker://memid01/nbody-hpc:latest
singularity test nbody.sif
```

On LEONARDO or another cluster, prefer the site-recommended host-MPI workflow
when available, and report the exact command and binding policy used for both
native and container runs.

Unified Slurm submission helper:

```sh
# Orfeo examples
bash jobs/submit.sh --cluster orfeo --bench probe
bash jobs/submit.sh --cluster orfeo --bench scaling --partition GENOA --cpus 64 --time 01:59:00 -- RANKS="1 2 4 8 16 32 64"
bash jobs/submit.sh --cluster orfeo --bench hybrid --partition GENOA --cpus 64
bash jobs/submit.sh --cluster orfeo --bench ablation --partition GENOA --cpus 64
bash jobs/submit.sh --cluster orfeo --bench evidence --partition GENOA --cpus 8
bash jobs/submit.sh --cluster orfeo --bench memory --partition GENOA --cpus 64
bash jobs/submit.sh --cluster orfeo --bench container --partition GENOA --cpus 4 -- IMAGE=nbody.sif
bash jobs/submit.sh --cluster orfeo --bench osu --partition GENOA --cpus 2 -- IMAGE=nbody.sif

# Leonardo examples, override account/qos when the active budget differs
bash jobs/submit.sh --cluster leonardo --bench scaling --cpus 64 --time 01:59:00
bash jobs/submit.sh --cluster leonardo --bench hybrid --cpus 64 --time 01:59:00
bash jobs/submit.sh --cluster leonardo --bench container --cpus 4 -- IMAGE=nbody.sif
```

`jobs/submit.sh` passes cluster-specific account, partition, QoS and module
settings to `sbatch`, then calls `run_benchmarks.sh` and `analyze.py` inside the
allocation. Extra benchmark parameters are passed after `--` as environment
assignments.

## Solver notes

The implemented time integrator is Drift-Kick-Drift:

1. drift positions by `dt/2`;
2. compute accelerations at the half-step positions;
3. kick velocities by `dt`;
4. drift positions by `dt/2` with the updated velocities.

The energy check uses the same softened potential as the force law:

```text
U = - sum_{i<j} G m^2 / sqrt(|r_i-r_j|^2 + eps^2)
```

The reported verification metric is

```text
abs(E(t) - E(0)) / max(abs(E(0)), dtype_min_normal)
```

A warning is printed if the maximum observed drift exceeds `--energy-tol` (default `1e-4`). This does not terminate the run, because large drift is often an intentional teaching signal: reduce `dt`, increase `eps`, or inspect the initial conditions.

## Intended optimisation path

The baseline is serial on purpose. Natural extensions are:

- **Pay attention to data qualifiers, like `const` and `restrict`, to help the compiler optimize the code**

- convert `compute_accelerations_naive` into an OpenMP loop without inner-loop
  atomics;
- compare Newton-third-law reuse against thread-private force buffers;
  when is it convenient, against the price of using atomics for a non-local write?
- split accumulators to shorten the floating-point dependency chain;
- compare scalar `sqrt` with an approximate reciprocal-square-root path and
  verify that energy conservation remains meaningful;
- preserve the SoA layout when adding MPI ring-shift communication;
- can you measure the achieved FLOP/s before and after each change.
- Instrument your code so that you can tie every section and assess their scalability separately, instead of just the total run-time
