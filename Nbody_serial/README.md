# Serial C11 direct N-body baseline

This directory contains the stand-alone programs for the direct gravitational N-body exercise:

- `nbody_direct_serial.c`: serial softened direct solver using a DKD leapfrog
  step and a relative energy-drift verifier.
- `nbody_direct_hybrid.c`: MPI + OpenMP direct solver for Exercise 1. Each MPI
  rank owns one permanent SoA chunk, rotates source chunks with a ring-shift
  communication pattern, parallelises the home-particle force loop with OpenMP,
  and reports per-section timings plus a pair-interaction rate.
- `collect_system_info.sh`: writes the hardware/software stack requested in the
  report.
- `benchmark_scaling.sh`: runs repeated strong/weak scaling experiments and
  writes a CSV that can be plotted for speedup and efficiency.
- `analyze_benchmark.py`: reduces the raw benchmark CSV to medians, standard
  deviations, speedup, and efficiency.
- `Dockerfile` and `Singularity.def`: starter container recipes for the
  required native-vs-container comparison.
- `benchmark_container.sh`: repeated native-vs-Singularity timing table helper.
- `docker_build.sh`, `docker_smoke.sh`, `benchmark_docker.sh`: Docker build,
  correctness, and native-vs-Docker timing helpers.
- `analyze_container_overhead.py`: reduces Docker/Singularity overhead CSVs to
  medians, standard deviations, and overhead percentages.
- `REPORT_TEMPLATE.md`: checklist-style report skeleton matching the exam
  deliverables.
- `generate_ic.c`: initial-condition generator supporting two models:
  Plummer sphere (`--model 0`) and uniform ball + Maxwellian (`--model 1`).

The codes are intended as *almost complete* exam skeletons. The direct force kernel is deliberately correct but naive. It uses an O(N^2) all-pairs loop, scalar `sqrt`, one accumulator per component, and no Newton-third-law reuse. 
The comments in `compute_accelerations_naive` mark this as the kernel whose optimization is part of the assignment, along with the hybrid parallelization.

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

Generate a uniform ball with Maxwellian velocities. If `--sigma` is negative or omitted, the generator uses the uniform-sphere virial estimate
`sigma^2 = G M / (5 R)`.

```sh
./generate_ic --model 1 --n 1000 --seed 456 --radius 1.0 --mass 1.0 --output ball_1000.bin
./nbody_direct_serial --input ball_1000.bin --nsteps 100 --dt 1e-4 --eps 0.05 --mass 1.0 --energy-every 10
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
  --mass 1.0 --energy-every 10 --integrator kdk --comm sendrecv \
  --output final_state_hybrid.bin
```

`--integrator kdk` follows the wording of the exam text. `--integrator dkd`
keeps the original baseline-compatible Drift-Kick-Drift path available for
comparison. `--comm sendrecv` uses a simple blocking ring exchange, while
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
./collect_system_info.sh system_info.txt
RANKS="1 2 4" THREADS="1 2" REPEATS=5 STRONG_N=4000 \
  WEAK_PER_RANK=1000 NSTEPS=50 ENERGY_EVERY=10 \
  ./benchmark_scaling.sh
./analyze_benchmark.py benchmark_results.csv benchmark_summary.csv
./plot_scaling.py benchmark_summary.csv scaling
make vec-report
```

The generated `benchmark_results.csv` contains one line per run, including
status, energy drift, section timings, and kernel rate. Use the median (or
trimmed mean and standard deviation) across the repeated rows in the report.
`plot_scaling.py` writes SVG plots for strong/weak speedup and efficiency.

Optional kernel experiments:

```sh
OMP_NUM_THREADS=4 mpirun -np 1 ./nbody_direct_hybrid \
  --input plummer_1000.bin --nsteps 20 --dt 1e-4 --eps 0.05 \
  --integrator kdk --kernel newton --rsqrt exact --quiet

OMP_NUM_THREADS=4 mpirun -np 2 ./nbody_direct_hybrid \
  --input plummer_1000.bin --nsteps 20 --dt 1e-4 --eps 0.05 \
  --integrator kdk --kernel direct --rsqrt approx --quiet
```

`--kernel newton` is intentionally single-rank only: in a distributed ring,
Newton-third-law reuse also requires returning the opposite force contribution
to the remote owner rank. The direct MPI ring path is the main scalable solver.

Docker workflow:

```sh
./docker_build.sh
./docker_smoke.sh
RANKS="1 2 4" THREADS=1 REPEATS=5 N=1000 NSTEPS=20 ./benchmark_docker.sh
./analyze_container_overhead.py docker_overhead.csv docker_overhead_summary.csv
```

The Docker benchmark runs the native executable and then the same executable
inside the Docker image using the same input file mounted into `/data`. Open MPI
inside Docker runs as root, so the helper sets the standard
`OMPI_ALLOW_RUN_AS_ROOT` variables.

Singularity/Apptainer workflow:

```sh
singularity build nbody.sif Singularity.def
singularity run nbody.sif --help
RANKS="1 2 4" THREADS=1 REPEATS=5 ./benchmark_container.sh
./analyze_container_overhead.py container_overhead.csv container_overhead_summary.csv
```

On LEONARDO or another cluster, prefer the site-recommended host-MPI workflow
when available, and report the exact command and binding policy used for both
native and container runs.

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

A warning is printed if the maximum observed drift exceeds `--energy-tol` (default `1e-3`). This does not terminate the run, because large drift is often an intentional teaching signal: reduce `dt`, increase `eps`, or inspect the initial conditions.

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
