# Direct N-body Gravitational Simulation with MPI, OpenMP, and Docker

Course: High Performance Computing 1 / Introduction to Parallelism  
Exercise: Direct N-body gravitational simulation  
Implementation directory: `/home/emma/HPC/HPC_exam/Nbody_serial`

## 1. Objective

The goal of this project is to implement and evaluate a direct gravitational
N-body solver for equal-mass particles in three-dimensional space. The force
law uses Plummer softening,

```text
1 / sqrt(|r_i - r_j|^2 + eps^2)
```

and the implementation is intentionally a direct all-pairs code, not a tree,
PM, or fast multipole method. This makes the exercise suitable for studying
OpenMP loop parallelism, MPI ring communication, SIMD/vectorisation issues,
container overhead, and strong/weak scaling behavior.

The submitted code provides:

- a serial baseline solver, `nbody_direct_serial`;
- a hybrid MPI+OpenMP solver, `nbody_direct_hybrid`;
- KDK and DKD leapfrog integration modes;
- MPI ring-shift communication;
- blocking and non-blocking/overlap communication modes;
- direct and Newton-third-law kernel variants;
- exact and approximate reciprocal-square-root paths;
- timing instrumentation by section;
- correctness checks based on relative energy drift;
- scripts for strong/weak scaling, Docker/native comparison, CSV analysis,
  plotting, system identification, and vectorisation reporting.

## 2. Hardware Platform

The measurements in this report were collected under WSL2 on a laptop, not on
LEONARDO. The results are therefore useful for functional validation and local
scaling trends, but they should not be interpreted as cluster-grade performance
numbers.

### Where To Run What (Orfeo vs Leonardo)

To keep the workflow clean and reproducible:

- Run on **Orfeo**: build, smoke tests, short correctness runs, and quick option checks.
- Run on **Leonardo**: final strong/weak scaling campaigns, repeated measurements,
  container-overhead benchmarks, and production plots/tables for submission.

Operationally:

- Orfeo is the development and validation stage.
- Leonardo is the final performance-measurement stage.

Collected with `./collect_system_info.sh system_info.txt`.

| Item | Value |
|---|---|
| OS/kernel | Linux Emma `6.18.33.2-microsoft-standard-WSL2` |
| Architecture | x86_64 |
| CPU | 12th Gen Intel Core i7-1255U |
| Visible CPUs | 12 |
| Sockets | 1 |
| Cores per socket | 6 |
| Threads per core | 2 |
| NUMA nodes | 1 |
| L1d cache | 288 KiB total, 6 instances |
| L2 cache | 7.5 MiB total, 6 instances |
| L3 cache | 12 MiB |
| Memory visible to WSL | 7.6 GiB |
| Swap | 2.0 GiB |

`numactl -H` was not available in this WSL environment. The `lscpu` output
reports one NUMA node containing CPUs `0-11`.

## 3. Software Stack

| Component | Version / setting |
|---|---|
| C compiler | GCC `11.4.0` |
| MPI compiler wrapper | `mpicc`, backed by GCC `11.4.0` |
| MPI runtime | Open MPI `4.1.6` |
| OpenMP runtime | GCC `libgomp` via `-fopenmp` |
| Container runtime | Docker Desktop / Docker Engine from WSL |
| Base Docker image | `ubuntu:24.04` |

The Docker image installs:

```text
build-essential openmpi-bin libopenmpi-dev ca-certificates
```

## 4. Build Configuration

The default build uses double precision:

```bash
make clean
make
```

The effective compiler settings are:

```text
STD       = -std=c11
CFLAGS    = -O3 -march=native -Wall -Wextra -Wpedantic
OMPFLAGS  = -fopenmp
LDLIBS    = -lm
PRECISION = double
CPPFLAGS  = -DNBODY_USE_DOUBLE
```

The hybrid solver is compiled with:

```bash
mpicc -std=c11 -DNBODY_USE_DOUBLE -O3 -march=native \
  -Wall -Wextra -Wpedantic -fopenmp \
  -o nbody_direct_hybrid nbody_direct_hybrid.c -lm
```

The vectorisation report is generated with:

```bash
make vec-report
```

which writes `vectorization_report.txt`.

## 5. Code Structure

Important files:

| File | Purpose |
|---|---|
| `nbody_direct_serial.c` | Serial reference solver |
| `nbody_direct_hybrid.c` | MPI+OpenMP solver |
| `generate_ic.c` | Initial-condition generator |
| `nbody_common.h` | Common precision and binary-format definitions |
| `benchmark_scaling.sh` | Repeated strong/weak scaling runs |
| `analyze_benchmark.py` | Median, standard deviation, speedup, efficiency |
| `plot_scaling.py` | SVG scaling plots |
| `collect_system_info.sh` | Hardware/software inventory |
| `benchmark_docker.sh` | Native-vs-Docker repeated timings |
| `analyze_container_overhead.py` | Container overhead summary |
| `Dockerfile` | Docker image definition |
| `Singularity.def` | Singularity/Apptainer recipe |

The binary input/output format stores six single-precision floats per particle:

```text
x y z vx vy vz
```

The solver performs all arithmetic in the selected `dtype`, which defaults to
double precision.

## 6. Numerical Algorithm

The code solves the direct softened gravitational problem:

```text
a_i = sum_{j != i} G m (r_j - r_i) / (|r_j-r_i|^2 + eps^2)^(3/2)
```

The main submitted integrator is KDK, selected by:

```bash
--integrator kdk
```

The original DKD path is kept for baseline comparison:

```bash
--integrator dkd
```

The KDK step is:

1. compute acceleration at current positions;
2. kick velocities by `dt/2`;
3. drift positions by `dt`;
4. recompute accelerations;
5. kick velocities by `dt/2`.

The DKD step is:

1. drift positions by `dt/2`;
2. compute accelerations;
3. kick velocities by `dt`;
4. drift positions by `dt/2`.

Both are second-order leapfrog forms. KDK is used for the main results because
it follows the project wording.

## 7. Parallel Design

### MPI Ring Shift

Each MPI rank permanently owns a contiguous home chunk of particles. At each
force evaluation, every rank rotates a buffer chunk around the MPI ring. During
each ring stage, the rank accumulates interactions between its home particles
and the current source buffer.

This preserves the SoA layout and avoids gathering all particles on every rank
as a permanent data structure.

Two communication modes are implemented:

```bash
--comm sendrecv
--comm overlap
```

`sendrecv` uses blocking `MPI_Sendrecv`. `overlap` posts `MPI_Irecv` and
`MPI_Isend` for the next chunk, computes the current chunk, then waits for the
exchange to complete. This exposes potential communication/computation overlap,
although on the local WSL laptop the problem sizes are too small for strong
conclusions about network behavior.

### OpenMP

OpenMP parallelises the home-particle loop. Each thread owns a subset of target
particles and accumulates local `ax`, `ay`, and `az` values. The main distributed
kernel avoids inner-loop atomics.

The benchmarked configurations used:

```text
OMP_NUM_THREADS = 1 or 2
MPI ranks       = 1, 2, or 4
```

No explicit `OMP_PLACES` or `OMP_PROC_BIND` was set during the local WSL runs.
For cluster runs these should be set and reported explicitly, for example:

```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

## 8. Kernel Variants

### Direct Kernel

The main scalable kernel is:

```bash
--kernel direct
```

It computes all target-source interactions directly and works for all MPI rank
counts.

### Newton Third Law Kernel

The code also includes:

```bash
--kernel newton
```

This uses Newton's third law within one rank and stores thread-private force
buffers to avoid OpenMP atomics in the inner loop. It is intentionally limited
to `mpirun -np 1`. In a distributed ring, Newton reuse would also require
sending the opposite force contribution back to the remote owner rank. That is
a different communication pattern and was not used as the main scalable solver.

### Reciprocal Square Root

Two inverse-square-root modes are implemented:

```bash
--rsqrt exact
--rsqrt approx
```

`exact` uses the `dtype_sqrt` path. `approx` uses a lower-precision reciprocal
square-root seed plus Newton refinement. The approximate mode is included for
performance/accuracy discussion and must always be checked against energy
drift.

## 9. Correctness Verification

The quantitative correctness metric is the maximum relative drift of the total
mechanical energy:

```text
abs(E(t) - E(0)) / max(abs(E(0)), dtype_min_normal)
```

The default tolerance is:

```text
1e-3
```

Across all strong/weak scaling runs:

```text
maximum relative energy drift = 2.546540785007945e-07
failed runs                   = 0
```

Across all Docker/native container-overhead runs:

```text
maximum relative energy drift = 9.248936342456412e-08
failed runs                   = 0
```

All reported benchmark rows therefore satisfy the correctness criterion.

## 10. Benchmark Methodology

The main scaling benchmark was run with:

```bash
RANKS="1 2 4" THREADS="1 2" REPEATS=5 STRONG_N=4000 \
  WEAK_PER_RANK=1000 NSTEPS=50 ENERGY_EVERY=10 \
  INTEGRATOR=kdk COMM=overlap KERNEL=direct RSQRT=exact \
  ./benchmark_scaling.sh
```

Then:

```bash
./analyze_benchmark.py benchmark_results.csv benchmark_summary.csv
./plot_scaling.py benchmark_summary.csv scaling
```

Each configuration was repeated five times. The tables below report median
wall time and sample standard deviation over those five repetitions.

Generated plots:

```text
scaling_strong_speedup.svg
scaling_strong_efficiency.svg
scaling_weak_speedup.svg
scaling_weak_efficiency.svg
```

## 11. Strong Scaling Results

Strong scaling fixed the total particle count at `N = 4000` and varied the
number of MPI ranks and OpenMP threads.

| Ranks | Threads/rank | Resources | Median total time (s) | Std. dev. (s) | Median force time (s) | Gpair/s | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 2.442623 | 0.070958 | 2.299850 | 0.354717 | 1.000 | 1.000 |
| 1 | 2 | 2 | 1.436501 | 0.045448 | 1.330727 | 0.613046 | 1.700 | 0.850 |
| 2 | 1 | 2 | 1.416991 | 0.044286 | 1.304706 | 0.625272 | 1.724 | 0.862 |
| 2 | 2 | 4 | 1.455799 | 0.037000 | 1.340327 | 0.608654 | 1.678 | 0.419 |
| 4 | 1 | 4 | 1.311381 | 0.025392 | 1.182102 | 0.690123 | 1.863 | 0.466 |
| 4 | 2 | 8 | 0.868922 | 0.045352 | 0.768473 | 1.061581 | 2.811 | 0.351 |

### Strong Scaling Discussion

The best measured local configuration was `4` MPI ranks with `2` OpenMP threads
per rank, reaching a median time of `0.868922 s` and a speedup of `2.81x`
relative to the single-rank/single-thread baseline.

The two-resource configurations scale well:

- `1 rank x 2 threads`: efficiency `0.850`;
- `2 ranks x 1 thread`: efficiency `0.862`.

At four and eight total resources, efficiency drops substantially. This is
expected on the WSL laptop for several reasons:

- the machine is a mobile CPU with heterogeneous frequency behavior and shared
  thermal/power limits;
- WSL adds virtualization overhead;
- `N=4000` is still small enough that MPI and OpenMP overheads are visible;
- all ranks are on the same socket/NUMA domain, so this is not a true cluster
  scaling experiment;
- the energy diagnostic, although not dominant, remains visible in the total
  time.

The force section dominates the execution time in all strong-scaling runs. For
example, the single-resource run spends `2.299850 s` of `2.442623 s` in the
force kernel, about `94%` of total runtime. This confirms that the benchmark is
mostly measuring the intended O(N^2) kernel.

## 12. Weak Scaling Results

Weak scaling used `1000` particles per rank:

```text
N = ranks * 1000
```

| N | Ranks | Threads/rank | Resources | Median total time (s) | Std. dev. (s) | Median force time (s) | Gpair/s | Relative speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 1 | 1 | 1 | 0.144845 | 0.003179 | 0.135905 | 0.374888 | 1.000 | 1.000 |
| 1000 | 1 | 2 | 2 | 0.091659 | 0.017861 | 0.084157 | 0.605403 | 1.580 | 0.790 |
| 2000 | 2 | 1 | 2 | 0.354558 | 0.017793 | 0.327074 | 0.623401 | 0.409 | 0.204 |
| 2000 | 2 | 2 | 4 | 0.369543 | 0.017898 | 0.342455 | 0.595401 | 0.392 | 0.098 |
| 4000 | 4 | 1 | 4 | 1.330804 | 0.070928 | 1.220753 | 0.668273 | 0.109 | 0.027 |
| 4000 | 4 | 2 | 8 | 0.986230 | 0.088811 | 0.863777 | 0.944452 | 0.147 | 0.018 |

### Weak Scaling Discussion

For direct N-body, weak scaling is subtle. If the number of particles per rank
is kept fixed, each rank still interacts its local particles with all global
particles. Therefore the per-rank work grows approximately linearly with the
number of ranks:

```text
work per rank ~ N_local * N_global
```

So, unlike nearest-neighbor stencil codes, constant time is not expected when
`N_global` grows with the number of ranks. The data reflects this: moving from
`N=1000` to `N=4000` significantly increases time.

The `4 ranks x 2 threads` weak configuration reaches the highest median kernel
rate in the weak table, `0.944452 Gpair/s`, but total time still increases
because the all-pairs work grows with global problem size.

This is a useful oral-exam point: direct summation exposes a clean compute-bound
kernel, but it is algorithmically O(N^2). Better weak-scaling behavior would
require changing the algorithm, for example to a tree method, particle-mesh
method, or fast multipole method, which the assignment explicitly excludes.

## 13. Docker Container Results

Docker was built and tested successfully. The smoke test inside Docker passed:

```text
status=OK
max_relative_energy_drift=1.2177566861936688e-08
```

Docker/native benchmark command:

```bash
sudo env RANKS="1 2 4" THREADS=1 REPEATS=5 N=1000 NSTEPS=20 \
  ./benchmark_docker.sh

./analyze_container_overhead.py docker_overhead.csv docker_overhead_summary.csv
```

| N | Ranks | Threads/rank | Native median (s) | Docker median (s) | Native std. dev. (s) | Docker std. dev. (s) | Overhead |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 1 | 1 | 0.061218 | 0.061774 | 0.002636 | 0.002378 | 0.91% |
| 1000 | 2 | 1 | 0.039516 | 0.047260 | 0.013883 | 0.005052 | 19.60% |
| 1000 | 4 | 1 | 0.041608 | 0.046650 | 0.001026 | 0.011859 | 12.12% |

### Container Discussion

Docker overhead is small for the single-rank case, below `1%`. For two and four
ranks it rises to roughly `20%` and `12%`, respectively. These numbers are from
WSL2 and Docker Desktop, so they include virtualization and Docker/WSL I/O and
process-launch effects, not only container overhead in the strict HPC sense.

For an HPC system such as LEONARDO, Docker is usually not the runtime used on
compute nodes. The project also includes `Singularity.def` for
Singularity/Apptainer. The Dockerfile remains useful as a portable build recipe
and for local reproducibility, while Singularity/Apptainer is the more typical
cluster execution mechanism.

## 14. Vectorisation Report

The vectorisation report was generated with:

```bash
make vec-report
```

The report shows many missed vectorisation opportunities in the force loops.
The main reasons include:

- loop control flow;
- calls to `sqrt`/`sqrtf`;
- OpenMP parallel regions and reductions;
- MPI calls and memory-clobbering library calls;
- lack of a convenient vector type in some loops;
- control flow caused by excluding self-interactions and by optional kernel
  paths.

Representative messages include:

```text
not vectorized: control flow in loop
statement clobbers memory: sqrt(...)
statement clobbers memory: MPI_Sendrecv(...)
```

This is consistent with the expected difficulty of vectorising a transparent
direct N-body kernel written in scalar C with runtime-selectable modes. The SoA
layout and split accumulators help the compiler, but the force kernel still
contains scalar square-root operations and branches. Possible future work would
be to separate specialised kernels into compile-time variants and remove
runtime conditionals from the innermost loops.

## 15. Bottleneck Analysis

The solver prints maximum-rank timings:

```text
# timing_max_seconds total=... io=... drift=... force=... kick=... energy=...
```

In the main strong-scaling baseline:

```text
total = 2.442623 s
force = 2.299850 s
```

The force kernel accounts for roughly:

```text
2.299850 / 2.442623 ~= 94.2%
```

Thus the primary bottleneck is the direct all-pairs force computation. Drift and
kick are negligible because they are O(N), while the force step is O(N^2). The
energy diagnostic is also O(N^2), and it can become visible when called too
often. For production benchmarks the diagnostic was therefore run every ten
steps:

```bash
--energy-every 10
```

This still verifies correctness while reducing diagnostic overhead.

Communication overhead is present in the MPI ring, but on this local WSL run it
is difficult to separate cleanly from process scheduling and virtualization
effects. The `--comm overlap` mode is implemented to expose the comparison, but
larger problem sizes and a real cluster network would be needed for a stronger
conclusion.

## 16. Optimisation Choices

### SoA Layout

The solver stores positions, velocities, and accelerations in separate arrays:

```text
x[], y[], z[], vx[], vy[], vz[], ax[], ay[], az[]
```

This avoids the cache and vectorisation disadvantages of an AoS particle
structure when the kernel streams over only coordinate arrays.

### Accumulator Splitting

The direct kernel uses split accumulators (`ax0/ax1`, etc.) to reduce the
floating-point dependency chain inside the inner loop. This is intended to help
instruction-level parallelism and FMA throughput.

### Newton Third Law

The Newton variant reduces arithmetic in the single-rank case by reusing
`F_ij = -F_ji`. To avoid atomics, it uses thread-private buffers and reduces
them at the end. This is deliberately not the main distributed kernel because
Newton reuse across MPI ranks requires sending the opposite force contribution
back to the remote owner rank.

### Reciprocal Square Root

The exact path is used for the main results. The approximate path is available
for comparison and must be judged by both speed and energy drift. The collected
main results used:

```text
--rsqrt exact
```

### Communication Overlap

The main scaling runs used:

```text
--comm overlap
```

This posts non-blocking receives/sends before computing the current source
chunk. It is not guaranteed to improve performance for small local runs, but it
implements the required experiment and provides a basis for measurement on a
cluster.

## 17. Reproducibility Commands

Build:

```bash
cd /home/emma/HPC/HPC_exam/Nbody_serial
make clean
make
```

Smoke test:

```bash
make run-smoke
```

System information:

```bash
./collect_system_info.sh system_info.txt
```

Scaling:

```bash
RANKS="1 2 4" THREADS="1 2" REPEATS=5 STRONG_N=4000 \
  WEAK_PER_RANK=1000 NSTEPS=50 ENERGY_EVERY=10 \
  INTEGRATOR=kdk COMM=overlap KERNEL=direct RSQRT=exact \
  ./benchmark_scaling.sh

./analyze_benchmark.py benchmark_results.csv benchmark_summary.csv
./plot_scaling.py benchmark_summary.csv scaling
```

Vectorisation:

```bash
make vec-report
```

Docker:

```bash
sudo ./docker_build.sh
sudo ./docker_smoke.sh

sudo env RANKS="1 2 4" THREADS=1 REPEATS=5 N=1000 NSTEPS=20 \
  ./benchmark_docker.sh

./analyze_container_overhead.py docker_overhead.csv docker_overhead_summary.csv
```

## 18. Conclusions

The implemented project satisfies the core requirements of the direct N-body
assignment:

- direct softened gravitational interaction;
- KDK leapfrog integration;
- quantitative energy-conservation verification;
- MPI ring-shift communication;
- OpenMP force-loop parallelisation without inner-loop atomics in the main
  direct kernel;
- SoA memory layout;
- section-level instrumentation;
- strong and weak scaling scripts with five repetitions;
- Docker container recipe and native-vs-container overhead measurements;
- vectorisation report and optimisation discussion.

The measured results show that the force kernel is the dominant cost, as
expected for a direct O(N^2) algorithm. Strong scaling is useful up to the local
resources tested, but efficiency drops as overheads and the limitations of the
WSL laptop environment become visible. Weak scaling is limited by the algorithm:
keeping particles per rank fixed still increases the number of remote source
particles each rank must process.

The Docker overhead is small in the single-rank case and moderate for multiple
ranks under WSL2/Docker Desktop. These measurements should be repeated on the
target HPC system with Singularity/Apptainer or the course-recommended
container runtime if official cluster results are required.

