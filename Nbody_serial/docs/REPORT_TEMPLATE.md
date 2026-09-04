# Direct N-body MPI+OpenMP Report Template

## 1. Hardware

- CPU model:
- Sockets:
- Cores per socket:
- SMT on/off:
- NUMA layout: attach `numactl -H` or `collect_system_info.sh` output.
- Memory:

## 2. Software Stack

- Compiler:
- MPI implementation:
- OpenMP runtime:
- OS/kernel:
- Container runtime, if used:

## 3. Build And Run Configuration

- Compiler flags:
- Precision:
- Integrator: `kdk` for the main results; optional `dkd` comparison.
- Communication mode: `sendrecv` and/or `overlap`.
- Kernel: `direct`; optional single-rank `newton` comparison.
- Inverse square root: `exact`; optional `approx` comparison.
- MPI ranks:
- OpenMP threads:
- Binding: `OMP_PLACES`, `OMP_PROC_BIND`, MPI binding options.
- Warmups and repetitions: `WARMUPS`, `REPEATS`, outlier rule.

## 4. Correctness

Report the maximum relative energy drift:

```text
max_relative_energy_drift = ...
tolerance = 1e-3
status = OK/WARNING
```

Use the same initial condition, time step, softening length, and number of
steps when comparing variants.

## 5. Strong Scaling

- Fixed total N:
- Repetitions per point:
- Statistic: median and standard deviation or trimmed mean.
- Warmup/outliers: state how many warmup runs were discarded and how many
  measured runs were flagged as outliers in `benchmark_summary.csv`.
- Include speedup and efficiency plots.
- Discuss where the force kernel stops scaling and whether diagnostics become
  visible in the timing.

## 6. Weak Scaling

- Fixed particles per rank:
- Repetitions per point:
- Include weak efficiency plot.
- Discuss communication growth in the ring pattern and the observed departure
  from ideal weak scaling.

## 7. Optimisation Discussion

- SoA layout: explain why the solver stores separate arrays.
- AoS vs SoA measurement: cite `layout_summary.csv`.
- Direct kernel: explain why it avoids inner-loop atomics.
- Newton third law: compare the single-rank `--kernel newton` variant with
  `--kernel direct`, then explain why distributed Newton reuse needs force
  contributions to be returned to remote owners.
- `rsqrt`: compare `--rsqrt exact` and `--rsqrt approx`; state the energy drift.
- Communication overlap: compare `--comm sendrecv` and `--comm overlap`.
- Communication wait: use the `comm_wait_median` and
  `comm_bandwidth_GBps` columns from `benchmark_summary.csv`.
- Energy diagnostic overhead: cite `energy_overhead_summary.csv`.
- MPI microbenchmarks: cite `osu_microbench.csv` latency/bandwidth results.
- Vectorisation: attach or summarise `make vec-report`.

## 8. Container Overhead

Run the same case natively and through Singularity/Apptainer for at
least three process/thread configurations. Report median time, standard
deviation, and overhead percentage:

```text
overhead_percent = 100 * (container_time - native_time) / native_time
```

Singularity/Apptainer helper:

```sh
RANKS="1 2 4" THREADS=1 REPEATS=5 N=1000 NSTEPS=20 bash ./run_benchmarks.sh container
python3 analyze.py summarize container container_overhead.csv container_overhead_summary.csv
```

Additional evidence helpers:

```sh
THREADS="1 2 4" REPEATS=5 WARMUPS=2 N=50000 bash ./run_benchmarks.sh layout
python3 analyze.py summarize layout layout_results.csv layout_summary.csv
RANKS=8 THREADS=1 REPEATS=5 WARMUPS=2 N=50000 NSTEPS=50 bash ./run_benchmarks.sh energy
python3 analyze.py summarize energy energy_overhead.csv energy_overhead_summary.csv
bash ./run_benchmarks.sh osu --mode native
bash ./run_benchmarks.sh osu --mode both --image nbody.sif --out osu_microbench_container.csv
```

## 9. Bottlenecks

Use the printed section timings:

```text
# timing_max_seconds total=... io=... drift=... force=... comm_wait=... kick=... energy=...
```

Explain whether the bottleneck is force computation, energy diagnostics,
communication, I/O, or a mixture.
