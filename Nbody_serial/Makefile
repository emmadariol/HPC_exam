CC        ?= gcc
MPICC     ?= mpicc
STD       ?= -std=c11
CFLAGS    ?= -O3 -march=native -Wall -Wextra -Wpedantic
OMPFLAGS  ?= -fopenmp
LDLIBS    ?= -lm
PRECISION ?= double
PROGRAMS := nbody_direct_serial nbody_direct_hybrid nbody_layout_benchmark generate_ic memory_bandwidth
HEADERS  := nbody_common.h

# Compile-time precision is selected through preprocessor symbols consumed by
# nbody_common.h.  File storage remains float in both cases.
ifeq ($(PRECISION),float)
PRECISION_CPPFLAGS := -DNBODY_USE_FLOAT
else ifeq ($(PRECISION),double)
PRECISION_CPPFLAGS := -DNBODY_USE_DOUBLE
else
$(error PRECISION must be either double or float)
endif

.PHONY: all clean run-smoke vec-report bench-scaling bench-hybrid bench-ablation bench-layout bench-energy bench-memory bench-container bench-osu bench-arch bench-perf

all: $(PROGRAMS)

# The serial executable is also built with OpenMP flags because it contains
# optional SIMD pragmas guarded by _OPENMP; it remains a single-process solver.
nbody_direct_serial: nbody_direct_serial.c $(HEADERS)
	$(CC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS)

# The hybrid solver links with MPI and enables OpenMP inside each rank.
nbody_direct_hybrid: nbody_direct_hybrid.c $(HEADERS)
	$(MPICC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS)

# The layout benchmark is OpenMP-only: it compares memory layout effects without
# MPI communication noise.
nbody_layout_benchmark: nbody_layout_benchmark.c $(HEADERS)
	$(CC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS)

# Initial-condition generation is serial and deterministic.
generate_ic: generate_ic.c $(HEADERS)
	$(CC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

# Standalone STREAM-style OpenMP benchmark used to report RAM bandwidth.
memory_bandwidth: memory_bandwidth.c
	$(CC) $(STD) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS)

# MPI launcher used by the smoke test; override it on systems that require srun.
MPI_RUN ?= mpirun -np 2

# Fast correctness/build smoke test.  It is deliberately small enough to run on
# a login node or short allocation, before spending cluster queue time.
run-smoke: all
	./generate_ic --model 0 --n 128 --seed 42 --output plummer_128.bin
	./nbody_direct_serial --input plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --output plummer_128_final.bin --quiet
	OMP_NUM_THREADS=2 $(MPI_RUN) ./nbody_direct_hybrid --input plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --comm sendrecv --output plummer_128_hybrid_final.bin --quiet
	OMP_NUM_THREADS=2 $(MPI_RUN) ./nbody_direct_hybrid --input plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --comm overlap --quiet
	OMP_NUM_THREADS=2 ./nbody_layout_benchmark --input plummer_128.bin --layout both --warmups 1 --inner-repeats 1 >/dev/null

# Ask GCC/MPICC which loops were vectorised or missed; the reports are useful
# when discussing compiler assistance and remaining scalar bottlenecks.
vec-report:
	mkdir -p docs
	$(CC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -fopt-info-vec-optimized -fopt-info-vec-missed -c nbody_direct_serial.c -o /tmp/nbody_direct_serial_vec.o 2> docs/vectorization_serial_report.txt
	$(MPICC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -fopt-info-vec-optimized -fopt-info-vec-missed -c nbody_direct_hybrid.c -o /tmp/nbody_direct_hybrid_vec.o 2> docs/vectorization_report.txt
	rm -f /tmp/nbody_direct_serial_vec.o /tmp/nbody_direct_hybrid_vec.o

# Convenience targets wrap the unified benchmark orchestrator and the unified
# analysis CLI.  Cluster submission is handled separately by jobs/submit.sh.
bench-scaling: all
	bash ./run_benchmarks.sh scaling
	python3 analyze.py summarize scaling benchmark_results.csv benchmark_summary.csv
	python3 analyze.py plot scaling benchmark_summary.csv scaling

bench-hybrid: all
	bash ./run_benchmarks.sh hybrid

bench-ablation: all
	bash ./run_benchmarks.sh ablation
	python3 analyze.py summarize ablation results_ablation.csv results_ablation_summary.csv
	python3 analyze.py plot ablation results_ablation.csv results_ablation

bench-layout: all
	bash ./run_benchmarks.sh layout
	python3 analyze.py summarize layout layout_results.csv layout_summary.csv
	python3 analyze.py plot layout layout_summary.csv layout_force_time

bench-energy: all
	bash ./run_benchmarks.sh energy
	python3 analyze.py summarize energy energy_overhead.csv energy_overhead_summary.csv
	python3 analyze.py plot energy energy_overhead_summary.csv energy_overhead

bench-memory: memory_bandwidth
	bash ./run_benchmarks.sh memory
	python3 analyze.py summarize memory memory_bandwidth.csv memory_bandwidth_summary.csv
	python3 analyze.py plot memory memory_bandwidth_summary.csv memory_bandwidth

bench-container: all
	bash ./run_benchmarks.sh container
	python3 analyze.py summarize container container_overhead.csv container_overhead_summary.csv
	python3 analyze.py plot container container_overhead_summary.csv container_overhead

bench-osu:
	bash ./run_benchmarks.sh osu
	python3 analyze.py summarize osu osu_microbench.csv osu_microbench_summary.csv
	python3 analyze.py plot osu osu_microbench_summary.csv osu_microbench

bench-arch:
	bash ./run_benchmarks.sh arch
	python3 analyze.py summarize arch arch_target_comparison.csv arch_target_comparison_summary.csv
	python3 analyze.py plot arch arch_target_comparison_summary.csv arch_target_comparison

bench-perf: all
	bash ./run_benchmarks.sh perf

clean:
	rm -f $(PROGRAMS) *.o *.bin
