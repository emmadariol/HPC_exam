CC        ?= gcc
MPICC     ?= mpicc
STD       ?= -std=c11
CFLAGS    ?= -O3 -march=native -Wall -Wextra -Wpedantic
OMPFLAGS  ?= -fopenmp
LDLIBS    ?= -lm
PRECISION ?= double
PROGRAMS := nbody_direct_serial nbody_direct_hybrid nbody_layout_benchmark generate_ic
HEADERS  := nbody_common.h

ifeq ($(PRECISION),float)
PRECISION_CPPFLAGS := -DNBODY_USE_FLOAT
else ifeq ($(PRECISION),double)
PRECISION_CPPFLAGS := -DNBODY_USE_DOUBLE
else
$(error PRECISION must be either double or float)
endif

.PHONY: all clean run-smoke vec-report bench-scaling bench-hybrid bench-ablation bench-layout bench-energy bench-container bench-osu bench-arch bench-perf

all: $(PROGRAMS)  # build all programs

nbody_direct_serial: nbody_direct_serial.c $(HEADERS)  # serial solver
	$(CC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS)

nbody_direct_hybrid: nbody_direct_hybrid.c $(HEADERS)  # MPI + OpenMP solver
	$(MPICC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS)

nbody_layout_benchmark: nbody_layout_benchmark.c $(HEADERS)  # AoS vs SoA benchmark
	$(CC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS)

generate_ic: generate_ic.c $(HEADERS)  # initial conditions
	$(CC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

MPI_RUN ?= mpirun -np 2

run-smoke: all  # quick test on 128 particles
	./generate_ic --model 0 --n 128 --seed 42 --output plummer_128.bin
	./nbody_direct_serial --input plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --output plummer_128_final.bin --quiet
	OMP_NUM_THREADS=2 $(MPI_RUN) ./nbody_direct_hybrid --input plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --comm sendrecv --output plummer_128_hybrid_final.bin --quiet
	OMP_NUM_THREADS=2 $(MPI_RUN) ./nbody_direct_hybrid --input plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --comm overlap --quiet
	OMP_NUM_THREADS=2 ./nbody_layout_benchmark --input plummer_128.bin --layout both --warmups 1 --inner-repeats 1 >/dev/null

vec-report:  # compiler vectorisation reports
	mkdir -p results_final
	$(CC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -fopt-info-vec-optimized -fopt-info-vec-missed -c nbody_direct_serial.c -o /tmp/nbody_direct_serial_vec.o 2> results_final/vectorization_serial_report.txt
	$(MPICC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -fopt-info-vec-optimized -fopt-info-vec-missed -c nbody_direct_hybrid.c -o /tmp/nbody_direct_hybrid_vec.o 2> results_final/vectorization_report.txt
	rm -f /tmp/nbody_direct_serial_vec.o /tmp/nbody_direct_hybrid_vec.o

bench-scaling: all  # strong and weak scaling
	bash ./run_benchmarks.sh scaling
	python3 analyze.py summarize scaling benchmark_results.csv benchmark_summary.csv
	python3 analyze.py plot scaling benchmark_summary.csv scaling

bench-hybrid: all  # rank x thread mappings
	bash ./run_benchmarks.sh hybrid

bench-ablation: all  # optimisation tests
	bash ./run_benchmarks.sh ablation
	python3 analyze.py summarize ablation results_ablation.csv results_ablation_summary.csv
	python3 analyze.py plot ablation results_ablation.csv results_ablation

bench-layout: all  # AoS vs SoA
	bash ./run_benchmarks.sh layout
	python3 analyze.py summarize layout layout_results.csv layout_summary.csv
	python3 analyze.py plot layout layout_summary.csv layout_force_time

bench-energy: all  # energy check cost
	bash ./run_benchmarks.sh energy
	python3 analyze.py summarize energy energy_overhead.csv energy_overhead_summary.csv
	python3 analyze.py plot energy energy_overhead_summary.csv energy_overhead

bench-container: all  # native vs container
	bash ./run_benchmarks.sh container
	python3 analyze.py summarize container container_overhead.csv container_overhead_summary.csv
	python3 analyze.py plot container container_overhead_summary.csv container_overhead

bench-osu:  # OSU latency and bandwidth
	bash ./run_benchmarks.sh osu
	python3 analyze.py summarize osu osu_microbench.csv osu_microbench_summary.csv
	python3 analyze.py plot osu osu_microbench_summary.csv osu_microbench

bench-arch:  # native vs x86-64-v3
	bash ./run_benchmarks.sh arch
	python3 analyze.py summarize arch arch_target_comparison.csv arch_target_comparison_summary.csv
	python3 analyze.py plot arch arch_target_comparison_summary.csv arch_target_comparison

bench-perf: all  # hardware counters
	bash ./run_benchmarks.sh perf

clean:  # remove binaries and inputs
	rm -f $(PROGRAMS) *.o *.bin
