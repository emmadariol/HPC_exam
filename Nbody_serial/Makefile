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

.PHONY: all clean run-smoke vec-report

all: $(PROGRAMS)

nbody_direct_serial: nbody_direct_serial.c $(HEADERS)
	$(CC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

nbody_direct_hybrid: nbody_direct_hybrid.c $(HEADERS)
	$(MPICC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS)

nbody_layout_benchmark: nbody_layout_benchmark.c $(HEADERS)
	$(CC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS)

generate_ic: generate_ic.c $(HEADERS)
	$(CC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

# Variabile per il launcher MPI, sovrascrivibile da riga di comando
MPI_RUN ?= mpirun -np 2

run-smoke: all
	./generate_ic --model 0 --n 128 --seed 42 --output plummer_128.bin
	./nbody_direct_serial --input plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --output plummer_128_final.bin --quiet
	OMP_NUM_THREADS=2 $(MPI_RUN) ./nbody_direct_hybrid --input plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --integrator kdk --comm sendrecv --output plummer_128_hybrid_final.bin --quiet
	OMP_NUM_THREADS=2 $(MPI_RUN) ./nbody_direct_hybrid --input plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --integrator dkd --comm overlap --quiet
	OMP_NUM_THREADS=2 ./nbody_layout_benchmark --input plummer_128.bin --layout both --warmups 1 --inner-repeats 1 >/dev/null

vec-report:
	$(MPICC) $(STD) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -fopt-info-vec-optimized -fopt-info-vec-missed -c nbody_direct_hybrid.c -o /tmp/nbody_direct_hybrid_vec.o 2> vectorization_report.txt
	rm -f /tmp/nbody_direct_hybrid_vec.o

clean:
	rm -f $(PROGRAMS) *.o *.bin
