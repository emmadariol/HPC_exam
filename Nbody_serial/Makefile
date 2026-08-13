CC        ?= cc
MPICC     ?= mpicc
STD       ?= -std=c11
CPPFLAGS  ?=
CFLAGS    ?= -O3 -march=native -Wall -Wextra -Wpedantic
OMPFLAGS  ?= -fopenmp
LDLIBS    ?= -lm
PRECISION ?= double

PROGRAMS := nbody_direct_serial nbody_direct_hybrid generate_ic
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
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

nbody_direct_hybrid: nbody_direct_hybrid.c $(HEADERS)
	$(MPICC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS)

generate_ic: generate_ic.c $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)


run-smoke: all
	./generate_ic --model 0 --n 128 --seed 42 --output plummer_128.bin
	./nbody_direct_serial --input plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --output plummer_128_final.bin --quiet
	OMP_NUM_THREADS=2 srun -n 2 ./nbody_direct_hybrid --input plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --integrator kdk --comm sendrecv --output plummer_128_hybrid_final.bin --quiet
	OMP_NUM_THREADS=2 srun -n 2 ./nbody_direct_hybrid --input plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --integrator dkd --comm overlap --quiet
	./generate_ic --model 1 --n 128 --seed 43 --output ball_128.bin
	./nbody_direct_serial --input ball_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --output ball_128_final.bin --quiet
	OMP_NUM_THREADS=2 srun -n 2 ./nbody_direct_hybrid --input ball_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --integrator kdk --comm overlap --quiet
	OMP_NUM_THREADS=2 srun -n 1 ./nbody_direct_hybrid --input ball_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --integrator kdk --kernel newton --rsqrt approx --quiet
vec-report:
	$(MPICC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -fopt-info-vec-optimized -fopt-info-vec-missed -c nbody_direct_hybrid.c -o /tmp/nbody_direct_hybrid_vec.o 2> vectorization_report.txt
	rm -f /tmp/nbody_direct_hybrid_vec.o

clean:
	rm -f $(PROGRAMS) *.o *.bin
