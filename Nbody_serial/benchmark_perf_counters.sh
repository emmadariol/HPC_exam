#!/bin/bash
# Optional hardware-counter snapshot for the direct force kernel.
# Intended for clusters where perf_event permissions allow user measurements.
set -euo pipefail

INPUT="${INPUT:-perf_counter_input.bin}"
N="${N:-20000}"
NSTEPS="${NSTEPS:-20}"
RANKS="${RANKS:-1}"
THREADS="${THREADS:-1}"
OUT="${OUT:-perf_counters.txt}"
EVENTS="${EVENTS:-cycles,instructions,cache-references,cache-misses,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_double}"

if ! command -v perf >/dev/null 2>&1; then
  echo "perf is not available in PATH" >&2
  exit 127
fi

make all
./generate_ic --model 0 --n "$N" --seed 4242 --output "$INPUT" >/dev/null

set +e
OMP_NUM_THREADS="$THREADS" perf stat -e "$EVENTS" -o "$OUT" \
  mpirun -np "$RANKS" ./nbody_direct_hybrid \
  --input "$INPUT" --nsteps "$NSTEPS" --energy-every "$NSTEPS" \
  --integrator kdk --comm sendrecv --kernel direct --rsqrt exact --quiet
rc=$?
set -e

rm -f "$INPUT"
exit "$rc"
