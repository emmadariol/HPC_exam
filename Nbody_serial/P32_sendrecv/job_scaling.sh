#!/usr/bin/env bash
set -euo pipefail

cd /u/dssc/emmadariol/HPC_exam/Nbody_serial
module purge; module load openMPI/4.1.6; module load singularity/4.3.1 2>/dev/null || true
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
# Export benchmark parameters so child scripts such as run_benchmarks.sh and
# analyze.py receive the values passed after -- to jobs/submit.sh.
export RESULT_DIR=/u/dssc/emmadariol/HPC_exam/Nbody_serial/runs/overlap_20260923_132533/P32_sendrecv SCALING_KINDS=strong STRONG_N=100000 NSTEPS=20 RANKS=32 THREADS=1 REPEATS=5 WARMUPS=1 COMM=sendrecv KERNEL=direct RSQRT=exact ACCUMULATORS=4 DT=0.0001 EPS=0.05 ENERGY_EVERY=20 KEEP_ALL_REPETITIONS=1 USE_CONTAINER=0
OUT="${RESULT_DIR}/scaling.csv" bash ./run_benchmarks.sh scaling; python3 analyze.py summarize scaling "$RESULT_DIR/scaling.csv" "$RESULT_DIR/scaling_summary.csv"; python3 analyze.py plot scaling "$RESULT_DIR/scaling_summary.csv" "$RESULT_DIR/scaling"
