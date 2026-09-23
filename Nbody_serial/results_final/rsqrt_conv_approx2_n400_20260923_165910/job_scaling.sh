#!/usr/bin/env bash
set -euo pipefail

cd /u/dssc/emmadariol/HPC_exam/Nbody_serial
module purge; module load openMPI/4.1.6; module load singularity/4.3.1 2>/dev/null || true
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
# Export benchmark parameters so child scripts such as run_benchmarks.sh and
# analyze.py receive the values passed after -- to jobs/submit.sh.
export RESULT_DIR=/u/dssc/emmadariol/HPC_exam/Nbody_serial/runs/rsqrt_conv_approx2_n400_20260923_165910 SCALING_KINDS=strong STRONG_N=10000 NSTEPS=400 DT=0.000025 RANKS=1 THREADS=8 COMM=sendrecv RSQRT=approx2 ACCUMULATORS=4 ENERGY_EVERY=1 EPS=0.05 REPEATS=3 WARMUPS=0 KEEP_ALL_REPETITIONS=1 USE_CONTAINER=0
OUT="${RESULT_DIR}/scaling.csv" bash ./run_benchmarks.sh scaling; python3 analyze.py summarize scaling "$RESULT_DIR/scaling.csv" "$RESULT_DIR/scaling_summary.csv"; python3 analyze.py plot scaling "$RESULT_DIR/scaling_summary.csv" "$RESULT_DIR/scaling"
