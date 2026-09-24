#!/usr/bin/env bash
set -euo pipefail

cd /u/dssc/emmadariol/HPC_exam/Nbody_serial
module purge; module load openMPI/4.1.6; module load singularity/4.3.1 2>/dev/null || true
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
# Export benchmark parameters so child scripts such as run_benchmarks.sh and
# analyze.py receive the values passed after -- to jobs/submit.sh.
export RESULT_DIR=/u/dssc/emmadariol/HPC_exam/Nbody_serial/runs/arch_exact_20260923_170254 N=100000 NSTEPS=10 RANKS=32 THREADS=1 RSQRT=exact REPEATS=5 WARMUPS=1 KEEP_ALL_REPETITIONS=1 USE_CONTAINER=0 DT=0.0001 EPS=0.05
OUT="$RESULT_DIR/arch_target_comparison.csv" bash ./run_benchmarks.sh arch; python3 analyze.py summarize arch "$RESULT_DIR/arch_target_comparison.csv" "$RESULT_DIR/arch_target_comparison_summary.csv"; python3 analyze.py plot arch "$RESULT_DIR/arch_target_comparison_summary.csv" "$RESULT_DIR/arch_target_comparison"
