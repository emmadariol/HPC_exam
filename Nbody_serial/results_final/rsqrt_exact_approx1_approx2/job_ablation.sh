#!/usr/bin/env bash
set -euo pipefail

cd /u/dssc/emmadariol/HPC_exam/Nbody_serial
module purge; module load openMPI/4.1.6; module load singularity/4.3.1 2>/dev/null || true
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
# Export benchmark parameters so child scripts such as run_benchmarks.sh and
# analyze.py receive the values passed after -- to jobs/submit.sh.
export RESULT_DIR=/u/dssc/emmadariol/HPC_exam/Nbody_serial/runs/rsqrt_exact_approx1_approx2 N=10000 RANKS=1 THREADS=1 NSTEPS=5 REPEATS=5 WARMUPS=1
OUT="${RESULT_DIR}/ablation.csv" bash ./run_benchmarks.sh ablation; python3 analyze.py summarize ablation "$RESULT_DIR/ablation.csv" "$RESULT_DIR/ablation_summary.csv"; python3 analyze.py plot ablation "$RESULT_DIR/ablation.csv" "$RESULT_DIR/ablation"
