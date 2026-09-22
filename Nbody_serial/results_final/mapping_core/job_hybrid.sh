#!/usr/bin/env bash
set -euo pipefail

cd /u/dssc/emmadariol/HPC_exam/Nbody_serial
module purge; module load openMPI/4.1.6; module load singularity/4.3.1 2>/dev/null || true
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
# Export benchmark parameters so child scripts such as run_benchmarks.sh and
# analyze.py receive the values passed after -- to jobs/submit.sh.
export RESULT_DIR=/u/dssc/emmadariol/HPC_exam/Nbody_serial/runs/mapping_core HYBRID_PAIRS=64x1 N=20000 NSTEPS=20 REPEATS=5 WARMUPS=1 VERIFY_MAPPING=core
RESULT_PREFIX="${RESULT_DIR}/hybrid" bash ./run_benchmarks.sh hybrid
