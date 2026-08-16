#!/bin/bash
#SBATCH --job-name=evidence_leo
#SBATCH --account=uTS26_Tornator
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_bprod
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=leo_5_evidence_%j.out
#SBATCH --error=leo_5_evidence_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load profile/base
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load osu-micro-benchmarks 2>/dev/null || true

export OMP_PLACES=cores
export OMP_PROC_BIND=spread

make clean
make all

bash ./collect_system_info.sh system_info_leonardo.txt

THREADS="1 2 4 8" REPEATS=5 WARMUPS=2 N=50000 INNER_REPEATS=3 \
  bash ./benchmark_layout.sh
python3 analyze_layout.py layout_results.csv layout_summary.csv

RANKS=8 THREADS=1 REPEATS=5 WARMUPS=2 N=50000 NSTEPS=50 \
  ENERGY_LIST="1 5 10 50" bash ./benchmark_energy.sh
python3 analyze_energy.py energy_overhead.csv energy_overhead_summary.csv

if command -v osu_latency >/dev/null 2>&1 && command -v osu_bw >/dev/null 2>&1; then
  bash ./benchmark_osu.sh --mode native
else
  echo "OSU Micro-Benchmarks not found; load the cluster OSU module or set OSU_LATENCY/OSU_BW." >&2
fi
