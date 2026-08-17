#!/bin/bash
#SBATCH --job-name=evidence_orfeo
#SBATCH --account=dssc
#SBATCH --partition=epyc
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=orfeo_5_evidence_%j.out
#SBATCH --error=orfeo_5_evidence_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load openMPI/4.1.6

export OMP_PLACES=cores
export OMP_PROC_BIND=spread

make clean
make all

bash ./collect_system_info.sh system_info_orfeo.txt

THREADS="1 2 4 8" REPEATS=5 WARMUPS=2 N=50000 INNER_REPEATS=3 \
  bash ./benchmark_layout.sh
python3 analyze_layout.py layout_results.csv layout_summary.csv

RANKS=8 THREADS=1 REPEATS=5 WARMUPS=2 N=50000 NSTEPS=50 \
  ENERGY_LIST="1 5 10 50" bash ./benchmark_energy.sh
python3 analyze_energy.py energy_overhead.csv energy_overhead_summary.csv

if command -v osu_latency >/dev/null 2>&1 && command -v osu_bw >/dev/null 2>&1; then
  bash ./benchmark_osu.sh --mode native
else
  echo "OSU Micro-Benchmarks not found; load an OSU module or set OSU_LATENCY/OSU_BW." >&2
fi
