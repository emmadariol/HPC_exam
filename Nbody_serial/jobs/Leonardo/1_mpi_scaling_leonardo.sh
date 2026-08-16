#!/bin/bash
#SBATCH --job-name=mpi_scale_leo
#SBATCH --account=uTS26_Tornator
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_bprod
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=leo_1_mpi_%j.out
#SBATCH --error=leo_1_mpi_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load profile/base
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=1

make clean
make all

bash ./collect_system_info.sh system_info_leonardo_dcgp.txt

RANKS="${RANKS:-1 2 4 8 16 32 64 112}" \
THREADS=1 \
REPEATS="${REPEATS:-5}" \
WARMUPS="${WARMUPS:-1}" \
STRONG_N="${STRONG_N:-100000}" \
WEAK_PER_RANK="${WEAK_PER_RANK:-10000}" \
NSTEPS="${NSTEPS:-50}" \
DT="${DT:-1e-4}" \
EPS="${EPS:-0.05}" \
ENERGY_EVERY="${ENERGY_EVERY:-10}" \
INTEGRATOR="${INTEGRATOR:-kdk}" \
COMM="${COMM:-overlap}" \
KERNEL=direct \
RSQRT="${RSQRT:-exact}" \
OUT="${OUT:-results_1_mpi_leonardo.csv}" \
  bash ./benchmark_scaling.sh

python3 analyze_benchmark.py "${OUT:-results_1_mpi_leonardo.csv}" results_1_mpi_leonardo_summary.csv
python3 plot_scaling.py results_1_mpi_leonardo_summary.csv results_1_mpi_leonardo
