#!/bin/bash
#SBATCH --job-name=mpi_scale_orfeo
#SBATCH --account=dssc
#SBATCH --partition=EPYC
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=orfeo_1_mpi_%j.out
#SBATCH --error=orfeo_1_mpi_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RESULT_DIR="${RESULT_DIR:-.}"
mkdir -p "$RESULT_DIR"
RESULT_PREFIX="${RESULT_PREFIX:-${RESULT_DIR}/results_1_mpi_orfeo}"

module purge
module load openMPI/4.1.6

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=1

make clean
make all

bash ./collect_system_info.sh "${RESULT_DIR}/system_info_orfeo_epyc.txt"

RANKS="${RANKS:-1 2 4 8 16 32 64 128}" \
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
OUT="${OUT:-${RESULT_PREFIX}.csv}" \
  bash ./benchmark_scaling.sh

python3 analyze_benchmark.py "${OUT:-${RESULT_PREFIX}.csv}" "${RESULT_PREFIX}_summary.csv"
python3 plot_scaling.py "${RESULT_PREFIX}_summary.csv" "$RESULT_PREFIX"
