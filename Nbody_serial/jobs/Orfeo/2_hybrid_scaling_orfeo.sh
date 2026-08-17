#!/bin/bash
#SBATCH --job-name=hybrid_scale_orfeo
#SBATCH --account=dssc
#SBATCH --partition=EPYC
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=orfeo_2_hybrid_%j.out
#SBATCH --error=orfeo_2_hybrid_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RESULT_DIR="${RESULT_DIR:-.}"
mkdir -p "$RESULT_DIR"
RESULT_PREFIX="${RESULT_PREFIX:-${RESULT_DIR}/results_2_hybrid_orfeo}"

module purge
module load openMPI/4.1.6

export OMP_PLACES=cores
export OMP_PROC_BIND=spread

make clean
make all

bash ./collect_system_info.sh "${RESULT_DIR}/system_info_orfeo_epyc.txt"

summary="${RESULT_PREFIX}_summary.csv"
rm -f "${RESULT_PREFIX}"_P*_T*.csv "${RESULT_PREFIX}"_P*_T*_summary.csv "$summary"

for pair in ${HYBRID_PAIRS:-128x1 64x2 32x4 16x8 8x16 4x32 2x64}; do
  P="${pair%x*}"
  T="${pair#*x}"
  export OMP_NUM_THREADS="$T"
  raw="${RESULT_PREFIX}_P${P}_T${T}.csv"
  partial="${RESULT_PREFIX}_P${P}_T${T}_summary.csv"

  RANKS="$P" \
  THREADS="$T" \
  SRUN_CPUS_PER_TASK="$T" \
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
  OUT="$raw" \
    bash ./benchmark_scaling.sh

  python3 analyze_benchmark.py "$raw" "$partial"
  if [[ ! -s "$summary" ]]; then
    cp "$partial" "$summary"
  else
    tail -n +2 "$partial" >> "$summary"
  fi
done

python3 plot_scaling.py "$summary" "$RESULT_PREFIX"
