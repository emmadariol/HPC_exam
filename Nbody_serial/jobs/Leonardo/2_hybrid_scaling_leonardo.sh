#!/bin/bash
#SBATCH --job-name=hybrid_scale_leo
#SBATCH --account=uTS26_Tornator
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_bprod
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=leo_2_hybrid_%j.out
#SBATCH --error=leo_2_hybrid_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load profile/base
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

export OMP_PLACES=cores
export OMP_PROC_BIND=spread

make clean
make all

bash ./collect_system_info.sh system_info_leonardo_dcgp.txt

summary="results_2_hybrid_leonardo_summary.csv"
rm -f results_2_hybrid_leonardo_P*_T*.csv results_2_hybrid_leonardo_P*_T*_summary.csv "$summary"

for pair in ${HYBRID_PAIRS:-112x1 56x2 28x4 16x7 14x8 8x14 4x28 2x56}; do
  P="${pair%x*}"
  T="${pair#*x}"
  export OMP_NUM_THREADS="$T"
  raw="results_2_hybrid_leonardo_P${P}_T${T}.csv"
  partial="results_2_hybrid_leonardo_P${P}_T${T}_summary.csv"

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

python3 plot_scaling.py "$summary" results_2_hybrid_leonardo
