#!/bin/bash
#SBATCH --job-name=container_leo
#SBATCH --account=uTS26_Tornator
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_bprod
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=leo_4_container_%j.out
#SBATCH --error=leo_4_container_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load profile/base
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load apptainer

# Esplorazione completa fino a 112 rank per saturare il nodo DCGP
export RANKS="1 2 4 8 16 32 64 112"
export THREADS=1
export REPEATS=5
export N=50000
export N_PER_RANK=10000
export NSTEPS=50

./benchmark_container.sh
python3 analyze_container_overhead.py container_overhead.csv
python3 plot_container_overhead.py container_overhead_summary.csv
