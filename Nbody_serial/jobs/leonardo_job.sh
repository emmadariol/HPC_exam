#!/bin/bash
#SBATCH --job-name=nbody-leonardo
#SBATCH --account=uTS26_Tornator
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_bprod
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=leonardo_job.%j.out
#SBATCH --error=leonardo_job.%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Pulizia e caricamento rigoroso dei moduli host
module purge
module load profile/base
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load python/3.11.6--gcc--12.2.0  # Assicurati che questo modulo sia disponibile per l'analisi

# Parametri corretti e allineati alle specifiche dell'Esercizio 1
RANKS="1 4 16 64 112 224 448" # Esplorazione che satura i 4 nodi (112 core/nodo)
THREADS="1 2 4"
REPEATS=5
STRONG_N=100000        # Dimensione adeguata per non farsi schiacciare dall'overhead
WEAK_PER_RANK=10000    # Dimensione adeguata al chunk locale
NSTEPS=50
ENERGY_EVERY=10

export OMP_PLACES=cores
# 'spread' è fisicamente superiore a 'close' per sfruttare la banda di memoria passante 
# nei codici compute-bound prima di saturare i core.
export OMP_PROC_BIND=spread 

echo "Collecting system info..."
./collect_system_info.sh > system_info_leonardo.txt

echo "Running benchmark sweep (Leonardo)"
RANKS="$RANKS" THREADS="$THREADS" REPEATS="$REPEATS" \
STRONG_N="$STRONG_N" WEAK_PER_RANK="$WEAK_PER_RANK" \
NSTEPS="$NSTEPS" ENERGY_EVERY="$ENERGY_EVERY" \
./benchmark_scaling.sh | tee leonardo_benchmark.log

echo "Analyzing and plotting (Leonardo)"
python3 analyze_benchmark.py benchmark_results.csv benchmark_summary_leonardo.csv
python3 plot_scaling.py benchmark_summary_leonardo.csv leonardo_scaling

echo "Packaging outputs"
tar -czf "leonardo_results_${SLURM_JOB_ID}.tgz" benchmark_results.csv benchmark_summary_leonardo.csv *.svg system_info_leonardo.txt leonardo_benchmark.log

echo "Done"