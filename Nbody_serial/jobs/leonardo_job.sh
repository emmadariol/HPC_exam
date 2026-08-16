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

module purge
module load profile/base
# Rimossa la dipendenza CUDA, inutile per partizioni pure CPU
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.6--gcc--12.2.0 

export OMP_PLACES=cores
export OMP_PROC_BIND=spread 

echo "Collecting system info..."
./collect_system_info.sh > system_info_leonardo.txt

echo "Running benchmark sweep (Leonardo)"
# ATTENZIONE: Nello script benchmark_scaling.sh la chiamata a srun DEVE includere 
# l'opzione --cpus-per-task=$SRUN_CPUS_PER_TASK per associare correttamente i thread.

# 1. Test: 1 rank per dominio NUMA (4 NUMA/nodo = 16 rank totali su 4 nodi, 28 thread/rank)
export SRUN_CPUS_PER_TASK=28
RANKS=16 THREADS=28 REPEATS=5 STRONG_N=100000 WEAK_PER_RANK=10000 NSTEPS=50 ENERGY_EVERY=10 \
./benchmark_scaling.sh | tee leonardo_benchmark_numa.log

# 2. Test: 1 rank per Socket (2 Socket/nodo = 8 rank totali su 4 nodi, 56 thread/rank)
export SRUN_CPUS_PER_TASK=56
RANKS=8 THREADS=56 REPEATS=5 STRONG_N=100000 WEAK_PER_RANK=10000 NSTEPS=50 ENERGY_EVERY=10 \
./benchmark_scaling.sh | tee leonardo_benchmark_socket.log

# 3. Test: 1 rank per Core (112 Core/nodo = 448 rank totali su 4 nodi, 1 thread/rank)
export SRUN_CPUS_PER_TASK=1
RANKS=448 THREADS=1 REPEATS=5 STRONG_N=100000 WEAK_PER_RANK=10000 NSTEPS=50 ENERGY_EVERY=10 \
./benchmark_scaling.sh | tee leonardo_benchmark_core.log

# Unisco i log per l'analisi
cat leonardo_benchmark_*.log > leonardo_benchmark_total.log

echo "Analyzing and plotting (Leonardo)"
python3 analyze_benchmark.py benchmark_results.csv benchmark_summary_leonardo.csv
python3 plot_scaling.py benchmark_summary_leonardo.csv leonardo_scaling

echo "Packaging outputs"
tar -czf "leonardo_results_${SLURM_JOB_ID}.tgz" benchmark_results.csv benchmark_summary_leonardo.csv *.svg system_info_leonardo.txt leonardo_benchmark_total.log

echo "Done"