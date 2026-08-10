#!/bin/bash
# Leonardo SLURM job template for production benchmarking and derivables
# Edit partition/account and node counts according to Leonardo's policy.
# Usage: sbatch leonardo_job.sh

#SBATCH --job-name=nbody-leonardo
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=standard
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --output=leonardo_job.%j.out

set -euo pipefail
cd $SLURM_SUBMIT_DIR

# Load site MPI module if available (adjust module name as needed)
MPI_MODULE="openmpi/4.1.6"
if type module >/dev/null 2>&1; then
	module load "${MPI_MODULE}"
fi

# Recommended production sweep for Leonardo (full derivables)
RANKS="1 2 4 8"
THREADS="1 2 4"
REPEATS=5
STRONG_N=4000
WEAK_PER_RANK=1000
NSTEPS=50
ENERGY_EVERY=10

export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "Compiling (production)..."
make clean && make -j

echo "Collecting system info..."
./collect_system_info.sh > system_info_leonardo.txt

echo "Running benchmark sweep (Leonardo)"
RANKS="$RANKS" THREADS="$THREADS" REPEATS=$REPEATS STRONG_N=$STRONG_N WEAK_PER_RANK=$WEAK_PER_RANK NSTEPS=$NSTEPS ENERGY_EVERY=$ENERGY_EVERY ./benchmark_scaling.sh | tee leonardo_benchmark.log

echo "Analyzing and plotting (Leonardo)"
python3 analyze_benchmark.py benchmark_results.csv benchmark_summary_leonardo.csv
python3 plot_scaling.py benchmark_summary_leonardo.csv leonardo_scaling

echo "Packaging outputs"
tar -czf leonardo_results_${SLURM_JOB_ID}.tgz benchmark_results.csv benchmark_summary_leonardo.csv *.svg system_info_leonardo.txt leonardo_benchmark.log

echo "Done"
