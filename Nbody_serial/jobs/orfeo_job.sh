#!/bin/bash
# Orfeo SLURM job template for quick development and validation
# Edit partition/account as needed for your Orfeo setup.
# Requests: small node count, short walltime for quick iteration.
# Usage: sbatch orfeo_job.sh

#SBATCH --job-name=nbody-orfeo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --output=orfeo_job.%j.out

set -euo pipefail
cd $SLURM_SUBMIT_DIR

# Load site MPI module if available (adjust module name as needed)
MPI_MODULE="openmpi/4.1.6"
if type module >/dev/null 2>&1; then
	module load "${MPI_MODULE}"
fi

# Recommended parameter sweep for Orfeo (fast, debug-focused)
RANKS="1 2"
THREADS="1 2"
REPEATS=3
STRONG_N=1000
NSTEPS=20
ENERGY_EVERY=5

export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "Compiling..."
make clean && make -j

echo "Collecting system info..."
./collect_system_info.sh > system_info_orfeo.txt

echo "Running benchmark sweep (Orfeo)"
RANKS="$RANKS" THREADS="$THREADS" REPEATS=$REPEATS STRONG_N=$STRONG_N NSTEPS=$NSTEPS ENERGY_EVERY=$ENERGY_EVERY ./benchmark_scaling.sh | tee orfeo_benchmark.log

echo "Analyzing and plotting"
python3 analyze_benchmark.py benchmark_results.csv benchmark_summary_orfeo.csv
python3 plot_scaling.py benchmark_summary_orfeo.csv orfeo_scaling

echo "Packaging outputs"
tar -czf orfeo_results_${SLURM_JOB_ID}.tgz benchmark_results.csv benchmark_summary_orfeo.csv *.svg system_info_orfeo.txt orfeo_benchmark.log

echo "Done"
