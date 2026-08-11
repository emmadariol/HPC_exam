#!/bin/bash
#SBATCH --job-name=container
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:00:00
#SBATCH --partition=EPYC
#SBATCH --output=orfeo_4_container_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load openmpi/4.1.6
module load singularity || module load apptainer || true

export RANKS="1 2 4 8 16 32"
export THREADS=1
export REPEATS=5
export N=50000
export NSTEPS=50

# Use the helper that compares native vs container (benchmark_container.sh)
./benchmark_container.sh
