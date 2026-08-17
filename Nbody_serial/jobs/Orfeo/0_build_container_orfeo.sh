#!/bin/bash
#SBATCH --job-name=build_sif_orfeo
#SBATCH --account=dssc
#SBATCH --partition=epyc
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH --output=orfeo_0_build_container_%j.out
#SBATCH --error=orfeo_0_build_container_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load openMPI/4.1.6
module load singularity/4.3.1

make clean
make all

image="${IMAGE:-nbody.sif}"
singularity build --force "$image" Singularity.def
singularity test "$image"

echo "built and tested $image"
