#!/bin/bash
#SBATCH --job-name=build_sif_leo
#SBATCH --account=uTS26_Tornator
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_bprod
#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH --output=leo_0_build_container_%j.out
#SBATCH --error=leo_0_build_container_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load profile/base
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load apptainer

make clean
make all

runtime="apptainer"
command -v apptainer >/dev/null 2>&1 || runtime="singularity"

image="${IMAGE:-nbody.sif}"
"$runtime" build --force "$image" Singularity.def
"$runtime" test "$image"

echo "built and tested $image"
