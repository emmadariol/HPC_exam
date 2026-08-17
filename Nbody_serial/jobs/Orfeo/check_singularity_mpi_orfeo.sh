#!/bin/bash
#SBATCH --job-name=check_mpi_sing
#SBATCH --account=dssc
#SBATCH --partition=EPYC
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=orfeo_check_singularity_mpi_%j.out
#SBATCH --error=orfeo_check_singularity_mpi_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load openMPI/4.1.6
module load singularity/4.3.1

IMAGE="${PWD}/nbody.sif"

make nbody_direct_hybrid >/dev/null

if [[ ! -f "$IMAGE" ]]; then
  echo "missing $IMAGE; run jobs/Orfeo/0_build_container_orfeo.sh first" >&2
  exit 1
fi

echo "=== MPI Linkage Check Native vs Singularity (on Orfeo EPYC node) ==="
echo "Image: $IMAGE"
echo

echo "1. Binary linkage OUTSIDE container (host binary):"
ldd ./nbody_direct_hybrid | grep -i mpi || echo "No libmpi direct link found"
echo

echo "2. Binary linkage INSIDE container (orchestrated by srun):"
srun --cpu-bind=verbose,cores -n 1 singularity exec "$IMAGE" bash -c 'ldd /opt/nbody/nbody_direct_hybrid | grep -i mpi || echo "No libmpi direct link found"'
echo

echo "3. Testing srun + singularity exec with -n 2 (multi-rank):"
srun --cpu-bind=verbose,cores -n 2 singularity exec "$IMAGE" /opt/nbody/nbody_direct_hybrid --help 2>&1 | head -5
echo

echo "=== Check Complete ==="
