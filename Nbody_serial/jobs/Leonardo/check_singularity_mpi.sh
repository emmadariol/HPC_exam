#!/bin/bash
#SBATCH --job-name=check_mpi_sing
#SBATCH --account=uts26_tornator_0
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_bprod
#SBATCH --gres=tmpfs:10g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=check_singularity_mpi_%j.out
#SBATCH --error=check_singularity_mpi_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Pulizia e caricamento del modulo esatto disponibile su Leonardo
module purge
module load profile/base
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load apptainer 2>/dev/null || true

IMAGE="${PWD}/nbody.sif"
runtime="apptainer"
command -v apptainer >/dev/null 2>&1 || runtime="singularity"

make nbody_direct_hybrid >/dev/null

if [[ ! -f "$IMAGE" ]]; then
  echo "missing $IMAGE; run jobs/Leonardo/0_build_container_leonardo.sh first" >&2
  exit 1
fi

echo "=== MPI Linkage Check Native vs Singularity (on DCGP Compute Node) ==="
echo "Image: $IMAGE"
echo ""

# Confronta esplicitamente il linker host con quello visto nel container.
# Entrambi devono risolvere le MPI del modulo Leonardo, non quelle apt dell'immagine.
echo "1. Binary linkage OUTSIDE container (host binary):"
ldd ./nbody_direct_hybrid | grep -i mpi || echo "No libmpi direct link found"
echo ""
echo "2. Binary linkage INSIDE container (orchestrated by srun):"
srun --cpu-bind=verbose,cores -n 1 "$runtime" exec "$IMAGE" bash -c 'ldd /opt/nbody/nbody_direct_hybrid | grep -i mpi || echo "No libmpi direct link found"'
echo ""

# Test funzionale multi-rank
echo "3. Testing srun + singularity exec with -n 2 (multi-rank):"
srun --cpu-bind=verbose,cores -n 2 "$runtime" exec "$IMAGE" /opt/nbody/nbody_direct_hybrid --help 2>&1 | head -5
echo ""

echo "=== Check Complete ==="
