#!/bin/bash
#SBATCH --job-name=check_mpi_sing
#SBATCH --account=uTS26_Tornator
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_bprod
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

IMAGE="${PWD}/nbody.sif"

echo "=== MPI Linkage Check Inside Singularity (on DCGP Compute Node) ==="
echo "Image: $IMAGE"
echo ""

# SLURM (tramite srun) avvia il container e inietta le librerie host.
# ldd deve mostrare percorsi che puntano a /leonardo/prod/opt/...
echo "1. Binary linkage INSIDE container (orchestrated by srun):"
srun -n 1 singularity exec "$IMAGE" bash -c 'ldd /opt/nbody/nbody_direct_hybrid | grep -i mpi || echo "No libmpi direct link found"'
echo ""

# Test funzionale multi-rank
echo "2. Testing srun + singularity exec with -n 2 (multi-rank):"
srun -n 2 singularity exec "$IMAGE" /opt/nbody/nbody_direct_hybrid --help 2>&1 | head -5
echo ""

echo "=== Check Complete ==="