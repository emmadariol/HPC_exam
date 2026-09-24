#!/usr/bin/env bash
#SBATCH -A dssc
#SBATCH -p GENOA
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=test_numa_p8_t8_%j.out
#SBATCH --error=test_numa_p8_t8_%j.err

set -x
module purge
module load openMPI/4.1.6

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

srun --cpu-bind=verbose,cores \
  ./nbody_direct_hybrid \
  --input plummer_10000.bin \
  --nsteps 5 \
  --comm overlap \
  --kernel direct \
  --rsqrt exact \
  --accumulators 4
