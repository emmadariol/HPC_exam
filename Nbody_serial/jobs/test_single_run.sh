#!/bin/bash
#SBATCH --account=uTS26_Tornator
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:30:00
#SBATCH --output=test_single_%j.out

cd "$SLURM_SUBMIT_DIR"
module purge
module load gcc openmpi

make clean && make -j
./generate_ic --model 0 --n 100000 --seed 42 --output test.bin >/dev/null

echo "Starting single strong scaling test..."
time srun -n 4 ./nbody_direct_hybrid --input test.bin --nsteps 50 --dt 1e-4 --eps 0.05 --quiet
rm -f test.bin
