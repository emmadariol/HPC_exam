#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --partition=EPYC
#SBATCH --output=orfeo_smoke_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load openMPI/4.1.6

echo "=== 1. Compilazione ==="
make clean && make -j

echo "=== 2. Smoke Test Seriale (da Makefile) ==="
make run-smoke

echo "=== 3. Generazione dataset di prova (N=1000) ==="
./generate_ic --model 0 --n 1000 --seed 123 --output ic_1000.bin >/dev/null

echo "=== 4. Test Ibrido (4 Ranks x 2 Threads) ==="
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=2

# Testa le flag avanzate per assicurarsi che non causino segfault
mpirun -np 4 --bind-to core ./nbody_direct_hybrid \
  --input ic_1000.bin --nsteps 20 --dt 1e-4 --eps 0.05 \
  --energy-every 10 --integrator kdk --comm overlap --kernel direct --rsqrt exact

echo "=== 5. Mini-scaling (Verifica script e parsing CSV) ==="
# Parametri minimi per testare la generazione e il formato del file di output
export RANKS="1 2 4"
export THREADS="1 2"
export REPEATS=2
export STRONG_N=1000
export WEAK_PER_RANK=500
export NSTEPS=10
export ENERGY_EVERY=5

./benchmark_scaling.sh

rm -f ic_1000.bin
echo "=== VALIDAZIONE COMPLETATA ==="