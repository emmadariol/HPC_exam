#!/bin/bash
#SBATCH --job-name=mpi_scale
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=02:00:00
#SBATCH --partition=EPYC
#SBATCH --output=orfeo_1_mpi_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load openMPI/4.1.6

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=1

STRONG_N=100000
LOAD_PER_CORE=10000
NSTEPS=50
DT=1e-4
EPS=0.05
CSV_OUT="results_1_mpi.csv"

echo "Experiment,MPI_Ranks,OpenMP_Threads,Total_N,Time_Sec" > "$CSV_OUT"
make clean && make -j

# ==========================================
# STRONG SCALING MPI (5 Ripetizioni)
# ==========================================
input_strong="ic_strong_N${STRONG_N}.bin"
./generate_ic --model 0 --n "$STRONG_N" --seed 42 --output "$input_strong" >/dev/null

for P in 1 2 4 8 16 32 64 128; do
    for REP in {1..5}; do
        LOG=$(mpirun -np "$P" --bind-to core ./nbody_direct_hybrid --input "$input_strong" --nsteps "$NSTEPS" --dt "$DT" --eps "$EPS" --quiet 2>&1)
        TIME_SEC=$(printf "%s" "$LOG" | grep -o 'total=[^ ]*' | head -n1 | cut -d= -f2 || true)
        [ -z "$TIME_SEC" ] && TIME_SEC=$(printf "%s" "$LOG" | grep -i 'Time' | head -n1 | awk '{print $2}' || true)
        echo "Strong,$P,1,$STRONG_N,$TIME_SEC" >> "$CSV_OUT"
    done
done
rm -f "$input_strong"

# ==========================================
# WEAK SCALING MPI (5 Ripetizioni)
# ==========================================
for P in 1 2 4 8 16 32 64 128; do
    WEAK_N=$((P * LOAD_PER_CORE))
    input_weak="ic_weak_N${WEAK_N}_P${P}.bin"
    ./generate_ic --model 0 --n "$WEAK_N" --seed 42 --output "$input_weak" >/dev/null
    
    for REP in {1..5}; do
        LOG=$(mpirun -np "$P" --bind-to core ./nbody_direct_hybrid --input "$input_weak" --nsteps "$NSTEPS" --dt "$DT" --eps "$EPS" --quiet 2>&1)
        TIME_SEC=$(printf "%s" "$LOG" | grep -o 'total=[^ ]*' | head -n1 | cut -d= -f2 || true)
        [ -z "$TIME_SEC" ] && TIME_SEC=$(printf "%s" "$LOG" | grep -i 'Time' | head -n1 | awk '{print $2}' || true)
        echo "Weak,$P,1,$WEAK_N,$TIME_SEC" >> "$CSV_OUT"
    done
    rm -f "$input_weak"
done
