#!/bin/bash
#SBATCH --job-name=hybrid_scale
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=01:30:00
#SBATCH --partition=EPYC
#SBATCH --output=orfeo_2_hybrid_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load openMPI/4.1.6

export OMP_PLACES=cores
export OMP_PROC_BIND=spread

STRONG_N=100000
NSTEPS=50
DT=1e-4
EPS=0.05
CSV_OUT="results_2_hybrid.csv"

echo "Experiment,MPI_Ranks,OpenMP_Threads,Total_N,Time_Sec" > "$CSV_OUT"

input_strong="ic_strong_hybrid_N${STRONG_N}.bin"
./generate_ic --model 0 --n "$STRONG_N" --seed 42 --output "$input_strong" >/dev/null

# Saturiamo 128 core cambiando il ratio P/T
for T in 2 4 8 16 32; do
    P=$((128 / T))
    export OMP_NUM_THREADS=$T
    
    for REP in {1..5}; do
        LOG=$(mpirun --map-by ppr:$P:node:pe=$T --bind-to core ./nbody_direct_hybrid --input "$input_strong" --nsteps "$NSTEPS" --dt "$DT" --eps "$EPS" --quiet 2>&1)
        TIME_SEC=$(printf "%s" "$LOG" | grep -o 'total=[^ ]*' | head -n1 | cut -d= -f2 || true)
        [ -z "$TIME_SEC" ] && TIME_SEC=$(printf "%s" "$LOG" | grep -i 'Time' | head -n1 | awk '{print $2}' || true)
        echo "Hybrid,$P,$T,$STRONG_N,$TIME_SEC" >> "$CSV_OUT"
    done
done
rm -f "$input_strong"
