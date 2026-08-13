#!/bin/bash
#SBATCH --job-name=hybrid_scale_leo
#SBATCH --account=uTS26_Tornator
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_bprod
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=leo_2_hybrid_%j.out
#SBATCH --error=leo_2_hybrid_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load profile/base
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

export OMP_PLACES=cores
export OMP_PROC_BIND=spread

STRONG_N=100000
NSTEPS=50
DT=1e-4
EPS=0.05
CSV_OUT="results_2_hybrid_leonardo.csv"

echo "Experiment,MPI_Ranks,OpenMP_Threads,Total_N,Time_Sec" > "$CSV_OUT"

input_strong="ic_strong_hybrid_N${STRONG_N}.bin"
./generate_ic --model 0 --n "$STRONG_N" --seed 42 --output "$input_strong" >/dev/null

# Rapporto P/T per saturare i 112 core del nodo DCGP
for T in 2 4 7 8 14 28 56; do
    P=$((112 / T))
    export OMP_NUM_THREADS=$T
    
    for REP in {1..5}; do
        # Utilizzo nativo di srun per gestire il binding di MPI e OpenMP
        LOG=$(srun --ntasks="$P" --cpus-per-task="$T" ./nbody_direct_hybrid --input "$input_strong" --nsteps "$NSTEPS" --dt "$DT" --eps "$EPS" --quiet 2>&1)
        
        TIME_SEC=$(printf "%s" "$LOG" | grep -o 'total=[^ ]*' | head -n1 | cut -d= -f2 || true)
        [ -z "$TIME_SEC" ] && TIME_SEC=$(printf "%s" "$LOG" | grep -i 'Time' | head -n1 | awk '{print $2}' || true)
        
        echo "Hybrid,$P,$T,$STRONG_N,$TIME_SEC" >> "$CSV_OUT"
    done
done

rm -f "$input_strong"