#!/bin/bash
#SBATCH --job-name=ablation_leo
#SBATCH --account=uTS26_Tornator
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:00:00
#SBATCH --output=leo_3_ablation_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc
module load openmpi
export OMP_NUM_THREADS=1

STRONG_N=50000
NSTEPS=50
CSV_OUT="results_3_ablation_leonardo.csv"

echo "Test_Type,Config,Time_Sec" > "$CSV_OUT"
input_strong="ic_ablation_N${STRONG_N}.bin"
./generate_ic --model 0 --n "$STRONG_N" --seed 123 --output "$input_strong" >/dev/null

# Newton vs Direct (-np 1)
for KERNEL in direct newton; do
    for REP in {1..5}; do
        LOG=$(srun -n 1 ./nbody_direct_hybrid --kernel "$KERNEL" --input "$input_strong" --nsteps "$NSTEPS" --quiet 2>&1)
        TIME_SEC=$(printf "%s" "$LOG" | grep -o 'total=[^ ]*' | cut -d= -f2 || true)
        [ -z "$TIME_SEC" ] && TIME_SEC=$(printf "%s" "$LOG" | grep -i 'Time' | awk '{print $2}' || true)
        echo "Kernel,$KERNEL,$TIME_SEC" >> "$CSV_OUT"
    done
done

# Exact vs Approx rsqrt
for RSQRT in exact approx; do
    for REP in {1..5}; do
        LOG=$(srun -n 32 ./nbody_direct_hybrid --rsqrt "$RSQRT" --input "$input_strong" --nsteps "$NSTEPS" --quiet 2>&1)
        TIME_SEC=$(printf "%s" "$LOG" | grep -o 'total=[^ ]*' | cut -d= -f2 || true)
        [ -z "$TIME_SEC" ] && TIME_SEC=$(printf "%s" "$LOG" | grep -i 'Time' | awk '{print $2}' || true)
        echo "Math,$RSQRT,$TIME_SEC" >> "$CSV_OUT"
    done
done

# Overlap vs Sendrecv
for COMM in sendrecv overlap; do
    for REP in {1..5}; do
        LOG=$(srun -n 32 ./nbody_direct_hybrid --comm "$COMM" --input "$input_strong" --nsteps "$NSTEPS" --quiet 2>&1)
        TIME_SEC=$(printf "%s" "$LOG" | grep -o 'total=[^ ]*' | cut -d= -f2 || true)
        [ -z "$TIME_SEC" ] && TIME_SEC=$(printf "%s" "$LOG" | grep -i 'Time' | awk '{print $2}' || true)
        echo "Comm,$COMM,$TIME_SEC" >> "$CSV_OUT"
    done
done
rm -f "$input_strong"