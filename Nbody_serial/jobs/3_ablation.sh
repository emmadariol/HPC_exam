#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:00:00
#SBATCH --partition=EPYC
#SBATCH --output=orfeo_3_ablation_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load openmpi/4.1.6
export OMP_NUM_THREADS=1

STRONG_N=50000
NSTEPS=50
CSV_OUT="results_3_ablation.csv"

echo "Test_Type,Config,Time_Sec" > "$CSV_OUT"
input_strong="ic_ablation_N${STRONG_N}.bin"
./generate_ic --model 0 --n "$STRONG_N" --seed 123 --output "$input_strong" >/dev/null

# 3.1: Newton vs Direct (Limita a -np 1 come da specifiche)
for KERNEL in direct newton; do
    for REP in {1..5}; do
        LOG=$(mpirun -np 1 ./nbody_direct_hybrid --kernel "$KERNEL" --input "$input_strong" --nsteps "$NSTEPS" --quiet 2>&1)
        TIME_SEC=$(printf "%s" "$LOG" | grep -o 'total=[^ ]*' | cut -d= -f2 || true)
        [ -z "$TIME_SEC" ] && TIME_SEC=$(printf "%s" "$LOG" | grep -i 'Time' | awk '{print $2}' || true)
        echo "Kernel,$KERNEL,$TIME_SEC" >> "$CSV_OUT"
    done
done

# 3.2: Exact vs Approx rsqrt (su 32 core)
for RSQRT in exact approx; do
    for REP in {1..5}; do
        LOG=$(mpirun -np 32 ./nbody_direct_hybrid --rsqrt "$RSQRT" --input "$input_strong" --nsteps "$NSTEPS" --quiet 2>&1)
        TIME_SEC=$(printf "%s" "$LOG" | grep -o 'total=[^ ]*' | cut -d= -f2 || true)
        [ -z "$TIME_SEC" ] && TIME_SEC=$(printf "%s" "$LOG" | grep -i 'Time' | awk '{print $2}' || true)
        echo "Math,$RSQRT,$TIME_SEC" >> "$CSV_OUT"
    done
done

# 3.3: Overlap vs Sendrecv (su 32 core)
for COMM in sendrecv overlap; do
    for REP in {1..5}; do
        LOG=$(mpirun -np 32 ./nbody_direct_hybrid --comm "$COMM" --input "$input_strong" --nsteps "$NSTEPS" --quiet 2>&1)
        TIME_SEC=$(printf "%s" "$LOG" | grep -o 'total=[^ ]*' | cut -d= -f2 || true)
        [ -z "$TIME_SEC" ] && TIME_SEC=$(printf "%s" "$LOG" | grep -i 'Time' | awk '{print $2}' || true)
        echo "Comm,$COMM,$TIME_SEC" >> "$CSV_OUT"
    done
done
rm -f "$input_strong"
