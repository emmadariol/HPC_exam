#!/bin/bash
#SBATCH --job-name=ablation_leo
#SBATCH --account=uTS26_Tornator
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_bprod
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=leo_3_ablation_%j.out
#SBATCH --error=leo_3_ablation_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Pulizia e caricamento moduli esatti per l'ambiente DCGP
module purge
module load profile/base
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

export OMP_NUM_THREADS=1

# Numero di core del nodo DCGP per saturare le risorse nel test distribuito
RANKS=112 

STRONG_N=50000
NSTEPS=50
CSV_OUT="results_3_ablation_leonardo.csv"

echo "Test_Type,Config,Time_Sec" > "$CSV_OUT"
input_strong="ic_ablation_N${STRONG_N}.bin"
./generate_ic --model 0 --n "$STRONG_N" --seed 123 --output "$input_strong" >/dev/null

run_ablation_case() {
    local test_type="$1"
    local config="$2"
    shift 2
    local log rc time_sec
    set +e
    log="$("$@" 2>&1)"
    rc=$?
    set -e
    if (( rc != 0 )); then
        printf "warning: %s/%s failed rc=%s\n%s\n" "$test_type" "$config" "$rc" "$log" >&2
        echo "$test_type,$config,nan" >> "$CSV_OUT"
        return 0
    fi
    time_sec=$(printf "%s" "$log" | grep -o 'total=[^ ]*' | cut -d= -f2 || true)
    [ -z "$time_sec" ] && time_sec=$(printf "%s" "$log" | grep -i 'Time' | awk '{print $2}' || true)
    [ -z "$time_sec" ] && time_sec="nan"
    echo "$test_type,$config,$time_sec" >> "$CSV_OUT"
}

# ==========================================
# Newton vs Direct (Seriale: -np 1)
# ==========================================
for KERNEL in direct newton; do
    for REP in {1..5}; do
        run_ablation_case Kernel "$KERNEL" \
          srun --cpu-bind=verbose,cores -n 1 --cpus-per-task=1 ./nbody_direct_hybrid \
          --kernel "$KERNEL" --input "$input_strong" --nsteps "$NSTEPS" --quiet
    done
done

# ==========================================
# Exact vs Approx rsqrt (Distribuito)
# ==========================================
for RSQRT in exact approx; do
    for REP in {1..5}; do
        run_ablation_case Math "$RSQRT" \
          srun --cpu-bind=verbose,cores -n "$RANKS" --cpus-per-task=1 ./nbody_direct_hybrid \
          --rsqrt "$RSQRT" --input "$input_strong" --nsteps "$NSTEPS" --quiet
    done
done

# ==========================================
# Overlap vs Sendrecv (Distribuito)
# ==========================================
for COMM in sendrecv overlap; do
    for REP in {1..5}; do
        run_ablation_case Comm "$COMM" \
          srun --cpu-bind=verbose,cores -n "$RANKS" --cpus-per-task=1 ./nbody_direct_hybrid \
          --comm "$COMM" --input "$input_strong" --nsteps "$NSTEPS" --quiet
    done
done

rm -f "$input_strong"
