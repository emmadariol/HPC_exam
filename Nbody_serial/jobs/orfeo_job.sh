#!/bin/bash
# Orfeo SLURM job template for quick development and validation
# Edit partition/account as needed for your Orfeo setup.
# Requests: small node count, short walltime for quick iteration.
# Usage: sbatch orfeo_job.sh

#SBATCH --job-name=nbody-orfeo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --output=orfeo_job.%j.out

#!/bin/bash
# Orfeo SLURM job for local node scaling runs (runnable and self-contained)
# Usage: sbatch jobs/orfeo_job.sh

#SBATCH --job-name=nbody_scale
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --partition=EPYC
#SBATCH --exclusive
#SBATCH --output=orfeo_scaling_%j.out

set -euo pipefail

# Spostati nella cartella di compilazione
cd $SLURM_SUBMIT_DIR

# Ambiente rigoroso: carica/controlla i moduli (se disponibili)
if type module >/dev/null 2>&1; then
	module purge
	module load openmpi/4.1.6
fi

export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Parametri del benchmark (modifica se necessario)
STRONG_N=${STRONG_N:-100000}
LOAD_PER_RANK=${LOAD_PER_RANK:-10000}
NSTEPS=${NSTEPS:-50}
DT=${DT:-1e-4}
EPS=${EPS:-0.05}
ENERGY_EVERY=${ENERGY_EVERY:-10}
THREADS=${THREADS:-1}
CSV_OUT=${CSV_OUT:-benchmark_results_scaling.csv}

echo "Experiment,MPI_Ranks,OpenMP_Threads,Total_N,Time_Sec" > "$CSV_OUT"

echo "Compiling (local)..."
make clean && make -j

echo "Collecting system info..."
./collect_system_info.sh > system_info_orfeo.txt || true

echo "=== Avvio Strong Scaling (N fisso a $STRONG_N) ==="
export OMP_NUM_THREADS=$THREADS

# Limitare i P ai valori ragionevoli su singolo nodo
for P in 1 2 4 8 16 32; do
	echo "Running Strong: P=$P, T=$OMP_NUM_THREADS, N=$STRONG_N"
	input="ic_strong_N${STRONG_N}_P${P}.bin"
	./generate_ic --model 0 --n "$STRONG_N" --seed $((1000 + P)) --output "$input" >/dev/null

	LOG=$(OMP_NUM_THREADS="$OMP_NUM_THREADS" mpirun -np "$P" ./nbody_direct_hybrid \
		--input "$input" --nsteps "$NSTEPS" --dt "$DT" --eps "$EPS" \
		--energy-every "$ENERGY_EVERY" --quiet 2>&1)

	TIME_SEC=$(printf "%s" "$LOG" | grep -o 'total=[^ ]*' | head -n1 | cut -d= -f2 || true)
	echo "Strong,$P,$OMP_NUM_THREADS,$STRONG_N,$TIME_SEC" >> "$CSV_OUT"
	rm -f "$input"
done

echo "=== Avvio Weak Scaling (N/P fisso a $LOAD_PER_RANK) ==="
for P in 1 2 4 8 16 32; do
	WEAK_N=$((P * LOAD_PER_RANK))
	echo "Running Weak: P=$P, T=$OMP_NUM_THREADS, Total_N=$WEAK_N"
	input="ic_weak_N${WEAK_N}_P${P}.bin"
	./generate_ic --model 0 --n "$WEAK_N" --seed $((2000 + P)) --output "$input" >/dev/null

	LOG=$(OMP_NUM_THREADS="$OMP_NUM_THREADS" mpirun -np "$P" ./nbody_direct_hybrid \
		--input "$input" --nsteps "$NSTEPS" --dt "$DT" --eps "$EPS" \
		--energy-every "$ENERGY_EVERY" --quiet 2>&1)

	TIME_SEC=$(printf "%s" "$LOG" | grep -o 'total=[^ ]*' | head -n1 | cut -d= -f2 || true)
	echo "Weak,$P,$OMP_NUM_THREADS,$WEAK_N,$TIME_SEC" >> "$CSV_OUT"
	rm -f "$input"
done

echo "Benchmark completato. Dati salvati in $CSV_OUT"

# Analisi opzionale
if command -v python3 >/dev/null 2>&1; then
	python3 analyze_benchmark.py "$CSV_OUT" benchmark_summary_orfeo.csv || true
	python3 plot_scaling.py benchmark_summary_orfeo.csv orfeo_scaling || true
fi

echo "Packaging outputs"
tar -czf orfeo_results_${SLURM_JOB_ID:-local}.tgz "$CSV_OUT" benchmark_summary_orfeo.csv *.svg system_info_orfeo.txt || true

echo "Done"
