#!/usr/bin/env bash
# Orchestrator script to build, run a benchmark sweep, analyze and plot results
# Can be run interactively on a login node or inside a job script on the compute node.
# Usage: ./jobs/run_all.sh [outdir]

set -euo pipefail
cd "$(dirname "$0")/.."

OUTDIR=${1:-results_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTDIR"

# Defaults (override by exporting environment variables before calling)
: ${RANKS:="1 2"}
: ${THREADS:="1 2"}
: ${REPEATS:=2}
: ${STRONG_N:=1000}
: ${WEAK_PER_RANK:=500}
: ${NSTEPS:=10}
: ${ENERGY_EVERY:=5}

echo "OUTDIR=$OUTDIR"
echo "RANKS=$RANKS THREADS=$THREADS REPEATS=$REPEATS STRONG_N=$STRONG_N NSTEPS=$NSTEPS"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "Compiling..."
make clean && make -j

echo "Collecting system info..."
./collect_system_info.sh > "$OUTDIR/system_info_local.txt"

echo "Generating small IC for tests..."
./generate_ic --n 128 --model 0 --output "$OUTDIR/ic_128.bin"

echo "Running benchmark sweep"
RANKS="$RANKS" THREADS="$THREADS" REPEATS=$REPEATS STRONG_N=$STRONG_N WEAK_PER_RANK=$WEAK_PER_RANK NSTEPS=$NSTEPS ENERGY_EVERY=$ENERGY_EVERY ./benchmark_scaling.sh | tee "$OUTDIR/benchmark.log"

echo "Copying raw results"
cp benchmark_results.csv "$OUTDIR/"

echo "Analyzing"
python3 analyze_benchmark.py benchmark_results.csv "$OUTDIR/benchmark_summary.csv"
python3 plot_scaling.py "$OUTDIR/benchmark_summary.csv" "$OUTDIR/scaling"

echo "Copying plots"
cp *.svg "$OUTDIR/" || true

echo "Packaging"
tar -czf "$OUTDIR/artifacts.tgz" -C "$OUTDIR" .

echo "All done. Results in $OUTDIR"
