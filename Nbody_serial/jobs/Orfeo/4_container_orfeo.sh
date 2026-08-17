#!/bin/bash
#SBATCH --job-name=container_orfeo
#SBATCH --account=dssc
#SBATCH --partition=EPYC
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=orfeo_4_container_%j.out
#SBATCH --error=orfeo_4_container_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RESULT_DIR="${RESULT_DIR:-.}"
mkdir -p "$RESULT_DIR"
RESULT_PREFIX="${RESULT_PREFIX:-${RESULT_DIR}/container_overhead}"

module purge
module load openMPI/4.1.6
module load singularity/4.3.1

export RANKS="${RANKS:-1 2 4 8 16 32 64 128}"
export THREADS="${THREADS:-1}"
export REPEATS="${REPEATS:-5}"
export N="${N:-50000}"
export N_PER_RANK="${N_PER_RANK:-10000}"
export NSTEPS="${NSTEPS:-50}"
export IMAGE="${IMAGE:-nbody.sif}"

if [[ ! -f "$IMAGE" ]]; then
  echo "missing $IMAGE; build it first with jobs/Orfeo/0_build_container_orfeo.sh" >&2
  exit 1
fi

OUT="${OUT:-${RESULT_PREFIX}.csv}" \
LAUNCH_OUT="${LAUNCH_OUT:-${RESULT_PREFIX}_launch.csv}" \
  bash ./benchmark_container.sh
python3 analyze_container_overhead.py "${OUT:-${RESULT_PREFIX}.csv}" "${RESULT_PREFIX}_summary.csv"
python3 plot_container_overhead.py "${RESULT_PREFIX}_summary.csv" "$RESULT_PREFIX"

if command -v osu_latency >/dev/null 2>&1 && command -v osu_bw >/dev/null 2>&1; then
  bash ./benchmark_osu.sh --mode both --image "$IMAGE" --out "${RESULT_DIR}/osu_microbench_container.csv"
else
  echo "OSU Micro-Benchmarks not found natively; container OSU exists, but native-vs-container comparison needs native OSU binaries." >&2
fi
