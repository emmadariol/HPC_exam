#!/bin/bash
#SBATCH --job-name=probe_specs_orfeo
#SBATCH --account=dssc
#SBATCH --partition=EPYC
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=orfeo_0_probe_specs_%j.out
#SBATCH --error=orfeo_0_probe_specs_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RESULT_DIR="${RESULT_DIR:-.}"
mkdir -p "$RESULT_DIR"

module purge
module load openMPI/4.1.6
module load singularity/4.3.1 2>/dev/null || true

out="${RESULT_DIR}/orfeo_specs_${SLURM_JOB_ID}.txt"

{
  echo "# Orfeo allocation probe"
  date -Is
  echo
  echo "## Slurm allocation"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
  echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-}"
  echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-}"
  echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
  echo
  echo "## Slurm cluster view"
  sinfo -N -l | head -50 || true
  echo
  echo "## Node hardware"
  hostname
  lscpu || true
  echo
  echo "## NUMA"
  numactl -H || true
  echo
  echo "## Memory"
  free -h || true
  echo
  echo "## Modules"
  module list 2>&1 || true
  echo
  echo "## Compilers and runtimes"
  cc --version | head -3 || true
  gcc --version | head -3 || true
  mpicc --version | head -5 || true
  mpirun --version | head -5 || true
  python3 --version || true
  command -v singularity && singularity --version || true
  command -v osu_latency || true
  command -v osu_bw || true
  echo
  echo "## Binding smoke"
  srun --cpu-bind=verbose,cores -n 2 --cpus-per-task=1 hostname || true
} > "$out"

echo "wrote $out"
