#!/bin/bash
#SBATCH --job-name=probe_specs_leo
#SBATCH --account=uts26_tornator
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_bprod
#SBATCH --gres=tmpfs:10g
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --output=leo_0_probe_specs_%j.out
#SBATCH --error=leo_0_probe_specs_%j.err



set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load profile/base
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load apptainer 2>/dev/null || true
module load osu-micro-benchmarks 2>/dev/null || true

out="leonardo_specs_${SLURM_JOB_ID}.txt"

{
  echo "# Leonardo allocation probe"
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
  gcc --version | head -3 || true
  mpicc --version | head -5 || true
  mpirun --version | head -5 || true
  python3 --version || true
  command -v apptainer && apptainer --version || true
  command -v singularity && singularity --version || true
  command -v osu_latency || true
  command -v osu_bw || true
  echo
  echo "## Binding smoke"
  srun --cpu-bind=verbose,cores -n 2 --cpus-per-task=1 hostname || true
} > "$out"

echo "wrote $out"
