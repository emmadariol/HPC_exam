#!/usr/bin/env bash
# submit_matrix.sh
# Generate a matrix of runs (local or sbatch) for the project.
# By default this writes sbatch commands to submit_matrix_sbatch.sh.
# Use --submit to actually submit the jobs.

set -euo pipefail

BASEDIR="$(dirname "$0")/.."
cd "$BASEDIR"

OUTFILE="jobs/submit_matrix_sbatch.sh"
SUBMIT=false
TEMPLATE="leonardo" # orfeo|leonardo

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=true; shift ;;
    --template) TEMPLATE="$2"; shift 2 ;;
    --out) OUTFILE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

cat > "$OUTFILE" <<'EOF'
#!/bin/bash
# Generated sbatch commands
EOF

echo "Generating sbatch command file: $OUTFILE"

# Parameter sets (customize as needed)
RANKS_LIST=(1 2 4 8)
THREADS_LIST=(1 2 4)
N_LIST=(128 256 512 1000 2000)
NSTEPS_LIST=(10 20 50)
INTEGRATORS=(kdk dkd)
COMM_LIST=(sendrecv overlap)
KERNELS=(direct newton)
RSQRT_LIST=(exact approx)
REPEATS=3

# Resource presets (can be adjusted per site)
if [[ "$TEMPLATE" == "orfeo" ]]; then
  NODES=1
  NTASKS_PER_NODE=4
  CPUS_PER_TASK=2
  TIME="01:00:00"
  JOBSCRIPT="jobs/orfeo_job.sh"
else
  NODES=4
  NTASKS_PER_NODE=16
  CPUS_PER_TASK=4
  TIME="04:00:00"
  JOBSCRIPT="jobs/leonardo_job.sh"
fi

for N in "${N_LIST[@]}"; do
  for NSTEPS in "${NSTEPS_LIST[@]}"; do
    for RANKS in "${RANKS_LIST[@]}"; do
      for THREADS in "${THREADS_LIST[@]}"; do
        for INTEGRATOR in "${INTEGRATORS[@]}"; do
          for COMM in "${COMM_LIST[@]}"; do
            for KERNEL in "${KERNELS[@]}"; do
              for RSQRT in "${RSQRT_LIST[@]}"; do
                # newton only on single rank
                if [[ "$KERNEL" == "newton" && "$RANKS" -ne 1 ]]; then
                  continue
                fi
                JOBNAME="nbody_N${N}_r${RANKS}_t${THREADS}_${INTEGRATOR}_${COMM}_${KERNEL}_${RSQRT}_s${NSTEPS}"
                EXPORTS=("RANKS=${RANKS}" "THREADS=${THREADS}" "REPEATS=${REPEATS}" "STRONG_N=${N}" "NSTEPS=${NSTEPS}" "ENERGY_EVERY=10" "INTEGRATOR=${INTEGRATOR}" "COMM=${COMM}" "KERNEL=${KERNEL}" "RSQRT=${RSQRT}")
                EXPORT_STR=$(IFS=,; echo "${EXPORTS[*]}")
                SBATCH_CMD=(sbatch --nodes=${NODES} --ntasks-per-node=${NTASKS_PER_NODE} --cpus-per-task=${CPUS_PER_TASK} --time=${TIME} --output=${JOBNAME}.%j.out --export=${EXPORT_STR} ${JOBSCRIPT})
                printf "%s\n" "${SBATCH_CMD[*]}" >> "$OUTFILE"
                if $SUBMIT; then
                  echo "Submitting: ${SBATCH_CMD[*]}"
                  ${SBATCH_CMD[*]}
                fi
              done
            done
          done
        done
      done
    done
  done
done

chmod +x "$OUTFILE"
echo "Wrote $OUTFILE"
if $SUBMIT; then
  echo "Submitted jobs (may take a few seconds for scheduler responses)."
else
  echo "To submit the generated jobs: bash $OUTFILE" 
fi
