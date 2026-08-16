#!/usr/bin/env bash
set -euo pipefail

# Compare native and Singularity/Apptainer executions under the same Slurm
# placement.  The weak case keeps N_PER_RANK particles on each MPI rank.
image="${IMAGE:-nbody.sif}"
out="${OUT:-container_overhead.csv}"
launch_out="${LAUNCH_OUT:-container_launch_overhead.csv}"
ranks_list="${RANKS:-1 2 4}"
threads="${THREADS:-1}"
repeats="${REPEATS:-5}"
strong_n="${N:-1000}"
n_per_rank="${N_PER_RANK:-1000}"
nsteps="${NSTEPS:-20}"
dt="${DT:-1e-4}"
eps="${EPS:-0.05}"
energy_every="${ENERGY_EVERY:-10}"
launch_repeats="${LAUNCH_REPEATS:-10}"

if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
  echo "singularity/apptainer not found; cannot run container overhead benchmark" >&2
  exit 1
fi
runtime="singularity"
command -v singularity >/dev/null 2>&1 || runtime="apptainer"

make nbody_direct_hybrid generate_ic >/dev/null
echo "kind,mode,N,ranks,threads,repeat,total,status,max_rel_drift" > "$out"

run_solver() {
  local kind="$1" mode="$2" n="$3" ranks="$4" rep="$5"
  local input="container_${kind}_N${n}_P${ranks}_seed${rep}.bin"
  local log executable
  ./generate_ic --model 0 --n "$n" --seed "$((7700 + rep))" --output "$input" >/dev/null
  if [[ "$mode" == "native" ]]; then
    executable="./nbody_direct_hybrid"
  else
    executable="$runtime exec $image /opt/nbody/nbody_direct_hybrid"
  fi
  set +e
  log="$(OMP_NUM_THREADS="$threads" srun --cpu-bind=verbose,cores --ntasks="$ranks" --cpus-per-task="$threads" $executable \
    --input "$input" --nsteps "$nsteps" --dt "$dt" --eps "$eps" \
    --energy-every "$energy_every" --integrator kdk --comm sendrecv --quiet 2>&1)"
  local rc=$?
  set -e
  if (( rc != 0 )); then
    printf "%s,%s,%s,%s,%s,%s,nan,RUN_FAILED,nan\n" \
      "$kind" "$mode" "$n" "$ranks" "$threads" "$rep"
    printf "warning: %s %s run failed for N=%s ranks=%s repeat=%s rc=%s\n%s\n" \
      "$kind" "$mode" "$n" "$ranks" "$rep" "$rc" "$log" >&2
    rm -f "$input"
    return 0
  fi
  python3 - "$kind" "$mode" "$n" "$ranks" "$threads" "$rep" "$log" <<'PY'
import re
import sys
kind, mode, n, ranks, threads, rep, log = sys.argv[1:]
field = r"([^ \r\n]+)"
final = re.search(r"max_relative_energy_drift=" + field + r".*status=" + field, log)
timing = re.search(r"total=" + field, log)
if not (final and timing):
    print(",".join([kind, mode, n, ranks, threads, rep, "nan", "PARSE_FAILED", "nan"]))
    print("warning: could not parse solver output:\n" + log, file=sys.stderr)
    raise SystemExit(0)
print(",".join([kind, mode, n, ranks, threads, rep, timing.group(1), final.group(2), final.group(1)]))
PY
  rm -f "$input"
}

for ranks in $ranks_list; do
  for rep in $(seq 1 "$repeats"); do
    run_solver strong native "$strong_n" "$ranks" "$rep" >> "$out"
    run_solver strong container "$strong_n" "$ranks" "$rep" >> "$out"
    weak_n=$((n_per_rank * ranks))
    run_solver weak native "$weak_n" "$ranks" "$rep" >> "$out"
    run_solver weak container "$weak_n" "$ranks" "$rep" >> "$out"
  done
done

echo "repeat,seconds" > "$launch_out"
for rep in $(seq 1 "$launch_repeats"); do
  seconds=$( { time -p "$runtime" exec "$image" true; } 2>&1 | awk '/^real / {print $2}' ) || true
  if [[ -z "$seconds" ]]; then
    echo "warning: could not measure container launch time for repeat $rep" >&2
    seconds="nan"
  fi
  echo "$rep,$seconds" >> "$launch_out"
done

echo "wrote $out and $launch_out"
