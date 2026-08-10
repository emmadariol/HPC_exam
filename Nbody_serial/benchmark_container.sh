#!/usr/bin/env bash
set -euo pipefail

image="${IMAGE:-nbody.sif}"
out="${OUT:-container_overhead.csv}"
ranks_list="${RANKS:-1 2 4}"
threads="${THREADS:-1}"
repeats="${REPEATS:-5}"
n="${N:-1000}"
nsteps="${NSTEPS:-20}"
dt="${DT:-1e-4}"
eps="${EPS:-0.05}"
energy_every="${ENERGY_EVERY:-10}"
input="${INPUT:-container_input.bin}"

if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
  echo "singularity/apptainer not found; cannot run container overhead benchmark" >&2
  exit 1
fi

runtime="singularity"
if ! command -v singularity >/dev/null 2>&1; then
  runtime="apptainer"
fi

make nbody_direct_hybrid generate_ic >/dev/null
./generate_ic --model 0 --n "$n" --seed 777 --output "$input" >/dev/null

echo "mode,N,ranks,threads,repeat,total,status,max_rel_drift" > "$out"

run_solver() {
  local mode="$1"
  local ranks="$2"
  local rep="$3"
  local log
  if [[ "$mode" == "native" ]]; then
    log="$(OMP_NUM_THREADS="$threads" mpirun -np "$ranks" ./nbody_direct_hybrid \
      --input "$input" --nsteps "$nsteps" --dt "$dt" --eps "$eps" \
      --energy-every "$energy_every" --integrator kdk --comm sendrecv --quiet)"
  else
    log="$(OMP_NUM_THREADS="$threads" mpirun -np "$ranks" "$runtime" run "$image" \
      --input "$input" --nsteps "$nsteps" --dt "$dt" --eps "$eps" \
      --energy-every "$energy_every" --integrator kdk --comm sendrecv --quiet)"
  fi
  python3 - "$mode" "$n" "$ranks" "$threads" "$rep" "$log" <<'PY'
import re
import sys
mode, n, ranks, threads, rep, log = sys.argv[1:]
field = r"([^ \r\n]+)"
final = re.search(r"max_relative_energy_drift=" + field + r".*status=" + field, log)
timing = re.search(r"total=" + field, log)
if not (final and timing):
    raise SystemExit("could not parse solver output:\n" + log)
print(",".join([mode, n, ranks, threads, rep, timing.group(1), final.group(2), final.group(1)]))
PY
}

for ranks in $ranks_list; do
  for rep in $(seq 1 "$repeats"); do
    run_solver native "$ranks" "$rep" >> "$out"
    run_solver container "$ranks" "$rep" >> "$out"
  done
done

rm -f "$input"
echo "wrote $out"
