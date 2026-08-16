#!/usr/bin/env bash
set -euo pipefail

model="${MODEL:-0}"
n="${N:-50000}"
nsteps="${NSTEPS:-50}"
dt="${DT:-1e-4}"
eps="${EPS:-0.05}"
ranks="${RANKS:-1}"
threads="${THREADS:-1}"
repeats="${REPEATS:-5}"
warmups="${WARMUPS:-1}"
energy_list="${ENERGY_LIST:-1 5 10 $nsteps}"
integrator="${INTEGRATOR:-kdk}"
comm="${COMM:-sendrecv}"
kernel="${KERNEL:-direct}"
rsqrt="${RSQRT:-exact}"
out="${OUT:-energy_overhead.csv}"
input="${INPUT:-energy_N${n}.bin}"

make nbody_direct_hybrid generate_ic >/dev/null
./generate_ic --model "$model" --n "$n" --seed "${SEED:-5151}" --output "$input" >/dev/null

echo "N,nsteps,ranks,threads,repeat,energy_every,total,force,energy,status,max_rel_drift" > "$out"

run_once() {
  local energy_every="$1"
  local repeat="$2"
  local log
  log="$(OMP_NUM_THREADS="$threads" srun --cpu-bind=verbose,cores \
    --ntasks="$ranks" --cpus-per-task="${SRUN_CPUS_PER_TASK:-$threads}" \
    ./nbody_direct_hybrid --input "$input" --nsteps "$nsteps" \
    --dt "$dt" --eps "$eps" --energy-every "$energy_every" \
    --integrator "$integrator" --comm "$comm" --kernel "$kernel" \
    --rsqrt "$rsqrt" --quiet)"

  python3 - "$n" "$nsteps" "$ranks" "$threads" "$repeat" "$energy_every" "$log" <<'PY'
import re
import sys

n, nsteps, ranks, threads, repeat, energy_every, log = sys.argv[1:]
field = r"([^ \r\n]+)"
final = re.search(r"max_relative_energy_drift=" + field + r".*status=" + field, log)
timing = re.search(r"total=" + field + r" io=" + field + r" drift=" + field +
                   r" force=" + field + r" comm_wait=" + field +
                   r" kick=" + field + r" energy=" + field, log)
if not (final and timing):
    raise SystemExit("could not parse solver output:\n" + log)
groups = timing.groups()
print(",".join([
    n, nsteps, ranks, threads, repeat, energy_every,
    groups[0], groups[3], groups[6], final.group(2), final.group(1),
]))
PY
}

for energy_every in $energy_list; do
  for rep in $(seq 1 "$warmups"); do
    run_once "$energy_every" "warmup_${rep}" >/dev/null
  done
  for rep in $(seq 1 "$repeats"); do
    run_once "$energy_every" "$rep" >> "$out"
  done
done

rm -f "$input"
echo "wrote $out"
