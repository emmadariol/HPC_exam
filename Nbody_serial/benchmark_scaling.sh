#!/usr/bin/env bash
set -euo pipefail

model="${MODEL:-0}"
strong_n="${STRONG_N:-2000}"
weak_per_rank="${WEAK_PER_RANK:-1000}"
nsteps="${NSTEPS:-20}"
dt="${DT:-1e-4}"
eps="${EPS:-0.05}"
energy_every="${ENERGY_EVERY:-10}"
repeats="${REPEATS:-5}"
rank_list="${RANKS:-1 2 4}"
thread_list="${THREADS:-1 2}"
integrator="${INTEGRATOR:-kdk}"
comm="${COMM:-sendrecv}"
kernel="${KERNEL:-direct}"
rsqrt="${RSQRT:-exact}"
out="${OUT:-benchmark_results.csv}"

# RIMOSSO: make nbody_direct_hybrid generate_ic >/dev/null

echo "kind,N,ranks,threads,repeat,integrator,comm,kernel,rsqrt,total,io,drift,force,kick,energy,gpairs,status,max_rel_drift" > "$out"

run_case() {
  local kind="$1"
  local n="$2"
  local ranks="$3"
  local threads="$4"
  local repeat="$5"
  local input="${kind}_N${n}_seed${repeat}.bin"
  local log

  if [[ "$kernel" == "newton" && "$ranks" != "1" ]]; then
    echo "skipping kernel=newton with ranks=$ranks because this variant is single-rank only" >&2
    return
  fi

  ./generate_ic --model "$model" --n "$n" --seed "$((1000 + repeat))" --output "$input" >/dev/null
  
  # CORRETTO: Uso di srun per l'integrazione nativa con SLURM
  log="$(OMP_NUM_THREADS="$threads" srun --ntasks="$ranks" --cpus-per-task="$threads" ./nbody_direct_hybrid \
    --input "$input" --nsteps "$nsteps" --dt "$dt" --eps "$eps" \
    --energy-every "$energy_every" --integrator "$integrator" --comm "$comm" \
    --kernel "$kernel" --rsqrt "$rsqrt" --quiet)"

  python3 - "$kind" "$n" "$ranks" "$threads" "$repeat" "$integrator" "$comm" "$kernel" "$rsqrt" "$log" <<'PY'
import re
import sys

kind, n, ranks, threads, rep, integrator, comm, kernel, rsqrt, log = sys.argv[1:]
field = r"([^ \r\n]+)"
final = re.search(r"max_relative_energy_drift=" + field + r".*status=" + field, log)
timing = re.search(r"total=" + field + r" io=" + field + r" drift=" + field +
                   r" force=" + field + r" kick=" + field + r" energy=" + field, log)
rate = re.search(r"gpair_interactions_per_second=" + field, log)
if not (final and timing and rate):
    raise SystemExit("could not parse solver output:\n" + log)
print(",".join([
    kind, n, ranks, threads, rep, integrator, comm, kernel, rsqrt,
    *timing.groups(), rate.group(1), final.group(2), final.group(1)
]))
PY
  rm -f "$input"
}

for ranks in $rank_list; do
  for threads in $thread_list; do
    for rep in $(seq 1 "$repeats"); do
      run_case "strong" "$strong_n" "$ranks" "$threads" "$rep" >> "$out"
    done
  done
done

for ranks in $rank_list; do
  n=$((weak_per_rank * ranks))
  for threads in $thread_list; do
    for rep in $(seq 1 "$repeats"); do
      run_case "weak" "$n" "$ranks" "$threads" "$rep" >> "$out"
    done
  done
done

echo "wrote $out"