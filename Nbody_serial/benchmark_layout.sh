#!/usr/bin/env bash
set -euo pipefail

model="${MODEL:-0}"
n="${N:-50000}"
threads="${THREADS:-1 2 4}"
repeats="${REPEATS:-5}"
warmups="${WARMUPS:-1}"
inner_repeats="${INNER_REPEATS:-3}"
eps="${EPS:-0.05}"
rsqrt="${RSQRT:-exact}"
out="${OUT:-layout_results.csv}"
input="${INPUT:-layout_N${n}.bin}"

make nbody_layout_benchmark generate_ic >/dev/null
./generate_ic --model "$model" --n "$n" --seed "${SEED:-4242}" --output "$input" >/dev/null

echo "layout,N,threads,repeat,warmups,inner_repeats,rsqrt,force,gpairs,checksum" > "$out"
for t in $threads; do
  for rep in $(seq 1 "$repeats"); do
    OMP_NUM_THREADS="$t" ./nbody_layout_benchmark \
      --input "$input" --layout both --warmups "$warmups" \
      --inner-repeats "$inner_repeats" --eps "$eps" --rsqrt "$rsqrt" |
      awk -F, -v rep="$rep" 'BEGIN { OFS="," } { print $1,$2,$3,rep,$4,$5,$6,$7,$8,$9 }' >> "$out"
  done
done

rm -f "$input"
echo "wrote $out"
