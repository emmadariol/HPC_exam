#!/usr/bin/env bash
set -euo pipefail

mode="${MODE:-native}"
image="${IMAGE:-nbody.sif}"
out="${OUT:-osu_microbench.csv}"
latency_tool="${OSU_LATENCY:-osu_latency}"
bw_tool="${OSU_BW:-osu_bw}"
container_latency_tool="${CONTAINER_OSU_LATENCY:-$latency_tool}"
container_bw_tool="${CONTAINER_OSU_BW:-$bw_tool}"

runtime=""
if command -v apptainer >/dev/null 2>&1; then
  runtime="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  runtime="singularity"
fi

require_tool() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1 && [[ ! -x "$tool" ]]; then
    echo "cannot find $tool; load an OSU Micro-Benchmarks module or set OSU_LATENCY/OSU_BW" >&2
    exit 1
  fi
}

parse_osu() {
  local mode_name="$1"
  local bench="$2"
  local metric="$3"
  awk -v mode="$mode_name" -v bench="$bench" -v metric="$metric" '
    BEGIN { OFS="," }
    /^[[:space:]]*[0-9]+[[:space:]]+/ {
      print mode, bench, metric, $1, $2
    }
  '
}

run_native() {
  local tool="$1"
  srun --cpu-bind=verbose,cores -n 2 "$tool"
}

run_container() {
  local tool="$1"
  if [[ -z "$runtime" ]]; then
    echo "apptainer/singularity not found; cannot run OSU inside container" >&2
    exit 1
  fi
  srun --cpu-bind=verbose,cores -n 2 "$runtime" exec "$image" "$tool"
}

echo "mode,benchmark,metric,bytes,value" > "$out"

if [[ "$mode" == "native" || "$mode" == "both" ]]; then
  require_tool "$latency_tool"
  require_tool "$bw_tool"
  run_native "$latency_tool" | parse_osu native latency latency_us >> "$out"
  run_native "$bw_tool" | parse_osu native bandwidth bandwidth_MBps >> "$out"
fi

if [[ "$mode" == "container" || "$mode" == "both" ]]; then
  run_container "$container_latency_tool" | parse_osu container latency latency_us >> "$out"
  run_container "$container_bw_tool" | parse_osu container bandwidth bandwidth_MBps >> "$out"
fi

echo "wrote $out"
