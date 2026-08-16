#!/usr/bin/env bash
set -euo pipefail

mode="${MODE:-native}"
image="${IMAGE:-nbody.sif}"
out="${OUT:-osu_microbench.csv}"
latency_tool="${OSU_LATENCY:-osu_latency}"
bw_tool="${OSU_BW:-osu_bw}"
container_latency_tool="${CONTAINER_OSU_LATENCY:-$latency_tool}"
container_bw_tool="${CONTAINER_OSU_BW:-$bw_tool}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="$2"; shift 2 ;;
    --mode=*)
      mode="${1#*=}"; shift ;;
    --image)
      image="$2"; shift 2 ;;
    --image=*)
      image="${1#*=}"; shift ;;
    --out)
      out="$2"; shift 2 ;;
    --out=*)
      out="${1#*=}"; shift ;;
    --osu-latency)
      latency_tool="$2"; container_latency_tool="${CONTAINER_OSU_LATENCY:-$latency_tool}"; shift 2 ;;
    --osu-bw)
      bw_tool="$2"; container_bw_tool="${CONTAINER_OSU_BW:-$bw_tool}"; shift 2 ;;
    --help)
      echo "usage: benchmark_osu.sh [--mode native|container|both] [--image nbody.sif] [--out osu_microbench.csv]"
      exit 0 ;;
    *)
      echo "unknown option: $1" >&2
      exit 1 ;;
  esac
done

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

run_one() {
  local mode_name="$1"
  local bench="$2"
  local metric="$3"
  local tool="$4"
  local log rc
  set +e
  if [[ "$mode_name" == "native" ]]; then
    log="$(run_native "$tool" 2>&1)"
  else
    log="$(run_container "$tool" 2>&1)"
  fi
  rc=$?
  set -e
  if (( rc != 0 )); then
    printf "warning: OSU %s %s failed rc=%s\n%s\n" "$mode_name" "$bench" "$rc" "$log" >&2
    return 0
  fi
  printf "%s\n" "$log" | parse_osu "$mode_name" "$bench" "$metric"
}

echo "mode,benchmark,metric,bytes,value" > "$out"

if [[ "$mode" == "native" || "$mode" == "both" ]]; then
  require_tool "$latency_tool"
  require_tool "$bw_tool"
  run_one native latency latency_us "$latency_tool" >> "$out"
  run_one native bandwidth bandwidth_MBps "$bw_tool" >> "$out"
fi

if [[ "$mode" == "container" || "$mode" == "both" ]]; then
  run_one container latency latency_us "$container_latency_tool" >> "$out"
  run_one container bandwidth bandwidth_MBps "$container_bw_tool" >> "$out"
fi

echo "wrote $out"
