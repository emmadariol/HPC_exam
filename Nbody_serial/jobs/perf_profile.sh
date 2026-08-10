#!/usr/bin/env bash
set -euo pipefail

out=${1:-perf_stat.txt}
shift || true

echo "Running perf stat (requires root or perf capabilities). Output: $out"

perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses,br_inst_retired.any,uops_issued.any,uops_executed.any,floating_point_ops -r 5 -- "$@" 2> "$out"

echo "Wrote $out"
