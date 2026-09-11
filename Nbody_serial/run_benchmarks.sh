#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: ./run_benchmarks.sh COMMAND [options]

Commands:
  scaling     strong/weak MPI or MPI+OpenMP scaling
  hybrid      fixed-resource P x T sweep; writes one merged summary
  ablation    compare kernel, math, communication and accumulator variants
  layout      AoS-vs-SoA memory-layout benchmark
  energy      energy-diagnostic overhead benchmark
  memory      STREAM-style RAM-bandwidth benchmark
  container   native-vs-container solver overhead plus launch overhead
  osu         OSU latency/bandwidth native, container, or both
  arch        native build target comparison: -march=native vs x86-64-v3
  perf        optional perf-stat hardware-counter snapshot

Options are passed as --key value and become upper-case environment variables.
Examples:
  ./run_benchmarks.sh scaling --ranks "1 2 4 8" --threads 1 --out results.csv
  ./run_benchmarks.sh container --image nbody.sif --runtime singularity
  ./run_benchmarks.sh osu --mode both --image nbody.sif --out osu.csv
  ./run_benchmarks.sh memory --threads 64 --out memory_bandwidth.csv
EOF
}

# The first positional argument selects the benchmark family.  Everything after
# it is interpreted as generic key/value configuration.
cmd="${1:-}"
if [[ -z "$cmd" || "$cmd" == "--help" || "$cmd" == "-h" ]]; then
  usage
  exit 0
fi
shift

# Convert CLI options to exported uppercase variables.  Example:
#   --strong-n 50000  -> STRONG_N=50000
# This makes the script easy to call from Make, Slurm, or an interactive shell.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --*=*)
      key="${1%%=*}"
      val="${1#*=}"
      shift
      ;;
    --*)
      key="$1"
      val="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  key="${key#--}"
  key="${key//-/_}"
  key="$(printf "%s" "$key" | tr '[:lower:]' '[:upper:]')"
  export "$key=$val"
done

# Common defaults.  Every value can be overridden either as an environment
# variable or via the command-line conversion above.
launcher="${LAUNCHER:-srun}"
cpu_bind="${CPU_BIND:---cpu-bind=verbose,cores}"
# The benchmark suite uses only the Plummer initial condition (model 0).
model=0
dt="${DT:-1e-4}"
eps="${EPS:-0.05}"
nsteps="${NSTEPS:-100}"
repeats="${REPEATS:-5}"
warmups="${WARMUPS:-1}"
energy_every="${ENERGY_EVERY:-100}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load helpers in the current shell: this adds no process per benchmark run.
source "$script_dir/benchmark_common.sh"

write_failed_scaling_row() {
  # Failed solver runs are recorded as CSV rows instead of aborting the whole
  # sweep.  This preserves partial evidence and makes transient Slurm/MPI
  # failures visible in post-processing.
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,unknown,nan,nan,nan,nan,nan,nan,nan,nan,RUN_FAILED,nan\n" "$@"
}

bench_scaling() {
  # Strong scaling keeps N fixed while resources grow; weak scaling grows N
  # proportionally to ranks.  Both are produced by the same loop so the report
  # can compare behavior without changing code paths.
  local strong_n="${STRONG_N:-100000}"
  local weak_per_rank="${WEAK_PER_RANK:-10000}"
  local ranks_list="${RANKS:-1 2 4 8 16 32 64}"
  local threads_list="${THREADS:-1}"
  local comm="${COMM:-overlap}"
  local kernel="${KERNEL:-direct}"
  local rsqrt="${RSQRT:-exact}"
  local accumulators="${ACCUMULATORS:-4}"
  local out="${OUT:-benchmark_results.csv}"
  local use_container="${USE_CONTAINER:-0}"
  local image="${IMAGE:-${CONTAINER_IMAGE:-nbody.sif}}"
  local runtime=""
  if [[ "$use_container" == "1" ]]; then
    runtime="$(detect_runtime)" || { echo "no container runtime found" >&2; exit 127; }
    [[ "$runtime" == "docker" || -f "$image" ]] || { echo "missing image: $image" >&2; exit 1; }
  fi

  make nbody_direct_hybrid generate_ic >/dev/null
  printf "kind,N,nsteps,ranks,threads,repeat,integrator,comm,kernel,rsqrt,accumulators,dtype,total,io,drift,force,comm_wait,kick,energy,gpairs,status,max_rel_drift\n" > "$out"

  run_case() {
    # One measured solver execution.  It creates a deterministic input, runs
    # native or container mode, parses the concise solver summary, and removes
    # the temporary binary input afterwards.
    local kind="$1" n="$2" ranks="$3" threads="$4" rep="$5" record="${6:-1}"
    local input="${kind}_N${n}_P${ranks}_T${threads}_seed${rep}.bin"
    local log rc prefix
    if [[ "$kernel" == "newton" && "$ranks" != "1" ]]; then
      echo "skip kernel=newton with ranks=$ranks: this variant is single-rank only" >&2
      return 0
    fi
    ./generate_ic --model "$model" --n "$n" --seed "$((1000 + rep))" --output "$input" >/dev/null
    # Temporarily disable `set -e` around the launched job: one failed run must
    # become a RUN_FAILED CSV row, not kill the entire benchmark sweep.
    set +e
    if [[ "$use_container" == "1" ]]; then
      log="$(run_hybrid_solver container "$runtime" "$image" "$ranks" "$threads" "$input" \
        --comm "$comm" --kernel "$kernel" --rsqrt "$rsqrt" \
        --accumulators "$accumulators" 2>&1)"
    else
      log="$(run_hybrid_solver native "" "" "$ranks" "$threads" "$input" \
        --comm "$comm" --kernel "$kernel" --rsqrt "$rsqrt" \
        --accumulators "$accumulators" 2>&1)"
    fi
    rc=$?
    set -e
    prefix="$kind,$n,$nsteps,$ranks,$threads,$rep,kdk,$comm,$kernel,$rsqrt,$accumulators"
    if (( rc != 0 )); then
      if [[ "$record" == "1" ]]; then
        write_failed_scaling_row "$kind" "$n" "$nsteps" "$ranks" "$threads" "$rep" kdk "$comm" "$kernel" "$rsqrt" "$accumulators" >> "$out"
      fi
      printf "warning: scaling failed kind=%s N=%s P=%s T=%s rep=%s rc=%s\n%s\n" "$kind" "$n" "$ranks" "$threads" "$rep" "$rc" "$log" >&2
    else
      if [[ "$record" == "1" ]]; then
        printf "%s\n" "$log" | parse_solver_csv "$prefix" >> "$out"
      fi
    fi
    rm -f "$input"
  }

  # Warmups are executed but not recorded; repeated measurements are recorded
  # and later summarized with medians/outlier handling in analyze.py.
  for kind in strong weak; do
    for ranks in $ranks_list; do
      local n="$strong_n"
      [[ "$kind" == "weak" ]] && n=$((weak_per_rank * ranks))
      for threads in $threads_list; do
        for rep in $(seq 1 "$warmups"); do run_case "$kind" "$n" "$ranks" "$threads" "$rep" 0 >/dev/null; done
        for rep in $(seq 1 "$repeats"); do run_case "$kind" "$n" "$ranks" "$threads" "$rep"; done
      done
    done
  done
  echo "wrote $out"
}

bench_hybrid() {
  # Hybrid scaling compares several MPI-rank x OpenMP-thread decompositions at a
  # fixed total core budget.  Each pair gets its own raw CSV, while the summary
  # file is merged once so plotting remains simple.
  local prefix="${RESULT_PREFIX:-results_hybrid}"
  local summary="${SUMMARY:-${prefix}_summary.csv}"
  local pairs="${HYBRID_PAIRS:-64x1 32x2 16x4 8x8 4x16}"
  rm -f "${prefix}"_P*_T*.csv "${prefix}"_P*_T*_summary.csv "$summary"
  for pair in $pairs; do
    local ranks="${pair%x*}"
    local threads="${pair#*x}"
    local raw="${prefix}_P${ranks}_T${threads}.csv"
    local partial="${prefix}_P${ranks}_T${threads}_summary.csv"
    RANKS="$ranks" THREADS="$threads" SRUN_CPUS_PER_TASK="$threads" OUT="$raw" bench_scaling
    python3 analyze.py summarize scaling "$raw" "$partial"
    if [[ ! -s "$summary" ]]; then cp "$partial" "$summary"; else tail -n +2 "$partial" >> "$summary"; fi
  done
  python3 analyze.py plot hybrid "$summary" "$prefix"
}

bench_ablation() {
  # Ablation isolates one optimization dimension at a time: algorithmic kernel,
  # inverse-square-root implementation, communication mode, and per-thread
  # accumulator count.  The output is intentionally simple for bar-plotting.
  local ranks="${RANKS:-64}"
  local n="${N:-10000}"
  local out="${OUT:-results_ablation.csv}"
  local input="ic_ablation_N${n}.bin"
  make nbody_direct_hybrid generate_ic >/dev/null
  ./generate_ic --model "$model" --n "$n" --seed "${SEED:-123}" --output "$input" >/dev/null
  printf "Test_Type,Config,Time_Sec\n" > "$out"

  ablation_case() {
    # Record median-ready total times for one variant.  Failures are written as
    # NaN so the plotter can skip them while the raw CSV still documents them.
    local test_type="$1" config="$2" ranks="$3"
    shift 3
    local log rc time_sec
    set +e
    log="$(run_hybrid_solver native "" "" "$ranks" "${THREADS:-1}" "$input" "$@" 2>&1)"
    rc=$?
    set -e
    if (( rc != 0 )); then
      printf "%s,%s,nan\n" "$test_type" "$config" >> "$out"
      printf "warning: ablation %s/%s failed rc=%s\n%s\n" "$test_type" "$config" "$rc" "$log" >&2
    else
      time_sec="$(printf "%s\n" "$log" | awk 'BEGIN{FS="[ =]+"} /^# timing_max_seconds/ {for(i=1;i<=NF;i++) if($i=="total") print $(i+1)}')"
      printf "%s,%s,%s\n" "$test_type" "$config" "${time_sec:-nan}" >> "$out"
    fi
  }

  for rep in $(seq 1 "$repeats"); do ablation_case Kernel direct 1 --kernel direct; done
  for rep in $(seq 1 "$repeats"); do ablation_case Kernel newton 1 --kernel newton; done
  for rep in $(seq 1 "$repeats"); do ablation_case Math exact "$ranks" --rsqrt exact; done
  for rep in $(seq 1 "$repeats"); do ablation_case Math approx "$ranks" --rsqrt approx; done
  for rep in $(seq 1 "$repeats"); do ablation_case Comm sendrecv "$ranks" --comm sendrecv; done
  for rep in $(seq 1 "$repeats"); do ablation_case Comm overlap "$ranks" --comm overlap; done
  for rep in $(seq 1 "$repeats"); do ablation_case Accumulators 1 "$ranks" --accumulators 1; done
  for rep in $(seq 1 "$repeats"); do ablation_case Accumulators 2 "$ranks" --accumulators 2; done
  for rep in $(seq 1 "$repeats"); do ablation_case Accumulators 4 "$ranks" --accumulators 4; done
  for rep in $(seq 1 "$repeats"); do ablation_case Accumulators 8 "$ranks" --accumulators 8; done
  rm -f "$input"
  echo "wrote $out"
}

bench_layout() {
  # Memory-layout evidence: compare Array-of-Structures (AoS) and
  # Structure-of-Arrays (SoA) force kernels over a thread sweep, including
  # checksums to show that the layouts compute equivalent accelerations.
  local n="${N:-10000}"
  local threads_list="${THREADS:-1 2 4 8}"
  local inner_repeats="${INNER_REPEATS:-3}"
  local rsqrt="${RSQRT:-exact}"
  local out="${OUT:-layout_results.csv}"
  local input="${INPUT:-layout_N${n}.bin}"
  make nbody_layout_benchmark generate_ic >/dev/null
  ./generate_ic --model "$model" --n "$n" --seed "${SEED:-4242}" --output "$input" >/dev/null
  printf "layout,N,threads,repeat,warmups,inner_repeats,rsqrt,force,gpairs,checksum\n" > "$out"
  for threads in $threads_list; do
    for rep in $(seq 1 "$repeats"); do
      OMP_NUM_THREADS="$threads" ./nbody_layout_benchmark \
        --input "$input" --layout both --warmups "$warmups" \
        --inner-repeats "$inner_repeats" --eps "$eps" --rsqrt "$rsqrt" |
        awk -F, -v rep="$rep" 'BEGIN{OFS=","}{print $1,$2,$3,rep,$4,$5,$6,$7,$8,$9}' >> "$out"
    done
  done
  rm -f "$input"
  echo "wrote $out"
}

bench_energy() {
  # Energy diagnostics are scientifically useful but expensive because they add
  # an O(N^2) potential-energy pass.  This sweep quantifies how often the report
  # can afford to compute the diagnostic.
  local n="${N:-10000}"
  local ranks="${RANKS:-8}"
  local threads="${THREADS:-1}"
  local list="${ENERGY_LIST:-}"
  local out="${OUT:-energy_overhead.csv}"
  local input="${INPUT:-energy_N${n}.bin}"
  make nbody_direct_hybrid generate_ic >/dev/null
  ./generate_ic --model "$model" --n "$n" --seed "${SEED:-5151}" --output "$input" >/dev/null
  printf "N,nsteps,ranks,threads,repeat,energy_every,total,force,energy,status,max_rel_drift\n" > "$out"
  if [[ -z "$list" ]]; then
    list="1 5 10 $nsteps"
  fi
  # Values larger than nsteps are equivalent to "final step only"; normalize and
  # de-duplicate them to avoid wasting allocations on repeated energy settings.
  list="$(for ee in $list; do if (( ee > nsteps )); then echo "$nsteps"; else echo "$ee"; fi; done | awk '!seen[$0]++')"
  for ee in $list; do
    for rep in $(seq 1 "$warmups"); do
      energy_every="$ee" run_hybrid_solver native "" "" "$ranks" "$threads" "$input" >/dev/null 2>&1 || true
    done
    for rep in $(seq 1 "$repeats"); do
      set +e
      log="$(energy_every="$ee" run_hybrid_solver native "" "" "$ranks" "$threads" "$input" 2>&1)"
      rc=$?
      set -e
      if (( rc != 0 )); then
        printf "%s,%s,%s,%s,%s,%s,nan,nan,nan,RUN_FAILED,nan\n" "$n" "$nsteps" "$ranks" "$threads" "$rep" "$ee" >> "$out"
        printf "warning: energy failed energy_every=%s rep=%s rc=%s\n%s\n" "$ee" "$rep" "$rc" "$log" >&2
      else
        printf "%s\n" "$log" | awk -v p="$n,$nsteps,$ranks,$threads,$rep,$ee" '
          BEGIN{FS="[ =]+";OFS=",";total=force=energy=status=drift=""}
          /^# final:/ {for(i=1;i<=NF;i++){if($i=="max_relative_energy_drift") drift=$(i+1); else if($i=="status") status=$(i+1)}}
          /^# timing_max_seconds/ {for(i=1;i<=NF;i++){if($i=="total") total=$(i+1); else if($i=="force") force=$(i+1); else if($i=="energy") energy=$(i+1)}}
          END{if(total==""||force==""||energy==""||status=="") print p,"nan","nan","nan","PARSE_FAILED","nan"; else print p,total,force,energy,status,drift}
        ' >> "$out"
      fi
    done
  done
  rm -f "$input"
  echo "wrote $out"
}

bench_memory() {
  # Standalone STREAM-style RAM-bandwidth benchmark.  This is separate from the
  # N-body kernels because the written report must document hardware memory
  # bandwidth directly, not only application-level pair-interaction throughput.
  local out="${OUT:-memory_bandwidth.csv}"
  local threads="${THREADS:-${MEMORY_THREADS:-64}}"
  local stream_n="${STREAM_N:-67108864}"
  local stream_inner="${STREAM_INNER:-10}"
  [[ "$threads" != *[[:space:]]* ]] || { echo "memory benchmark requires one scalar THREADS value, got: $threads" >&2; exit 2; }
  make memory_bandwidth >/dev/null
  printf "kernel,N,threads,repeat,inner_repeats,best_seconds,GBps,checksum,status\n" > "$out"
  for rep in $(seq 1 "$warmups"); do
    STREAM_N="$stream_n" STREAM_INNER="$stream_inner" \
      OMP_NUM_THREADS="$threads" OMP_PLACES="${OMP_PLACES:-cores}" OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
      "$launcher" $cpu_bind --ntasks=1 --cpus-per-task="$threads" ./memory_bandwidth >/dev/null 2>&1 || true
  done
  for rep in $(seq 1 "$repeats"); do
    local log rc
    set +e
    log="$(STREAM_N="$stream_n" STREAM_INNER="$stream_inner" \
      OMP_NUM_THREADS="$threads" OMP_PLACES="${OMP_PLACES:-cores}" OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
      "$launcher" $cpu_bind --ntasks=1 --cpus-per-task="$threads" ./memory_bandwidth 2>&1)"
    rc=$?
    set -e
    if (( rc != 0 )); then
      printf "RUN_FAILED,%s,%s,%s,%s,nan,nan,nan,RUN_FAILED\n" "$stream_n" "$threads" "$rep" "$stream_inner" >> "$out"
      printf "warning: memory bandwidth failed rep=%s rc=%s\n%s\n" "$rep" "$rc" "$log" >&2
    else
      printf "%s\n" "$log" |
        awk -F, -v rep="$rep" '
          BEGIN { OFS="," }
          $1 ~ /^(copy|scale|add|triad)$/ && NF >= 7 { print $1,$2,$3,rep,$4,$5,$6,$7,"OK" }
        ' >> "$out"
    fi
  done
  echo "wrote $out"
}

bench_container() {
  # Solver container overhead benchmark required by the container part of the
  # assignment.  It compares native and container executions for the same
  # deterministic inputs, plus a separate launch-only overhead measurement.
  local image="${IMAGE:-nbody.sif}"
  local out="${OUT:-container_overhead.csv}"
  local launch_out="${LAUNCH_OUT:-container_launch_overhead.csv}"
  local ranks_list="${RANKS:-1 2 4}"
  local threads="${THREADS:-1}"
  local strong_n="${N:-100000}"
  local n_per_rank="${N_PER_RANK:-10000}"
  local launch_repeats="${LAUNCH_REPEATS:-10}"
  local runtime
  runtime="$(detect_runtime)" || { echo "no container runtime found" >&2; exit 127; }
  [[ "$runtime" == "docker" || -f "$image" ]] || { echo "missing image: $image" >&2; exit 1; }
  make nbody_direct_hybrid generate_ic >/dev/null
  printf "kind,mode,N,ranks,threads,repeat,total,status,max_rel_drift\n" > "$out"
  container_case() {
    # Pair native/container timings as closely as possible: same N, same rank
    # count, same repeat index, same generated input seed.
    local kind="$1" mode="$2" n="$3" ranks="$4" rep="$5"
    local input="container_${kind}_N${n}_P${ranks}_seed${rep}.bin"
    local log rc prefix
    ./generate_ic --model "$model" --n "$n" --seed "$((7700 + rep))" --output "$input" >/dev/null
    set +e
    if [[ "$mode" == "native" ]]; then
      log="$(run_hybrid_solver native "" "" "$ranks" "$threads" "$input" --comm sendrecv 2>&1)"
    else
      log="$(run_hybrid_solver container "$runtime" "$image" "$ranks" "$threads" "$input" --comm sendrecv 2>&1)"
    fi
    rc=$?
    set -e
    prefix="$kind,$mode,$n,$ranks,$threads,$rep"
    if (( rc != 0 )); then
      printf "%s,nan,RUN_FAILED,nan\n" "$prefix" >> "$out"
      printf "warning: container case failed %s rc=%s\n%s\n" "$prefix" "$rc" "$log" >&2
    else
      printf "%s\n" "$log" | parse_solver_csv "$prefix" total >> "$out"
    fi
    rm -f "$input"
  }
  for ranks in $ranks_list; do
    for rep in $(seq 1 "$repeats"); do
      container_case strong native "$strong_n" "$ranks" "$rep"
      container_case strong container "$strong_n" "$ranks" "$rep"
      container_case weak native "$((n_per_rank * ranks))" "$ranks" "$rep"
      container_case weak container "$((n_per_rank * ranks))" "$ranks" "$rep"
    done
  done
  printf "repeat,seconds\n" > "$launch_out"
  for rep in $(seq 1 "$launch_repeats"); do
    seconds="$( { time -p run_container_single "$runtime" "$image" true; } 2>&1 | awk '/^real /{print $2}' )" || seconds="nan"
    printf "%s,%s\n" "$rep" "${seconds:-nan}" >> "$launch_out"
  done
  echo "wrote $out and $launch_out"
}

parse_osu() {
  local mode_name="$1" bench="$2" metric="$3"
  # OSU tools print whitespace-separated tables after comment/header lines; only
  # numeric rows become CSV records.
  awk -v mode="$mode_name" -v bench="$bench" -v metric="$metric" '
    BEGIN { OFS="," }
    /^[[:space:]]*[0-9]+[[:space:]]+/ { print mode, bench, metric, $1, $2 }
  '
}

bench_osu() {
  # OSU Micro-Benchmarks provide a direct MPI communication comparison
  # independent of the N-body code.  The Docker/SIF image includes OSU binaries
  # so native-vs-container latency and bandwidth can be reported.
  local mode="${MODE:-both}"
  local image="${IMAGE:-nbody.sif}"
  local out="${OUT:-osu_microbench.csv}"
  local latency="${OSU_LATENCY:-osu_latency}"
  local bw="${OSU_BW:-osu_bw}"
  local container_latency="${CONTAINER_OSU_LATENCY:-osu_latency}"
  local container_bw="${CONTAINER_OSU_BW:-osu_bw}"
  local runtime
  runtime="$(detect_runtime 2>/dev/null || true)"
  printf "mode,benchmark,metric,bytes,value\n" > "$out"
  run_osu_one() {
    # Run one OSU executable in native or container mode.  Warnings go to stderr;
    # successful numeric rows are appended to the shared CSV.
    local mode_name="$1" bench="$2" metric="$3" tool="$4" record="${5:-1}"
    local log rc
    set +e
    if [[ "$mode_name" == "native" ]]; then
      local distribution_args=()
      local saved_ntasks_per_node="${SRUN_NTASKS_PER_NODE:-}"
      SRUN_NTASKS_PER_NODE="${OSU_NTASKS_PER_NODE:-}"
      mapfile -t distribution_args < <(launcher_distribution_args)
      SRUN_NTASKS_PER_NODE="$saved_ntasks_per_node"
      log="$("$launcher" $cpu_bind -n 2 "${distribution_args[@]}" "$tool" 2>&1)"
    else
      log="$(SRUN_NTASKS_PER_NODE="${OSU_NTASKS_PER_NODE:-}" \
        run_container_mpi "$runtime" "$image" 2 1 "$tool" 2>&1)"
    fi
    rc=$?
    set -e
    if (( rc != 0 )); then
      printf "warning: OSU %s/%s failed rc=%s\n%s\n" "$mode_name" "$bench" "$rc" "$log" >&2
    elif [[ "$record" == "1" ]]; then
      printf "%s\n" "$log" | parse_osu "$mode_name" "$bench" "$metric" >> "$out"
    fi
  }
  run_osu_repeated() {
    # OSU performs many internal iterations per invocation, but the project
    # deliverables ask for statistics over repeated measurements.  Repeat the
    # whole launched benchmark so analyze.py can report median and stdev across
    # comparable native/container runs.
    local mode_name="$1" bench="$2" metric="$3" tool="$4"
    local rep
    for rep in $(seq 1 "$warmups"); do
      run_osu_one "$mode_name" "$bench" "$metric" "$tool" 0
    done
    for rep in $(seq 1 "$repeats"); do
      run_osu_one "$mode_name" "$bench" "$metric" "$tool" 1
    done
  }
  if [[ "$mode" == "native" || "$mode" == "both" ]]; then
    command -v "$latency" >/dev/null 2>&1 || [[ -x "$latency" ]] || { echo "missing $latency" >&2; exit 1; }
    command -v "$bw" >/dev/null 2>&1 || [[ -x "$bw" ]] || { echo "missing $bw" >&2; exit 1; }
    run_osu_repeated native latency latency_us "$latency"
    run_osu_repeated native bandwidth bandwidth_MBps "$bw"
  fi
  if [[ "$mode" == "container" || "$mode" == "both" ]]; then
    [[ -n "$runtime" ]] || { echo "no container runtime found" >&2; exit 127; }
    run_osu_repeated container latency latency_us "$container_latency"
    run_osu_repeated container bandwidth bandwidth_MBps "$container_bw"
  fi
  echo "wrote $out"
}

bench_arch() {
  # Isolate the compilation-target effect requested by the container section:
  # compare the same native run built with -march=native and -march=x86-64-v3.
  # This keeps the architectural penalty separate from Singularity runtime
  # overhead, which is measured by bench_container().
  local out="${OUT:-arch_target_comparison.csv}"
  local n="${N:-100000}"
  local ranks="${RANKS:-64}"
  local threads="${THREADS:-1}"
  local base_cflags="${BASE_CFLAGS:--O3 -Wall -Wextra -Wpedantic}"
  local input="arch_compare_N${n}.bin"
  printf "target,N,nsteps,ranks,threads,repeat,total,status,max_rel_drift\n" > "$out"
  for target in native x86-64-v3; do
    make clean >/dev/null
    CFLAGS="$base_cflags -march=$target" make nbody_direct_hybrid generate_ic >/dev/null
    ./generate_ic --model "$model" --n "$n" --seed "${SEED:-5151}" --output "$input" >/dev/null
    for rep in $(seq 1 "$warmups"); do
      run_hybrid_solver native "" "" "$ranks" "$threads" "$input" \
        --comm overlap --kernel direct --rsqrt exact --accumulators 4 >/dev/null 2>&1 || true
    done
    for rep in $(seq 1 "$repeats"); do
      local log rc prefix
      set +e
      log="$(run_hybrid_solver native "" "" "$ranks" "$threads" "$input" \
        --comm overlap --kernel direct --rsqrt exact --accumulators 4 2>&1)"
      rc=$?
      set -e
      prefix="$target,$n,$nsteps,$ranks,$threads,$rep"
      if (( rc != 0 )); then
        printf "%s,nan,RUN_FAILED,nan\n" "$prefix" >> "$out"
        printf "warning: arch target=%s failed rep=%s rc=%s\n%s\n" "$target" "$rep" "$rc" "$log" >&2
      else
        printf "%s\n" "$log" | parse_solver_csv "$prefix" total >> "$out"
      fi
    done
  done
  rm -f "$input"
  make clean >/dev/null
  make nbody_direct_hybrid generate_ic >/dev/null
  echo "wrote $out"
}

bench_perf() {
  # Optional hardware-counter snapshot.  It is not required for the main report
  # but can help discuss bottlenecks such as instruction count, cache misses, or
  # branch misses when the cluster allows `perf`.
  local input="${INPUT:-perf_counter_input.bin}"
  local n="${N:-20000}"
  local ranks="${RANKS:-1}"
  local threads="${THREADS:-1}"
  local out="${OUT:-perf_counters.txt}"
  local events="${EVENTS:-cycles,instructions,cache-references,cache-misses,branches,branch-misses}"
  command -v perf >/dev/null 2>&1 || { echo "perf is not available in PATH" >&2; exit 127; }
  make all >/dev/null
  ./generate_ic --model "$model" --n "$n" --seed 4242 --output "$input" >/dev/null
  OMP_NUM_THREADS="$threads" perf stat -e "$events" -o "$out" \
    "$launcher" -n "$ranks" ./nbody_direct_hybrid \
    --input "$input" --nsteps "$nsteps" --energy-every "$nsteps" \
    --comm sendrecv --kernel direct --rsqrt exact --quiet
  rm -f "$input"
}

case "$cmd" in
  scaling) bench_scaling ;;
  hybrid) bench_hybrid ;;
  ablation) bench_ablation ;;
  layout) bench_layout ;;
  energy) bench_energy ;;
  memory) bench_memory ;;
  container) bench_container ;;
  osu) bench_osu ;;
  arch) bench_arch ;;
  perf) bench_perf ;;
  *) echo "unknown command: $cmd" >&2; usage >&2; exit 2 ;;
esac
