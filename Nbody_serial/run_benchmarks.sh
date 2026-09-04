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
  container   native-vs-container solver overhead plus launch overhead
  osu         OSU latency/bandwidth native, container, or both
  perf        optional perf-stat hardware-counter snapshot

Options are passed as --key value and become upper-case environment variables.
Examples:
  ./run_benchmarks.sh scaling --ranks "1 2 4 8" --threads 1 --out results.csv
  ./run_benchmarks.sh container --image nbody.sif --runtime singularity
  ./run_benchmarks.sh osu --mode both --image nbody.sif --out osu.csv
EOF
}

cmd="${1:-}"
if [[ -z "$cmd" || "$cmd" == "--help" || "$cmd" == "-h" ]]; then
  usage
  exit 0
fi
shift

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

launcher="${LAUNCHER:-srun}"
cpu_bind="${CPU_BIND:---cpu-bind=verbose,cores}"
model="${MODEL:-0}"
dt="${DT:-1e-4}"
eps="${EPS:-0.05}"
nsteps="${NSTEPS:-50}"
repeats="${REPEATS:-5}"
warmups="${WARMUPS:-1}"
energy_every="${ENERGY_EVERY:-10}"

detect_runtime() {
  if [[ -n "${RUNTIME:-}" ]]; then
    printf "%s" "$RUNTIME"
  elif command -v singularity >/dev/null 2>&1; then
    printf "singularity"
  elif command -v apptainer >/dev/null 2>&1; then
    printf "apptainer"
  elif command -v docker >/dev/null 2>&1; then
    printf "docker"
  else
    return 1
  fi
}

parse_solver_csv() {
  local prefix="$1"
  awk -v prefix="$prefix" '
    BEGIN {
      FS = "[ =]+";
      OFS = ",";
      dtype = "unknown";
      total = io = drift = force = comm_wait = kick = energy = gpairs = max_rel_drift = status = "";
    }
    /^# final:/ {
      for (i = 1; i <= NF; ++i) {
        if ($i == "arithmetic_dtype") dtype = $(i + 1);
        else if ($i == "max_relative_energy_drift") max_rel_drift = $(i + 1);
        else if ($i == "status") status = $(i + 1);
      }
    }
    /^# timing_max_seconds/ {
      for (i = 1; i <= NF; ++i) {
        if ($i == "total") total = $(i + 1);
        else if ($i == "io") io = $(i + 1);
        else if ($i == "drift") drift = $(i + 1);
        else if ($i == "force") force = $(i + 1);
        else if ($i == "comm_wait") comm_wait = $(i + 1);
        else if ($i == "kick") kick = $(i + 1);
        else if ($i == "energy") energy = $(i + 1);
      }
    }
    /^# kernel_rate/ {
      for (i = 1; i <= NF; ++i) {
        if ($i == "gpair_interactions_per_second") gpairs = $(i + 1);
      }
    }
    END {
      if (total == "" || force == "" || gpairs == "" || status == "") {
        print prefix, dtype, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "PARSE_FAILED", "nan";
      } else {
        print prefix, dtype, total, io, drift, force, comm_wait, kick, energy, gpairs, status, max_rel_drift;
      }
    }
  '
}

parse_total_only_csv() {
  local prefix="$1"
  awk -v prefix="$prefix" '
    BEGIN { FS = "[ =]+"; OFS = ","; total = status = drift = "" }
    /^# final:/ {
      for (i = 1; i <= NF; ++i) {
        if ($i == "max_relative_energy_drift") drift = $(i + 1);
        else if ($i == "status") status = $(i + 1);
      }
    }
    /^# timing_max_seconds/ {
      for (i = 1; i <= NF; ++i) if ($i == "total") total = $(i + 1);
    }
    END {
      if (total == "" || status == "") print prefix, "nan", "PARSE_FAILED", "nan";
      else print prefix, total, status, drift;
    }
  '
}

run_mpi_solver() {
  local ranks="$1"
  local threads="$2"
  shift 2
  OMP_NUM_THREADS="$threads" "$launcher" $cpu_bind --ntasks="$ranks" \
    --cpus-per-task="${SRUN_CPUS_PER_TASK:-$threads}" "$@"
}

run_container_mpi() {
  local runtime="$1"
  local image="$2"
  local ranks="$3"
  local threads="$4"
  shift 4
  case "$runtime" in
    singularity|apptainer)
      OMP_NUM_THREADS="$threads" "$launcher" $cpu_bind --ntasks="$ranks" \
        --cpus-per-task="${SRUN_CPUS_PER_TASK:-$threads}" \
        "$runtime" exec "$image" "$@"
      ;;
    docker)
      docker run --rm \
        -e "OMP_NUM_THREADS=$threads" \
        -e OMPI_ALLOW_RUN_AS_ROOT=1 \
        -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
        -v "$PWD:/work" -w /work "$image" \
        mpirun --allow-run-as-root -np "$ranks" "$@"
      ;;
    *)
      echo "unknown runtime: $runtime" >&2
      return 127
      ;;
  esac
}

run_container_single() {
  local runtime="$1"
  local image="$2"
  shift 2
  case "$runtime" in
    singularity|apptainer)
      "$runtime" exec "$image" "$@"
      ;;
    docker)
      docker run --rm -v "$PWD:/work" -w /work "$image" "$@"
      ;;
    *)
      echo "unknown runtime: $runtime" >&2
      return 127
      ;;
  esac
}

write_failed_scaling_row() {
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,nan,nan,nan,nan,nan,nan,nan,nan,RUN_FAILED,nan\n" "$@"
}

bench_scaling() {
  local strong_n="${STRONG_N:-100000}"
  local weak_per_rank="${WEAK_PER_RANK:-10000}"
  local ranks_list="${RANKS:-1 2 4 8 16 32 64}"
  local threads_list="${THREADS:-1}"
  local integrator="${INTEGRATOR:-kdk}"
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
    local kind="$1" n="$2" ranks="$3" threads="$4" rep="$5" record="${6:-1}"
    local input="${kind}_N${n}_P${ranks}_T${threads}_seed${rep}.bin"
    local log rc prefix
    if [[ "$kernel" == "newton" && "$ranks" != "1" ]]; then
      echo "skip kernel=newton with ranks=$ranks: this variant is single-rank only" >&2
      return 0
    fi
    ./generate_ic --model "$model" --n "$n" --seed "$((1000 + rep))" --output "$input" >/dev/null
    set +e
    if [[ "$use_container" == "1" ]]; then
      log="$(run_container_mpi "$runtime" "$image" "$ranks" "$threads" \
        /opt/nbody/nbody_direct_hybrid \
        --input "$input" --nsteps "$nsteps" --dt "$dt" --eps "$eps" \
        --energy-every "$energy_every" --integrator "$integrator" --comm "$comm" \
        --kernel "$kernel" --rsqrt "$rsqrt" --accumulators "$accumulators" --quiet 2>&1)"
    else
      log="$(run_mpi_solver "$ranks" "$threads" ./nbody_direct_hybrid \
        --input "$input" --nsteps "$nsteps" --dt "$dt" --eps "$eps" \
        --energy-every "$energy_every" --integrator "$integrator" --comm "$comm" \
        --kernel "$kernel" --rsqrt "$rsqrt" --accumulators "$accumulators" --quiet 2>&1)"
    fi
    rc=$?
    set -e
    prefix="$kind,$n,$nsteps,$ranks,$threads,$rep,$integrator,$comm,$kernel,$rsqrt,$accumulators"
    if (( rc != 0 )); then
      if [[ "$record" == "1" ]]; then
        write_failed_scaling_row "$kind" "$n" "$nsteps" "$ranks" "$threads" "$rep" "$integrator" "$comm" "$kernel" "$rsqrt" "$accumulators" >> "$out"
      fi
      printf "warning: scaling failed kind=%s N=%s P=%s T=%s rep=%s rc=%s\n%s\n" "$kind" "$n" "$ranks" "$threads" "$rep" "$rc" "$log" >&2
    else
      if [[ "$record" == "1" ]]; then
        printf "%s\n" "$log" | parse_solver_csv "$prefix" >> "$out"
      fi
    fi
    rm -f "$input"
  }

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
  local prefix="${RESULT_PREFIX:-results_hybrid}"
  local summary="${SUMMARY:-${prefix}_summary.csv}"
  local pairs="${HYBRID_PAIRS:-64x1 32x2 16x4 8x8 4x16 2x32 1x64}"
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
  local ranks="${RANKS:-64}"
  local n="${N:-50000}"
  local out="${OUT:-results_ablation.csv}"
  local input="ic_ablation_N${n}.bin"
  make nbody_direct_hybrid generate_ic >/dev/null
  ./generate_ic --model "$model" --n "$n" --seed "${SEED:-123}" --output "$input" >/dev/null
  printf "Test_Type,Config,Time_Sec\n" > "$out"

  ablation_case() {
    local test_type="$1" config="$2" ranks="$3"
    shift 3
    local log rc time_sec
    set +e
    log="$(run_mpi_solver "$ranks" "${THREADS:-1}" ./nbody_direct_hybrid --input "$input" --nsteps "$nsteps" --quiet "$@" 2>&1)"
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
  for rep in $(seq 1 "$repeats"); do ablation_case Accumulators 4 "$ranks" --accumulators 4; done
  rm -f "$input"
  echo "wrote $out"
}

bench_layout() {
  local n="${N:-50000}"
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
  local n="${N:-50000}"
  local ranks="${RANKS:-8}"
  local threads="${THREADS:-1}"
  local list="${ENERGY_LIST:-1 5 10 $nsteps}"
  local out="${OUT:-energy_overhead.csv}"
  local input="${INPUT:-energy_N${n}.bin}"
  make nbody_direct_hybrid generate_ic >/dev/null
  ./generate_ic --model "$model" --n "$n" --seed "${SEED:-5151}" --output "$input" >/dev/null
  printf "N,nsteps,ranks,threads,repeat,energy_every,total,force,energy,status,max_rel_drift\n" > "$out"
  for ee in $list; do
    for rep in $(seq 1 "$warmups"); do
      run_mpi_solver "$ranks" "$threads" ./nbody_direct_hybrid --input "$input" --nsteps "$nsteps" --dt "$dt" --eps "$eps" --energy-every "$ee" --quiet >/dev/null 2>&1 || true
    done
    for rep in $(seq 1 "$repeats"); do
      set +e
      log="$(run_mpi_solver "$ranks" "$threads" ./nbody_direct_hybrid --input "$input" --nsteps "$nsteps" --dt "$dt" --eps "$eps" --energy-every "$ee" --quiet 2>&1)"
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

bench_container() {
  local image="${IMAGE:-nbody.sif}"
  local out="${OUT:-container_overhead.csv}"
  local launch_out="${LAUNCH_OUT:-container_launch_overhead.csv}"
  local ranks_list="${RANKS:-1 2 4}"
  local threads="${THREADS:-1}"
  local strong_n="${N:-10000}"
  local n_per_rank="${N_PER_RANK:-1000}"
  local launch_repeats="${LAUNCH_REPEATS:-10}"
  local runtime
  runtime="$(detect_runtime)" || { echo "no container runtime found" >&2; exit 127; }
  [[ "$runtime" == "docker" || -f "$image" ]] || { echo "missing image: $image" >&2; exit 1; }
  make nbody_direct_hybrid generate_ic >/dev/null
  printf "kind,mode,N,ranks,threads,repeat,total,status,max_rel_drift\n" > "$out"
  container_case() {
    local kind="$1" mode="$2" n="$3" ranks="$4" rep="$5"
    local input="container_${kind}_N${n}_P${ranks}_seed${rep}.bin"
    local log rc prefix
    ./generate_ic --model "$model" --n "$n" --seed "$((7700 + rep))" --output "$input" >/dev/null
    set +e
    if [[ "$mode" == "native" ]]; then
      log="$(run_mpi_solver "$ranks" "$threads" ./nbody_direct_hybrid --input "$input" --nsteps "$nsteps" --dt "$dt" --eps "$eps" --energy-every "$energy_every" --integrator kdk --comm sendrecv --quiet 2>&1)"
    else
      log="$(run_container_mpi "$runtime" "$image" "$ranks" "$threads" /opt/nbody/nbody_direct_hybrid --input "$input" --nsteps "$nsteps" --dt "$dt" --eps "$eps" --energy-every "$energy_every" --integrator kdk --comm sendrecv --quiet 2>&1)"
    fi
    rc=$?
    set -e
    prefix="$kind,$mode,$n,$ranks,$threads,$rep"
    if (( rc != 0 )); then
      printf "%s,nan,RUN_FAILED,nan\n" "$prefix" >> "$out"
      printf "warning: container case failed %s rc=%s\n%s\n" "$prefix" "$rc" "$log" >&2
    else
      printf "%s\n" "$log" | parse_total_only_csv "$prefix" >> "$out"
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
  awk -v mode="$mode_name" -v bench="$bench" -v metric="$metric" '
    BEGIN { OFS="," }
    /^[[:space:]]*[0-9]+[[:space:]]+/ { print mode, bench, metric, $1, $2 }
  '
}

bench_osu() {
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
    local mode_name="$1" bench="$2" metric="$3" tool="$4"
    local log rc
    set +e
    if [[ "$mode_name" == "native" ]]; then
      log="$("$launcher" $cpu_bind -n 2 "$tool" 2>&1)"
    else
      log="$(run_container_mpi "$runtime" "$image" 2 1 "$tool" 2>&1)"
    fi
    rc=$?
    set -e
    if (( rc != 0 )); then
      printf "warning: OSU %s/%s failed rc=%s\n%s\n" "$mode_name" "$bench" "$rc" "$log" >&2
    else
      printf "%s\n" "$log" | parse_osu "$mode_name" "$bench" "$metric" >> "$out"
    fi
  }
  if [[ "$mode" == "native" || "$mode" == "both" ]]; then
    command -v "$latency" >/dev/null 2>&1 || [[ -x "$latency" ]] || { echo "missing $latency" >&2; exit 1; }
    command -v "$bw" >/dev/null 2>&1 || [[ -x "$bw" ]] || { echo "missing $bw" >&2; exit 1; }
    run_osu_one native latency latency_us "$latency"
    run_osu_one native bandwidth bandwidth_MBps "$bw"
  fi
  if [[ "$mode" == "container" || "$mode" == "both" ]]; then
    [[ -n "$runtime" ]] || { echo "no container runtime found" >&2; exit 127; }
    run_osu_one container latency latency_us "$container_latency"
    run_osu_one container bandwidth bandwidth_MBps "$container_bw"
  fi
  echo "wrote $out"
}

bench_perf() {
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
    --integrator kdk --comm sendrecv --kernel direct --rsqrt exact --quiet
  rm -f "$input"
}

case "$cmd" in
  scaling) bench_scaling ;;
  hybrid) bench_hybrid ;;
  ablation) bench_ablation ;;
  layout) bench_layout ;;
  energy) bench_energy ;;
  container) bench_container ;;
  osu) bench_osu ;;
  perf) bench_perf ;;
  *) echo "unknown command: $cmd" >&2; usage >&2; exit 2 ;;
esac
